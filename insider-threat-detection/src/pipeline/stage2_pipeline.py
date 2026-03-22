#!/usr/bin/env python3
"""
Stage 2: Feature Engineering Pipeline

Orchestrates StatisticalFeatures, SequenceEncoder, and GraphBuilder to produce
the full feature set for insider-threat-detection training and evaluation.

Optimized for: 16 GB RAM + NVIDIA RTX 4060 (8 GB VRAM)

Output files (all under data/features/):
  user_features_daily.parquet       one row per user per day  (~365,000 rows)
  user_features_weekly.parquet       suffix _7d
  user_features_monthly.parquet      suffix _30d
  daily_sequences.parquet            (user, date, sequence_tensor_path)
  graph_monthly_snapshots/graph_YYYY_MM.pt
  hourly_profiles.parquet           hour_0 … hour_23 per user
  day_of_week_profiles.parquet      day_0 … day_6  per user
  ground_truth.parquet              with user_hash join key

CLI:
  python -m src.pipeline.stage2_pipeline run
  python -m src.pipeline.stage2_pipeline run --windows 24h,7d,30d
  python -m src.pipeline.stage2_pipeline check
"""

from __future__ import annotations

import gc
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ── Project Root ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import polars as pl
import torch
import psutil
from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env", override=True)

from src.utils.config import PipelineConfig, get_config
from src.utils.logging import setup_logging, get_logger
from src.utils.helpers import (
    compute_memory_usage,
    format_bytes,
    format_duration,
)
from src.features.statistical import (
    compute_all_window_features,
    create_user_date_ground_truth,
    merge_psychometric_features,
    merge_ldap_features,
    extend_features_to_windows,
    compute_behavioral_drift_score,
    ensure_all_columns,
    WINDOWS,
    StatisticalFeatures,
    WindowConfig,
)
from src.features.sequence_encoder import SequenceEncoder, EncoderConfig
from src.features.graph_builder import GraphBuilder

# ── Logging ─────────────────────────────────────────────────────────────────────
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    log_format=os.getenv("LOG_FORMAT", "console"),
    log_dir=_PROJECT_ROOT / "logs",
)
logger = get_logger(__name__)


# ==============================================================================
# Data Loading
# ==============================================================================

def load_normalized_parquet(base_dir: Path) -> dict[str, pl.DataFrame]:
    """
    Load every normalized parquet file emitted by Stage 1.

    Expected files (all under base_dir):
      logon, device, file, email, http, ldap, psychometric, ground_truth
    """
    logger.info("loading_normalized_parquet", path=str(base_dir))
    dfs: dict[str, pl.DataFrame] = {}
    expected = ["logon", "device", "file", "email", "http", "ldap", "psychometric"]

    for name in expected:
        p = base_dir / f"{name}.parquet"
        if p.exists():
            dfs[name] = pl.scan_parquet(p).collect()
            n = len(dfs[name])
            mb = dfs[name].estimated_size() / 1_048_576
            logger.info("file_loaded", name=name, rows=n, size_mb=round(mb, 1))
        else:
            logger.warning("file_missing", name=name)

    gt_path = base_dir / "ground_truth.parquet"
    if gt_path.exists():
        dfs["ground_truth"] = pl.scan_parquet(gt_path).collect()
        logger.info("ground_truth_loaded", rows=len(dfs["ground_truth"]))
    else:
        logger.warning("ground_truth_missing")

    return dfs


# ==============================================================================
# Ground Truth Builder
# ==============================================================================

def build_ground_truth(
    gt_raw: pl.DataFrame,
    users: pl.Series,
    date_range: tuple[datetime, datetime],
) -> pl.DataFrame:
    """
    Expand ground-truth insider labels into a user × date fact-table.

    CERT insiders.csv format (r4.2):
      user, start, end, type
    We expand every calendar day in [start, end] so that downstream
    features can LEFT-JOIN on (user, date) without any date-range logic.

    Output columns: user_hash, date, is_insider, threat_type
    """
    start_dt, end_dt = date_range

    if gt_raw is None or len(gt_raw) == 0:
        # All-benign table covering every user × date
        all_dates = pl.date_range(
            start=start_dt, end=end_dt, interval="1d", eager=True
        ).alias("date")
        return (
            users.to_frame("user_hash")
            .join(all_dates, how="cross")
            .with_columns([
                pl.lit(0).alias("is_insider"),
                pl.lit("none").alias("threat_type"),
            ])
            .sort(["user_hash", "date"])
        )

    # Parse start/end as dates (handle mixed formats)
    # Convert using a format hint or fallback to strict=False
    gt = gt_raw.with_columns([
        pl.col("start").str.to_datetime(strict=False).dt.date().alias("start_date"),
        pl.col("end").str.to_datetime(strict=False).dt.date().alias("end_date"),
    ])

    rows = []
    for row in gt.iter_rows(named=True):
        u = row["user"]
        s = row["start_date"]
        e = row["end_date"]
        t = row.get("type", "unknown")
        if s is None or e is None:
            continue
        # Expand date range
        n_days = (e - s).days + 1
        dates = [s + timedelta(days=d) for d in range(n_days)]
        for d in dates:
            rows.append({"user_hash": u, "date": d, "is_insider": 1, "threat_type": t})

    gt_expanded = (
        pl.DataFrame(rows)
        .unique(subset=["user_hash", "date"])
    )

    # Cross-join with all users to get non-insider rows
    all_dates_df = pl.date_range(
        start=start_dt, end=end_dt, interval="1d", eager=True
    ).to_frame("date")

    full = (
        users.to_frame("user_hash")
        .join(all_dates_df, how="cross")
        .join(gt_expanded, on=["user_hash", "date"], how="left")
        .with_columns([
            pl.col("is_insider").fill_null(0).cast(pl.UInt8),
            pl.col("threat_type").fill_null("none"),
        ])
        .sort(["user_hash", "date"])
    )

    return full


# ==============================================================================
# Feature Computation
# ==============================================================================

def compute_user_features_for_window(
    feature_calc: StatisticalFeatures,
    dfs: dict[str, pl.DataFrame],
    window: str,
    date_col: str = "window_start",
) -> pl.DataFrame:
    """
    Compute and merge all per-user-per-window statistical features.

    Returns a DataFrame with columns:
      user, window_start, <logon features>, <device features>, …
    """
    suffix = f"_{window}"
    cfg = WindowConfig.from_str(window)

    # Logon
    logon_feat: pl.DataFrame = pl.DataFrame()
    if "logon" in dfs and len(dfs["logon"]) > 0:
        logon_feat = feature_calc.compute_logon_features(dfs["logon"], window)

    # Device
    device_feat: pl.DataFrame = pl.DataFrame()
    if "device" in dfs and len(dfs["device"]) > 0:
        device_feat = feature_calc.compute_device_features(dfs["device"], window)

    # File
    file_feat: pl.DataFrame = pl.DataFrame()
    if "file" in dfs and len(dfs["file"]) > 0:
        file_feat = feature_calc.compute_file_features(dfs["file"], window)

    # Email
    email_feat: pl.DataFrame = pl.DataFrame()
    if "email" in dfs and len(dfs["email"]) > 0:
        email_feat = feature_calc.compute_email_features(dfs["email"], window)

    # HTTP
    http_feat: pl.DataFrame = pl.DataFrame()
    if "http" in dfs and len(dfs["http"]) > 0:
        http_feat = feature_calc.compute_http_features(dfs["http"], window)

    # Merge
    merged = feature_calc.merge_all_features(
        logon_df=logon_feat,
        device_df=device_feat if len(device_feat) > 0 else None,
        file_df=file_feat if len(file_feat) > 0 else None,
        email_df=email_feat if len(email_feat) > 0 else None,
        http_df=http_feat if len(http_feat) > 0 else None,
        window=window,
    )

    # Rename window_start → date for consistency
    if "window_start" in merged.columns:
        merged = merged.rename({"window_start": "date"})

    # Drop any internal helper columns
    drop_cols = [c for c in merged.columns if c.startswith("_")]
    if drop_cols:
        merged = merged.drop(drop_cols)

    return merged


def compute_hourly_profiles(
    feature_calc: StatisticalFeatures,
    dfs: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """Compute 24-bin hourly activity distribution per user."""
    if "logon" not in dfs or len(dfs["logon"]) == 0:
        return pl.DataFrame()

    df = dfs["logon"]
    hourly = feature_calc.compute_hourly_profile(df)

    # Pivot so columns are hour_0 … hour_23
    hour_cols = {}
    for c in hourly.columns:
        if c == "user":
            continue
        try:
            h = int(c)
            hour_cols[c] = f"hour_{h}"
        except ValueError:
            pass

    if hour_cols:
        hourly = hourly.rename(hour_cols)

    # Fill missing hours with 0
    for h in range(24):
        col = f"hour_{h}"
        if col not in hourly.columns:
            hourly = hourly.with_columns(pl.lit(0).alias(col))

    # Reorder
    ordered = ["user"] + [f"hour_{h}" for h in range(24)]
    return hourly.select(ordered)


def compute_day_of_week_profiles(
    dfs: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """
    Compute 7-bin day-of-week activity distribution per user.

    Columns: user, day_0 (Monday) … day_6 (Sunday)
    """
    if "logon" not in dfs or len(dfs["logon"]) == 0:
        return pl.DataFrame()

    df = dfs["logon"]

    dow = (
        df.lazy()
        .with_columns(
            pl.col("timestamp").dt.weekday().alias("day_of_week")
        )
        .group_by(["user", "day_of_week"])
        .agg(pl.col("id").count().alias("activity_count"))
        .pivot(
            values="activity_count",
            index="user",
            columns="day_of_week",
            aggregate_function="first",
        )
        .fill_null(0)
        .collect()
    )

    # Rename day columns
    rename_map = {}
    for c in dow.columns:
        if c == "user":
            continue
        try:
            d = int(c)
            rename_map[c] = f"day_{d}"
        except ValueError:
            pass

    if rename_map:
        dow = dow.rename(rename_map)

    # Ensure all 7 day columns exist
    for d in range(7):
        col = f"day_{d}"
        if col not in dow.columns:
            dow = dow.with_columns(pl.lit(0).alias(col))

    ordered = ["user"] + [f"day_{d}" for d in range(7)]
    return dow.select(ordered)


# ==============================================================================
# Sequence Encoding
# ==============================================================================

def encode_daily_sequences(
    encoder: SequenceEncoder,
    dfs: dict[str, pl.DataFrame],
    output_dir: Path,
) -> pl.DataFrame:
    """
    Encode all user event sequences as daily fixed-length tensors.

    Returns a DataFrame with columns:
      user, date, seq_path  (seq_path = path to .pt tensor on disk)
    """
    # Use the encoder's encode() method with the dfs dictionary
    print("Encoding sequences using encoder.encode()...")
    seq_df = encoder.encode(dfs)

    # Create sequence tensor directory
    seq_dir = output_dir / "sequence_tensors"
    seq_dir.mkdir(parents=True, exist_ok=True)

    # Return the dataframe (encoder already handles tensor generation)
    return seq_df


# ==============================================================================
# Graph Construction
# ==============================================================================

def build_monthly_graph_snapshots(
    graph_builder: BehavioralGraphBuilder,
    dfs: dict[str, pl.DataFrame],
    output_dir: Path,
    date_range: tuple[datetime, datetime],
) -> list[Path]:
    """
    Build one HeteroData graph snapshot per calendar month.

    Each snapshot covers all events in that month and is saved as
    graph_YYYY_MM.pt under graph_monthly_snapshots/.
    """
    snapshot_dir = output_dir / "graph_monthly_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    start_dt, end_dt = date_range
    year_months: list[tuple[int, int]] = []
    cur = datetime(start_dt.year, start_dt.month, 1)
    while cur <= end_dt:
        year_months.append((cur.year, cur.month))
        # Advance to next month
        if cur.month == 12:
            cur = datetime(cur.year + 1, 1, 1)
        else:
            cur = datetime(cur.year, cur.month + 1, 1)

    saved_paths: list[Path] = []
    logon_df = dfs.get("logon")
    http_df = dfs.get("http")

    for year, month in year_months:
        month_start = datetime(year, month, 1)
        if month == 12:
            month_end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
        else:
            month_end = datetime(year, month + 1, 1) - timedelta(seconds=1)

        # Filter logon to this month
        if logon_df is not None and len(logon_df) > 0:
            month_logon = logon_df.filter(
                pl.col("timestamp") >= month_start,
                pl.col("timestamp") <= month_end,
            )
        else:
            month_logon = pl.DataFrame()

        if http_df is not None and len(http_df) > 0:
            month_http = http_df.filter(
                pl.col("timestamp") >= month_start,
                pl.col("timestamp") <= month_end,
            )
        else:
            month_http = None

        if len(month_logon) == 0:
            logger.warning("skipping_empty_month", year=year, month=month)
            continue

        hetero_data = graph_builder.build_hetero_data(
            logon_df=month_logon,
            http_df=month_http,
        )

        out_path = snapshot_dir / f"graph_{year}_{month:02d}.pt"
        torch.save(hetero_data, out_path)
        saved_paths.append(out_path)
        logger.info(
            "graph_snapshot_saved",
            year=year, month=month,
            nodes=user_count_from_data(hetero_data),
            path=str(out_path),
        )

    return saved_paths


def user_count_from_data(data: Any) -> int:
    """Safely extract user node count from a HeteroData object."""
    try:
        return data["user"].x.shape[0]  # type: ignore[union-attr]
    except Exception:
        return 0


# ==============================================================================
# Main Pipeline
# ==============================================================================

def run_stage2(
    data_dir: Path,
    output_dir: Path,
    windows: list[str] = ["24h", "7d", "30d"],
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    """
    Execute Stage 2: Feature Engineering.

    Steps:
      1. Load normalized parquet from Stage 1
      2. Compute per-window user features (daily / weekly / monthly)
      3. Encode daily event sequences
      4. Build monthly graph snapshots
      5. Compute hourly and day-of-week profiles
      6. Build ground-truth fact table
      7. Write all outputs

    Args:
        data_dir:  Path to Stage 1 output (normalized/)
        output_dir: Path to write feature outputs (features/)
        windows:    Time windows for statistical features
        config:     Optional PipelineConfig (loaded from YAML if None)

    Returns:
        Stats dict with timing, row counts, memory usage
    """
    start_time = time.monotonic()
    config = config or get_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "windows": {},
        "outputs": {},
        "total_seconds": 0,
    }

    logger.info(
        "stage2_started",
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        windows=windows,
    )

    # ── 1. Load Normalized Data ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 2: Feature Engineering")
    print("=" * 60)

    load_start = time.monotonic()
    dfs = load_normalized_parquet(data_dir)
    load_elapsed = time.monotonic() - load_start
    mem_now = compute_memory_usage()
    print(
        f"\n[OK] Loaded {len(dfs)} datasets in {format_duration(load_elapsed)}"
        f" | MEM: {mem_now['rss_mb']:,.0f} MB"
    )

    # Determine date range from logon data (or config)
    if "logon" in dfs and len(dfs["logon"]) > 0:
        ts_col = dfs["logon"]["timestamp"]
        date_min = ts_col.min()
        date_max = ts_col.max()
    else:
        date_min = datetime.fromisoformat(config.dataset.start_date)
        date_max = datetime.fromisoformat(config.dataset.end_date)

    date_range = (date_min, date_max)

    # Collect all unique users across datasets
    all_users: list[str] = []
    for name in ["logon", "device", "file", "email", "http"]:
        if name in dfs and len(dfs[name]) > 0:
            all_users.extend(dfs[name]["user"].unique().to_list())
    unique_users = list(set(all_users))
    print(f"    Unique users: {len(unique_users):,}")
    del all_users, unique_users
    gc.collect()

    # ── 2. Statistical Features ──────────────────────────────────────────────
    feature_calc = StatisticalFeatures(
        working_hours_start=config.preprocessing.working_hours.start_hour,
        working_hours_end=config.preprocessing.working_hours.end_hour,
        work_days=list(config.preprocessing.working_hours.work_days),
    )

    window_suffix_map = {
        "24h": "daily",
        "7d": "weekly",
        "30d": "monthly",
    }

    for window in windows:
        w_start = time.monotonic()
        suffix = window_suffix_map.get(window, window)
        out_file = output_dir / f"user_features_{suffix}.parquet"

        print(f"\n[>] Window: {window} -> {out_file.name}")
        features = compute_user_features_for_window(feature_calc, dfs, window)

        features.write_parquet(
            out_file,
            compression=config.output.parquet_compression,
        )
        w_elapsed = time.monotonic() - w_start
        mb = out_file.stat().st_size / 1_048_576

        stats["windows"][window] = {
            "rows": len(features),
            "cols": len(features.columns),
            "seconds": round(w_elapsed, 1),
            "output_mb": round(mb, 1),
        }
        print(
            f"    {len(features):,} rows × {len(features.columns)} cols"
            f" | {format_duration(w_elapsed)} | {mb:.1f} MB"
        )
        logger.info(
            "window_features_saved",
            window=window,
            rows=len(features),
            cols=len(features.columns),
            path=str(out_file),
            elapsed_s=w_elapsed,
        )
        del features
        gc.collect()

    # ── 3. Ground Truth ─────────────────────────────────────────────────────
    print("\n[>] Building ground-truth table …")
    gt_start = time.monotonic()

    # Collect unique users from all sources
    all_users_list = []
    for name in ["logon", "device", "file", "email", "http"]:
        if name in dfs and len(dfs[name]) > 0:
            users = dfs[name]["user_hash"].unique().to_list()
            all_users_list.extend(users)
    gt_users = pl.Series("user_hash", list(set(all_users_list)))

    gt_df = build_ground_truth(
        gt_raw=dfs.get("ground_truth"),
        users=gt_users,
        date_range=date_range,
    )
    del gt_users
    gc.collect()

    gt_path = output_dir / "ground_truth.parquet"
    gt_df.write_parquet(gt_path, compression=config.output.parquet_compression)
    gt_elapsed = time.monotonic() - gt_start
    stats["outputs"]["ground_truth"] = {
        "rows": len(gt_df),
        "cols": len(gt_df.columns),
        "seconds": round(gt_elapsed, 1),
    }
    print(
        f"    {len(gt_df):,} rows × {len(gt_df.columns)} cols"
        f" | {format_duration(gt_elapsed)}"
    )
    logger.info("ground_truth_saved", rows=len(gt_df), path=str(gt_path))
    del gt_df
    gc.collect()

    # ── 4. Daily Sequences (SKIPPED - requires schema fixes) ──────────────────
    print("\n[>] Encoding daily sequences … (SKIPPED - requires schema alignment)")
    stats["outputs"]["daily_sequences"] = {
        "status": "skipped",
        "reason": "Sequence encoder requires column schema alignment",
    }

    # ── 5. Monthly Graph Snapshots (SKIPPED - requires PyG fix) ───────────────
    print("\n[>] Building monthly graph snapshots … (SKIPPED - PyG not fully installed)")
    stats["outputs"]["graph_snapshots"] = {
        "status": "skipped",
        "reason": "PyTorch Geometric extensions not properly loaded",
    }

    # ── 6. Final Output Summary ──────────────────────────────────────────────
    total_elapsed = time.monotonic() - start_time
    stats["memory_peak_mb"] = stats.get("memory_peak_mb", 0)

    print("\n" + "=" * 60)
    print("STAGE 2 COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    for name, info in stats["outputs"].items():
        if "path" in info:
            print(f"  {name}: {info['rows']:,} rows × {info['cols']} cols | {info.get('seconds', 0)}s")
        elif "status" in info:
            print(f"  {name}: {info['status']} - {info.get('reason', '')}")

    print(f"\nTotal time: {format_duration(total_elapsed)}")
    print(f"Peak memory: {stats['memory_peak_mb']:,.0f} MB")

    logger.info(
        "stage2_complete",
        total_seconds=round(total_elapsed, 1),
        peak_memory_mb=stats['memory_peak_mb'],
        outputs=stats['outputs'],
    )

    return stats


# ==============================================================================
# CLI
# ==============================================================================

def check_outputs(output_dir: Path) -> dict[str, Any]:
    """Check feature outputs and print a human-readable report."""
    results: dict[str, Any] = {}

    print("\n" + "=" * 60)
    print("Stage 2 Output Check")
    print("=" * 60)
    print(f"\n  Output directory: {output_dir}\n")

    # Expected files
    expected = [
        "user_features_daily.parquet",
        "user_features_weekly.parquet",
        "user_features_monthly.parquet",
        "daily_sequences.parquet",
        "hourly_profiles.parquet",
        "day_of_week_profiles.parquet",
        "ground_truth.parquet",
    ]

    print("  Feature files:")
    print("  " + "-" * 54)
    all_ok = True
    for fname in expected:
        p = output_dir / fname
        if p.exists():
            size = p.stat().st_size
            n_rows = _count_parquet_rows(p)
            print(f"    [OK]  {fname:<38} {n_rows:>8,} rows  {format_bytes(size):>10}")
            results[fname] = {"exists": True, "rows": n_rows, "size_bytes": size}
        else:
            print(f"    [--]  {fname:<38} NOT FOUND")
            results[fname] = {"exists": False}
            all_ok = False

    # Graph snapshots
    graph_dir = output_dir / "graph_monthly_snapshots"
    print(f"\n  Graph snapshots:  {graph_dir}")
    print("  " + "-" * 54)
    if graph_dir.exists():
        graphs = sorted(graph_dir.glob("graph_????_??.pt"))
        if graphs:
            for g in graphs:
                size = g.stat().st_size
                print(f"    [OK]  {g.name:<38} {format_bytes(size):>10}")
            results["graph_snapshots"] = {
                "count": len(graphs),
                "files": [g.name for g in graphs],
            }
        else:
            print("    [--]  No graph snapshot files found")
            all_ok = False
    else:
        print("    [--]  Directory not found")
        all_ok = False

    # Sequence tensors
    seq_dir = output_dir / "sequence_tensors"
    print(f"\n  Sequence tensors: {seq_dir}")
    print("  " + "-" * 54)
    if seq_dir.exists():
        tensors = list(seq_dir.glob("seq_*.pt"))
        print(f"    [OK]  {len(tensors):,} tensor files")
        results["sequence_tensors"] = {"count": len(tensors)}
    else:
        print("    [--]  Directory not found")
        all_ok = False

    print("\n  " + "=" * 54)
    if all_ok:
        print("  All expected outputs present.")
    else:
        print("  WARNING: Some expected outputs are missing.")
    print("=" * 60)

    return results


def _count_parquet_rows(path: Path) -> int:
    """Count rows in a parquet file without loading all data."""
    try:
        return pl.scan_parquet(path).select(pl.len()).collect().item()
    except Exception:
        return -1


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 2: Feature Engineering Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.pipeline.stage2_pipeline run
  python -m src.pipeline.stage2_pipeline run --windows 24h,7d,30d
  python -m src.pipeline.stage2_pipeline check
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # run
    run_parser = subparsers.add_parser("run", help="Run the feature pipeline")
    run_parser.add_argument(
        "--windows",
        default="24h,7d,30d",
        help="Comma-separated time windows (default: 24h,7d,30d)",
    )
    run_parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Stage 1 output directory (normalized/)",
    )
    run_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Stage 2 output directory (features/)",
    )
    run_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml",
    )

    # check
    check_parser = subparsers.add_parser("check", help="Check feature outputs")
    check_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Feature output directory to check",
    )

    args = parser.parse_args()

    # Default to "run" if no subcommand
    if args.command is None:
        args.command = "run"

    # Load config
    config_path = Path(args.config) if getattr(args, "config", None) else None
    config = PipelineConfig.from_yaml(config_path)

    if args.command == "check":
        out_dir = Path(args.output_dir) if args.output_dir else (
            config.output.base_dir / "features"
        )
        check_outputs(out_dir)

    elif args.command == "run":
        windows = [w.strip() for w in args.windows.split(",")]

        data_dir = (
            Path(args.data_dir) if args.data_dir
            else config.output.base_dir / config.output.normalized_dir
        )
        out_dir = (
            Path(args.output_dir) if args.output_dir
            else config.output.base_dir / "features"
        )

        if not data_dir.exists():
            print(f"[X] Error: Stage 1 output not found: {data_dir}")
            print("    Run Stage 1 first: python -m src.pipeline.stage1_pipeline run")
            sys.exit(1)

        # Validate windows
        valid = {"24h", "7d", "30d"}
        for w in windows:
            if w not in valid:
                print(f"[X] Invalid window '{w}'. Valid: {sorted(valid)}")
                sys.exit(1)

        stats = run_stage2(
            data_dir=data_dir,
            output_dir=out_dir,
            windows=windows,
            config=config,
        )

        # Quick check
        check_outputs(out_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
