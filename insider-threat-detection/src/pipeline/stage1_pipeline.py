#!/usr/bin/env python3
"""
Stage 1: Data Loading and Normalization Pipeline

CERT Insider Threat Detection System
Optimized for: 16 GB RAM + NVIDIA RTX 4060 (8 GB VRAM)

Usage:
    # Full pipeline (skip http.csv for memory efficiency)
    python -m src.pipeline.stage1_pipeline run

    # Include http.csv (requires ~20 GB RAM)
    python -m src.pipeline.stage1_pipeline run --load-http

    # Sample mode (fastest, for testing)
    python -m src.pipeline.stage1_pipeline run --sample

    # Check dataset
    python -m src.pipeline.stage1_pipeline check

    # GPU info
    python -m src.pipeline.stage1_pipeline gpu-info
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ── Project Root ───────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import polars as pl
import torch
import psutil
from dotenv import load_dotenv

# ── Load Environment Variables ────────────────────────────────────────────────
load_dotenv(_PROJECT_ROOT / ".env", override=True)

from src.utils.config import PipelineConfig, get_config
from src.utils.logging import setup_logging, get_logger
from src.utils.helpers import (
    compute_memory_usage,
    format_bytes,
    format_duration,
)
from src.cert_dataset.loaders import CertDatasetLoader
from src.cert_dataset.privacy import PrivacyManager, AuditLogger
from src.cert_dataset import CERT_SCHEMAS

# ── Setup Logging ─────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "console")
setup_logging(level=LOG_LEVEL, log_format=LOG_FORMAT)
logger = get_logger(__name__)


# ── GPU Utilities ─────────────────────────────────────────────────────────────

def check_gpu() -> dict:
    """
    Check GPU availability and return GPU info.

    Returns:
        Dict with GPU status: available, device_name, memory_total,
        memory_available, cuda_available
    """
    info = {
        "available": False,
        "device_name": "CPU",
        "memory_total_mb": 0,
        "memory_available_mb": 0,
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
    }

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        info["available"] = True
        info["device_count"] = device_count

        # Get first GPU (device 0)
        gpu_id = int(os.getenv("TORCH_DEVICE", "cuda").split(":")[-1] if ":" in os.getenv("TORCH_DEVICE", "cuda") else 0)
        info["device_name"] = torch.cuda.get_device_name(gpu_id)
        mem_total = torch.cuda.get_device_properties(gpu_id).total_memory
        info["memory_total_mb"] = round(mem_total / 1_048_576, 0)
        info["memory_available_mb"] = round(
            (mem_total - torch.cuda.memory_allocated(gpu_id)) / 1_048_576, 0
        )

    return info


def print_gpu_info() -> None:
    """Print GPU information to console."""
    info = check_gpu()

    print("\n" + "=" * 60)
    print("GPU Configuration")
    print("=" * 60)

    if info["cuda_available"]:
        print(f"  CUDA Available:  [OK] Yes")
        print(f"  GPU Count:        {info['device_count']}")
        print(f"  GPU 0:           {info['device_name']}")
        print(f"  VRAM Total:      {info['memory_total_mb']:,.0f} MB")
        print(f"  VRAM Available:  {info['memory_available_mb']:,.0f} MB")
    else:
        print(f"  CUDA Available:   [X] No (using CPU)")

    print("=" * 60)


def optimize_memory() -> None:
    """Optimize memory settings for Polars and PyTorch."""
    # Polars memory limit (50% of system RAM)
    system_ram_gb = psutil.virtual_memory().total / 1_073_741_824
    polars_limit_gb = int(system_ram_gb * 0.4)
    os.environ["POLARS_MEMORY_LIMIT"] = f"{polars_limit_gb}GB"

    # PyTorch memory settings
    if torch.cuda.is_available():
        # Enable memory caching
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Set GPU device
        gpu_id = int(os.getenv("TORCH_DEVICE", "cuda").split(":")[-1] if ":" in os.getenv("TORCH_DEVICE", "cuda") else 0)
        torch.cuda.set_device(gpu_id)

    # Force garbage collection
    import gc
    gc.collect()

    logger.info(
        "memory_optimized",
        system_ram_gb=round(system_ram_gb, 1),
        polars_limit_gb=polars_limit_gb,
        gpu_available=check_gpu()["available"],
    )


# ── Dataset Checker ────────────────────────────────────────────────────────────

def check_dataset(config: PipelineConfig) -> dict:
    """
    Check dataset availability and print statistics.

    Returns:
        Dict with file status and sizes
    """
    print("\n" + "=" * 60)
    print("Dataset Check: CERT r4.2")
    print("=" * 60)

    root = config.dataset.root
    ldap = config.dataset.ldap_dir
    answers = config.dataset.answers_dir

    results = {}

    # Check main files
    files = {
        "logon.csv": root / "logon.csv",
        "device.csv": root / "device.csv",
        "file.csv": root / "file.csv",
        "email.csv": root / "email.csv",
        "http.csv": root / "http.csv",
        "psychometric.csv": root / "psychometric.csv",
    }

    print(f"\n  Dataset Root: {root}")
    print(f"  LDAP Dir:     {ldap}")
    print(f"  Answers Dir:  {answers}\n")

    print("  Main CSV Files:")
    print("  " + "-" * 54)
    all_exist = True
    total_size = 0
    for name, path in files.items():
        if path.exists():
            size = path.stat().st_size
            total_size += size
            size_str = format_bytes(size)
            print(f"    [OK] {name:<25} {size_str:>12}")
            results[name] = {"exists": True, "size_bytes": size, "size_str": size_str}
        else:
            print(f"    [X]  {name:<25} NOT FOUND")
            results[name] = {"exists": False, "size_bytes": 0, "size_str": "MISSING"}
            all_exist = False

    # Check LDAP directory
    print(f"\n  LDAP Directory:")
    print("  " + "-" * 54)
    ldap_files = sorted(ldap.glob("*.csv")) if ldap.exists() else []
    if ldap_files:
        print(f"    [OK] {len(ldap_files)} monthly snapshots found")
        for f in ldap_files[:3]:
            print(f"       - {f.name}")
        if len(ldap_files) > 3:
            print(f"       ... and {len(ldap_files) - 3} more")
        results["ldap"] = {"exists": True, "count": len(ldap_files)}
    else:
        print(f"    [X]  No LDAP snapshots found")
        results["ldap"] = {"exists": False, "count": 0}

    # Check ground truth
    gt_path = answers / "insiders.csv"
    print(f"\n  Ground Truth:")
    print("  " + "-" * 54)
    if gt_path.exists():
        print(f"    [OK] insiders.csv found ({format_bytes(gt_path.stat().st_size)})")
        results["ground_truth"] = {"exists": True}
    else:
        print(f"    [!]  insiders.csv not found (evaluation will be limited)")
        results["ground_truth"] = {"exists": False}

    # System resources
    print(f"\n  System Resources:")
    print("  " + "-" * 54)
    mem = psutil.virtual_memory()
    gpu = check_gpu()

    print(f"    RAM Total:    {format_bytes(mem.total)}")
    print(f"    RAM Available: {format_bytes(mem.available)}")
    print(f"    RAM Used:     {format_bytes(mem.used)} ({mem.percent:.0f}%)")
    print(f"    GPU:          {gpu['device_name'] if gpu['available'] else 'Not available'}")
    if gpu["available"]:
        print(f"    VRAM Total:   {gpu['memory_total_mb']:,.0f} MB")

    print(f"\n  Total Dataset Size: {format_bytes(total_size)}")
    print("=" * 60)

    return results


# ── Stage 1: Load and Normalize ──────────────────────────────────────────────

def run_stage1(
    config: PipelineConfig,
    load_http: bool = False,
    http_sample: bool = False,
    output_dir: Path | None = None,
) -> dict:
    """
    Execute Stage 1: Data Loading and Normalization.

    This is the main entry point for the preprocessing pipeline.

    Args:
        config: Pipeline configuration
        load_http: Whether to load http.csv (14.5 GB)
        http_sample: If loading http.csv, load only sample rows
        output_dir: Output directory for normalized parquet files

    Returns:
        Dict with processing statistics
    """
    start_time = time.monotonic()
    stats = {"stages": {}, "total_seconds": 0}

    print("\n" + "=" * 60)
    print("STAGE 1: Data Loading and Normalization")
    print(f"Dataset: CERT r4.2 | Output: {output_dir or config.output.base_dir}")
    print("=" * 60)

    # Optimize memory
    optimize_memory()
    mem_before = compute_memory_usage()
    logger.info("memory_before", rss_mb=mem_before["rss_mb"])

    # Initialize components
    privacy = PrivacyManager(
        salt=config.preprocessing.pseudonymization_salt,
        enable_pseudonymization=config.privacy.pseudonymize,
    )

    audit = AuditLogger(
        log_path=config.privacy.audit_log_path,
        log_format="json",
    )

    loader = CertDatasetLoader(config=config)

    # ── Load Datasets ────────────────────────────────────────────────────────
    print("\n[...] Loading datasets...")

    load_start = time.monotonic()

    # Load all datasets (http.csv optional)
    dfs = loader.load_all(load_http=load_http, http_sample=http_sample)

    load_elapsed = time.monotonic() - load_start
    logger.info(
        "data_loading_complete",
        elapsed_seconds=round(load_elapsed, 1),
        datasets=list(dfs.keys()),
        memory_mb=compute_memory_usage()["rss_mb"],
    )

    print(f"\n[OK] Loaded datasets in {format_duration(load_elapsed)}")
    print(f"   Memory: {compute_memory_usage()['rss_mb']:,.0f} MB")

    # ── Print Loading Summary ────────────────────────────────────────────────
    print("\n[=] Dataset Summary:")
    print("  " + "-" * 54)
    for name, df in dfs.items():
        if isinstance(df, pl.DataFrame) and len(df) > 0:
            mem_mb = df.estimated_size() / 1_048_576
            print(f"    {name:<20} {len(df):>10,} rows  {mem_mb:>8.1f} MB")
        else:
            print(f"    {name:<20} {'N/A':>10}")

    # ── Apply Pseudonymization ────────────────────────────────────────────────
    print("\n[#] Applying pseudonymization...")
    pseudo_start = time.monotonic()

    for name, df in dfs.items():
        if not isinstance(df, pl.DataFrame) or len(df) == 0:
            continue

        if name in ["logon", "device", "file", "email", "http"]:
            # Add pseudonymized columns
            df = privacy.pseudonymize_dataframe(
                df,
                columns={"user": "user", "pc": "pc"},
            )
            dfs[name] = df

    pseudo_elapsed = time.monotonic() - pseudo_start
    print(f"   [OK] Pseudonymization complete in {format_duration(pseudo_elapsed)}")
    print(f"   [#] Hash caches: {privacy.get_cache_stats()}")

    # ── Save Normalized Data ────────────────────────────────────────────────
    if output_dir is None:
        output_dir = config.output.base_dir / config.output.normalized_dir

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[=] Saving normalized data to:")
    print(f"   {output_dir}")

    parquet_compression = config.output.parquet_compression

    save_start = time.monotonic()

    for name, df in dfs.items():
        if not isinstance(df, pl.DataFrame) or len(df) == 0:
            continue

        out_path = output_dir / f"{name}.parquet"
        df.write_parquet(
            out_path,
            compression=parquet_compression,
        )
        size_str = format_bytes(out_path.stat().st_size)
        print(f"   [OK] {name}.parquet -> {size_str}")

        audit.log_data_load(
            source=name,
            rows=len(df),
            file_size_mb=out_path.stat().st_size / 1_048_576,
        )

    save_elapsed = time.monotonic() - save_start
    print(f"\n   [=] Saved {len(dfs)} files in {format_duration(save_elapsed)}")

    # ── Final Stats ─────────────────────────────────────────────────────────
    total_elapsed = time.monotonic() - start_time
    mem_after = compute_memory_usage()

    stats["stages"]["load"] = round(load_elapsed, 1)
    stats["stages"]["pseudo"] = round(pseudo_elapsed, 1)
    stats["stages"]["save"] = round(save_elapsed, 1)
    stats["total_seconds"] = round(total_elapsed, 1)
    stats["memory_peak_mb"] = mem_after["rss_mb"]
    stats["datasets_loaded"] = list(dfs.keys())
    stats["load_stats"] = loader.get_load_stats()

    # ── Print Final Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 1 COMPLETE")
    print("=" * 60)
    print(f"  Total Time:    {format_duration(total_elapsed)}")
    print(f"  Peak Memory:   {mem_after['rss_mb']:,.0f} MB")
    print(f"  GPU Used:      {'[OK] Yes' if check_gpu()['available'] else '[X] No'}")
    print(f"  Output Dir:   {output_dir}")
    print("=" * 60)

    # Log to audit
    audit.logger.info(f"STAGE1_COMPLETE | elapsed={total_elapsed:.1f}s | "
                      f"memory_peak={mem_after['rss_mb']:.0f}MB")

    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main entry point for the Stage 1 pipeline CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CERT r4.2 Data Loading and Normalization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check dataset availability
  python -m src.pipeline.stage1_pipeline check

  # Check GPU configuration
  python -m src.pipeline.stage1_pipeline gpu-info

  # Run full pipeline (skip http.csv for memory efficiency)
  python -m src.pipeline.stage1_pipeline run

  # Run with http.csv (requires ~20 GB RAM)
  python -m src.pipeline.stage1_pipeline run --load-http

  # Fast test run with sampling
  python -m src.pipeline.stage1_pipeline run --sample

  # Custom output directory
  python -m src.pipeline.stage1_pipeline run --output ./my_data
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # check command
    subparsers.add_parser("check", help="Check dataset availability")

    # gpu-info command
    subparsers.add_parser("gpu-info", help="Show GPU configuration")

    # run command
    run_parser = subparsers.add_parser("run", help="Run the preprocessing pipeline")
    run_parser.add_argument(
        "--load-http",
        action="store_true",
        help="Load http.csv (14.5 GB, requires ~20 GB RAM)",
    )
    run_parser.add_argument(
        "--sample",
        action="store_true",
        help="Run in sample mode (fast, for testing)",
    )
    run_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for normalized parquet files",
    )
    run_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: configs/config.yaml)",
    )

    args = parser.parse_args()

    # Default command
    if args.command is None:
        args.command = "run"

    # Load config
    config_path = Path(getattr(args, "config", None)) if getattr(args, "config", None) else None
    config = PipelineConfig.from_yaml(config_path)

    # Execute command
    if args.command == "check":
        check_dataset(config)

    elif args.command == "gpu-info":
        print_gpu_info()

    elif args.command == "run":
        # Determine http.csv loading strategy
        load_http = getattr(args, "load_http", False)
        http_sample = getattr(args, "sample", False)

        # In sample mode, always load http.csv with sampling
        if args.sample:
            load_http = True
            http_sample = True

        output_dir = Path(args.output) if args.output else None

        stats = run_stage1(
            config=config,
            load_http=load_http,
            http_sample=http_sample,
            output_dir=output_dir,
        )

        print(f"\n[i] Pipeline Statistics:")
        print(f"   Load time:     {stats['stages'].get('load', 0):.1f}s")
        print(f"   Pseudo time:   {stats['stages'].get('pseudo', 0):.1f}s")
        print(f"   Save time:     {stats['stages'].get('save', 0):.1f}s")
        print(f"   Peak memory:   {stats['memory_peak_mb']:,.0f} MB")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
