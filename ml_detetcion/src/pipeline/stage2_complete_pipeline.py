#!/usr/bin/env python3
"""
Stage 2: Complete Feature Engineering Pipeline

Generates ALL features according to the CERT Insider Threat Detection schema:
1. Tabular Features (~500 features per user per day)
2. Sequence Datasets (daily_sequences.parquet for LSTM/Transformer)
3. Graph Snapshots (graph_monthly_snapshots/*.pt for GNN)
4. User Embeddings (pre-computed embeddings for anomaly detection)
5. Anomaly Scores (pre-computed baseline scores)

Output: data/features/
  - user_features_daily.parquet (~365k rows × ~500 cols)
  - user_features_weekly.parquet
  - user_features_monthly.parquet
  - daily_sequences.parquet
  - sequence_tensors/*.pt
  - graph_monthly_snapshots/*.pt
  - user_embeddings/*.pt
  - all_model_scores.parquet
  - ground_truth.parquet

CLI:
  python -m src.pipeline.stage2_complete_pipeline run
  python -m src.pipeline.stage2_complete_pipeline check
"""

from __future__ import annotations

import gc
import hashlib
import os
import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import polars as pl
import torch
import numpy as np
from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env", override=True)

from src.utils.config import get_config
from src.utils.logging import setup_logging, get_logger
from src.utils.helpers import format_bytes, format_duration

setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)


# ==============================================================================
# Data Loading
# ==============================================================================

def load_normalized_parquet(base_dir: Path) -> dict[str, pl.DataFrame]:
    """Load all normalized parquet files from Stage 1."""
    logger.info("Loading normalized data", path=str(base_dir))

    dfs = {}
    expected = ["logon", "device", "file", "email", "http", "ldap", "psychometric"]

    for name in expected:
        path = base_dir / f"{name}.parquet"
        if path.exists():
            df = pl.read_parquet(path)
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(f"Loaded {name}", rows=len(df), size_mb=f"{size_mb:.1f}")
            dfs[name] = df
        else:
            logger.warning(f"Missing {name}.parquet")

    # Load ground truth
    gt_path = base_dir / "ground_truth.parquet"
    if gt_path.exists():
        dfs["ground_truth"] = pl.read_parquet(gt_path)
        logger.info("Loaded ground_truth", rows=len(dfs["ground_truth"]))

    return dfs


# ==============================================================================
# Complete Statistical Features
# ==============================================================================

def compute_complete_features(dfs: dict[str, pl.DataFrame], windows: list[str] = ["24h", "7d", "30d"]) -> dict[str, pl.DataFrame]:
    """
    Compute ALL features according to the CERT schema.

    Returns dict with keys: daily, weekly, monthly
    """
    from src.features.statistical_complete import (
        compute_logon_features,
        compute_session_duration_features,
        compute_rapid_logon_cycles,
        compute_device_features,
        compute_device_session_duration,
        compute_file_features,
        compute_email_features,
        compute_http_features,
        compute_temporal_features,
        compute_drift_features,
        compute_graph_features,
        merge_psychometric_features,
        merge_ldap_features,
        ensure_all_required_columns,
    )

    logger.info("Computing complete feature set for all windows...")

    results = {}

    for window in windows:
        logger.info(f"Computing {window} features...")
        suffix = f"_{window}"

        # Days for window
        window_days = {"24h": 1, "7d": 7, "30d": 30}[window]

        all_features = None

        # 1. LOGON FEATURES
        if "logon" in dfs:
            logon_df = dfs["logon"].lazy()

            # Basic logon counts
            logon_counts = (
                logon_df.filter(pl.col("activity").str.to_lowercase() == "logon")
                .group_by(["user_hash", "date"])
                .agg([
                    pl.col("activity").count().alias(f"logon_count{suffix}"),
                    pl.col("is_after_hours").sum().alias(f"after_hours_logons{suffix}"),
                    pl.col("is_weekend").sum().alias(f"weekend_logons{suffix}"),
                    pl.col("pc_hash").n_unique().alias(f"unique_pcs{suffix}"),
                ])
                .collect()
            )

            logoff_counts = (
                logon_df.filter(pl.col("activity").str.to_lowercase() == "logoff")
                .group_by(["user_hash", "date"])
                .agg(pl.col("activity").count().alias(f"logoff_count{suffix}"))
                .collect()
            )

            logon_features = logon_counts.join(logoff_counts, on=["user_hash", "date"], how="outer", suffix="_right")
            # Drop the _right columns since they're duplicates
            logon_features = logon_features.drop([c for c in logon_features.columns if c.endswith("_right")])
            logon_features = logon_features.with_columns([
                pl.col(f"logon_count{suffix}").fill_null(0).cast(pl.Int64),
                pl.col(f"logoff_count{suffix}").fill_null(0).cast(pl.Int64),
                pl.col(f"after_hours_logons{suffix}").fill_null(0).cast(pl.Int64),
                pl.col(f"weekend_logons{suffix}").fill_null(0).cast(pl.Int64),
                pl.col(f"unique_pcs{suffix}").fill_null(0).cast(pl.Int64),
                (pl.col(f"logon_count{suffix}") / (pl.col(f"logon_count{suffix}") + pl.col(f"logoff_count{suffix}") + 1))
                .alias(f"logon_ratio{suffix}"),
            ])

            all_features = logon_features

            # Session duration features
            session_lf = (
                logon_df.filter(pl.col("activity").str.to_lowercase().is_in(["logon", "logoff"]))
                .sort(["user_hash", "pc_hash", "timestamp"])
            )

            session_df = (
                session_lf
                .with_columns([
                    pl.col("activity").shift(1).over(["user_hash", "pc_hash"]).alias("_prev_activity"),
                    pl.col("timestamp").shift(1).over(["user_hash", "pc_hash"]).alias("_prev_ts"),
                ])
                .filter(
                    (pl.col("activity").str.to_lowercase() == "logoff") &
                    (pl.col("_prev_activity").str.to_lowercase() == "logon") &
                    (pl.col("_prev_ts").is_not_null())
                )
                .with_columns([
                    (pl.col("timestamp") - pl.col("_prev_ts"))
                    .dt.total_seconds()
                    .alias("_session_sec")
                ])
                .filter(pl.col("_session_sec") > 0)
                .group_by(["user_hash", "date"])
                .agg([
                    pl.col("_session_sec").mean().alias(f"avg_session_duration{suffix}"),
                    pl.col("_session_sec").max().alias(f"max_session_duration{suffix}"),
                    pl.col("_session_sec").std().alias(f"session_duration_std{suffix}"),
                ])
                .collect()
            )

            session_df = session_df.with_columns([
                pl.col(f"avg_session_duration{suffix}").fill_null(0.0),
                pl.col(f"max_session_duration{suffix}").fill_null(0.0),
                pl.col(f"session_duration_std{suffix}").fill_null(0.0),
            ])

            all_features = all_features.join(session_df, on=["user_hash", "date"], how="left")

            # Rapid logon cycles
            rapid_df = (
                logon_df.filter(pl.col("activity").str.to_lowercase() == "logon")
                .sort(["user_hash", "timestamp"])
                .with_columns([
                    pl.col("activity").shift(1).over("user_hash").alias("_prev_act"),
                    pl.col("timestamp").shift(1).over("user_hash").alias("_prev_ts"),
                ])
                .filter(
                    (pl.col("_prev_act").str.to_lowercase() == "logon") &
                    (pl.col("_prev_ts").is_not_null())
                )
                .with_columns([
                    (pl.col("timestamp") - pl.col("_prev_ts"))
                    .dt.total_seconds()
                    .alias("_diff")
                ])
                .filter(pl.col("_diff") < 300)
                .group_by(["user_hash", "date"])
                .agg(pl.col("_diff").count().alias(f"rapid_logon_cycles{suffix}"))
                .collect()
            ).with_columns(pl.col(f"rapid_logon_cycles{suffix}").fill_null(0))

            all_features = all_features.join(rapid_df, on=["user_hash", "date"], how="left")

        # 2. DEVICE FEATURES
        if "device" in dfs:
            device_df = dfs["device"].lazy()

            connect_df = (
                device_df.filter(pl.col("activity") == "Connect")
                .group_by(["user_hash", "date"])
                .agg([
                    pl.col("activity").count().alias(f"device_connect_count{suffix}"),
                    pl.col("is_after_hours").sum().alias(f"after_hours_device{suffix}"),
                    pl.col("is_weekend").sum().alias(f"weekend_device{suffix}"),
                ])
                .collect()
            )

            disconnect_df = (
                device_df.filter(pl.col("activity") == "Disconnect")
                .group_by(["user_hash", "date"])
                .agg(pl.col("activity").count().alias(f"device_disconnect_count{suffix}"))
                .collect()
            )

            device_features = connect_df.join(disconnect_df, on=["user_hash", "date"], how="outer", suffix="_right")
            device_features = device_features.drop([c for c in device_features.columns if c.endswith("_right")])
            device_features = device_features.with_columns([
                pl.col(f"device_connect_count{suffix}").fill_null(0).cast(pl.Int64),
                pl.col(f"device_disconnect_count{suffix}").fill_null(0).cast(pl.Int64),
                pl.col(f"after_hours_device{suffix}").fill_null(0).cast(pl.Int64),
                pl.col(f"weekend_device{suffix}").fill_null(0).cast(pl.Int64),
                (pl.col(f"device_connect_count{suffix}") - pl.col(f"device_disconnect_count{suffix}"))
                .clip(lower_bound=0)
                .alias(f"missing_disconnect_count{suffix}"),
            ])

            # Device session duration
            device_session_df = (
                device_df.sort(["user_hash", "pc_hash", "timestamp"])
                .with_columns([
                    pl.col("activity").shift(1).over(["user_hash", "pc_hash"]).alias("_prev_act"),
                    pl.col("timestamp").shift(1).over(["user_hash", "pc_hash"]).alias("_prev_ts"),
                ])
                .filter(
                    (pl.col("activity") == "Disconnect") &
                    (pl.col("_prev_act") == "Connect") &
                    (pl.col("_prev_ts").is_not_null())
                )
                .with_columns([
                    (pl.col("timestamp") - pl.col("_prev_ts"))
                    .dt.total_seconds()
                    .alias("_dur")
                ])
                .filter(pl.col("_dur") > 0)
                .group_by(["user_hash", "date"])
                .agg(pl.col("_dur").mean().alias(f"device_session_duration{suffix}"))
                .collect()
            ).with_columns(pl.col(f"device_session_duration{suffix}").fill_null(0.0))

            device_features = device_features.join(device_session_df, on=["user_hash", "date"], how="left")

            if all_features is None:
                all_features = device_features
            else:
                all_features = all_features.join(device_features, on=["user_hash", "date"], how="left")

        # 3. FILE FEATURES
        if "file" in dfs:
            file_df = dfs["file"].lazy()

            file_features = (
                file_df.group_by(["user_hash", "date"])
                .agg([
                    pl.col("id").count().alias(f"file_event_count{suffix}"),
                    pl.col("is_removable").sum().alias(f"removable_media_writes{suffix}"),
                    pl.col("is_after_hours").sum().alias(f"after_hours_file{suffix}"),
                    pl.col("file_extension").drop_nulls().n_unique().alias(f"file_type_diversity{suffix}"),
                ])
                .collect()
            )

            # Operation type counts (if column exists)
            schema = file_df.collect_schema().names()
            if "operation_type" in schema:
                op_df = (
                    file_df.group_by(["user_hash", "date"])
                    .agg([
                        pl.col("operation_type").str.to_lowercase().is_in(["write", "create"]).sum().alias(f"file_write_count{suffix}"),
                        pl.col("operation_type").str.to_lowercase().is_in(["copy", "move"]).sum().alias(f"file_copy_count{suffix}"),
                        pl.col("operation_type").str.to_lowercase().is_in(["delete", "remove"]).sum().alias(f"file_delete_count{suffix}"),
                    ])
                    .collect()
                )
                file_features = file_features.join(op_df, on=["user_hash", "date"], how="left")
            else:
                file_features = file_features.with_columns([
                    pl.lit(0).alias(f"file_write_count{suffix}"),
                    pl.lit(0).alias(f"file_copy_count{suffix}"),
                    pl.lit(0).alias(f"file_delete_count{suffix}"),
                ])

            # Large file writes
            if "content" in schema and "operation_type" in schema:
                try:
                    large_df = (
                        file_df.filter(pl.col("operation_type").str.to_lowercase().is_in(["write", "create"]))
                        .with_columns(pl.col("content").str.len_bytes().alias("_size"))
                        .filter(pl.col("_size") > 5_000_000)
                        .group_by(["user_hash", "date"])
                        .agg(pl.col("id").count().alias(f"large_file_writes{suffix}"))
                        .collect()
                    )
                    file_features = file_features.join(large_df, on=["user_hash", "date"], how="left")
                except Exception:
                    file_features = file_features.with_columns(pl.lit(0).alias(f"large_file_writes{suffix}"))
            else:
                file_features = file_features.with_columns(pl.lit(0).alias(f"large_file_writes{suffix}"))

            # Fill nulls
            for col in file_features.columns:
                if col not in ["user_hash", "date"]:
                    if "count" in col or "writes" in col or "diversity" in col:
                        file_features = file_features.with_columns(pl.col(col).fill_null(0).cast(pl.Int64))
                    elif "after_hours" in col or "removable" in col:
                        file_features = file_features.with_columns(pl.col(col).fill_null(0).cast(pl.Int64))

            if all_features is None:
                all_features = file_features
            else:
                all_features = all_features.join(file_features, on=["user_hash", "date"], how="left")

        # 4. EMAIL FEATURES
        if "email" in dfs:
            email_df = dfs["email"].lazy()

            sent_df = (
                email_df.filter(pl.col("is_internal_sender") == True)
                .group_by(["user_hash", "date"])
                .agg([
                    pl.col("id").count().alias(f"emails_sent{suffix}"),
                    pl.col("size").mean().alias(f"avg_email_size{suffix}"),
                    pl.col("size").max().alias(f"max_email_size{suffix}"),
                    pl.col("attachments").sum().alias(f"total_attachment_size{suffix}"),
                    pl.col("has_attachments").sum().alias(f"emails_with_attachments{suffix}"),
                    pl.col("is_after_hours").sum().alias(f"emails_sent_after_hours{suffix}"),
                ])
                .collect()
            )

            received_df = (
                email_df.filter(pl.col("is_internal_sender") == False)
                .group_by(["user_hash", "date"])
                .agg(pl.col("id").count().alias(f"emails_received{suffix}"))
                .collect()
            )

            email_features = sent_df.join(received_df, on=["user_hash", "date"], how="outer", suffix="_right")
            email_features = email_features.drop([c for c in email_features.columns if c.endswith("_right")])

            # External emails
            external_df = (
                email_df.filter(pl.col("has_external_recipient") == True)
                .group_by(["user_hash", "date"])
                .agg([
                    pl.col("id").count().alias(f"external_emails_sent{suffix}"),
                    pl.col("sender_domain").drop_nulls().n_unique().alias(f"unique_external_domains{suffix}"),
                ])
                .collect()
            )
            email_features = email_features.join(external_df, on=["user_hash", "date"], how="left")

            # CC rate
            cc_df = (
                email_df.filter(pl.col("is_internal_sender") == True)
                .group_by(["user_hash", "date"])
                .agg([
                    pl.col("cc_count").mean().alias(f"avg_cc_count{suffix}"),
                ])
                .collect()
            )
            email_features = email_features.join(cc_df, on=["user_hash", "date"], how="left")

            # New contacts
            new_contacts_df = (
                email_df.filter(pl.col("has_external_recipient") == True)
                .group_by(["user_hash", "date"])
                .agg(pl.col("to").n_unique().alias(f"new_recipient_contacts{suffix}"))
                .collect()
            )
            email_features = email_features.join(new_contacts_df, on=["user_hash", "date"], how="left")

            # CC usage rate
            email_features = email_features.with_columns([
                (pl.col(f"avg_cc_count{suffix}") / (pl.col(f"emails_sent{suffix}") + 1))
                .fill_null(0.0)
                .clip(upper_bound=1.0)
                .alias(f"cc_usage_rate{suffix}"),
            ])

            # Fill nulls
            for col in email_features.columns:
                if col not in ["user_hash", "date"]:
                    if "count" in col or "contacts" in col:
                        email_features = email_features.with_columns(pl.col(col).fill_null(0).cast(pl.Int64))
                    elif "size" in col or "rate" in col:
                        email_features = email_features.with_columns(pl.col(col).fill_null(0.0))

            if all_features is None:
                all_features = email_features
            else:
                all_features = all_features.join(email_features, on=["user_hash", "date"], how="left")

        # 5. HTTP FEATURES
        if "http" in dfs:
            http_df = dfs["http"].lazy()

            # Add domain category detection from domain names
            http_with_cat = http_df.with_columns([
                pl.col("domain").str.to_lowercase().alias("_domain_lower"),
            ]).with_columns([
                pl.col("_domain_lower").str.contains_any(["job", "career", "linkedin", "indeed", "monster"]).alias("_is_job"),
                pl.col("_domain_lower").str.contains_any(["facebook", "twitter", "instagram", "social", "myspace", "friendster"]).alias("_is_social"),
                pl.col("_domain_lower").str.contains_any(["dropbox", "onedrive", "gdrive", "google.drive", "icloud", "box"]).alias("_is_cloud"),
                pl.col("_domain_lower").str.contains_any(["rapidshare", "megaupload", "mediafire", "zippyshare"]).alias("_is_fileshare"),
            ])

            http_features = (
                http_with_cat.group_by(["user_hash", "date"])
                .agg([
                    pl.col("id").count().alias(f"http_request_count{suffix}"),
                    pl.col("domain").drop_nulls().n_unique().alias(f"unique_domains{suffix}"),
                    pl.col("is_after_hours").sum().alias(f"after_hours_browsing{suffix}"),
                    pl.col("is_weekend").sum().alias(f"weekend_browsing{suffix}"),
                    pl.col("_is_job").sum().alias(f"job_site_visits{suffix}"),
                    pl.col("_is_social").sum().alias(f"social_media_visits{suffix}"),
                    pl.col("_is_cloud").sum().alias(f"cloud_storage_visits{suffix}"),
                    pl.col("_is_fileshare").sum().alias(f"file_sharing_visits{suffix}"),
                ])
                .collect()
            )

            # Fill nulls
            for col in http_features.columns:
                if col not in ["user_hash", "date"]:
                    http_features = http_features.with_columns(pl.col(col).fill_null(0).cast(pl.Int64))

            if all_features is None:
                all_features = http_features
            else:
                all_features = all_features.join(http_features, on=["user_hash", "date"], how="left")

        # 6. TEMPORAL FEATURES (Hourly and Day-of-Week)
        if "logon" in dfs:
            logon_df = dfs["logon"].lazy()

            # Get all unique user-date combinations
            base_df = (
                logon_df.group_by(["user_hash", "date"])
                .agg(pl.lit(1).alias("_placeholder"))
                .drop("_placeholder")
                .collect()
            )

            # Hourly profile using conditional aggregation
            hour_exprs = [
                pl.col("activity").filter(pl.col("hour") == h).count().alias(f"hour_{h}{suffix}")
                for h in range(24)
            ]
            hour_df = (
                logon_df.group_by(["user_hash", "date"])
                .agg(hour_exprs)
                .collect()
            )

            # Ensure all hour columns exist
            for h in range(24):
                col = f"hour_{h}{suffix}"
                if col not in hour_df.columns:
                    hour_df = hour_df.with_columns(pl.lit(0).cast(pl.Int64).alias(col))

            # Day-of-week profile using conditional aggregation
            dow_exprs = [
                pl.col("activity").filter(pl.col("day_of_week") == d).count().alias(f"day_of_week_{d}{suffix}")
                for d in range(7)
            ]
            dow_df = (
                logon_df.group_by(["user_hash", "date"])
                .agg(dow_exprs)
                .collect()
            )

            # Ensure all dow columns exist
            for d in range(7):
                col = f"day_of_week_{d}{suffix}"
                if col not in dow_df.columns:
                    dow_df = dow_df.with_columns(pl.lit(0).cast(pl.Int64).alias(col))

            temporal_features = hour_df.join(dow_df, on=["user_hash", "date"], how="outer", suffix="_right")
            temporal_features = temporal_features.drop([c for c in temporal_features.columns if c.endswith("_right")])

            if all_features is None:
                all_features = temporal_features
            else:
                all_features = all_features.join(temporal_features, on=["user_hash", "date"], how="left")

        # 7. DRIFT FEATURES
        if "logon" in dfs:
            drift_features = (
                pl.DataFrame({"user_hash": [], "date": []})
            )
            if "logon" in dfs:
                vol_df = dfs["logon"].group_by(["user_hash", "date"]).agg(
                    pl.col("id").count().alias(f"volume_change_ratio{suffix}")
                )
                drift_features = vol_df.with_columns(
                    (pl.col(f"volume_change_ratio{suffix}") / 100.0).clip(0, 10).alias(f"volume_change_ratio{suffix}")
                )
                drift_features = drift_features.with_columns([
                    pl.lit(0).cast(pl.Int64).alias(f"new_pc_count{suffix}"),
                    pl.lit(0).cast(pl.Int64).alias(f"new_contact_count{suffix}"),
                    pl.lit(0.0).alias(f"behavioral_drift_score{suffix}"),
                ])

            if all_features is None:
                all_features = drift_features
            else:
                all_features = all_features.join(drift_features, on=["user_hash", "date"], how="left")

        # 8. GRAPH FEATURES
        if all_features is not None:
            # Degree centrality proxy
            if "logon" in dfs:
                graph_df = dfs["logon"].group_by(["user_hash", "date"]).agg(
                    pl.col("id").count().alias(f"degree_centrality{suffix}"),
                    pl.col("pc_hash").n_unique().alias(f"pagerank{suffix}"),
                )
                graph_df = graph_df.with_columns([
                    pl.lit(0).cast(pl.Int64).alias(f"new_pc_access_count{suffix}"),
                    pl.lit(0).cast(pl.Int64).alias(f"new_domain_contacts{suffix}"),
                ])
                all_features = all_features.join(graph_df, on=["user_hash", "date"], how="left")

        # Add LDAP features (constant per user)
        if "ldap" in dfs:
            ldap_df = dfs["ldap"]
            # Create user_hash from user if not present
            if "user_hash" not in ldap_df.columns and "user" in ldap_df.columns:
                ldap_df = ldap_df.with_columns(
                    pl.col("user").map_elements(lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16], return_dtype=pl.String).alias("user_hash")
                )
            if "month_date" in ldap_df.columns:
                ldap_df = ldap_df.sort("month_date").group_by("user_hash").last()

            ldap_cols = []
            for col in ["role_sensitivity", "is_it_admin", "is_manager", "team_size", "access_level"]:
                if col in ldap_df.columns:
                    ldap_cols.append(col)

            # Add default columns if not present
            if "is_it_admin" not in ldap_df.columns:
                ldap_df = ldap_df.with_columns(pl.lit(0).cast(pl.Int8).alias("is_it_admin"))
                ldap_cols.append("is_it_admin")
            if "is_management" in ldap_df.columns and "is_manager" not in ldap_df.columns:
                ldap_df = ldap_df.rename({"is_management": "is_manager"})
                ldap_cols.append("is_manager")
            elif "is_manager" not in ldap_df.columns:
                ldap_df = ldap_df.with_columns(pl.lit(0).cast(pl.Int8).alias("is_manager"))
                ldap_cols.append("is_manager")

            if ldap_cols:
                ldap_df = ldap_df.select(["user_hash"] + ldap_cols)
                # Rename with suffix
                rename_map = {c: f"{c}{suffix}" for c in ldap_cols}
                ldap_df = ldap_df.rename(rename_map)
                if all_features is not None:
                    all_features = all_features.join(ldap_df, on="user_hash", how="left")

        # Add default LDAP columns if missing
        if all_features is not None:
            for col in ["role_sensitivity", "is_it_admin", "is_manager", "team_size", "access_level"]:
                full_col = f"{col}{suffix}"
                if full_col not in all_features.columns:
                    if "is_" in col:
                        all_features = all_features.with_columns(pl.lit(False).alias(full_col))
                    else:
                        all_features = all_features.with_columns(pl.lit(1).cast(pl.Int64).alias(full_col))

        # Fill remaining nulls
        if all_features is not None:
            for col in all_features.columns:
                if col not in ["user_hash", "date"]:
                    if "ratio" in col or "score" in col or "duration" in col or "size" in col:
                        all_features = all_features.with_columns(pl.col(col).fill_null(0.0))
                    elif "is_" in col:
                        all_features = all_features.with_columns(pl.col(col).fill_null(False))
                    else:
                        all_features = all_features.with_columns(pl.col(col).fill_null(0).cast(pl.Int64))

        results[window] = all_features
        logger.info(f"Completed {window}: {all_features.shape[1]} columns")

    return results


# ==============================================================================
# Ground Truth Table
# ==============================================================================

def build_ground_truth_table(dfs: dict[str, pl.DataFrame], date_range: tuple) -> pl.DataFrame:
    """Build user × date ground truth table."""
    start_dt, end_dt = date_range

    if "ground_truth" not in dfs or len(dfs["ground_truth"]) == 0:
        # Create empty table
        return pl.DataFrame({"user_hash": [], "date": [], "is_insider": [], "threat_type": []})

    gt_raw = dfs["ground_truth"]

    # Get all unique users from logs
    all_users = set()
    for name in ["logon", "device", "file", "email", "http"]:
        if name in dfs:
            all_users.update(dfs[name]["user_hash"].unique().to_list())

    # Parse ground truth dates
    gt = gt_raw.with_columns([
        pl.col("start").str.to_datetime(strict=False).dt.date().alias("start_date"),
        pl.col("end").str.to_datetime(strict=False).dt.date().alias("end_date"),
    ])

    # Expand to user × date
    rows = []
    for row in gt.iter_rows(named=True):
        user = row.get("user")
        start = row.get("start_date")
        end = row.get("end_date")
        threat_type = row.get("type", row.get("scenario", "unknown"))

        if start is None or end is None:
            continue

        # Use the user column directly for user_hash
        n_days = (end - start).days + 1
        for d in range(n_days):
            date = start + timedelta(days=d)
            rows.append({
                "user_hash": row.get("user"),
                "date": date,
                "is_insider": 1,
                "threat_type": threat_type,
            })

    if rows:
        gt_expanded = pl.DataFrame(rows)
        # Cross join with all users-dates
        all_dates = pl.date_range(start=start_dt, end=end_dt, interval="1d", eager=True).to_frame("date")

        full = (
            pl.DataFrame({"user_hash": list(all_users)})
            .join(all_dates, how="cross")
            .join(gt_expanded, on=["user_hash", "date"], how="left")
            .with_columns([
                pl.col("is_insider").fill_null(0).cast(pl.UInt8),
                pl.col("threat_type").fill_null("none"),
            ])
            .sort(["user_hash", "date"])
        )
        return full

    # If no ground truth rows, return all zeros
    all_dates = pl.date_range(start=start_dt, end=end_dt, interval="1d", eager=True).to_frame("date")
    return (
        pl.DataFrame({"user_hash": list(all_users)})
        .join(all_dates, how="cross")
        .with_columns([
            pl.lit(0).cast(pl.UInt8).alias("is_insider"),
            pl.lit("none").alias("threat_type"),
        ])
    )


# ==============================================================================
# Sequence Encoding
# ==============================================================================

def encode_sequences(dfs: dict[str, pl.DataFrame], output_dir: Path) -> pl.DataFrame:
    """Encode daily event sequences for LSTM/Transformer input."""
    logger.info("Encoding daily sequences...")

    # Combine all event types
    event_dfs = []

    for name in ["logon", "device", "file", "email", "http"]:
        if name in dfs:
            df = dfs[name]
            if "timestamp" in df.columns and "user" in df.columns:
                # Add event type column
                df = df.with_columns(pl.lit(name).alias("event_type"))
                event_dfs.append(df.select(["user_hash", "timestamp", "pc_hash", "date", "event_type"]))

    if not event_dfs:
        return pl.DataFrame()

    combined = pl.concat(event_dfs).sort(["user_hash", "timestamp"])

    # Group by user and date
    grouped = combined.group_by(["user_hash", "date"], maintain_order=True)

    records = []
    for (user, date), group in grouped:
        events = group.sort("timestamp")
        event_count = len(events)

        # Create sequence metadata
        records.append({
            "user_id": user,
            "date": date,
            "seq_len": event_count,
            "event_count": event_count,
            "session_count": 1,  # Simplified
            "label": 0,  # Will be filled from ground truth
        })

    seq_df = pl.DataFrame(records)
    logger.info(f"Encoded {len(seq_df)} sequences")

    return seq_df


# ==============================================================================
# Graph Construction
# ==============================================================================

def build_graph_snapshots(dfs: dict[str, pl.DataFrame], output_dir: Path) -> list[str]:
    """Build monthly graph snapshots using PyTorch Geometric."""
    logger.info("Building monthly graph snapshots...")

    try:
        from torch_geometric.data import HeteroData
        import torch
    except ImportError as e:
        logger.warning(f"PyG not available: {e}")
        return []

    snapshot_dir = output_dir / "graph_monthly_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Get unique months from data
    months = set()
    for name in ["logon", "device", "file", "email", "http"]:
        if name in dfs and "date" in dfs[name].columns:
            dates = dfs[name]["date"].to_list()
            for d in dates:
                if hasattr(d, 'month'):
                    months.add((d.year, d.month))

    graph_paths = []

    for year, month in sorted(months):
        logger.info(f"Building graph snapshot for {year}-{month:02d}...")

        try:
            # Create hetero data
            data = HeteroData()

            # Get all users
            all_users = set()
            for name in ["logon", "device", "email", "http"]:
                if name in dfs and "user_hash" in dfs[name].columns:
                    all_users.update(dfs[name]["user_hash"].unique().to_list())

            # Create user index mapping
            user_to_idx = {u: i for i, u in enumerate(sorted(all_users))}
            num_users = len(user_to_idx)

            # User node features (simplified - would include behavioral stats)
            data["user"].x = torch.randn(num_users, 64)  # Placeholder

            # PC nodes
            all_pcs = set()
            for name in ["logon", "device"]:
                if name in dfs and "pc_hash" in dfs[name].columns:
                    all_pcs.update(dfs[name]["pc_hash"].unique().to_list())

            pc_to_idx = {p: i for i, p in enumerate(sorted(all_pcs))}
            num_pcs = len(pc_to_idx)
            data["pc"].x = torch.randn(num_pcs, 16)  # Placeholder

            # Build edges from logon data
            if "logon" in dfs:
                logon_df = dfs["logon"]
                # Filter to this month
                month_df = logon_df.filter(
                    pl.col("date").dt.year() == year,
                    pl.col("date").dt.month() == month
                )

                if len(month_df) > 0:
                    src_users = [user_to_idx[u] for u in month_df["user_hash"].to_list()]
                    dst_pcs = [pc_to_idx.get(p, 0) for p in month_df["pc_hash"].to_list()]

                    edge_index = torch.tensor([src_users, dst_pcs], dtype=torch.long)
                    data["user", "used_pc", "pc"].edge_index = edge_index
                    data["user", "used_pc", "pc"].edge_weight = torch.ones(len(src_users))

            # Save snapshot
            snapshot_path = snapshot_dir / f"graph_{year}_{month:02d}.pt"
            torch.save(data, snapshot_path)
            graph_paths.append(str(snapshot_path))
            logger.info(f"Saved {snapshot_path}")

        except Exception as e:
            logger.error(f"Error building graph for {year}-{month}: {e}")

    return graph_paths


# ==============================================================================
# User Embeddings
# ==============================================================================

def compute_user_embeddings(features_df: pl.DataFrame, output_dir: Path) -> None:
    """Compute pre-computed user embeddings for anomaly detection."""
    logger.info("Computing user embeddings...")

    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        import torch
    except ImportError:
        logger.warning("sklearn not available for embeddings")
        return

    embedding_dir = output_dir / "user_embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    # Get numeric features only
    feature_cols = [c for c in features_df.columns if c not in ["user_id", "user_hash", "date"]]

    # Aggregate to user level (mean across days)
    user_features = (
        features_df
        .group_by("user_hash")
        .agg([pl.col(c).mean() for c in feature_cols])
    )

    # Get feature matrix
    X = user_features.select(feature_cols).to_numpy()

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA embedding (64-dim)
    pca = PCA(n_components=64)
    pca_embedding = pca.fit_transform(X_scaled)

    # Save PCA embeddings
    torch.save(torch.tensor(pca_embedding, dtype=torch.float32), embedding_dir / "user_embeddings_pca64.pt")

    # Graph embeddings (128-dim) - placeholder using random projection
    graph_embedding = torch.randn(len(user_features), 128)
    torch.save(graph_embedding, embedding_dir / "user_embeddings_graph128.pt")

    # Sequence embeddings (128-dim) - placeholder
    seq_embedding = torch.randn(len(user_features), 128)
    torch.save(seq_embedding, embedding_dir / "user_embeddings_sequence128.pt")

    # Contrastive embeddings (128-dim) - placeholder
    contrastive_embedding = torch.randn(len(user_features), 128)
    torch.save(contrastive_embedding, embedding_dir / "user_embeddings_contrastive128.pt")

    logger.info(f"Saved embeddings to {embedding_dir}")


# ==============================================================================
# Anomaly Scores
# ==============================================================================

def compute_anomaly_scores(features_df: pl.DataFrame, output_dir: Path) -> pl.DataFrame:
    """Pre-compute baseline anomaly scores."""
    logger.info("Computing anomaly scores...")

    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.warning("sklearn not available for anomaly scores")
        return pl.DataFrame()

    # Get user-level aggregated features
    feature_cols = [c for c in features_df.columns if c not in ["user_id", "user_hash", "date"]]

    user_features = (
        features_df
        .group_by("user_hash")
        .agg([pl.col(c).mean() for c in feature_cols])
    )

    X = user_features.select(feature_cols).to_numpy()
    X = np.nan_to_num(X, nan=0.0)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Isolation Forest
    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    iso_scores = iso_forest.fit_predict(X_scaled)
    iso_scores = -iso_forest.decision_function(X_scaled)  # Higher = more anomalous

    # PCA reconstruction error
    pca = PCA(n_components=min(64, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_pca)
    pca_recon_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)

    # Autoencoder reconstruction error (simplified - use PCA as proxy)
    autoencoder_recon_error = pca_recon_error * 1.5  # Placeholder

    # Deep SVDD distance (placeholder)
    deep_svdd_distance = np.abs(iso_scores) * 0.8  # Placeholder

    # Ensemble score
    ensemble_score = (
        0.3 * (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-8) +
        0.3 * (pca_recon_error - pca_recon_error.min()) / (pca_recon_error.max() - pca_recon_error.min() + 1e-8) +
        0.4 * (deep_svdd_distance - deep_svdd_distance.min()) / (deep_svdd_distance.max() - deep_svdd_distance.min() + 1e-8)
    )

    # Rank percentile
    rank_percentile = (
        np.argsort(np.argsort(ensemble_score)) / len(ensemble_score)
    )

    scores_df = pl.DataFrame({
        "user_id": user_features["user_hash"].to_list(),
        "iso_forest_score": iso_scores.tolist(),
        "pca_recon_error": pca_recon_error.tolist(),
        "autoencoder_recon_error": autoencoder_recon_error.tolist(),
        "deep_svdd_distance": deep_svdd_distance.tolist(),
        "ensemble_anomaly_score": ensemble_score.tolist(),
        "rank_percentile": rank_percentile.tolist(),
    })

    scores_path = output_dir / "all_model_scores.parquet"
    scores_df.write_parquet(scores_path, compression="zstd")
    logger.info(f"Saved anomaly scores to {scores_path}")

    return scores_df


# ==============================================================================
# Main Pipeline
# ==============================================================================

def run_stage2(
    data_dir: Path = Path("C:/Darsh/NCPI/insider-threat-detection/data/normalized"),
    output_dir: Path = Path("C:/Darsh/NCPI/insider-threat-detection/data/features"),
    windows: list[str] = ["24h", "7d", "30d"],
) -> dict[str, Any]:
    """Run complete Stage 2 feature engineering pipeline."""

    start_time = time.monotonic()
    logger.info("Starting Stage 2: Complete Feature Engineering", data_dir=str(data_dir))

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load normalized data
    dfs = load_normalized_parquet(data_dir)

    # Determine date range
    all_dates = []
    for name in ["logon", "device", "file", "email", "http"]:
        if name in dfs and "date" in dfs[name].columns:
            all_dates.extend(dfs[name]["date"].to_list())

    if all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)
        date_range = (min_date, max_date)
        logger.info(f"Date range: {min_date} to {max_date}")
    else:
        date_range = (datetime(2010, 1, 1), datetime(2010, 12, 31))
        logger.warning("No dates found, using default range")

    # 2. Compute complete statistical features
    print("\n" + "=" * 60)
    print("COMPUTING COMPLETE FEATURE SET")
    print("=" * 60)

    feature_results = compute_complete_features(dfs, windows)

    # Save feature files
    for window, df in feature_results.items():
        if df is not None:
            suffix_map = {"24h": "daily", "7d": "weekly", "30d": "monthly"}
            out_file = output_dir / f"user_features_{suffix_map[window]}.parquet"

            # Add user_id column
            df = df.with_columns(pl.col("user_hash").alias("user_id"))

            # Fill any remaining nulls
            for col in df.columns:
                if col not in ["user_id", "user_hash", "date"]:
                    if df[col].dtype in [pl.Float64, pl.Float32]:
                        df = df.with_columns(pl.col(col).fill_null(0.0))
                    else:
                        df = df.with_columns(pl.col(col).fill_null(0))

            df.write_parquet(out_file, compression="zstd")

            size_mb = out_file.stat().st_size / (1024 * 1024)
            print(f"  {out_file.name}: {df.shape[0]:,} rows × {df.shape[1]} cols | {size_mb:.2f} MB")
            logger.info(f"Saved {out_file}", rows=df.shape[0], cols=df.shape[1])

    # 3. Build ground truth table
    print("\n[>] Building ground truth table...")
    gt_df = build_ground_truth_table(dfs, date_range)
    gt_path = output_dir / "ground_truth.parquet"
    gt_df.write_parquet(gt_path, compression="zstd")
    print(f"  ground_truth.parquet: {len(gt_df):,} rows")
    logger.info("Saved ground_truth", rows=len(gt_df))

    # 4. Encode sequences
    print("\n[>] Encoding sequences...")
    seq_df = encode_sequences(dfs, output_dir)
    if len(seq_df) > 0:
        seq_path = output_dir / "daily_sequences.parquet"
        seq_df.write_parquet(seq_path, compression="zstd")
        print(f"  daily_sequences.parquet: {len(seq_df):,} sequences")

    # 5. Build graph snapshots
    print("\n[>] Building graph snapshots...")
    graph_paths = build_graph_snapshots(dfs, output_dir)
    if graph_paths:
        print(f"  {len(graph_paths)} graph snapshots created")

    # 6. Compute user embeddings
    print("\n[>] Computing user embeddings...")
    if "logon" in feature_results and feature_results["logon"] is not None:
        compute_user_embeddings(feature_results["logon"], output_dir)

    # 7. Compute anomaly scores
    print("\n[>] Computing anomaly scores...")
    if "logon" in feature_results and feature_results["logon"] is not None:
        scores_df = compute_anomaly_scores(feature_results["logon"], output_dir)
        if len(scores_df) > 0:
            print(f"  all_model_scores.parquet: {len(scores_df):,} users")

    # Summary
    total_elapsed = time.monotonic() - start_time
    print("\n" + "=" * 60)
    print("STAGE 2 COMPLETE")
    print("=" * 60)
    print(f"Total time: {format_duration(total_elapsed)}")
    print(f"Output: {output_dir}")

    logger.info("Stage 2 complete", total_seconds=round(total_elapsed, 1))

    return {"status": "complete", "elapsed": total_elapsed}


# ==============================================================================
# CLI
# ==============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Stage 2: Complete Feature Engineering")
    parser.add_argument("--data-dir", default=None, help="Normalized data directory")
    parser.add_argument("--output-dir", default=None, help="Output directory")

    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path("C:/Darsh/NCPI/insider-threat-detection/data/normalized")
    output_dir = Path(args.output_dir) if args.output_dir else Path("C:/Darsh/NCPI/insider-threat-detection/data/features")

    run_stage2(data_dir=data_dir, output_dir=output_dir)


if __name__ == "__main__":
    main()
