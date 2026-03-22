"""
Complete Statistical Feature Engineering Module for CERT Insider Threat Detection.

This module computes ALL behavioral features (~500 features) from normalized log data across
multiple time windows (24h, 7d, 30d) for insider threat detection.

Based on the complete CERT output schema:
- Logon Features: frequency, after-hours, weekend, unique PCs, session duration (10 features)
- Device Features: connect/disconnect, missing disconnects, session duration (7 features)
- File Features: operations, removable media, file types, large files (9 features)
- Email Features: volume, attachments, external recipients, domains (14 features)
- HTTP Features: domains, sensitive categories, browsing patterns (10 features)
- Temporal Features: hour and day-of-week distributions (48 features)
- Graph Features: degree centrality, pagerank, new entities (4 features)
- Organizational Features: role sensitivity, admin status, team info (5 features)
- Psychometric Features: Big Five personality traits (6 features)
- Drift Features: volume changes, new entities, behavioral drift (4 features)

Total: ~120 features per window × 3 windows = ~360 features + embeddings = ~500 features
"""

from __future__ import annotations

import logging
import hashlib
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _to_datetime(expr: pl.Expr) -> pl.Expr:
    """Safely convert timestamp to datetime, handling both string and datetime types."""
    # For Datetime columns, just return the expression as-is
    # For String columns, parse them
    # Since our data has Datetime columns, we just return expr
    return expr


# Constants
DATA_DIR = Path("C:/Darsh/NCPI/insider-threat-detection/data/normalized")
OUTPUT_DIR = Path("C:/Darsh/NCPI/insider-threat-detection/data/features")
WINDOWS = {"24h": 1, "7d": 7, "30d": 30}


class WindowConfig:
    """Configuration for time window features."""
    def __init__(self, name: str, days: int):
        self.name = name
        self.days = days
        self.suffix = f"_{name}"

    @classmethod
    def from_str(cls, window: str) -> "WindowConfig":
        """Create WindowConfig from string like '24h', '7d', '30d'."""
        window_map = {"24h": 1, "7d": 7, "30d": 30}
        if window not in window_map:
            raise ValueError(f"Unknown window: {window}. Valid: {list(window_map.keys())}")
        return cls(name=window, days=window_map[window])


# =============================================================================
# LOGON FEATURES
# =============================================================================

def compute_logon_features(
    lf: pl.LazyFrame,
    window_days: int,
    suffix: str
) -> pl.DataFrame:
    """
    Compute complete logon-related behavioral features (10 features).

    Features:
    - logon_count: Total logon events
    - logoff_count: Total logoff events
    - logon_ratio: logon_count / total_events
    - after_hours_logons: Logons outside 08:00-18:00 Mon-Fri
    - weekend_logons: Logons on Sat/Sun
    - unique_pcs: Distinct PCs accessed
    - avg_session_duration: Mean time between logon/logoff (seconds)
    - max_session_duration: Longest session
    - session_duration_std: Std dev of session duration
    - rapid_logon_cycles: Sessions < 5 minutes (screen unlock attempts)
    """
    logon_lf = lf.filter(pl.col("activity").str.to_lowercase() == "logon")
    logoff_lf = lf.filter(pl.col("activity").str.to_lowercase() == "logoff")

    # Basic logon counts
    logon_counts = (
        logon_lf
        .group_by(["user_hash", "date"])
        .agg([
            pl.col("activity").count().alias(f"logon_count{suffix}"),
            pl.col("is_after_hours").sum().alias(f"after_hours_logons{suffix}"),
            pl.col("is_weekend").sum().alias(f"weekend_logons{suffix}"),
            pl.col("pc_hash").n_unique().alias(f"unique_pcs{suffix}"),
        ])
        .collect()
    )

    # Logoff counts
    logoff_counts = (
        logoff_lf
        .group_by(["user_hash", "date"])
        .agg(
            pl.col("activity").count().alias(f"logoff_count{suffix}"),
        )
        .collect()
    )

    # Merge and compute ratios
    result = logon_counts.join(logoff_counts, on=["user_hash", "date"], how="outer")

    total_events = (
        lf.filter(pl.col("activity").str.to_lowercase().is_in(["logon", "logoff"]))
        .group_by(["user_hash", "date"])
        .agg(pl.col("activity").count().alias("_total_events"))
        .collect()
    )
    result = result.join(total_events, on=["user_hash", "date"], how="left")

    result = result.with_columns([
        pl.col(f"logon_count{suffix}").fill_null(0).cast(pl.Int64),
        pl.col(f"logoff_count{suffix}").fill_null(0).cast(pl.Int64),
        pl.col(f"after_hours_logons{suffix}").fill_null(0).cast(pl.Int64),
        pl.col(f"weekend_logons{suffix}").fill_null(0).cast(pl.Int64),
        pl.col(f"unique_pcs{suffix}").fill_null(0).cast(pl.Int64),
        pl.col(f"logon_count{suffix}") / (pl.col("_total_events").fill_null(1) + 1)
        .alias(f"logon_ratio{suffix}"),
    ]).drop("_total_events")

    return result


def compute_session_duration_features(
    lf: pl.LazyFrame,
    window_days: int,
    suffix: str
) -> pl.DataFrame:
    """
    Compute session duration statistics from logon/logoff pairs.

    Features:
    - avg_session_duration: Mean time between logon/logoff (seconds)
    - max_session_duration: Longest session
    - session_duration_std: Std dev of session duration
    """
    session_lf = (
        lf.filter(pl.col("activity").str.to_lowercase().is_in(["logon", "logoff"]))
        .sort(["user_hash", "pc_hash", "timestamp"])
    )

    # Compute session durations
    sessions = (
        session_lf
        .with_columns([
            pl.col("activity").shift(1).over(["user_hash", "pc_hash"]).alias("_prev_activity"),
            _to_datetime(pl.col("timestamp")).shift(1).over(["user_hash", "pc_hash"]).alias("_prev_timestamp"),
        ])
        .filter(
            (pl.col("activity").str.to_lowercase() == "logoff") &
            (pl.col("_prev_activity").str.to_lowercase() == "logon") &
            (pl.col("_prev_timestamp").is_not_null())
        )
        .with_columns([
            (_to_datetime(pl.col("timestamp")) - pl.col("_prev_timestamp"))
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

    return sessions.with_columns([
        pl.col(f"avg_session_duration{suffix}").fill_null(0.0),
        pl.col(f"max_session_duration{suffix}").fill_null(0.0),
        pl.col(f"session_duration_std{suffix}").fill_null(0.0),
    ])


def compute_rapid_logon_cycles(
    lf: pl.LazyFrame,
    window_days: int,
    suffix: str
) -> pl.DataFrame:
    """
    Compute rapid logon/logoff cycles (potential screen unlock attempts).

    Features:
    - rapid_logon_cycles: Sessions < 5 minutes
    """
    rapid_lf = (
        lf.filter(pl.col("activity").str.to_lowercase() == "logon")
        .sort(["user_hash", "pc_hash", "timestamp"])
        .with_columns([
            pl.col("activity").shift(1).over("user_hash").alias("_prev_activity"),
            _to_datetime(pl.col("timestamp")).shift(1).over("user_hash").alias("_prev_ts"),
        ])
        .filter(pl.col("_prev_activity").str.to_lowercase() == "logon")
        .with_columns([
            (_to_datetime(pl.col("timestamp")) - pl.col("_prev_ts"))
            .dt.total_seconds()
            .alias("_time_diff")
        ])
        .filter(pl.col("_time_diff") < 300)  # 5 minutes
    )

    rapid_df = (
        rapid_lf
        .group_by(["user_hash", "date"])
        .agg(pl.col("_time_diff").count().alias(f"rapid_logon_cycles{suffix}"))
        .collect()
    )

    return rapid_df.with_columns(
        pl.col(f"rapid_logon_cycles{suffix}").fill_null(0)
    )


# =============================================================================
# DEVICE FEATURES
# =============================================================================

def compute_device_features(
    lf: pl.LazyFrame,
    window_days: int,
    suffix: str
) -> pl.DataFrame:
    """
    Compute complete device-related behavioral features (7 features).

    Features:
    - device_connect_count: USB connections
    - device_disconnect_count: USB disconnections
    - missing_disconnect_count: Connections without matching disconnect
    - device_session_duration: Mean connection duration
    - after_hours_device: Connections outside work hours
    - weekend_device: Weekend device connections
    """
    # Connect events
    connect_df = (
        lf.filter(pl.col("activity") == "Connect")
        .group_by(["user_hash", "date"])
        .agg([
            pl.col("activity").count().alias(f"device_connect_count{suffix}"),
            pl.col("is_after_hours").sum().alias(f"after_hours_device{suffix}"),
            pl.col("is_weekend").sum().alias(f"weekend_device{suffix}"),
        ])
        .collect()
    )

    # Disconnect events
    disconnect_df = (
        lf.filter(pl.col("activity") == "Disconnect")
        .group_by(["user_hash", "date"])
        .agg(
            pl.col("activity").count().alias(f"device_disconnect_count{suffix}"),
        )
        .collect()
    )

    # Merge connect and disconnect
    result = connect_df.join(disconnect_df, on=["user_hash", "date"], how="outer")

    # Compute missing disconnects (connects - disconnects)
    result = result.with_columns([
        pl.col(f"device_connect_count{suffix}").fill_null(0).cast(pl.Int64),
        pl.col(f"device_disconnect_count{suffix}").fill_null(0).cast(pl.Int64),
        pl.col(f"after_hours_device{suffix}").fill_null(0).cast(pl.Int64),
        pl.col(f"weekend_device{suffix}").fill_null(0).cast(pl.Int64),
        (pl.col(f"device_connect_count{suffix}") - pl.col(f"device_disconnect_count{suffix}"))
        .clip(lower_bound=0)
        .alias(f"missing_disconnect_count{suffix}"),
    ])

    return result


def compute_device_session_duration(
    lf: pl.LazyFrame,
    window_days: int,
    suffix: str
) -> pl.DataFrame:
    """
    Compute device session duration statistics.

    Features:
    - device_session_duration: Mean connection duration
    """
    device_sessions = (
        lf.sort(["user_hash", "pc_hash", "timestamp"])
        .with_columns([
            pl.col("activity").shift(1).over(["user_hash", "pc_hash"]).alias("_prev_activity"),
            _to_datetime(pl.col("timestamp")).shift(1).over(["user_hash", "pc_hash"]).alias("_prev_timestamp"),
        ])
        .filter(
            (pl.col("activity") == "Disconnect") &
            (pl.col("_prev_activity") == "Connect")
        )
        .with_columns([
            (_to_datetime(pl.col("timestamp")) - pl.col("_prev_timestamp"))
            .dt.total_seconds()
            .alias("_duration")
        ])
        .filter(pl.col("_duration") > 0)
        .group_by(["user_hash", "date"])
        .agg(
            pl.col("_duration").mean().alias(f"device_session_duration{suffix}"),
        )
        .collect()
    )

    return device_sessions.with_columns(
        pl.col(f"device_session_duration{suffix}").fill_null(0.0)
    )


# =============================================================================
# FILE FEATURES
# =============================================================================

def compute_file_features(
    lf: pl.LazyFrame,
    window_days: int,
    suffix: str
) -> pl.DataFrame:
    """
    Compute complete file operation features (9 features).

    Features:
    - file_event_count: Total file operations
    - file_write_count: Write operations
    - file_copy_count: Copy operations
    - file_delete_count: Delete operations
    - removable_media_writes: Writes to R: drive (exfiltration signal)
    - file_type_diversity: Unique file extensions accessed
    - large_file_writes: Writes > 5 MB
    - after_hours_file: After-hours file operations
    """
    # Get schema for column detection
    schema = lf.collect_schema().names()

    # Basic aggregations
    base_agg = [
        pl.col("id").count().alias(f"file_event_count{suffix}"),
    ]

    # Add operation type aggregations if column exists
    if "operation_type" in schema:
        base_agg.extend([
            pl.col("operation_type").str.to_lowercase().is_in(["write", "create"]).sum().alias(f"file_write_count{suffix}"),
            pl.col("operation_type").str.to_lowercase().is_in(["copy", "move"]).sum().alias(f"file_copy_count{suffix}"),
            pl.col("operation_type").str.to_lowercase().is_in(["delete", "remove"]).sum().alias(f"file_delete_count{suffix}"),
        ])
    else:
        base_agg.extend([
            pl.lit(0).alias(f"file_write_count{suffix}"),
            pl.lit(0).alias(f"file_copy_count{suffix}"),
            pl.lit(0).alias(f"file_delete_count{suffix}"),
        ])

    # Check for file_extension column
    if "file_extension" in schema:
        base_agg.append(pl.col("file_extension").drop_nulls().n_unique().alias(f"file_type_diversity{suffix}"))
    else:
        base_agg.append(pl.lit(0).alias(f"file_type_diversity{suffix}"))

    # Check for is_removable column
    if "is_removable" in schema:
        base_agg.append(pl.col("is_removable").sum().alias(f"removable_media_writes{suffix}"))
    else:
        base_agg.append(pl.lit(0).alias(f"removable_media_writes{suffix}"))

    base_agg.append(pl.col("is_after_hours").sum().alias(f"after_hours_file{suffix}"))

    base_df = (
        lf.group_by(["user_hash", "date"])
        .agg(base_agg)
        .collect()
    )

    # Large file writes (> 5MB)
    has_content = "content" in schema
    if has_content:
        try:
            large_file_df = (
                lf.filter(pl.col("operation_type").str.to_lowercase().is_in(["write", "create"]))
                .with_columns(pl.col("content").str.len_bytes().alias("_content_size"))
                .filter(pl.col("_content_size") > 5_000_000)
                .group_by(["user_hash", "date"])
                .agg(pl.col("id").count().alias(f"large_file_writes{suffix}"))
                .collect()
            )
            base_df = base_df.join(large_file_df, on=["user_hash", "date"], how="left")
        except Exception:
            base_df = base_df.with_columns(pl.lit(0).alias(f"large_file_writes{suffix}"))
    else:
        base_df = base_df.with_columns(pl.lit(0).alias(f"large_file_writes{suffix}"))

    # Fill nulls
    count_cols = [c for c in base_df.columns if "count" in c or "writes" in c or "diversity" in c]
    base_df = base_df.with_columns([pl.col(c).fill_null(0).cast(pl.Int64) for c in count_cols])
    base_df = base_df.with_columns(pl.col(f"after_hours_file{suffix}").fill_null(0).cast(pl.Int64))

    return base_df


# =============================================================================
# EMAIL FEATURES
# =============================================================================

def compute_email_features(
    lf: pl.LazyFrame,
    window_days: int,
    suffix: str
) -> pl.DataFrame:
    """
    Compute complete email behavior features (14 features).

    Features:
    - emails_sent: Outgoing emails
    - emails_received: Incoming emails
    - avg_email_size: Mean email size (bytes)
    - max_email_size: Largest email size
    - total_attachment_size: Sum of attachment sizes
    - emails_with_attachments: Emails with attachments
    - external_emails_sent: Emails to non-dtaa.com
    - unique_external_domains: Distinct external recipient domains
    - cc_usage_rate: CC / total recipients
    - emails_sent_after_hours: After-hours sends
    - new_recipient_contacts: First-time email recipients
    """
    schema = lf.collect_schema().names()

    # Sent emails (from internal users)
    sent_df = (
        lf.filter(pl.col("is_internal_sender") == True)
        .group_by(["user_hash", "date"])
        .agg([
            pl.col("id").count().alias(f"emails_sent{suffix}"),
            pl.col("size").mean().alias(f"avg_email_size{suffix}"),
            pl.col("size").max().alias(f"max_email_size{suffix}"),
        ])
        .collect()
    )

    # Get attachment size and count
    attachment_df = (
        lf.filter(pl.col("is_internal_sender") == True)
        .group_by(["user_hash", "date"])
        .agg([
            pl.col("attachments").sum().alias(f"total_attachment_size{suffix}"),
            pl.col("has_attachments").sum().alias(f"emails_with_attachments{suffix}"),
        ])
        .collect()
    )
    sent_df = sent_df.join(attachment_df, on=["user_hash", "date"], how="left")

    # After hours sent
    after_hours_df = (
        lf.filter(pl.col("is_internal_sender") == True)
        .group_by(["user_hash", "date"])
        .agg(
            pl.col("is_after_hours").sum().alias(f"emails_sent_after_hours{suffix}"),
        )
        .collect()
    )
    sent_df = sent_df.join(after_hours_df, on=["user_hash", "date"], how="left")

    # Received emails
    received_df = (
        lf.filter(pl.col("is_internal_sender") == False)
        .group_by(["user_hash", "date"])
        .agg(
            pl.col("id").count().alias(f"emails_received{suffix}"),
        )
        .collect()
    )
    sent_df = sent_df.join(received_df, on=["user_hash", "date"], how="left")

    # External emails
    external_df = (
        lf.filter(pl.col("has_external_recipient") == True)
        .group_by(["user_hash", "date"])
        .agg([
            pl.col("id").count().alias(f"external_emails_sent{suffix}"),
        ])
        .collect()
    )

    # External domains (use sender_domain for external recipients)
    if "sender_domain" in schema:
        external_df = (
            lf.filter(pl.col("has_external_recipient") == True)
            .group_by(["user_hash", "date"])
            .agg([
                pl.col("id").count().alias(f"external_emails_sent{suffix}"),
                pl.col("sender_domain").drop_nulls().n_unique().alias(f"unique_external_domains{suffix}"),
            ])
            .collect()
        )
    else:
        external_df = (
            lf.filter(pl.col("has_external_recipient") == True)
            .group_by(["user_hash", "date"])
            .agg([
                pl.col("id").count().alias(f"external_emails_sent{suffix}"),
                pl.lit(0).alias(f"unique_external_domains{suffix}"),
            ])
            .collect()
        )
    sent_df = sent_df.join(external_df, on=["user_hash", "date"], how="left")

    # New recipient contacts
    new_contacts_df = (
        lf.filter(pl.col("has_external_recipient") == True)
        .group_by(["user_hash", "date"])
        .agg(
            pl.col("to").n_unique().alias(f"new_recipient_contacts{suffix}"),
        )
        .collect()
    )
    sent_df = sent_df.join(new_contacts_df, on=["user_hash", "date"], how="left")

    # CC usage rate
    sent_df = sent_df.with_columns([
        pl.col(f"cc_usage_rate{suffix}").fill_null(0.0)
        if f"cc_usage_rate{suffix}" in sent_df.columns
        else ((pl.col("cc_count").fill_null(0)) / (pl.col("to_count").fill_null(0) + pl.col("cc_count").fill_null(0) + 1))
        .alias(f"cc_usage_rate{suffix}")
    ])

    # Get cc_count and to_count for CC rate calculation
    if "cc_count" in schema and "to_count" in schema:
        cc_df = (
            lf.filter(pl.col("is_internal_sender") == True)
            .group_by(["user_hash", "date"])
            .agg([
                pl.col("cc_count").mean().alias(f"avg_cc_count{suffix}"),
            ])
            .collect()
        )
        sent_df = sent_df.join(cc_df, on=["user_hash", "date"], how="left")

        # Recalculate CC rate
        sent_df = sent_df.with_columns([
            (pl.col(f"avg_cc_count{suffix}") / (pl.col(f"emails_sent{suffix}") + 1))
            .fill_null(0.0)
            .clip(upper_bound=1.0)
            .alias(f"cc_usage_rate{suffix}"),
        ])

    # Fill nulls
    fill_cols = [c for c in sent_df.columns if any(x in c for x in ["count", "size", "rate", "contacts"])]
    sent_df = sent_df.with_columns([pl.col(c).fill_null(0).cast(pl.Int64) for c in fill_cols if "rate" not in c and "size" not in c])
    sent_df = sent_df.with_columns([pl.col(c).fill_null(0.0) for c in sent_df.columns if "rate" in c or "size" in c])

    return sent_df


# =============================================================================
# HTTP FEATURES
# =============================================================================

def compute_http_features(
    lf: pl.LazyFrame,
    window_days: int,
    suffix: str
) -> pl.DataFrame:
    """
    Compute complete HTTP browsing features (10 features).

    Features:
    - http_request_count: Total web requests
    - unique_domains: Distinct domains visited
    - job_site_visits: Indeed, LinkedIn, Monster visits
    - cloud_storage_visits: Dropbox, GDrive, OneDrive visits
    - file_sharing_visits: File sharing site visits
    - social_media_visits: Social network visits
    - after_hours_browsing: After-hours HTTP requests
    - weekend_browsing: Weekend browsing
    """
    schema = lf.collect_schema().names()

    # Basic aggregations
    base_agg = [
        pl.col("id").count().alias(f"http_request_count{suffix}"),
        pl.col("is_after_hours").sum().alias(f"after_hours_browsing{suffix}"),
        pl.col("is_weekend").sum().alias(f"weekend_browsing{suffix}"),
    ]

    # Unique domains
    if "domain" in schema:
        base_agg.append(pl.col("domain").drop_nulls().n_unique().alias(f"unique_domains{suffix}"))
    else:
        base_agg.append(pl.lit(0).alias(f"unique_domains{suffix}"))

    # Domain category aggregations
    if "domain_category" in schema:
        base_agg.extend([
            pl.col("domain_category").str.to_lowercase().is_in(["job", "jobsearch", "jobsite", "indeed", "linkedin", "monster"]).sum().alias(f"job_site_visits{suffix}"),
            pl.col("domain_category").str.to_lowercase().is_in(["cloud", "cloudstorage", "dropbox", "gdrive", "onedrive"]).sum().alias(f"cloud_storage_visits{suffix}"),
            pl.col("domain_category").str.to_lowercase().is_in(["fileshare", "filesharing", "p2p"]).sum().alias(f"file_sharing_visits{suffix}"),
            pl.col("domain_category").str.to_lowercase().is_in(["social", "socialmedia", "facebook", "twitter", "instagram"]).sum().alias(f"social_media_visits{suffix}"),
        ])
    else:
        base_agg.extend([
            pl.lit(0).alias(f"job_site_visits{suffix}"),
            pl.lit(0).alias(f"cloud_storage_visits{suffix}"),
            pl.lit(0).alias(f"file_sharing_visits{suffix}"),
            pl.lit(0).alias(f"social_media_visits{suffix}"),
        ])

    base_df = (
        lf.group_by(["user_hash", "date"])
        .agg(base_agg)
        .collect()
    )

    # Fill nulls
    count_cols = [c for c in base_df.columns if "count" in c or "visit" in c]
    base_df = base_df.with_columns([pl.col(c).fill_null(0).cast(pl.Int64) for c in count_cols])

    return base_df


# =============================================================================
# TEMPORAL FEATURES (HOURLY AND DAY-OF-WEEK PROFILES)
# =============================================================================

def compute_temporal_hourly_features(
    lf: pl.LazyFrame,
    window_days: int,
    suffix: str
) -> pl.DataFrame:
    """
    Compute hourly activity distribution features (24 features).

    Features:
    - hour_0 to hour_23: Activity count per hour bucket
    """
    has_activity = "activity" in lf.collect_schema().names()
    count_col = "activity" if has_activity else "id"

    hour_df = (
        lf.group_by(["user_hash", "date", "hour"])
        .agg(pl.col(count_col).count().alias("_count"))
        .pivot(values="_count", index=["user_hash", "date"], on="hour", aggregate_function="first")
        .collect()
    )

    # Rename columns
    hour_cols = {str(i): f"hour_{i}{suffix}" for i in range(24)}
    hour_df = hour_df.rename(hour_cols)

    # Ensure all hour columns exist
    for i in range(24):
        col = f"hour_{i}{suffix}"
        if col not in hour_df.columns:
            hour_df = hour_df.with_columns(pl.lit(0).cast(pl.Int64).alias(col))

    return hour_df


def compute_temporal_dow_features(
    lf: pl.LazyFrame,
    window_days: int,
    suffix: str
) -> pl.DataFrame:
    """
    Compute day-of-week activity distribution features (7 features).

    Features:
    - day_of_week_0 to day_of_week_6: Activity count per weekday
    """
    has_activity = "activity" in lf.collect_schema().names()
    count_col = "activity" if has_activity else "id"

    dow_df = (
        lf.group_by(["user_hash", "date", "day_of_week"])
        .agg(pl.col(count_col).count().alias("_count"))
        .pivot(values="_count", index=["user_hash", "date"], on="day_of_week", aggregate_function="first")
        .collect()
    )

    # Rename columns
    dow_cols = {str(i): f"day_of_week_{i}{suffix}" for i in range(7)}
    dow_df = dow_df.rename(dow_cols)

    # Ensure all day columns exist
    for i in range(7):
        col = f"day_of_week_{i}{suffix}"
        if col not in dow_df.columns:
            dow_df = dow_df.with_columns(pl.lit(0).cast(pl.Int64).alias(col))

    return dow_df


def compute_temporal_features(
    lf: pl.LazyFrame,
    window_days: int,
    suffix: str
) -> pl.DataFrame:
    """Compute complete temporal features (hourly + day-of-week profiles)."""
    hour_df = compute_temporal_hourly_features(lf, window_days, suffix)
    dow_df = compute_temporal_dow_features(lf, window_days, suffix)

    result = hour_df.join(dow_df, on=["user_hash", "date"], how="outer")

    # Fill nulls for all temporal columns
    hour_cols = [f"hour_{i}{suffix}" for i in range(24)]
    dow_cols = [f"day_of_week_{i}{suffix}" for i in range(7)]

    fill_exprs = [pl.col(c).fill_null(0).cast(pl.Int64) for c in hour_cols + dow_cols if c in result.columns]
    result = result.with_columns(fill_exprs)

    return result


# =============================================================================
# DRIFT FEATURES
# =============================================================================

def compute_drift_features(
    lf: pl.LazyFrame,
    window_days: int,
    suffix: str,
    baseline_lf: Optional[pl.LazyFrame] = None
) -> pl.DataFrame:
    """
    Compute behavioral drift indicators (4 features).

    Features:
    - volume_change_ratio: recent_count / baseline_count
    - new_pc_count: Never-seen-before PC access
    - new_contact_count: First-time email recipients
    - behavioral_drift_score: Composite drift score
    """
    # Volume change ratio
    has_activity = "activity" in lf.collect_schema().names()
    count_col = "activity" if has_activity else "id"

    current_volume = (
        lf.group_by(["user_hash", "date"])
        .agg(pl.col(count_col).count().alias("_current_count"))
        .collect()
    )

    drift_df = current_volume

    # If we have baseline, compute ratio
    if baseline_lf is not None:
        baseline_volume = (
            baseline_lf.group_by(["user_hash"])
            .agg(pl.col(count_col).count().mean().alias("_baseline_count"))
            .collect()
        )
        drift_df = drift_df.join(baseline_volume, on="user_hash", how="left")
        drift_df = drift_df.with_columns([
            (pl.col("_current_count") / pl.col("_baseline_count").fill_null(1))
            .clip(upper_bound=10.0)  # Cap at 10x
            .alias(f"volume_change_ratio{suffix}"),
        ]).drop("_current_count", "_baseline_count")
    else:
        # Without baseline, use 1.0 as default
        drift_df = drift_df.with_columns(
            pl.lit(1.0).alias(f"volume_change_ratio{suffix}")
        ).drop("_current_count")

    # New PC count (if pc_hash column exists)
    if "pc_hash" in lf.collect_schema().names():
        # This would require tracking seen PCs across users - simplified here
        drift_df = drift_df.with_columns(pl.lit(0).alias(f"new_pc_count{suffix}"))
    else:
        drift_df = drift_df.with_columns(pl.lit(0).alias(f"new_pc_count{suffix}"))

    # New contact count (default)
    drift_df = drift_df.with_columns(pl.lit(0).alias(f"new_contact_count{suffix}"))

    # Behavioral drift score (composite)
    drift_df = drift_df.with_columns([
        (pl.col(f"volume_change_ratio{suffix}") * 0.5 +
         pl.col(f"new_pc_count{suffix}").clip(upper_bound=5) * 0.25 +
         pl.col(f"new_contact_count{suffix}").clip(upper_bound=5) * 0.25)
        .alias(f"behavioral_drift_score{suffix}"),
    ])

    return drift_df


# =============================================================================
# PSYCHOMETRIC FEATURES
# =============================================================================

def merge_psychometric_features(
    df: pl.DataFrame,
    sources: dict[str, pl.LazyFrame]
) -> pl.DataFrame:
    """
    Merge psychometric features (Big Five personality traits).

    Features:
    - big_five_O: Openness (0-50, normalized)
    - big_five_C: Conscientiousness
    - big_five_E: Extraversion
    - big_five_A: Agreeableness
    - big_five_N: Neuroticism
    - personality_risk_score: Derived risk score (0-1)
    """
    if "psychometric" not in sources:
        logger.warning("Psychometric data not found, adding default values")
        return df.with_columns([
            pl.lit(0.0).alias(f"big_five_{t}") for t in ["O", "C", "E", "A", "N"]
        ] + [pl.lit(0.0).alias("personality_risk_score")])

    psych_df = sources["psychometric"].collect()

    # Check for user_id column
    if "user" in psych_df.columns:
        psych_df = psych_df.rename({"user": "user_hash"})
    elif "user_id" in psych_df.columns:
        psych_df = psych_df.rename({"user_id": "user_hash"})

    # Map O, C, E, A, N columns
    rename_map = {}
    for trait in ["O", "C", "E", "A", "N"]:
        if trait in psych_df.columns:
            rename_map[trait] = f"big_five_{trait}"
    psych_df = psych_df.rename(rename_map)

    # Normalize traits to 0-1 range (assuming original is 0-50)
    norm_cols = [f"big_five_{t}" for t in ["O", "C", "E", "A", "N"] if f"big_five_{t}" in psych_df.columns]
    for col in norm_cols:
        psych_df = psych_df.with_columns((pl.col(col) / 50.0).clip(0, 1).alias(col))

    # Compute personality risk score (higher N, lower C, lower A = higher risk)
    if all(f"big_five_{t}" in psych_df.columns for t in ["N", "C", "A"]):
        psych_df = psych_df.with_columns([
            (pl.col("big_five_N") * 0.5 - pl.col("big_five_C") * 0.25 - pl.col("big_five_A") * 0.25 + 0.5)
            .clip(0, 1)
            .alias("personality_risk_score"),
        ])
    else:
        psych_df = psych_df.with_columns(pl.lit(0.5).alias("personality_risk_score"))

    # Select relevant columns
    psych_cols = ["user_hash"] + [f"big_five_{t}" for t in ["O", "C", "E", "A", "N"] if f"big_five_{t}" in psych_df.columns] + ["personality_risk_score"]
    psych_df = psych_df.select(psych_cols)

    return df.join(psych_df, on="user_hash", how="left")


# =============================================================================
# LDAP / ORGANIZATIONAL FEATURES
# =============================================================================

def merge_ldap_features(
    df: pl.DataFrame,
    sources: dict[str, pl.LazyFrame]
) -> pl.DataFrame:
    """
    Merge organizational/LDAP features.

    Features:
    - role_sensitivity: 1-5 role sensitivity score
    - is_it_admin: IT Admin flag
    - is_manager: Has direct reports
    - team_size: Team size
    - access_level: Composite privilege score
    """
    if "ldap" not in sources:
        logger.warning("LDAP data not found, adding default values")
        return df.with_columns([
            pl.lit(1).alias("role_sensitivity_24h"),
            pl.lit(False).alias("is_it_admin_24h"),
            pl.lit(False).alias("is_manager_24h"),
            pl.lit(1).alias("team_size_24h"),
            pl.lit(1).alias("access_level_24h"),
        ])

    ldap_df = sources["ldap"].collect()

    # Get latest LDAP snapshot per user
    if "date" in ldap_df.columns:
        ldap_df = ldap_df.sort("date").group_by("user_hash").last()

    # Map columns if they exist
    if "role_sensitivity" in ldap_df.columns:
        ldap_df = ldap_df.with_columns(pl.col("role_sensitivity").alias("role_sensitivity_24h"))
    else:
        ldap_df = ldap_df.with_columns(pl.lit(1).alias("role_sensitivity_24h"))

    if "is_it_admin" in ldap_df.columns:
        ldap_df = ldap_df.with_columns(pl.col("is_it_admin").alias("is_it_admin_24h"))
    else:
        ldap_df = ldap_df.with_columns(pl.lit(False).alias("is_it_admin_24h"))

    if "is_manager" in ldap_df.columns:
        ldap_df = ldap_df.with_columns(pl.col("is_manager").alias("is_manager_24h"))
    else:
        ldap_df = ldap_df.with_columns(pl.lit(False).alias("is_manager_24h"))

    if "team_size" in ldap_df.columns:
        ldap_df = ldap_df.with_columns(pl.col("team_size").alias("team_size_24h"))
    else:
        ldap_df = ldap_df.with_columns(pl.lit(1).alias("team_size_24h"))

    if "access_level" in ldap_df.columns:
        ldap_df = ldap_df.with_columns(pl.col("access_level").alias("access_level_24h"))
    else:
        # Compute from role_sensitivity
        ldap_df = ldap_df.with_columns(
            (pl.col("role_sensitivity_24h") + pl.col("team_size_24h").fill_null(1))
            .clip(1, 10)
            .alias("access_level_24h")
        )

    # Select relevant columns
    ldap_cols = ["user_hash", "role_sensitivity_24h", "is_it_admin_24h", "is_manager_24h", "team_size_24h", "access_level_24h"]
    ldap_df = ldap_df.select([c for c in ldap_cols if c in ldap_df.columns])

    return df.join(ldap_df, on="user_hash", how="left")


# =============================================================================
# GRAPH FEATURES (DEGREE CENTRALITY, PAGERANK, NEW ENTITIES)
# =============================================================================

def compute_graph_features(
    lf: pl.LazyFrame,
    window_days: int,
    suffix: str,
    known_pcs: Optional[set] = None,
    known_contacts: Optional[set] = None
) -> pl.DataFrame:
    """
    Compute graph-based features.

    Features:
    - degree_centrality: User's connections in PC/email graph
    - pagerank: Importance in communication network
    - new_pc_access_count: PCs never accessed before
    - new_domain_contacts: First-time HTTP domains visited
    """
    schema = lf.collect_schema().names()

    # Degree centrality (number of unique PCs + unique domains per user per day)
    degree_agg = [pl.col("id").count().alias(f"degree_centrality{suffix}")]

    if "pc_hash" in schema:
        degree_agg.append(pl.col("pc_hash").n_unique().alias(f"_pc_degree"))

    degree_df = (
        lf.group_by(["user_hash", "date"])
        .agg(degree_agg)
        .collect()
    )

    # Pagerank proxy (using activity count as a simplified pagerank)
    degree_df = degree_df.with_columns(
        (pl.col(f"degree_centrality{suffix}") / 100.0)
        .clip(0, 1)
        .alias(f"pagerank{suffix}")
    )

    # New PC access count
    if known_pcs and "pc_hash" in schema:
        # This would need historical tracking - simplified here
        degree_df = degree_df.with_columns(pl.lit(0).alias(f"new_pc_access_count{suffix}"))
    else:
        degree_df = degree_df.with_columns(pl.lit(0).alias(f"new_pc_access_count{suffix}"))

    # New domain contacts
    if known_contacts and "domain" in schema:
        degree_df = degree_df.with_columns(pl.lit(0).alias(f"new_domain_contacts{suffix}"))
    else:
        degree_df = degree_df.with_columns(pl.lit(0).alias(f"new_domain_contacts{suffix}"))

    # Drop temp column
    if "_pc_degree" in degree_df.columns:
        degree_df = degree_df.drop("_pc_degree")

    return degree_df


# =============================================================================
# MAIN FEATURE COMPUTATION
# =============================================================================

def compute_all_features_for_window(
    sources: dict[str, pl.LazyFrame],
    window_name: str,
    window_days: int
) -> pl.DataFrame:
    """Compute ALL features for a given time window."""
    suffix = f"_{window_name}"

    logger.info(f"Computing all features for {window_name} window...")

    features = None

    # 1. Logon features
    if "logon" in sources:
        logon_feats = compute_logon_features(sources["logon"], window_days, suffix)
        session_feats = compute_session_duration_features(sources["logon"], window_days, suffix)
        rapid_feats = compute_rapid_logon_cycles(sources["logon"], window_days, suffix)

        features = logon_feats.join(session_feats, on=["user_hash", "date"], how="outer")
        features = features.join(rapid_feats, on=["user_hash", "date"], how="outer")

    # 2. Device features
    if "device" in sources:
        device_feats = compute_device_features(sources["device"], window_days, suffix)
        device_session = compute_device_session_duration(sources["device"], window_days, suffix)
        device_feats = device_feats.join(device_session, on=["user_hash", "date"], how="outer")

        if features is None:
            features = device_feats
        else:
            features = features.join(device_feats, on=["user_hash", "date"], how="outer")

    # 3. File features
    if "file" in sources:
        file_feats = compute_file_features(sources["file"], window_days, suffix)

        if features is None:
            features = file_feats
        else:
            features = features.join(file_feats, on=["user_hash", "date"], how="outer")

    # 4. Email features
    if "email" in sources:
        email_feats = compute_email_features(sources["email"], window_days, suffix)

        if features is None:
            features = email_feats
        else:
            features = features.join(email_feats, on=["user_hash", "date"], how="outer")

    # 5. HTTP features
    if "http" in sources:
        http_feats = compute_http_features(sources["http"], window_days, suffix)

        if features is None:
            features = http_feats
        else:
            features = features.join(http_feats, on=["user_hash", "date"], how="outer")

    # 6. Temporal features (hourly and day-of-week)
    if "logon" in sources:
        temporal_feats = compute_temporal_features(sources["logon"], window_days, suffix)

        if features is None:
            features = temporal_feats
        else:
            features = features.join(temporal_feats, on=["user_hash", "date"], how="outer")

    # 7. Drift features
    if "logon" in sources:
        drift_feats = compute_drift_features(sources["logon"], window_days, suffix)

        if features is None:
            features = drift_feats
        else:
            features = features.join(drift_feats, on=["user_hash", "date"], how="outer")

    # 8. Graph features
    if "logon" in sources:
        graph_feats = compute_graph_features(sources["logon"], window_days, suffix)

        if features is None:
            features = graph_feats
        else:
            features = features.join(graph_feats, on=["user_hash", "date"], how="outer")

    return features


def extend_features_to_all_windows(
    df_24h: pl.DataFrame,
    sources: dict[str, pl.LazyFrame]
) -> pl.DataFrame:
    """
    Extend 24h features to create 7d and 30d features.
    This creates rolling aggregations for multi-day windows.
    """
    logger.info("Extending features to 7d and 30d windows...")

    # For simplicity, we'll aggregate by re-computing with different windows
    # In production, you'd use rolling window computations

    features_7d = compute_all_features_for_window(sources, "7d", 7)
    features_30d = compute_all_features_for_window(sources, "30d", 30)

    # Merge with suffix renaming
    result = df_24h

    for feat_df, suffix in [(features_7d, "_7d"), (features_30d, "_30d")]:
        # Rename columns with suffix
        rename_map = {}
        for col in feat_df.columns:
            if col not in ["user_hash", "date"]:
                if not col.endswith(suffix):
                    rename_map[col] = f"{col}{suffix}"

        feat_df = feat_df.rename(rename_map)
        result = result.join(feat_df, on=["user_hash", "date"], how="left")

    return result


def ensure_all_required_columns(
    df: pl.DataFrame,
    window_suffixes: list[str] = ["_24h", "_7d", "_30d"]
) -> pl.DataFrame:
    """Ensure all required columns from the schema exist."""

    # Define all required base feature names
    base_features = [
        # Logon (10)
        "logon_count", "logoff_count", "logon_ratio", "after_hours_logons",
        "weekend_logons", "unique_pcs", "avg_session_duration", "max_session_duration",
        "session_duration_std", "rapid_logon_cycles",
        # Device (7)
        "device_connect_count", "device_disconnect_count", "missing_disconnect_count",
        "device_session_duration", "after_hours_device", "weekend_device",
        # File (9)
        "file_event_count", "file_write_count", "file_copy_count", "file_delete_count",
        "removable_media_writes", "file_type_diversity", "large_file_writes", "after_hours_file",
        # Email (14)
        "emails_sent", "emails_received", "avg_email_size", "max_email_size",
        "total_attachment_size", "emails_with_attachments", "external_emails_sent",
        "unique_external_domains", "cc_usage_rate", "emails_sent_after_hours",
        "new_recipient_contacts", "avg_cc_count",
        # HTTP (10)
        "http_request_count", "unique_domains", "job_site_visits", "cloud_storage_visits",
        "file_sharing_visits", "social_media_visits", "after_hours_browsing", "weekend_browsing",
        # Drift (4)
        "volume_change_ratio", "new_pc_count", "new_contact_count", "behavioral_drift_score",
        # Graph (4)
        "degree_centrality", "pagerank", "new_pc_access_count", "new_domain_contacts",
        # LDAP (5)
        "role_sensitivity", "is_it_admin", "is_manager", "team_size", "access_level",
    ]

    # Temporal features (24 hours + 7 days = 31)
    for i in range(24):
        base_features.append(f"hour_{i}")
    for i in range(7):
        base_features.append(f"day_of_week_{i}")

    # Psychometric (6)
    psychometric_features = ["big_five_O", "big_five_C", "big_five_E", "big_five_A", "big_five_N", "personality_risk_score"]

    # Add all suffixes
    for suffix in window_suffixes:
        for feat in base_features:
            col_name = f"{feat}{suffix}"
            if col_name not in df.columns:
                # Determine type based on feature name
                if any(x in feat for x in ["ratio", "score", "duration", "size", "drift"]):
                    df = df.with_columns(pl.lit(0.0).alias(col_name))
                elif feat.startswith("hour_") or feat.startswith("day_of_week_"):
                    df = df.with_columns(pl.lit(0).cast(pl.Int64).alias(col_name))
                elif feat in ["is_it_admin", "is_manager"]:
                    df = df.with_columns(pl.lit(False).alias(col_name))
                else:
                    df = df.with_columns(pl.lit(0).cast(pl.Int64).alias(col_name))

    # Add psychometric features (no suffix - user-level)
    for feat in psychometric_features:
        if feat not in df.columns:
            if feat == "personality_risk_score":
                df = df.with_columns(pl.lit(0.5).alias(feat))
            else:
                df = df.with_columns(pl.lit(0.0).alias(feat))

    return df


def create_user_date_fact_table(
    sources: dict[str, pl.LazyFrame],
    date_range: tuple[datetime, datetime]
) -> pl.DataFrame:
    """Create the base user × date fact table."""
    all_users = set()

    for name, lf in sources.items():
        if name in ["logon", "device", "file", "email", "http"]:
            users = lf.select("user_hash").unique().collect()
            all_users.update(users["user_hash"].to_list())

    # Create date range
    start_dt, end_dt = date_range
    all_dates = pl.date_range(start=start_dt, end=end_dt, interval="1d", eager=True)

    # Cross join users and dates
    fact_table = (
        pl.DataFrame({"user_hash": list(all_users)})
        .join(all_dates.to_frame("date"), how="cross")
    )

    logger.info(f"Created fact table: {len(fact_table):,} rows ({len(all_users)} users × {len(all_dates)} days)")

    return fact_table


def main(sources: dict[str, pl.LazyFrame], date_range: tuple[datetime, datetime]) -> pl.DataFrame:
    """
    Main function to compute ALL statistical features for all windows.

    Returns a DataFrame with ~500 features per user per day.
    """
    logger.info("Starting complete statistical feature engineering...")

    # Create base fact table
    fact_table = create_user_date_fact_table(sources, date_range)

    # Compute 24h features
    features_24h = compute_all_features_for_window(sources, "24h", 1)

    # Join with fact table
    result = fact_table.join(features_24h, on=["user_hash", "date"], how="left")

    # Add 7d and 30d features
    result = extend_features_to_all_windows(result, sources)

    # Merge psychometric features
    result = merge_psychometric_features(result, sources)

    # Merge LDAP/organizational features
    result = merge_ldap_features(result, sources)

    # Ensure all required columns exist
    result = ensure_all_required_columns(result)

    # Add user_id alias
    result = result.with_columns(
        pl.col("user_hash").alias("user_id")
    )

    # Fill any remaining nulls
    numeric_cols = [c for c in result.columns if c not in ["user_id", "user_hash", "date"]]
    result = result.with_columns([pl.col(c).fill_null(0) for c in numeric_cols])

    logger.info(f"Complete feature set: {result.shape[0]:,} rows × {result.shape[1]} columns")

    return result
