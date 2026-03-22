"""
Statistical Feature Engineering Module for CERT Insider Threat Detection.

This module computes behavioral features from normalized log data across
multiple time windows (24h, 7d, 30d) for insider threat detection.

Key Features:
- Logon Features: frequency, after-hours, weekend, unique PCs, session duration
- Device Features: connect/disconnect, missing disconnects, session duration
- File Features: operations, removable media, file types, large files
- Email Features: volume, attachments, external recipients, domains
- HTTP Features: domains, sensitive categories, browsing patterns
- Temporal Features: hour and day-of-week distributions
- Drift Features: volume changes, new entities, behavioral drift
- Organizational Features: role sensitivity, admin status, team info
- Psychometric Features: Big Five personality traits

Author: NCPI Insider Threat Detection Team
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import polars as pl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


class StatisticalFeatures:
    """
    Wrapper class for statistical feature computation.
    Provides class-based API for the pipeline.
    """

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        working_hours_start: int = 8,
        working_hours_end: int = 18,
        work_days: list = None,
    ):
        self.data_dir = data_dir
        self.working_hours_start = working_hours_start
        self.working_hours_end = working_hours_end
        self.work_days = work_days or [0, 1, 2, 3, 4]  # Mon-Fri

    def compute_logon_features(self, df: pl.DataFrame, window: str) -> pl.DataFrame:
        """Compute logon features for given window."""
        cfg = WindowConfig.from_str(window)
        return compute_logon_features(
            df.lazy() if hasattr(df, 'lazy') else df,
            cfg.days,
            cfg.suffix
        )

    def compute_device_features(self, df: pl.DataFrame, window: str) -> pl.DataFrame:
        """Compute device features for given window."""
        cfg = WindowConfig.from_str(window)
        return compute_device_features(
            df.lazy() if hasattr(df, 'lazy') else df,
            cfg.days,
            cfg.suffix
        )

    def compute_file_features(self, df: pl.DataFrame, window: str) -> pl.DataFrame:
        """Compute file features for given window."""
        cfg = WindowConfig.from_str(window)
        return compute_file_features(
            df.lazy() if hasattr(df, 'lazy') else df,
            cfg.days,
            cfg.suffix
        )

    def compute_email_features(self, df: pl.DataFrame, window: str) -> pl.DataFrame:
        """Compute email features for given window."""
        cfg = WindowConfig.from_str(window)
        return compute_email_features(
            df.lazy() if hasattr(df, 'lazy') else df,
            cfg.days,
            cfg.suffix
        )

    def compute_http_features(self, df: pl.DataFrame, window: str) -> pl.DataFrame:
        """Compute HTTP features for given window."""
        cfg = WindowConfig.from_str(window)
        return compute_http_features(
            df.lazy() if hasattr(df, 'lazy') else df,
            cfg.days,
            cfg.suffix
        )

    def compute_hourly_profile(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute hourly activity profile."""
        return compute_temporal_features(
            df.lazy() if hasattr(df, 'lazy') else df,
            1,
            "_24h"
        )

    def merge_all_features(
        self,
        logon_df: pl.DataFrame,
        device_df: Optional[pl.DataFrame] = None,
        file_df: Optional[pl.DataFrame] = None,
        email_df: Optional[pl.DataFrame] = None,
        http_df: Optional[pl.DataFrame] = None,
        window: str = "24h",
    ) -> pl.DataFrame:
        """Merge all feature DataFrames into one."""
        merged = logon_df
        for df, name in [(device_df, "device"), (file_df, "file"),
                         (email_df, "email"), (http_df, "http")]:
            if df is not None and len(df) > 0:
                on_cols = ["user_hash", "date"] if "date" in df.columns else ["user_hash"]
                # Use suffix to handle duplicate columns
                merged = merged.join(df, on=on_cols, how="left", suffix="_right")
                # Drop any right suffix columns (duplicates from join)
                right_cols = [c for c in merged.columns if c.endswith("_right")]
                if right_cols:
                    merged = merged.drop(right_cols)
        return merged


def load_parquet(path: Path) -> pl.LazyFrame:
    """Load a parquet file as a Polars LazyFrame."""
    return pl.scan_parquet(path)


def load_all_sources() -> dict[str, pl.LazyFrame]:
    """Load all normalized data sources."""
    sources = {}
    for name in ["logon", "device", "file", "email", "http"]:
        path = DATA_DIR / f"{name}.parquet"
        if path.exists():
            sources[name] = load_parquet(path)
            logger.info(f"Loaded {name} from {path}")
        else:
            logger.warning(f"{path} not found, skipping {name}")

    psych_path = DATA_DIR / "psychometric.parquet"
    ldap_path = DATA_DIR / "ldap.parquet"

    if psych_path.exists():
        sources["psychometric"] = load_parquet(psych_path)
        logger.info(f"Loaded psychometric from {psych_path}")

    if ldap_path.exists():
        sources["ldap"] = load_parquet(ldap_path)
        logger.info(f"Loaded ldap from {ldap_path}")

    return sources


def compute_logon_features(
    lf: pl.LazyFrame,
    window_days: int,
    suffix: str
) -> pl.DataFrame:
    """Compute logon-related behavioral features."""
    logon_df = (
        lf.filter(pl.col("activity").str.to_lowercase() == "logon")
        .group_by(["user_hash", "date"])
        .agg(
            pl.col("activity").count().alias(f"logon_count{suffix}"),
            pl.col("is_after_hours").sum().alias(f"after_hours_logons{suffix}"),
            pl.col("is_weekend").sum().alias(f"weekend_logons{suffix}"),
            pl.col("pc_hash").n_unique().alias(f"unique_pcs{suffix}"),
        )
        .collect()
    )

    logoff_df = (
        lf.filter(pl.col("activity").str.to_lowercase() == "logoff")
        .group_by(["user_hash", "date"])
        .agg(
            pl.col("activity").count().alias(f"logoff_count{suffix}"),
        )
        .collect()
    )

    result = logon_df.join(logoff_df, on=["user_hash", "date"], how="outer")

    result = result.with_columns([
        pl.col(f"logon_count{suffix}").fill_null(0),
        pl.col(f"logoff_count{suffix}").fill_null(0),
        pl.col(f"after_hours_logons{suffix}").fill_null(0),
        pl.col(f"weekend_logons{suffix}").fill_null(0),
        pl.col(f"unique_pcs{suffix}").fill_null(0),
        (pl.col(f"logon_count{suffix}") / (pl.col(f"logoff_count{suffix}") + 1))
        .alias(f"logon_ratio{suffix}"),
    ])

    return result


def compute_session_features(
    lf: pl.LazyFrame,
    window_days: int,
    suffix: str
) -> pl.DataFrame:
    """Compute session duration statistics from logon/logoff pairs."""
    session_lf = lf.filter(
        pl.col("activity").str.to_lowercase().is_in(["logon", "logoff"])
    ).sort(["user_hash", "pc_hash", "timestamp"])

    session_df = (
        session_lf
        .with_columns([
            pl.col("activity").shift(1).over(["user_hash", "pc_hash"]).alias("_prev_activity"),
            pl.col("timestamp").str.to_datetime().shift(1).over(["user_hash", "pc_hash"]).alias("_prev_timestamp"),
        ])
        .filter(
            (pl.col("activity").str.to_lowercase() == "logoff") &
            (pl.col("_prev_activity").str.to_lowercase() == "logon")
        )
        .with_columns([
            (pl.col("timestamp").str.to_datetime() - pl.col("_prev_timestamp"))
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
        pl.col(f"avg_session_duration{suffix}").fill_null(0),
        pl.col(f"max_session_duration{suffix}").fill_null(0),
        pl.col(f"session_duration_std{suffix}").fill_null(0),
    ])

    return session_df


def compute_rapid_logon_cycles(
    lf: pl.LazyFrame,
    window_days: int,
    suffix: str
) -> pl.DataFrame:
    """Compute rapid logon/logoff cycles (potential suspicious activity)."""
    rapid_lf = (
        lf.filter(pl.col("activity").str.to_lowercase().is_in(["logon", "logoff"]))
        .sort(["user_hash", "timestamp"])
        .with_columns([
            pl.col("activity").shift(1).over("user_hash").alias("_prev_activity"),
            pl.col("timestamp").str.to_datetime().shift(1).over("user_hash").alias("_prev_ts"),
        ])
        .filter(
            pl.col("activity").str.to_lowercase() == "logon"
        )
        .filter(
            pl.col("_prev_activity").str.to_lowercase() == "logon"
        )
        .with_columns([
            (pl.col("timestamp").str.to_datetime() - pl.col("_prev_ts"))
            .dt.total_seconds()
            .alias("_time_diff")
        ])
        .filter(pl.col("_time_diff") < 300)
    )

    rapid_df = (
        rapid_lf
        .group_by(["user_hash", "date"])
        .agg(pl.col("_time_diff").count().alias(f"rapid_logon_cycles{suffix}"))
        .collect()
    )

    rapid_df = rapid_df.with_columns(
        pl.col(f"rapid_logon_cycles{suffix}").fill_null(0)
    )

    return rapid_df


def compute_device_features(
    lf: pl.LazyFrame,
    window_days: int,
    suffix: str
) -> pl.DataFrame:
    """Compute device-related behavioral features."""
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

    disconnect_df = (
        lf.filter(pl.col("activity") == "Disconnect")
        .group_by(["user_hash", "date"])
        .agg(pl.col("activity").count().alias(f"device_disconnect_count{suffix}"))
        .collect()
    )

    device_df = connect_df.join(disconnect_df, on=["user_hash", "date"], how="outer")

    device_df = device_df.with_columns([
        pl.col(f"device_connect_count{suffix}").fill_null(0),
        pl.col(f"device_disconnect_count{suffix}").fill_null(0),
        pl.col(f"after_hours_device{suffix}").fill_null(0),
        pl.col(f"weekend_device{suffix}").fill_null(0),
        (pl.col(f"device_connect_count{suffix}") - pl.col(f"device_disconnect_count{suffix}"))
        .clip(0)
        .alias(f"missing_disconnect_count{suffix}"),
    ])

    return device_df


def compute_device_session_duration(
    lf: pl.LazyFrame,
    window_days: int,
    suffix: str
) -> pl.DataFrame:
    """Compute device session duration (connect to disconnect time)."""
    sorted_lf = lf.sort(["user_hash", "pc_hash", "timestamp"])

    device_session = (
        sorted_lf
        .with_columns([
            pl.col("activity").shift(1).over(["user_hash", "pc_hash"]).alias("_prev_activity"),
            pl.col("timestamp").str.to_datetime().shift(1).over(["user_hash", "pc_hash"]).alias("_prev_ts"),
        ])
        .filter(
            (pl.col("activity") == "Disconnect") &
            (pl.col("_prev_activity") == "Connect")
        )
        .with_columns([
            (pl.col("timestamp").str.to_datetime() - pl.col("_prev_ts"))
            .dt.total_seconds()
            .alias("_duration_sec")
        ])
        .filter(pl.col("_duration_sec") > 0)
        .group_by(["user_hash", "date"])
        .agg(pl.col("_duration_sec").sum().alias(f"device_session_duration{suffix}"))
        .collect()
    )

    device_session = device_session.with_columns(
        pl.col(f"device_session_duration{suffix}").fill_null(0)
    )

    return device_session


def compute_file_features(
    lf: pl.LazyFrame,
    window_days: int,
    suffix: str
) -> pl.DataFrame:
    """Compute file operation features."""
    # File parquet doesn't have 'activity' column - count by id instead
    has_activity = "activity" in lf.collect_schema().names()

    if has_activity:
        base_agg = [
            pl.col("activity").count().alias(f"file_event_count{suffix}"),
        ]
    else:
        base_agg = [
            pl.col("id").count().alias(f"file_event_count{suffix}"),
        ]

    # Check if operation_type column exists
    has_operation_type = "operation_type" in lf.collect_schema().names()

    if has_operation_type:
        base_agg.extend([
            pl.col("operation_type").str.to_lowercase().is_in(["write", "create"]).sum().alias(f"file_write_count{suffix}"),
            pl.col("operation_type").str.to_lowercase().is_in(["copy", "move"]).sum().alias(f"file_copy_count{suffix}"),
            pl.col("operation_type").str.to_lowercase().is_in(["delete", "remove"]).sum().alias(f"file_delete_count{suffix}"),
        ])
    else:
        # For file events without operation_type, use is_removable as proxy
        base_agg.extend([
            pl.col("is_removable").sum().alias(f"file_write_count{suffix}"),
            pl.lit(0).alias(f"file_copy_count{suffix}"),
            pl.lit(0).alias(f"file_delete_count{suffix}"),
        ])

    base_df = (
        lf.group_by(["user_hash", "date"])
        .agg(base_agg + [
            pl.col("is_removable").sum().alias(f"removable_media_writes{suffix}"),
            pl.col("file_extension").drop_nulls().n_unique().alias(f"file_type_diversity{suffix}"),
            pl.col("is_after_hours").sum().alias(f"after_hours_file{suffix}"),
        ])
        .collect()
    )

    # Large file writes - check content size if available
    has_content = "content" in lf.collect_schema().names()
    if has_operation_type and has_content:
        try:
            large_file_df = (
                lf.filter(pl.col("operation_type").str.to_lowercase().is_in(["write", "create"]))
                .with_columns(pl.col("content").str.len_bytes().alias("_content_size"))
                .filter(pl.col("_content_size") > 5_000_000)  # 5MB threshold
                .group_by(["user_hash", "date"])
                .agg(pl.col("id").count().alias(f"large_file_writes{suffix}"))
                .collect()
            )
            file_df = base_df.join(large_file_df, on=["user_hash", "date"], how="left")
        except Exception:
            file_df = base_df.with_columns(pl.lit(0).alias(f"large_file_writes{suffix}"))
    else:
        file_df = base_df.with_columns(pl.lit(0).alias(f"large_file_writes{suffix}"))

    count_cols = [c for c in file_df.columns if "count" in c or "writes" in c or "size" in c]
    file_df = file_df.with_columns([pl.col(c).fill_null(0) for c in count_cols])

    return file_df


def compute_email_features(
    lf: pl.LazyFrame,
    window_days: int,
    suffix: str
) -> pl.DataFrame:
    """Compute email behavior features."""
    # Email parquet doesn't have 'activity' - count by id instead
    sent_df = (
        lf.filter(pl.col("is_internal_sender") == True)
        .group_by(["user_hash", "date"])
        .agg([
            pl.col("id").count().alias(f"emails_sent{suffix}"),
            pl.col("size").mean().alias(f"avg_email_size{suffix}"),
            pl.col("size").max().alias(f"max_email_size{suffix}"),
            pl.col("attachments").sum().alias(f"total_attachment_size{suffix}"),
            pl.col("has_attachments").sum().alias(f"emails_with_attachments{suffix}"),
            pl.col("cc_count").mean().alias(f"avg_cc_count{suffix}"),
            pl.col("is_after_hours").sum().alias(f"emails_sent_after_hours{suffix}"),
        ])
        .collect()
    )

    received_df = (
        lf.filter(pl.col("is_internal_sender") == False)
        .group_by(["user_hash", "date"])
        .agg(pl.col("id").count().alias(f"emails_received{suffix}"))
        .collect()
    )

    external_df = (
        lf.filter(pl.col("has_external_recipient") == True)
        .group_by(["user_hash", "date"])
        .agg([
            pl.col("id").count().alias(f"external_emails_sent{suffix}"),
            pl.col("sender_domain").drop_nulls().n_unique().alias(f"unique_external_domains{suffix}"),
        ])
        .collect()
    )

    new_contacts_df = (
        lf.filter(pl.col("has_external_recipient") == True)
        .group_by(["user_hash", "date"])
        .agg(pl.col("to").n_unique().alias(f"new_recipient_contacts{suffix}"))
        .collect()
    )

    email_df = sent_df.join(received_df, on=["user_hash", "date"], how="outer")
    email_df = email_df.join(external_df, on=["user_hash", "date"], how="left")
    email_df = email_df.join(new_contacts_df, on=["user_hash", "date"], how="left")

    email_df = email_df.with_columns([
        (pl.col(f"avg_cc_count{suffix}") / (pl.col(f"emails_sent{suffix}") + 1))
        .fill_null(0)
        .alias(f"cc_usage_rate{suffix}")
    ])

    numeric_cols = [c for c in email_df.columns if any(x in c for x in ["count", "size", "rate"])]
    email_df = email_df.with_columns([pl.col(c).fill_null(0) for c in numeric_cols])

    return email_df


def compute_http_features(
    lf: pl.LazyFrame,
    window_days: int,
    suffix: str
) -> pl.DataFrame:
    """Compute HTTP browsing behavior features."""
    # HTTP parquet doesn't have 'activity' - count by id instead
    has_activity = "activity" in lf.collect_schema().names()

    base_agg = [
        pl.col("id").count().alias(f"http_request_count{suffix}"),
        pl.col("domain").drop_nulls().n_unique().alias(f"unique_domains{suffix}"),
        pl.col("is_after_hours").sum().alias(f"after_hours_browsing{suffix}"),
        pl.col("is_weekend").sum().alias(f"weekend_browsing{suffix}"),
    ]

    # Domain category aggregations if the column exists
    has_category = "domain_category" in lf.collect_schema().names()
    if has_category:
        base_agg.extend([
            pl.col("domain_category").str.to_lowercase().is_in(["job", "jobsearch", "jobsite"]).sum().alias(f"job_site_visits{suffix}"),
            pl.col("domain_category").str.to_lowercase().is_in(["cloud", "cloudstorage"]).sum().alias(f"cloud_storage_visits{suffix}"),
            pl.col("domain_category").str.to_lowercase().is_in(["fileshare", "filesharing", "p2p"]).sum().alias(f"file_sharing_visits{suffix}"),
            pl.col("domain_category").str.to_lowercase().is_in(["social", "socialmedia"]).sum().alias(f"social_media_visits{suffix}"),
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

    count_cols = [c for c in base_df.columns if "count" in c or "visit" in c]
    base_df = base_df.with_columns([pl.col(c).fill_null(0) for c in count_cols])

    return base_df


def compute_temporal_features(
    lf: pl.LazyFrame,
    window_days: int,
    suffix: str
) -> pl.DataFrame:
    """Compute one-hot encoded temporal features for hours and days."""
    # Use id instead of activity for counting (more universal)
    has_activity = "activity" in lf.collect_schema().names()
    count_col = "activity" if has_activity else "id"

    hour_df = (
        lf.group_by(["user_hash", "date", "hour"])
        .agg(pl.col(count_col).count().alias("_count"))
        .pivot(values="_count", index=["user_hash", "date"], on="hour", aggregate_function="first")
        .collect()
    )

    hour_cols = {}
    for i in range(24):
        col_name = f"hour_{i}{suffix}"
        hour_cols[str(i)] = col_name
    hour_df = hour_df.rename(hour_cols)

    all_hour_cols = [f"hour_{i}{suffix}" for i in range(24)]
    for col in all_hour_cols:
        if col not in hour_df.columns:
            hour_df = hour_df.with_columns(pl.lit(0).alias(col))

    dow_df = (
        lf.group_by(["user_hash", "date", "day_of_week"])
        .agg(pl.col(count_col).count().alias("_count"))
        .pivot(values="_count", index=["user_hash", "date"], on="day_of_week", aggregate_function="first")
        .collect()
    )

    dow_cols = {}
    for i in range(7):
        col_name = f"day_of_week_{i}{suffix}"
        dow_cols[str(i)] = col_name
    dow_df = dow_df.rename(dow_cols)

    all_dow_cols = [f"day_of_week_{i}{suffix}" for i in range(7)]
    for col in all_dow_cols:
        if col not in dow_df.columns:
            dow_df = dow_df.with_columns(pl.lit(0).alias(col))

    temporal_df = hour_df.join(dow_df, on=["user_hash", "date"], how="outer")

    return temporal_df


def compute_drift_features(
    lf: pl.LazyFrame,
    window_days: int,
    suffix: str
) -> pl.DataFrame:
    """Compute behavioral drift indicators."""
    volume_df = (
        lf.group_by(["user_hash", "date"])
        .agg(pl.col("activity").count().alias("_current_volume"))
        .with_columns(pl.col("date").str.to_date())
        .sort(["user_hash", "date"])
        .collect()
    )

    volume_with_baseline = (
        volume_df
        .with_columns([
            pl.col("_current_volume")
            .rolling_mean(window_size=30, by="date", min_periods=1)
            .over("user_hash")
            .alias("_rolling_avg")
        ])
    )

    drift_df = (
        volume_with_baseline
        .with_columns([
            ((pl.col("_current_volume") - pl.col("_rolling_avg")) / (pl.col("_rolling_avg") + 1))
            .alias(f"volume_change_ratio{suffix}")
        ])
        .select(["user_hash", "date", f"volume_change_ratio{suffix}"])
    )

    drift_df = drift_df.with_columns(
        pl.col(f"volume_change_ratio{suffix}").fill_null(0)
    )

    return drift_df


def compute_new_entity_features(
    lf: pl.LazyFrame,
    window_days: int,
    suffix: str,
    entity_type: str = "pc"
) -> pl.DataFrame:
    """Compute count of new entities seen for the first time."""
    entity_col = f"{entity_type}_hash"
    entity_col_name = "pc_hash" if entity_type == "pc" else entity_col

    entity_first_seen = (
        lf.group_by(["user_hash", entity_col_name])
        .agg(
            pl.col("date").str.to_date().min().alias("_first_seen")
        )
    )

    new_entities = (
        entity_first_seen
        .with_columns(pl.col("date").str.to_date().alias("_current_date"))
        .filter(pl.col("_first_seen") <= pl.col("_current_date"))
        .group_by(["user_hash", "_current_date"])
        .agg(pl.col(entity_col_name).n_unique().alias(f"new_{entity_type}_count{suffix}"))
        .with_columns(pl.col("_current_date").cast(pl.Utf8).alias("date"))
        .collect()
    )

    new_entities = new_entities.with_columns(
        pl.col(f"new_{entity_type}_count{suffix}").fill_null(0)
    )

    return new_entities


def compute_rolling_features(
    lf: pl.LazyFrame,
    window_days: int,
    suffix: str
) -> pl.DataFrame:
    """Compute rolling window features for 7d and 30d."""
    date_df = (
        lf.group_by(["user_hash", "date"])
        .agg([
            pl.col("activity").count().alias("_activity_count"),
            pl.col("is_after_hours").sum().alias("_after_hours_count"),
        ])
        .with_columns(pl.col("date").str.to_date())
        .sort(["user_hash", "date"])
        .collect()
    )

    rolling_window = 7 if "7d" in suffix else 30

    rolling_df = (
        date_df
        .with_columns([
            pl.col("_activity_count")
            .rolling_sum(window_size=rolling_window, by="date", min_periods=1)
            .over("user_hash")
            .alias(f"_rolling_activity_count{suffix}"),
        ])
        .select(["user_hash", "date", f"_rolling_activity_count{suffix}"])
        .with_columns(pl.col("date").cast(pl.Utf8))
    )

    return rolling_df


def merge_psychometric_features(
    features_df: pl.DataFrame,
    sources: dict[str, pl.LazyFrame]
) -> pl.DataFrame:
    """Merge psychometric features (Big Five personality traits)."""
    if "psychometric" not in sources:
        logger.warning("Psychometric data not available")
        return features_df.with_columns([
            pl.lit(0.0).alias(f"big_five_{trait}") for trait in ["O", "C", "E", "A", "N"]
        ] + [pl.lit(0.0).alias("personality_risk_score")])

    psych_df = sources["psychometric"].collect()

    psych_df = psych_df.rename({
        "user_id": "user_hash",
        "O": "big_five_O",
        "C": "big_five_C",
        "E": "big_five_E",
        "A": "big_five_A",
        "N": "big_five_N",
    })

    return features_df.join(
        psych_df.select(["user_hash", "big_five_O", "big_five_C", "big_five_E",
                        "big_five_A", "big_five_N", "personality_risk_score"]),
        on="user_hash",
        how="left"
    )


def merge_ldap_features(
    features_df: pl.DataFrame,
    sources: dict[str, pl.LazyFrame]
) -> pl.DataFrame:
    """Merge LDAP organizational features."""
    if "ldap" not in sources:
        logger.warning("LDAP data not available")
        return features_df.with_columns([
            pl.lit(0).cast(pl.Int32).alias(col) for col in [
                "role_sensitivity_24h", "is_it_admin_24h", "is_manager_24h",
                "team_size_24h", "access_level_24h"
            ]
        ])

    ldap_df = sources["ldap"].collect()

    ldap_df = ldap_df.rename({"user_id": "user_hash"})

    ldap_df = ldap_df.with_columns([
        pl.col("role_sensitivity").cast(pl.Float64).fill_null(0).alias("role_sensitivity_24h"),
        pl.col("is_it_admin").cast(pl.Int32).fill_null(0).alias("is_it_admin_24h"),
        pl.col("is_manager").cast(pl.Int32).fill_null(0).alias("is_manager_24h"),
        pl.lit(1.0).alias("team_size_24h"),
        pl.col("role_sensitivity").cast(pl.Float64).fill_null(0).alias("access_level_24h"),
    ])

    ldap_cols = ["user_hash", "role_sensitivity_24h", "is_it_admin_24h",
                 "is_manager_24h", "team_size_24h", "access_level_24h"]

    return features_df.join(ldap_df.select(ldap_cols), on="user_hash", how="left")


def extend_features_to_windows(
    df: pl.DataFrame,
    base_window: str = "24h"
) -> pl.DataFrame:
    """Extend 24h features to 7d and 30d windows."""
    for suffix in ["_7d", "_30d"]:
        base_suffix = f"_{base_window}"
        org_base_cols = ["role_sensitivity", "is_it_admin", "is_manager", "team_size", "access_level"]

        for col in org_base_cols:
            base_col = f"{col}{base_suffix}"
            target_col = f"{col}{suffix}"
            if base_col in df.columns and target_col not in df.columns:
                df = df.with_columns(pl.col(base_col).alias(target_col))

        psych_cols = ["big_five_O", "big_five_C", "big_five_E", "big_five_A", "big_five_N", "personality_risk_score"]
        for col in psych_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).alias(f"{col}{suffix}"))

    return df


def compute_behavioral_drift_score(
    features_df: pl.DataFrame
) -> pl.DataFrame:
    """Compute overall behavioral drift score across all feature categories."""
    count_cols = [c for c in features_df.columns if any(x in c.lower() for x in ["count", "size", "duration"])]

    if count_cols and "user_hash" in features_df.columns:
        numeric_features = features_df.select(["user_hash", "date"] + count_cols)

        drift_score = (
            numeric_features
            .with_columns([
                pl.col(c).std().alias(f"{c}_std") for c in count_cols
            ])
            .with_columns([
                (pl.col(f"{c}_std") / (pl.col(c).mean() + 1)).alias(f"{c}_cv") for c in count_cols
            ])
        )

        cv_cols = [f"{c}_cv" for c in count_cols if f"{c}_cv" in drift_score.columns]

        if cv_cols:
            drift_result = (
                drift_score
                .select(["user_hash", "date"] + cv_cols)
                .with_columns([
                    pl.concat_list(cv_cols).list.mean().alias("behavioral_drift_score_24h")
                ])
                .select(["user_hash", "date", "behavioral_drift_score_24h"])
            )

            features_df = features_df.join(drift_result, on=["user_hash", "date"], how="left")
            features_df = features_df.with_columns(
                pl.col("behavioral_drift_score_24h").fill_null(0)
            )

    if "behavioral_drift_score_24h" in features_df.columns:
        features_df = features_df.with_columns([
            pl.col("behavioral_drift_score_24h").alias("behavioral_drift_score_7d"),
            pl.col("behavioral_drift_score_24h").alias("behavioral_drift_score_30d"),
        ])

    return features_df


def create_user_date_ground_truth(
    sources: dict[str, pl.LazyFrame]
) -> pl.DataFrame:
    """Create the base user-date combinations from all sources."""
    user_dates = []

    for name in ["logon", "device", "file", "email", "http"]:
        if name in sources:
            source_dates = (
                sources[name]
                .select(["user_hash", "date"])
                .distinct()
                .collect()
            )
            user_dates.append(source_dates)

    if user_dates:
        all_user_dates = user_dates[0]
        for ud in user_dates[1:]:
            all_user_dates = all_user_dates.join(ud, on=["user_hash", "date"], how="outer")
        return all_user_dates
    else:
        raise ValueError("No data sources available")


def ensure_all_columns(
    df: pl.DataFrame,
    required_columns: list[str]
) -> pl.DataFrame:
    """Ensure all required columns exist, filling with appropriate defaults."""
    for col in required_columns:
        if col not in df.columns:
            if any(x in col for x in ["count", "ratio", "size", "rate", "duration", "diversity"]):
                df = df.with_columns(pl.lit(0.0).alias(col))
            elif col.startswith("hour_") or col.startswith("day_of_week_"):
                df = df.with_columns(pl.lit(0).alias(col))
            elif "drift" in col or "score" in col:
                df = df.with_columns(pl.lit(0.0).alias(col))
            else:
                df = df.with_columns(pl.lit(0).cast(pl.Int32).alias(col))
    return df


def compute_all_window_features(
    sources: dict[str, pl.LazyFrame]
) -> dict[str, pl.DataFrame]:
    """Compute all features for all time windows."""
    all_features = {}

    for window_name, window_days in WINDOWS.items():
        suffix = f"_{window_name}"
        logger.info(f"Computing features for {window_name} window...")

        window_features = None

        if "logon" in sources:
            logon_feats = compute_logon_features(sources["logon"], window_days, suffix)
            session_feats = compute_session_features(sources["logon"], window_days, suffix)
            rapid_feats = compute_rapid_logon_cycles(sources["logon"], window_days, suffix)

            window_features = logon_feats.join(session_feats, on=["user_hash", "date"], how="left")
            window_features = window_features.join(rapid_feats, on=["user_hash", "date"], how="left")

        if "device" in sources:
            device_feats = compute_device_features(sources["device"], window_days, suffix)
            device_session = compute_device_session_duration(sources["device"], window_days, suffix)
            device_feats = device_feats.join(device_session, on=["user_hash", "date"], how="left")

            if window_features is None:
                window_features = device_feats
            else:
                window_features = window_features.join(device_feats, on=["user_hash", "date"], how="outer")

        if "file" in sources:
            file_feats = compute_file_features(sources["file"], window_days, suffix)

            if window_features is None:
                window_features = file_feats
            else:
                window_features = window_features.join(file_feats, on=["user_hash", "date"], how="outer")

        if "email" in sources:
            email_feats = compute_email_features(sources["email"], window_days, suffix)

            if window_features is None:
                window_features = email_feats
            else:
                window_features = window_features.join(email_feats, on=["user_hash", "date"], how="outer")

        if "http" in sources:
            http_feats = compute_http_features(sources["http"], window_days, suffix)

            if window_features is None:
                window_features = http_feats
            else:
                window_features = window_features.join(http_feats, on=["user_hash", "date"], how="outer")

        if "logon" in sources:
            temporal = compute_temporal_features(sources["logon"], window_days, suffix)

            if window_features is None:
                window_features = temporal
            else:
                window_features = window_features.join(temporal, on=["user_hash", "date"], how="outer")

        if "logon" in sources:
            drift = compute_drift_features(sources["logon"], window_days, suffix)

            if window_features is None:
                window_features = drift
            else:
                window_features = window_features.join(drift, on=["user_hash", "date"], how="outer")

        if "logon" in sources:
            new_pc = compute_new_entity_features(sources["logon"], window_days, suffix, "pc")

            if window_features is None:
                window_features = new_pc
            else:
                window_features = window_features.join(new_pc, on=["user_hash", "date"], how="outer")

        if "email" in sources:
            new_contact = compute_new_entity_features(sources["email"], window_days, suffix, "contact")

            if window_features is None:
                window_features = new_contact
            else:
                window_features = window_features.join(new_contact, on=["user_hash", "date"], how="outer")

        all_features[window_name] = window_features

    return all_features


def main() -> pl.DataFrame:
    """Main function to compute all statistical features."""
    logger.info("Starting statistical feature engineering...")

    sources = load_all_sources()

    user_dates = create_user_date_ground_truth(sources)
    logger.info(f"Base user-date combinations: {len(user_dates)} rows")

    window_features = compute_all_window_features(sources)

    logger.info("Merging features from all windows...")
    combined_features = user_dates

    for window_name, features_df in window_features.items():
        if features_df is not None:
            combined_features = combined_features.join(
                features_df,
                on=["user_hash", "date"],
                how="left"
            )

    logger.info("Merging psychometric features...")
    combined_features = merge_psychometric_features(combined_features, sources)

    logger.info("Merging LDAP features...")
    combined_features = merge_ldap_features(combined_features, sources)

    logger.info("Extending features to all windows...")
    combined_features = extend_features_to_windows(combined_features)

    logger.info("Computing behavioral drift score...")
    combined_features = compute_behavioral_drift_score(combined_features)

    required_24h_columns = [
        "user_hash", "date",
        "logon_count_24h", "logoff_count_24h", "logon_ratio_24h",
        "after_hours_logons_24h", "weekend_logons_24h", "unique_pcs_24h",
        "avg_session_duration_24h", "max_session_duration_24h",
        "session_duration_std_24h", "rapid_logon_cycles_24h",
        "device_connect_count_24h", "device_disconnect_count_24h",
        "missing_disconnect_count_24h", "device_session_duration_24h",
        "after_hours_device_24h", "weekend_device_24h",
        "file_event_count_24h", "file_write_count_24h", "file_copy_count_24h",
        "file_delete_count_24h", "removable_media_writes_24h",
        "file_type_diversity_24h", "large_file_writes_24h", "after_hours_file_24h",
        "emails_sent_24h", "emails_received_24h", "avg_email_size_24h",
        "max_email_size_24h", "total_attachment_size_24h",
        "emails_with_attachments_24h", "external_emails_sent_24h",
        "unique_external_domains_24h", "cc_usage_rate_24h",
        "emails_sent_after_hours_24h", "new_recipient_contacts_24h",
        "http_request_count_24h", "unique_domains_24h",
        "job_site_visits_24h", "cloud_storage_visits_24h",
        "file_sharing_visits_24h", "social_media_visits_24h",
        "after_hours_browsing_24h", "weekend_browsing_24h",
        "volume_change_ratio_24h", "new_pc_count_24h",
        "new_contact_count_24h", "behavioral_drift_score_24h",
        "role_sensitivity_24h", "is_it_admin_24h", "is_manager_24h",
        "team_size_24h", "access_level_24h",
    ]

    required_7d_columns = [c.replace("_24h", "_7d") for c in required_24h_columns
                           if "_24h" in c and "user_hash" not in c and "date" not in c]
    required_30d_columns = [c.replace("_24h", "_30d") for c in required_24h_columns
                             if "_24h" in c and "user_hash" not in c and "date" not in c]

    required_temporal_24h = [f"hour_{i}_24h" for i in range(24)] + [f"day_of_week_{i}_24h" for i in range(7)]
    required_temporal_7d = [f"hour_{i}_7d" for i in range(24)] + [f"day_of_week_{i}_7d" for i in range(7)]
    required_temporal_30d = [f"hour_{i}_30d" for i in range(24)] + [f"day_of_week_{i}_30d" for i in range(7)]

    required_columns = (
        required_24h_columns +
        required_7d_columns +
        required_30d_columns +
        required_temporal_24h +
        required_temporal_7d +
        required_temporal_30d +
        ["big_five_O", "big_five_C", "big_five_E", "big_five_A", "big_five_N",
         "personality_risk_score", "user_id"]
    )

    combined_features = ensure_all_columns(combined_features, required_columns)

    combined_features = combined_features.sort(["user_hash", "date"])

    combined_features = combined_features.with_columns(
        pl.col("user_hash").alias("user_id")
    )

    final_columns = [
        "user_id", "user_hash", "date",
    ]

    feature_bases = [
        "logon_count", "logoff_count", "logon_ratio",
        "after_hours_logons", "weekend_logons", "unique_pcs",
        "avg_session_duration", "max_session_duration",
        "session_duration_std", "rapid_logon_cycles",
        "device_connect_count", "device_disconnect_count",
        "missing_disconnect_count", "device_session_duration",
        "after_hours_device", "weekend_device",
        "file_event_count", "file_write_count", "file_copy_count",
        "file_delete_count", "removable_media_writes",
        "file_type_diversity", "large_file_writes", "after_hours_file",
        "emails_sent", "emails_received", "avg_email_size",
        "max_email_size", "total_attachment_size",
        "emails_with_attachments", "external_emails_sent",
        "unique_external_domains", "cc_usage_rate",
        "emails_sent_after_hours", "new_recipient_contacts",
        "http_request_count", "unique_domains",
        "job_site_visits", "cloud_storage_visits",
        "file_sharing_visits", "social_media_visits",
        "after_hours_browsing", "weekend_browsing",
        "volume_change_ratio", "new_pc_count",
        "new_contact_count", "behavioral_drift_score",
        "role_sensitivity", "is_it_admin", "is_manager",
        "team_size", "access_level",
    ]

    for base in feature_bases:
        for suffix in ["_24h", "_7d", "_30d"]:
            col = f"{base}{suffix}"
            if col in combined_features.columns:
                final_columns.append(col)

    for suffix in ["_24h", "_7d", "_30d"]:
        for i in range(24):
            col = f"hour_{i}{suffix}"
            if col in combined_features.columns:
                final_columns.append(col)
        for i in range(7):
            col = f"day_of_week_{i}{suffix}"
            if col in combined_features.columns:
                final_columns.append(col)

    for col in ["big_five_O", "big_five_C", "big_five_E", "big_five_A", "big_five_N",
                "personality_risk_score"]:
        if col in combined_features.columns:
            final_columns.append(col)

    final_columns = [c for c in final_columns if c in combined_features.columns]
    combined_features = combined_features.select(final_columns)

    numeric_cols = [c for c in combined_features.columns if c not in ["user_id", "user_hash", "date"]]
    combined_features = combined_features.with_columns([
        pl.col(c).fill_null(0) for c in numeric_cols
    ])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "user_features_daily.parquet"

    combined_features.write_parquet(output_path, compression="zstd")

    logger.info(f"Features written to {output_path}")
    logger.info(f"Shape: {combined_features.shape}")
    logger.info(f"Columns: {len(combined_features.columns)}")

    return combined_features


if __name__ == "__main__":
    result = main()
    print(f"\nFinal dataset shape: {result.shape}")
    print(f"Sample columns: {result.columns[:20]}")
