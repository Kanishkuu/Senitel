"""
Data loaders for the CERT Insider Threat Dataset r4.2.

Provides parallel chunked CSV loading with automatic schema inference,
column validation, and progress tracking. All loaders use Polars
lazy mode for memory efficiency.

Key design decisions:
- Chunked reading for large files (email.csv 1.36 GB, http.csv 14.5 GB)
- Automatic schema inference from sample rows before full load
- Column validation against expected CERT r4.2 schemas
- Progress tracking with throughput estimation
- Memory-efficient streaming for files that don't fit in RAM
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Iterator,
    Literal,
)

import polars as pl
from tqdm import tqdm

from src.utils.config import PipelineConfig, get_config
from src.utils.logging import get_logger
from src.utils.helpers import (
    ProgressTracker,
    compute_memory_usage,
    format_bytes,
    format_duration,
    parse_cert_timestamp,
    validate_pc_id,
    validate_user_id,
)
from src.cert_dataset import (
    CERT_SCHEMAS,
    SchemaRegistry,
    LogonSchema,
    DeviceSchema,
    FileSchema,
    EmailSchema,
    HttpSchema,
    LdapSchema,
    PsychometricSchema,
    GroundTruthSchema,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)


# ─── Exception Classes ───────────────────────────────────────────────────────

class CertLoadError(Exception):
    """Raised when loading or validating a CERT dataset fails."""


class SchemaValidationError(Exception):
    """Raised when a loaded dataset doesn't match expected schema."""


# ─── Core Loader Class ────────────────────────────────────────────────────────

class CertDatasetLoader:
    """
    Unified data loader for all CERT r4.2 dataset components.

    Supports:
    - Eager loading for small files (psychometric, LDAP)
    - Chunked loading for large files (email, http)
    - Lazy loading for schema inspection
    - Parallel loading of multiple file types
    - Automatic schema validation

    Example usage:
        loader = CertDatasetLoader()
        loader.load_logon()          # → pl.DataFrame
        loader.load_device()         # → pl.DataFrame
        loader.load_all()            # → dict[str, pl.DataFrame]

    Args:
        config: Pipeline configuration. Defaults to global config.
        strict_validation: If True, raise on schema mismatches.
                          If False, log warnings and continue.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        strict_validation: bool = False,
    ):
        self.config = config or get_config()
        self.strict_validation = strict_validation
        self._schemas: SchemaRegistry = CERT_SCHEMAS
        self._warnings: list[str] = []
        self._load_stats: dict[str, dict] = {}

        # Verify dataset root exists
        if not self.config.dataset.root.exists():
            raise CertLoadError(
                f"Dataset root directory not found: {self.config.dataset.root}\n"
                f"Please update dataset.root in configs/config.yaml"
            )

    # ─── Properties ───────────────────────────────────────────────────────────

    @cached_property
    def logon_path(self) -> Path:
        return self.config.dataset.root / "logon.csv"

    @cached_property
    def device_path(self) -> Path:
        return self.config.dataset.root / "device.csv"

    @cached_property
    def file_path(self) -> Path:
        return self.config.dataset.root / "file.csv"

    @cached_property
    def email_path(self) -> Path:
        return self.config.dataset.root / "email.csv"

    @cached_property
    def http_path(self) -> Path:
        return self.config.dataset.root / "http.csv"

    @cached_property
    def psychometric_path(self) -> Path:
        return self.config.dataset.root / "psychometric.csv"

    @cached_property
    def ldap_glob(self) -> Path:
        return self.config.dataset.ldap_dir / "*.csv"

    @cached_property
    def ground_truth_path(self) -> Path:
        return self.config.dataset.answers_dir / "insiders.csv"

    @cached_property
    def scenarios_dir(self) -> Path:
        return self.config.dataset.answers_dir

    # ─── Row Count Estimation ─────────────────────────────────────────────────

    def estimate_row_count(self, path: Path) -> int | None:
        """
        Estimate row count of a CSV file without loading it fully.

        Uses file size / average row size estimate for speed.
        """
        if not path.exists():
            return None
        file_size = path.stat().st_size
        # CERT CSV average row size varies by file type
        avg_row_sizes = {
            "logon.csv": 60,
            "device.csv": 60,
            "file.csv": 200,
            "email.csv": 180,
            "http.csv": 250,
        }
        fname = path.name
        avg_size = avg_row_sizes.get(fname, 150)
        return max(1, int(file_size / avg_size))

    # ─── Chunked CSV Reading (Polars 1.x compatible) ────────────────────────────

    def _read_csv_chunked(
        self,
        path: Path,
        chunk_size: int,
        tracker: "ProgressTracker | None" = None,
        **kwargs,
    ) -> list[pl.DataFrame]:
        """
        Read a CSV file in chunks using modern Polars API.

        Uses scan_csv().collect_batches() which is the recommended approach
        in Polars 1.x (replaces deprecated read_csv_batched).

        Args:
            path: Path to CSV file
            chunk_size: Number of rows per chunk
            tracker: Optional progress tracker
            **kwargs: Additional arguments passed to scan_csv

        Returns:
            List of DataFrame chunks
        """
        chunks: list[pl.DataFrame] = []

        lf = pl.scan_csv(path, **kwargs)

        # collect_batches returns an iterator of batches
        for batch in lf.collect_batches(chunk_size=chunk_size):
            # batch is a DataFrame or can be collected
            if isinstance(batch, pl.DataFrame):
                chunks.append(batch)
            else:
                # batch might be a Series or other type
                chunks.append(batch.to_frame())

            if tracker:
                tracker.update(len(chunks[-1]))

        return chunks

    # ─── Schema Inference ─────────────────────────────────────────────────────

    def _infer_dtype(
        self,
        path: Path,
        sample_rows: int = 100_000,
    ) -> dict[str, pl.DataType]:
        """
        Infer Polars dtypes from a sample of rows.

        This handles the MM/DD/YYYY HH:MM:SS date format in CERT files.
        """
        sample_df = pl.scan_csv(
            path,
            n_rows=sample_rows,
            infer_schema_length=sample_rows,
        ).collect()

        # Override date columns: keep as Utf8 (we parse ourselves)
        dtypes = {}
        for col in sample_df.columns:
            if sample_df[col].dtype == pl.Date:
                dtypes[col] = pl.Utf8
            else:
                dtypes[col] = sample_df[col].dtype

        return dtypes

    def _validate_schema(
        self,
        source: Literal["logon", "device", "file", "email", "http", "psychometric"],
        df: pl.DataFrame,
    ) -> None:
        """Validate loaded DataFrame against expected schema."""
        # Known derived columns added during processing
        derived_columns = {
            "timestamp", "hour", "day_of_week", "is_weekend",
            "is_working_hours", "is_after_hours", "is_holiday",
            "week", "month", "year",
            "domain", "url_path", "sender", "to_count", "cc_count",
            "bcc_count", "has_external_recipient", "has_attachments",
            "operation_type", "file_extension", "is_removable_media",
            "activity_type", "session_key",
        }

        # Filter out derived columns from actual columns
        raw_columns = [c for c in df.columns if c not in derived_columns]
        warnings = self._schemas.validate_columns(source, raw_columns)
        if warnings:
            msg = f"[{source}] Schema validation warnings:\n"
            for w in warnings:
                logger.warning("schema_validation_warning", detail=w)
                msg += f"  - {w}\n"
            self._warnings.extend(warnings)
            if self.strict_validation:
                raise SchemaValidationError(msg)

    # ─── Individual File Loaders ──────────────────────────────────────────────

    def load_logon(
        self,
        sample: bool = False,
        sample_rows: int = 500_000,
    ) -> pl.DataFrame:
        """
        Load and parse logon.csv.

        The logon.csv file contains user logon/logoff events:
        id, date, user, pc, activity (Logon/Logoff)

        Args:
            sample: If True, load only sample_rows for testing
            sample_rows: Number of rows to sample if sample=True

        Returns:
            DataFrame with columns:
            - id, date, user, pc, activity, timestamp, hour, dow,
              is_weekend, is_working_hours, is_after_hours
        """
        path = self.logon_path
        logger.info("loading_file", file="logon.csv", path=str(path))

        schema = self._schemas.logon
        chunk_size = self.config.preprocessing.chunk_size

        # First pass: determine how to load
        if sample:
            df = pl.read_csv(
                path,
                n_rows=sample_rows,
                schema_overrides={"date": pl.Utf8},
            )
        else:
            # Chunked loading with progress
            estimated = self.estimate_row_count(path)
            tracker = ProgressTracker(
                total=estimated or 1_000_000,
                description="logon.csv",
                log_interval=5.0,
            )

            chunks = self._read_csv_chunked(
                path,
                chunk_size=chunk_size,
                tracker=tracker,
                schema_overrides={"date": pl.Utf8},
            )
            tracker.finish()
            df = pl.concat(chunks, rechunk=True)

        # Parse timestamps
        df = self._parse_timestamps(df, "date")

        # Validate schema (before adding derived columns)
        self._validate_schema("logon", df)

        # Add temporal features
        df = self._add_temporal_features(df, "timestamp")

        self._load_stats["logon"] = {
            "rows": len(df),
            "file_size_mb": round(path.stat().st_size / 1_048_576, 2),
            "unique_users": df["user"].n_unique(),
            "unique_pcs": df["pc"].n_unique(),
            "logon_count": int((df["activity"] == "Logon").sum()),
            "logoff_count": int((df["activity"] == "Logoff").sum()),
        }

        logger.info(
            "logon_loaded",
            rows=len(df),
            users=df["user"].n_unique(),
            pcs=df["pc"].n_unique(),
            date_range=f"{df['timestamp'].min()} — {df['timestamp'].max()}",
        )
        return df

    def load_device(
        self,
        sample: bool = False,
        sample_rows: int = 500_000,
    ) -> pl.DataFrame:
        """
        Load and parse device.csv.

        The device.csv file contains USB/removable media events:
        id, date, user, pc, activity (Connect/Disconnect)

        Args:
            sample: If True, load only sample_rows
            sample_rows: Number of rows to sample if sample=True

        Returns:
            DataFrame with temporal features added
        """
        path = self.device_path
        logger.info("loading_file", file="device.csv", path=str(path))

        schema = self._schemas.device
        chunk_size = self.config.preprocessing.chunk_size

        if sample:
            df = pl.read_csv(
                path,
                n_rows=sample_rows,
                schema_overrides={"date": pl.Utf8},
            )
        else:
            estimated = self.estimate_row_count(path)
            tracker = ProgressTracker(
                total=estimated or 500_000,
                description="device.csv",
                log_interval=5.0,
            )

            chunks = self._read_csv_chunked(
                path,
                chunk_size=chunk_size,
                tracker=tracker,
                schema_overrides={"date": pl.Utf8},
            )
            tracker.finish()
            df = pl.concat(chunks, rechunk=True)

        df = self._parse_timestamps(df, "date")
        self._validate_schema("device", df)
        df = self._add_temporal_features(df, "timestamp")

        self._load_stats["device"] = {
            "rows": len(df),
            "file_size_mb": round(path.stat().st_size / 1_048_576, 2),
            "unique_users": df["user"].n_unique(),
            "connect_count": int((df["activity"] == "Connect").sum()),
            "disconnect_count": int((df["activity"] == "Disconnect").sum()),
        }

        logger.info(
            "device_loaded",
            rows=len(df),
            users=df["user"].n_unique(),
            connects=df["activity"].value_counts().filter(
                pl.col("activity") == "Connect"
            )["count"].sum(),
        )
        return df

    def load_file(
        self,
        sample: bool = False,
        sample_rows: int = 200_000,
    ) -> pl.DataFrame:
        """
        Load and parse file.csv.

        The file.csv file contains file copy operations to removable media:
        id, date, user, pc, filename, content

        Note: Unlike r5.2, r4.2 does NOT have an 'activity' column.
        The 'filename' column contains the full path; operations are
        inferred from the path (R:\ = removable media write).

        Args:
            sample: If True, load only sample_rows
            sample_rows: Number of rows to sample if sample=True

        Returns:
            DataFrame with temporal features and derived columns
        """
        path = self.file_path
        logger.info("loading_file", file="file.csv", path=str(path))

        if sample:
            df = pl.read_csv(
                path,
                n_rows=sample_rows,
                schema_overrides={"date": pl.Utf8},
            )
        else:
            estimated = self.estimate_row_count(path)
            tracker = ProgressTracker(
                total=estimated or 1_000_000,
                description="file.csv",
                log_interval=5.0,
            )

            chunks = self._read_csv_chunked(
                path,
                chunk_size=self.config.preprocessing.chunk_size,
                tracker=tracker,
                schema_overrides={"date": pl.Utf8},
            )
            tracker.finish()
            df = pl.concat(chunks, rechunk=True)

        df = self._parse_timestamps(df, "date")
        self._validate_schema("file", df)
        df = self._add_temporal_features(df, "timestamp")

        # Infer operation type from filename path
        # R:\ = removable media, C:\ = local (but file.csv is only removable copies)
        df = df.with_columns([
            pl.col("filename").str.starts_with("R:\\").alias("is_removable"),
            pl.col("filename")
            .str.extract(r"\.([a-zA-Z0-9]+)$", 1)
            .str.to_lowercase()
            .alias("file_extension"),
        ])

        # Parse file header from content (hex prefix)
        # Content format: "HEX_HEADER_1 HEX_HEADER_2 ... keywords..."
        df = df.with_columns([
            pl.col("content")
            .str.split(" ")
            .list.first()
            .alias("file_header"),
        ])

        self._load_stats["file"] = {
            "rows": len(df),
            "file_size_mb": round(path.stat().st_size / 1_048_576, 2),
            "unique_users": df["user"].n_unique(),
            "removable_writes": int(df["is_removable"].sum()),
        }

        logger.info(
            "file_loaded",
            rows=len(df),
            users=df["user"].n_unique(),
            removable_writes=int(df["is_removable"].sum()),
        )
        return df

    def load_email(
        self,
        sample: bool = False,
        sample_rows: int = 500_000,
    ) -> pl.DataFrame:
        """
        Load and parse email.csv.

        The email.csv file contains email events:
        id, date, user, pc, to, cc, bcc, from, size, attachment_count, content

        Note: r4.2 uses 'attachment_count' (integer) instead of
        attachment list like r5.2.

        Args:
            sample: If True, load only sample_rows
            sample_rows: Number of rows to sample if sample=True

        Returns:
            DataFrame with parsed recipient fields and temporal features
        """
        path = self.email_path
        logger.info("loading_file", file="email.csv", path=str(path))
        mem_before = compute_memory_usage()

        if sample:
            df = pl.read_csv(
                path,
                n_rows=sample_rows,
                schema_overrides={"date": pl.Utf8},
            )
        else:
            # email.csv is 1.36 GB — must use chunked reading
            estimated = self.estimate_row_count(path)
            tracker = ProgressTracker(
                total=estimated or 7_000_000,
                description="email.csv",
                log_interval=10.0,
            )

            chunks = self._read_csv_chunked(
                path,
                chunk_size=self.config.preprocessing.chunk_size,
                tracker=tracker,
                schema_overrides={"date": pl.Utf8},
            )
            tracker.finish()
            df = pl.concat(chunks, rechunk=True)

        mem_after = compute_memory_usage()
        logger.info(
            "email_memory",
            before_mb=mem_before["rss_mb"],
            after_mb=mem_after["rss_mb"],
            delta_mb=round(mem_after["rss_mb"] - mem_before["rss_mb"], 2),
        )

        df = self._parse_timestamps(df, "date")
        self._validate_schema("email", df)
        df = self._add_temporal_features(df, "timestamp")

        # Parse semicolon-delimited recipient fields
        # 'to', 'cc', 'bcc' can contain multiple addresses
        def count_recipients(expr: pl.Expr) -> pl.Expr:
            return (
                expr.str.split(";")
                .list.eval(pl.element().str.strip_chars())
                .list.eval(pl.element().str.strip_chars().str.len_bytes() > 0)
                .list.len()
            )

        email_schema = self._schemas.email
        df = df.with_columns([
            count_recipients(pl.col("to")).alias("to_count"),
            count_recipients(pl.col("cc")).alias("cc_count"),
            count_recipients(pl.col("bcc")).alias("bcc_count"),
            # Rename 'from' to avoid Python keyword conflict
            pl.col("from").str.strip_chars().alias("sender"),
            pl.col("from")
            .str.extract(r"@([a-zA-Z0-9.-]+)$", 1)
            .str.to_lowercase()
            .alias("sender_domain"),
            # Internal email detection
            pl.col("from")
            .str.contains(email_schema.internal_domain)
            .alias("is_internal_sender"),
            # Has attachments (attachments is already Int64 count)
            (pl.col("attachments") > 0).alias("has_attachments"),
        ])

        # Check for external recipients (non-dtaa.com domains)
        df = df.with_columns([
            # Simple approach: if 'to' has an @ and it's not just @dtaa.com addresses
            # Remove all @dtaa.com occurrences, if anything remains with @, it's external
            (
                pl.col("to")
                .str.replace_all(r"[^;]+@dtaa\.com", "", literal=False)
                .str.contains("@")
                .fill_null(False)
            ).alias("has_external_recipient"),
        ])

        self._load_stats["email"] = {
            "rows": len(df),
            "file_size_mb": round(path.stat().st_size / 1_048_576, 2),
            "unique_users": df["user"].n_unique(),
            "total_size_mb": round(df["size"].sum() / 1_048_576, 2),
            "with_attachments": int(
                (df["attachments"] > 0).sum()
            ),
            "with_external": int(df["has_external_recipient"].sum()),
        }

        logger.info(
            "email_loaded",
            rows=len(df),
            users=df["user"].n_unique(),
            with_attachments=int((df["attachments"] > 0).sum()),
        )
        return df

    def load_http(
        self,
        sample: bool = False,
        sample_rows: int = 200_000,
    ) -> pl.DataFrame:
        """
        Load and parse http.csv.

        The http.csv file contains web browsing events:
        id, date, user, pc, url, content

        This is the LARGEST file at 14.5 GB. Loading in full requires
        ~16 GB RAM. For machines with less RAM, use sample=True
        or process in chunks.

        Args:
            sample: If True, load only sample_rows
            sample_rows: Number of rows to sample if sample=True

        Returns:
            DataFrame with parsed URL components and temporal features
        """
        path = self.http_path
        file_size_gb = path.stat().st_size / 1_073_741_824
        logger.info(
            "loading_file_large",
            file="http.csv",
            path=str(path),
            size_gb=round(file_size_gb, 2),
        )

        if sample:
            logger.warning(
                "http_sample_mode",
                detail="Loading http.csv in sample mode (not full dataset)",
                sample_rows=sample_rows,
            )
            df = pl.read_csv(
                path,
                n_rows=sample_rows,
                schema_overrides={"date": pl.Utf8},
            )
        else:
            # WARNING: Full load requires ~16 GB RAM
            available_mem = psutil.virtual_memory().available / 1_073_741_824
            if available_mem < 20:
                logger.warning(
                    "low_memory_http",
                    available_gb=round(available_mem, 1),
                    recommended_gb=20,
                    detail="http.csv may not fit in memory. Consider sample=True",
                )

            estimated = self.estimate_row_count(path)
            tracker = ProgressTracker(
                total=estimated or 60_000_000,
                description="http.csv",
                log_interval=30.0,
            )

            chunks = self._read_csv_chunked(
                path,
                chunk_size=self.config.preprocessing.chunk_size,
                tracker=tracker,
                schema_overrides={"date": pl.Utf8},
            )
            tracker.finish()
            df = pl.concat(chunks, rechunk=True)

        df = self._parse_timestamps(df, "date")
        self._validate_schema("http", df)
        df = self._add_temporal_features(df, "timestamp")

        # Parse URL into components
        df = df.with_columns([
            # Extract domain
            pl.col("url")
            .str.extract(r"^https?://([^/]+)", 1)
            .str.to_lowercase()
            .alias("domain"),
            # Extract path
            pl.col("url")
            .str.extract(r"^https?://[^/]+(/.*)?$", 1)
            .alias("url_path"),
        ])

        self._load_stats["http"] = {
            "rows": len(df),
            "file_size_gb": round(file_size_gb, 2),
            "unique_users": df["user"].n_unique(),
            "unique_domains": df["domain"].n_unique(),
        }

        logger.info(
            "http_loaded",
            rows=len(df),
            users=df["user"].n_unique(),
            domains=df["domain"].n_unique(),
        )
        return df

    def load_psychometric(self) -> pl.DataFrame:
        """
        Load and validate psychometric.csv.

        Contains Big Five personality traits for all 1,000 users:
        employee_name, user_id, O, C, E, A, N (scores 0-50)

        Returns:
            DataFrame with Big Five scores and derived risk indicators
        """
        path = self.psychometric_path
        logger.info("loading_file", file="psychometric.csv", path=str(path))

        schema = self._schemas.psychometric

        df = pl.read_csv(path)

        # Validate required columns
        self._validate_schema("psychometric", df)

        # Normalize column name: user_id → user (for consistency)
        df = df.rename({"user_id": "user"})

        # Validate score ranges
        for trait in ["O", "C", "E", "A", "N"]:
            min_val = df[trait].min()
            max_val = df[trait].max()
            if min_val < schema.score_min or max_val > schema.score_max:
                logger.warning(
                    "psychometric_out_of_range",
                    trait=trait,
                    min=min_val,
                    max=max_val,
                    expected_range=f"{schema.score_min}-{schema.score_max}",
                )

        # Add derived psychometric risk indicators
        df = df.with_columns([
            # High Neuroticism + Low Conscientiousness → elevated risk
            ((pl.col("N") > schema.high_neuroticism_threshold) &
             (pl.col("C") < schema.low_conscientiousness_threshold))
            .cast(pl.Int8)
            .alias("risk_N_high_C_low"),
            # Low Agreeableness + High Openness → elevated risk
            ((pl.col("A") < schema.low_agreeableness_threshold) &
             (pl.col("O") > 40))
            .cast(pl.Int8)
            .alias("risk_A_low_O_high"),
        ])

        self._load_stats["psychometric"] = {
            "rows": len(df),
            "file_size_kb": round(path.stat().st_size / 1024, 2),
            "avg_scores": {
                trait: round(df[trait].mean(), 2)
                for trait in ["O", "C", "E", "A", "N"]
            },
        }

        logger.info(
            "psychometric_loaded",
            rows=len(df),
            avg_openness=df["O"].mean(),
            avg_conscientiousness=df["C"].mean(),
        )
        return df

    def load_ldap(
        self,
        combine: bool = True,
    ) -> pl.DataFrame | dict[str, pl.DataFrame]:
        """
        Load LDAP monthly snapshots.

        LDAP directory contains 18 monthly snapshots:
        2009-12.csv through 2011-05.csv

        Each snapshot contains organizational hierarchy for all employees
        at that point in time (user_id, role, department, supervisor, etc.)

        Args:
            combine: If True, concatenate all snapshots with a 'month' column.
                    If False, return dict of {filename: DataFrame}

        Returns:
            Combined DataFrame with 'month' column, or dict of snapshots
        """
        ldap_dir = self.config.dataset.ldap_dir
        logger.info("loading_ldap", path=str(ldap_dir))

        if not ldap_dir.exists():
            raise CertLoadError(f"LDAP directory not found: {ldap_dir}")

        csv_files = sorted(ldap_dir.glob("*.csv"))
        if not csv_files:
            raise CertLoadError(f"No CSV files found in LDAP directory: {ldap_dir}")

        logger.info("ldap_files_found", count=len(csv_files))

        if not combine:
            snapshots: dict[str, pl.DataFrame] = {}
            for fpath in tqdm(csv_files, desc="Loading LDAP snapshots"):
                month = fpath.stem  # e.g., "2010-01"
                df = pl.read_csv(fpath)
                df = df.rename({"user_id": "user"})
                snapshots[month] = df
            return snapshots

        # Combine all snapshots with month tracking
        frames: list[pl.DataFrame] = []
        for fpath in tqdm(csv_files, desc="Loading LDAP snapshots"):
            month = fpath.stem  # e.g., "2010-01"
            df = pl.read_csv(fpath)
            df = df.rename({"user_id": "user"})
            df = df.with_columns(pl.lit(month).alias("month"))
            frames.append(df)

        combined = pl.concat(frames, rechunk=True)

        # Add derived features
        combined = combined.with_columns([
            pl.col("role")
            .str.to_uppercase()
            .str.contains("ITADMIN")
            .cast(pl.Int8)
            .alias("is_it_admin"),
            pl.col("role")
            .str.to_uppercase()
            .str.contains_any(["MANAGER", "DIRECTOR", "VP", "VICE PRESIDENT"])
            .cast(pl.Int8)
            .alias("is_management"),
        ])

        # Parse month to datetime for temporal alignment
        combined = combined.with_columns([
            pl.col("month")
            .str.replace("-", "-")
            .str.to_date(format="%Y-%m")
            .alias("month_date"),
        ])

        self._load_stats["ldap"] = {
            "snapshots": len(csv_files),
            "date_range": (
                f"{combined['month'].min()} — {combined['month'].max()}"
            ),
            "total_rows": len(combined),
            "unique_users_per_snapshot": (
                combined.group_by("month").agg(
                    pl.col("user").n_unique().alias("user_count")
                )["user_count"].mean()
            ),
        }

        logger.info(
            "ldap_loaded",
            snapshots=len(csv_files),
            combined_rows=len(combined),
            date_range=f"{combined['month'].min()} — {combined['month'].max()}",
        )
        return combined

    def load_ground_truth(self) -> pl.DataFrame:
        """
        Load the ground truth / insiders.csv file.

        This file maps known insider threat actors to their scenarios:
        dataset, scenario, details, user, start, end

        Returns:
            DataFrame with insider actor labels and scenario metadata
        """
        path = self.ground_truth_path
        logger.info("loading_ground_truth", path=str(path))

        if not path.exists():
            logger.warning(
                "ground_truth_not_found",
                path=str(path),
                detail="Ground truth file not found. Some evaluation features will be unavailable.",
            )
            return pl.DataFrame()

        df = pl.read_csv(path)
        self._validate_schema("ground_truth", df)

        # Parse timestamps
        df = df.with_columns([
            pl.col("start").map_elements(
                parse_cert_timestamp, return_dtype=pl.Datetime
            ).alias("start_dt"),
            pl.col("end").map_elements(
                parse_cert_timestamp, return_dtype=pl.Datetime
            ).alias("end_dt"),
        ])

        # Filter to r4.2 insiders only (dataset column is Float64, e.g., 4.2)
        r42_insiders = df.filter(pl.col("dataset") == 4.2)

        self._load_stats["ground_truth"] = {
            "total_insiders": len(df),
            "r42_insiders": len(r42_insiders),
            "scenarios": df["scenario"].unique().to_list(),
        }

        logger.info(
            "ground_truth_loaded",
            total=len(df),
            r42=len(r42_insiders),
            scenarios=df["scenario"].unique().to_list(),
        )
        return df

    # ─── Bulk Loading ─────────────────────────────────────────────────────────

    def load_all(
        self,
        load_http: bool = False,
        http_sample: bool = False,
    ) -> dict[str, pl.DataFrame]:
        """
        Load all CERT dataset components.

        This is the primary entry point for full dataset loading.

        Args:
            load_http: Whether to load http.csv (14.5 GB). Default False
                       to save memory. Set True for full processing.
            http_sample: If load_http=True, load only sample rows

        Returns:
            Dictionary mapping source name to DataFrame
        """
        logger.info("loading_all_datasets", load_http=load_http)

        results: dict[str, pl.DataFrame] = {}

        # Load in order of increasing memory requirement
        results["psychometric"] = self.load_psychometric()
        results["logon"] = self.load_logon()
        results["device"] = self.load_device()
        results["file"] = self.load_file()
        results["email"] = self.load_email()
        results["ldap"] = self.load_ldap(combine=True)

        # Load ground truth (small)
        gt = self.load_ground_truth()
        if len(gt) > 0:
            results["ground_truth"] = gt

        # HTTP: only if explicitly requested
        if load_http:
            results["http"] = self.load_http(sample=http_sample)
        else:
            logger.info(
                "http_skipped",
                detail="http.csv not loaded. Set load_http=True to include.",
            )

        # Log summary
        total_rows = sum(
            self._load_stats.get(k, {}).get("rows", 0)
            for k in ["logon", "device", "file", "email", "http"]
        )
        logger.info(
            "all_datasets_loaded",
            sources=list(results.keys()),
            total_event_rows=total_rows,
            memory_mb=compute_memory_usage()["rss_mb"],
        )

        return results

    # ─── Internal Helpers ────────────────────────────────────────────────────

    def _parse_timestamps(
        self,
        df: pl.DataFrame,
        date_col: str = "date",
    ) -> pl.DataFrame:
        """
        Parse CERT MM/DD/YYYY HH:MM:SS timestamps to datetime.

        Uses Polars map_elements for robust parsing with fallback
        to None for unparseable values.
        """
        return df.with_columns([
            pl.col(date_col)
            .map_elements(parse_cert_timestamp, return_dtype=pl.Datetime)
            .alias("timestamp"),
        ])

    def _add_temporal_features(
        self,
        df: pl.DataFrame,
        ts_col: str = "timestamp",
    ) -> pl.DataFrame:
        """
        Add temporal feature columns based on a datetime column.

        Adds: hour, day_of_week, is_weekend, is_working_hours,
              is_after_hours, is_holiday, date
        """
        cfg = self.config.preprocessing
        wh = cfg.working_hours
        holidays_set = set(cfg.holidays)

        return df.with_columns([
            pl.col(ts_col).dt.hour().alias("hour"),
            pl.col(ts_col).dt.weekday().alias("day_of_week"),  # 0=Mon, 6=Sun
            pl.col(ts_col).dt.week().alias("week"),
            pl.col(ts_col).dt.month().alias("month"),
            pl.col(ts_col).dt.year().alias("year"),
            pl.col(ts_col).dt.date().alias("date"),
            # Weekend: Saturday (5) or Sunday (6) — compute from weekday directly
            (pl.col(ts_col).dt.weekday() >= 5).alias("is_weekend"),
            # Working hours — compute from hour directly
            pl.when(
                (pl.col(ts_col).dt.hour() >= wh.start_hour)
                & (pl.col(ts_col).dt.hour() < wh.end_hour)
                & pl.col(ts_col).dt.weekday().is_in(wh.work_days)
            )
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("is_working_hours"),
            # After hours — inverse of working hours
            pl.when(
                (pl.col(ts_col).dt.hour() >= wh.start_hour)
                & (pl.col(ts_col).dt.hour() < wh.end_hour)
                & pl.col(ts_col).dt.weekday().is_in(wh.work_days)
            )
            .then(pl.lit(False))
            .otherwise(pl.lit(True))
            .alias("is_after_hours"),
            # Holiday (date-based lookup)
            pl.col(ts_col)
            .dt.date()
            .cast(pl.Utf8)
            .is_in(holidays_set)
            .alias("is_holiday"),
        ])

    # ─── Statistics & Reporting ───────────────────────────────────────────────

    def get_load_stats(self) -> dict[str, dict]:
        """Return statistics from all load operations."""
        return dict(self._load_stats)

    def get_warnings(self) -> list[str]:
        """Return all warnings accumulated during loading."""
        return list(self._warnings)

    def print_summary(self) -> None:
        """Print a formatted summary of loaded datasets."""
        print("\n" + "=" * 70)
        print("CERT r4.2 Dataset Loading Summary")
        print("=" * 70)
        for source, stats in self._load_stats.items():
            print(f"\n  [{source.upper()}]")
            for key, val in stats.items():
                if isinstance(val, dict):
                    print(f"    {key}:")
                    for k2, v2 in val.items():
                        print(f"      {k2}: {v2}")
                elif isinstance(val, float):
                    print(f"    {key}: {val:,.2f}")
                else:
                    print(f"    {key}: {val:,}" if isinstance(val, int) else f"    {key}: {val}")
        print()
        if self._warnings:
            print(f"  ⚠️  {len(self._warnings)} warnings (see get_warnings())")
        print(f"  Memory: {compute_memory_usage()['rss_mb']:,.0f} MB RSS")
        print("=" * 70)


# ─── Import psutil for memory tracking ──────────────────────────────────────
import psutil  # noqa: E402, F401
