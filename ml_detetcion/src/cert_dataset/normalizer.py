"""
Log Normalizer for the CERT Insider Threat Dataset r4.2.

Provides temporal feature enrichment, entity standardization, and
data quality validation for all log sources.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl


class LogNormalizer:
    """
    Normalizes and enriches CERT log data with temporal and entity features.

    This class applies a consistent normalization pipeline across all log sources:
    - Timestamp parsing and validation
    - Temporal feature derivation (hour, day_of_week, is_weekend, etc.)
    - Entity standardization (user IDs, PC IDs)
    - Data quality flagging

    Optimized for memory efficiency with Polars lazy evaluation.
    """

    def __init__(
        self,
        working_hours_start: int = 8,
        working_hours_end: int = 18,
        work_days: list[int] | None = None,
        holidays: list[str] | None = None,
    ):
        """
        Initialize the LogNormalizer.

        Args:
            working_hours_start: Start of working hours (0-23)
            working_hours_end: End of working hours (0-23)
            work_days: List of work days as integers (0=Mon, 6=Sun)
            holidays: List of holiday dates as YYYY-MM-DD strings
        """
        self.working_hours_start = working_hours_start
        self.working_hours_end = working_hours_end
        self.work_days = work_days or [0, 1, 2, 3, 4]  # Mon-Fri
        self.holidays = set(holidays or [])

        # Build date expressions for Polars
        self._holiday_expr = pl.col("date").cast(pl.Date).is_in(
            [pl.date.from_iso_format(h) for h in self.holidays]
        ) if self.holidays else pl.lit(False)

    def _build_temporal_expressions(self) -> list[pl.Expr]:
        """Build Polars expressions for temporal features."""
        return [
            # Hour of day (0-23)
            pl.col("timestamp").dt.hour().alias("hour"),
            # Day of week (0=Mon, 6=Sun)
            pl.col("timestamp").dt.weekday().alias("day_of_week"),
            # Day of month
            pl.col("timestamp").dt.day().alias("day_of_month"),
            # Month
            pl.col("timestamp").dt.month().alias("month"),
            # Year
            pl.col("timestamp").dt.year().alias("year"),
            # Week of year
            pl.col("timestamp").dt.week().alias("week"),
            # Is weekend (Sat=5, Sun=6)
            (pl.col("day_of_week") >= 5).alias("is_weekend"),
            # Is working hours
            (
                (pl.col("hour") >= self.working_hours_start)
                & (pl.col("hour") < self.working_hours_end)
                & pl.col("day_of_week").is_in(self.work_days)
                & ~self._holiday_expr
            ).alias("is_working_hours"),
            # Is after hours
            (
                ~(
                    (pl.col("hour") >= self.working_hours_start)
                    & (pl.col("hour") < self.working_hours_end)
                    & pl.col("day_of_week").is_in(self.work_days)
                    & ~self._holiday_expr
                )
            ).alias("is_after_hours"),
            # Is holiday
            self._holiday_expr.alias("is_holiday"),
            # Time bucket (1-hour resolution)
            pl.col("timestamp").dt.truncate("1h").alias("time_bucket"),
            # Date only
            pl.col("timestamp").dt.date().alias("date"),
        ]

    def _parse_timestamp(self, df: pl.LazyFrame, date_col: str = "date") -> pl.LazyFrame:
        """
        Parse timestamp column from CERT format to datetime.

        CERT format: MM/DD/YYYY HH:MM:SS (12-hour)
        Target format: YYYY-MM-DD HH:MM:SS (24-hour, UTC-aware)
        """
        return df.with_columns(
            pl.col(date_col)
            .str.to_datetime(format="%m/%d/%Y %H:%M:%S", strict=False)
            .alias("timestamp")
        )

    def normalize_logon(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Normalize logon.csv data.

        Adds temporal features and standardizes entity IDs.
        """
        result = self._parse_timestamp(df)

        # Add temporal features
        result = result.with_columns(self._build_temporal_expressions())

        # Standardize entity columns
        result = result.with_columns(
            pl.col("user").str.to_uppercase().str.strip_chars().alias("user"),
            pl.col("pc").str.to_uppercase().str.strip_chars().alias("pc"),
        )

        # Add derived features
        result = result.with_columns(
            pl.col("activity")
            .cast(pl.Categorical)
            .alias("activity_type"),
            # Session key for pairing logon/logoff
            (pl.col("user") + "_" + pl.col("pc")).alias("session_key"),
        )

        return result

    def normalize_device(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Normalize device.csv data.

        Adds temporal features and device session support.
        """
        result = self._parse_timestamp(df)

        # Add temporal features
        result = result.with_columns(self._build_temporal_expressions())

        # Standardize entity columns
        result = result.with_columns(
            pl.col("user").str.to_uppercase().str.strip_chars().alias("user"),
            pl.col("pc").str.to_uppercase().str.strip_chars().alias("pc"),
            pl.col("activity").str.to_uppercase().alias("activity"),
        )

        # Add derived features
        result = result.with_columns(
            pl.col("activity")
            .cast(pl.Categorical)
            .alias("activity_type"),
        )

        return result

    def normalize_file(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Normalize file.csv data.

        Note: r4.2 file.csv has no activity column. Operations are
        inferred from the filename path (R:\ = removable media write).

        Adds temporal features and file operation classification.
        """
        result = self._parse_timestamp(df)

        # Add temporal features
        result = result.with_columns(self._build_temporal_expressions())

        # Standardize entity columns
        result = result.with_columns(
            pl.col("user").str.to_uppercase().str.strip_chars().alias("user"),
            pl.col("pc").str.to_uppercase().str.strip_chars().alias("pc"),
        )

        # Infer file operation from path
        # R:\ prefix indicates removable media (write operation)
        result = result.with_columns(
            pl.when(pl.col("filename").str.contains("^R:\\\\", literal=False))
            .then(pl.lit("REMOVABLE_WRITE"))
            .when(pl.col("filename").str.contains("^C:\\\\Windows", literal=False))
            .then(pl.lit("SYSTEM_ACCESS"))
            .otherwise(pl.lit("FILE_ACCESS"))
            .alias("operation_type")
        )

        # Extract file extension
        result = result.with_columns(
            pl.col("filename")
            .str.extract(r"\.([a-zA-Z0-9]+)$", 1)
            .str.to_lowercase()
            .alias("file_extension")
        )

        # Flag removable media writes (potential exfiltration)
        result = result.with_columns(
            pl.col("filename")
            .str.starts_with("R:\\")
            .alias("is_removable_media")
        )

        return result

    def normalize_email(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Normalize email.csv data.

        Handles semicolon-delimited recipient lists and classifies
        internal vs external emails.
        """
        result = self._parse_timestamp(df)

        # Add temporal features
        result = result.with_columns(self._build_temporal_expressions())

        # Standardize entity columns
        result = result.with_columns(
            pl.col("user").str.to_uppercase().str.strip_chars().alias("user"),
            pl.col("pc").str.to_uppercase().str.strip_chars().alias("pc"),
            # Rename 'from' to avoid Python keyword conflict
            pl.col("from").str.strip_chars().alias("sender"),
        )

        # Process recipient counts
        result = result.with_columns(
            pl.col("to")
            .str.split(";")
            .list.eval(pl.element().str.strip_chars())
            .list.unique()
            .list.len_bytes()
            .cast(pl.Int64)
            .alias("to_count"),
            pl.col("cc")
            .str.split(";")
            .list.eval(pl.element().str.strip_chars())
            .list.unique()
            .list.len_bytes()
            .cast(pl.Int64)
            .alias("cc_count"),
            pl.col("bcc")
            .str.split(";")
            .list.eval(pl.element().str.strip_chars())
            .list.unique()
            .list.len_bytes()
            .cast(pl.Int64)
            .fill_null(0)
            .alias("bcc_count"),
        )

        # Classify emails
        result = result.with_columns(
            # Has external recipients
            pl.col("to")
            .str.contains(
                r"@(?!dtaa\.com)", strict=False  # Negative lookahead for internal domain
            )
            .fill_null(False)
            .alias("has_external_recipient"),
            # Has attachments
            (pl.col("attachment_count") > 0).alias("has_attachments"),
            # Activity type
            pl.col("activity").cast(pl.Categorical).alias("activity_type"),
        )

        return result

    def normalize_http(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Normalize http.csv data.

        Extracts URL components and classifies domain categories.
        """
        result = self._parse_timestamp(df)

        # Add temporal features
        result = result.with_columns(self._build_temporal_expressions())

        # Standardize entity columns
        result = result.with_columns(
            pl.col("user").str.to_uppercase().str.strip_chars().alias("user"),
            pl.col("pc").str.to_uppercase().str.strip_chars().alias("pc"),
        )

        # Extract URL components
        result = result.with_columns(
            # Extract domain from URL
            pl.col("url")
            .str.extract(r"^https?://([^/]+)", 1)
            .str.to_lowercase()
            .alias("domain"),
            # Extract path
            pl.col("url")
            .str.extract(r"^https?://[^/]+(/.*)$", 1)
            .fill_null("")
            .alias("url_path"),
        )

        # Classify domain categories
        # Note: This is a simplified classification; real-world would use
        # a comprehensive domain database
        result = result.with_columns(
            pl.when(pl.col("domain").str.contains("facebook|linkedin|twitter", literal=False))
            .then(pl.lit("social_media"))
            .when(pl.col("domain").str.contains("dropbox|googledrive|onedrive|box\\.com", literal=False))
            .then(pl.lit("cloud_storage"))
            .when(pl.col("domain").str.contains("indeed|linkedin\\.com|monster|glassdoor", literal=False))
            .then(pl.lit("job_search"))
            .when(pl.col("domain").str.contains("wikileaks|mega\\.nz|mediafire", literal=False))
            .then(pl.lit("file_sharing"))
            .otherwise(pl.lit("other"))
            .alias("domain_category")
        )

        # Flag sensitive categories
        sensitive_categories = ["cloud_storage", "file_sharing", "job_search"]
        result = result.with_columns(
            pl.col("domain_category")
            .is_in(sensitive_categories)
            .alias("is_sensitive_category")
        )

        return result

    def normalize_ldap(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Normalize LDAP directory snapshots.

        Standardizes employee data and derives organizational features.
        """
        result = df.lazy()

        # Standardize columns
        result = result.with_columns(
            pl.col("user_id").str.to_uppercase().str.strip_chars().alias("user_id"),
            pl.col("email").str.to_lowercase().str.strip_chars().alias("email"),
        )

        # Derive organizational features
        result = result.with_columns(
            # Role sensitivity score (1-5)
            pl.when(pl.col("role").str.contains("Admin|Executive|Finance", literal=False))
            .then(pl.lit(5))
            .when(pl.col("role").str.contains("Manager|Director", literal=False))
            .then(pl.lit(4))
            .when(pl.col("role").str.contains("Engineer|Analyst", literal=False))
            .then(pl.lit(3))
            .when(pl.col("role").str.contains("Specialist|Coordinator", literal=False))
            .then(pl.lit(2))
            .otherwise(pl.lit(1))
            .alias("role_sensitivity"),
            # Is IT Admin
            pl.col("role")
            .str.contains("Admin|IT", literal=False)
            .alias("is_it_admin"),
            # Is manager
            pl.col("role")
            .str.contains("Manager|Director|Lead|Head", literal=False)
            .alias("is_manager"),
        )

        return result

    def normalize_psychometric(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Normalize psychometric.csv data.

        Computes derived risk indicators from Big Five personality traits.
        """
        result = df.lazy()

        # Standardize user ID
        result = result.with_columns(
            pl.col("user_id").str.to_uppercase().str.strip_chars().alias("user_id"),
        )

        # Compute risk indicators from Big Five traits
        result = result.with_columns(
            # High Neuroticism + Low Conscientiousness = elevated risk
            (
                (pl.col("N") > 35)
                & (pl.col("C") < 25)
            ).alias("neurotic_low_conscientious"),
            # High Openness + Low Agreeableness = potential rule-breaking
            (
                (pl.col("O") > 35)
                & (pl.col("A") < 25)
            ).alias("open_low_agreeable"),
            # Combined risk score (0-1 scale)
            (
                (pl.col("N") / 50.0)
                + ((50 - pl.col("C")) / 50.0)
                + ((50 - pl.col("A")) / 50.0)
            )
            .clip(0.0, 3.0)
            .truediv(3.0)
            .alias("personality_risk_score"),
        )

        return result

    def validate_schema(
        self,
        df: pl.DataFrame,
        expected_columns: list[str],
        source_name: str,
    ) -> list[str]:
        """
        Validate DataFrame against expected schema.

        Args:
            df: DataFrame to validate
            expected_columns: List of expected column names
            source_name: Name of the source file (for error messages)

        Returns:
            List of validation warnings (empty if valid)
        """
        warnings = []
        actual_columns = set(df.columns)
        expected = set(expected_columns)

        missing = expected - actual_columns
        extra = actual_columns - expected

        if missing:
            warnings.append(
                f"[{source_name}] Missing columns: {sorted(missing)}"
            )
        if extra:
            warnings.append(
                f"[{source_name}] Extra columns (may be OK): {sorted(extra)}"
            )

        return warnings

    def apply_pipeline(
        self,
        df: pl.LazyFrame,
        source: str,
    ) -> pl.LazyFrame:
        """
        Apply the full normalization pipeline based on source type.

        Args:
            df: LazyFrame to normalize
            source: Source type (logon, device, file, email, http, ldap, psychometric)

        Returns:
            Normalized LazyFrame
        """
        normalizers = {
            "logon": self.normalize_logon,
            "device": self.normalize_device,
            "file": self.normalize_file,
            "email": self.normalize_email,
            "http": self.normalize_http,
            "ldap": self.normalize_ldap,
            "psychometric": self.normalize_psychometric,
        }

        normalizer = normalizers.get(source)
        if normalizer is None:
            raise ValueError(
                f"Unknown source: {source}. "
                f"Valid sources: {list(normalizers.keys())}"
            )

        return normalizer(df)
