"""
Configuration management for the CERT Insider Threat Detection pipeline.

Loads all configuration from configs/config.yaml and provides
a typed, validated configuration object accessible throughout the pipeline.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ─── Project Root ──────────────────────────────────────────────────────────────
# Detect project root relative to this file: src/utils/config.py → project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class WorkingHoursConfig:
    """Working hours definition for temporal feature extraction."""
    start_hour: int = 8
    end_hour: int = 18
    work_days: tuple[int, ...] = (0, 1, 2, 3, 4)

    def is_working_hours(self, hour: int, dow: int) -> bool:
        """Check if given hour and day-of-week are within working hours."""
        return self.start_hour <= hour < self.end_hour and dow in self.work_days

    def is_after_hours(self, hour: int, dow: int) -> bool:
        """Check if given hour and day-of-week are after hours."""
        return not self.is_working_hours(hour, dow)


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset paths and metadata."""
    root: Path
    ldap_dir: Path
    answers_dir: Path
    version: str
    num_users: int
    start_date: str
    end_date: str


@dataclass(frozen=True)
class PreprocessingConfig:
    """Preprocessing hyperparameters."""
    chunk_size: int
    schema_inference_rows: int
    working_hours: WorkingHoursConfig
    holidays: tuple[str, ...]
    session_timeout_minutes: int
    screen_unlock_threshold_seconds: int
    pseudonymization_salt: str


@dataclass(frozen=True)
class PrivacyConfig:
    """Privacy and security settings."""
    pseudonymize: bool
    kdf_iterations: int
    audit_log_path: Path


@dataclass(frozen=True)
class OutputConfig:
    """Output paths and storage settings."""
    base_dir: Path
    normalized_dir: Path
    parquet_compression: str
    parquet_partitions: int


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""
    level: str
    format: str
    log_dir: Path


@dataclass(frozen=True)
class TelemetryConfig:
    """Performance monitoring settings."""
    track_memory: bool
    track_timing: bool
    memory_warning_threshold_gb: float


@dataclass(frozen=True)
class PipelineConfig:
    """
    Complete pipeline configuration.

    This is the single source of truth for all pipeline settings.
    All modules import their config from here — no hardcoded values.
    """
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    privacy: PrivacyConfig
    output: OutputConfig
    logging: LoggingConfig
    telemetry: TelemetryConfig

    @classmethod
    def from_yaml(cls, config_path: Path | str | None = None) -> PipelineConfig:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config.yaml. Defaults to configs/config.yaml
                        relative to project root.

        Returns:
            Fully validated PipelineConfig instance.
        """
        if config_path is None:
            config_path = _CONFIG_PATH
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Please ensure configs/config.yaml exists at the project root."
            )

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        dataset_cfg = raw.get("dataset", {})
        preprocess_cfg = raw.get("preprocessing", {})
        privacy_cfg = raw.get("privacy", {})
        output_cfg = raw.get("output", {})
        logging_cfg = raw.get("logging", {})
        telemetry_cfg = raw.get("telemetry", {})

        # Resolve relative paths from project root
        root = Path(dataset_cfg.get("root", ""))
        if not root.is_absolute():
            root = _PROJECT_ROOT / root

        ldap_dir = Path(dataset_cfg.get("ldap_dir", ""))
        if not ldap_dir.is_absolute():
            ldap_dir = _PROJECT_ROOT / ldap_dir

        answers_dir = Path(dataset_cfg.get("answers_dir", ""))
        if not answers_dir.is_absolute():
            answers_dir = _PROJECT_ROOT / answers_dir

        wh_cfg = preprocess_cfg.get("working_hours", {})
        working_hours = WorkingHoursConfig(
            start_hour=wh_cfg.get("start_hour", 8),
            end_hour=wh_cfg.get("end_hour", 18),
            work_days=tuple(wh_cfg.get("work_days", [0, 1, 2, 3, 4])),
        )

        holidays = tuple(preprocess_cfg.get("holidays", []))

        base_dir = Path(output_cfg.get("base_dir", "data"))
        if not base_dir.is_absolute():
            base_dir = _PROJECT_ROOT / base_dir

        log_dir = Path(logging_cfg.get("log_dir", "logs"))
        if not log_dir.is_absolute():
            log_dir = _PROJECT_ROOT / log_dir

        audit_log = Path(privacy_cfg.get("audit_log_path", "logs/audit.log"))
        if not audit_log.is_absolute():
            audit_log = _PROJECT_ROOT / audit_log

        return cls(
            dataset=DatasetConfig(
                root=root,
                ldap_dir=ldap_dir,
                answers_dir=answers_dir,
                version=dataset_cfg.get("version", "4.2"),
                num_users=dataset_cfg.get("num_users", 1000),
                start_date=dataset_cfg.get("start_date", "2010-01-01"),
                end_date=dataset_cfg.get("end_date", "2011-05-31"),
            ),
            preprocessing=PreprocessingConfig(
                chunk_size=preprocess_cfg.get("chunk_size", 1_000_000),
                schema_inference_rows=preprocess_cfg.get(
                    "schema_inference_rows", 100_000
                ),
                working_hours=working_hours,
                holidays=holidays,
                session_timeout_minutes=preprocess_cfg.get(
                    "session_timeout_minutes", 30
                ),
                screen_unlock_threshold_seconds=preprocess_cfg.get(
                    "screen_unlock_threshold_seconds", 60
                ),
                pseudonymization_salt=preprocess_cfg.get(
                    "pseudonymization_salt",
                    "CERT_R42_INSIDER_THREAT_SALT_2024",
                ),
            ),
            privacy=PrivacyConfig(
                pseudonymize=privacy_cfg.get("pseudonymize", True),
                kdf_iterations=privacy_cfg.get("kdf_iterations", 480_000),
                audit_log_path=audit_log,
            ),
            output=OutputConfig(
                base_dir=base_dir,
                normalized_dir=base_dir / output_cfg.get("normalized_dir", "normalized"),
                parquet_compression=output_cfg.get("parquet_compression", "zstd"),
                parquet_partitions=output_cfg.get("parquet_partitions", 4),
            ),
            logging=LoggingConfig(
                level=logging_cfg.get("level", "INFO"),
                format=logging_cfg.get("format", "json"),
                log_dir=log_dir,
            ),
            telemetry=TelemetryConfig(
                track_memory=telemetry_cfg.get("track_memory", True),
                track_timing=telemetry_cfg.get("track_timing", True),
                memory_warning_threshold_gb=telemetry_cfg.get(
                    "memory_warning_threshold_gb", 8
                ),
            ),
        )

    @property
    def cert_root(self) -> Path:
        """Alias for dataset.root for convenience."""
        return self.dataset.root

    def validate(self) -> list[str]:
        """
        Validate the configuration and return a list of warning messages.

        Does not raise — warnings are non-fatal.
        """
        warnings: list[str] = []

        # Check dataset directory exists
        if not self.dataset.root.exists():
            warnings.append(
                f"Dataset root does not exist: {self.dataset.root}"
            )

        # Check LDAP directory exists
        if not self.dataset.ldap_dir.exists():
            warnings.append(
                f"LDAP directory does not exist: {self.dataset.ldap_dir}"
            )

        # Check required CSV files exist
        required_files = [
            "logon.csv",
            "device.csv",
            "file.csv",
            "email.csv",
            "http.csv",
            "psychometric.csv",
        ]
        for fname in required_files:
            fpath = self.dataset.root / fname
            if not fpath.exists():
                warnings.append(f"Required file missing: {fpath}")

        # Validate working hours
        if not (0 <= self.preprocessing.working_hours.start_hour < 24):
            warnings.append(
                f"Invalid start_hour: "
                f"{self.preprocessing.working_hours.start_hour}"
            )
        if not (0 < self.preprocessing.working_hours.end_hour <= 24):
            warnings.append(
                f"Invalid end_hour: "
                f"{self.preprocessing.working_hours.end_hour}"
            )

        # Check chunk size is reasonable
        if self.preprocessing.chunk_size < 10_000:
            warnings.append(
                f"Chunk size {self.preprocessing.chunk_size} is very small"
            )

        return warnings


# ─── Global Config Instance ────────────────────────────────────────────────────
# Lazy-loaded singleton — loaded on first access
_config_instance: PipelineConfig | None = None


def get_config(config_path: Path | str | None = None) -> PipelineConfig:
    """
    Get the global pipeline configuration.

    Loads from configs/config.yaml on first call. Subsequent calls
    return the cached instance.

    Args:
        config_path: Optional override path to config.yaml

    Returns:
        PipelineConfig singleton instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = PipelineConfig.from_yaml(config_path)
    return _config_instance


def reset_config() -> None:
    """Reset the global config (useful for testing)."""
    global _config_instance
    _config_instance = None
