"""CERT Dataset package — schemas, loaders, normalizers, and privacy tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Literal

import polars as pl


# ─── Unified Event Taxonomy ────────────────────────────────────────────────────

class EventType(str, Enum):
    """
    Canonical event types across all CERT log sources.

    Each raw log activity is mapped to one of these unified types.
    This enables unified processing across heterogeneous log sources.
    """
    # Logon activities
    LOGON = "LOGON"
    LOGOFF = "LOGOFF"

    # Device activities
    DEVICE_CONNECT = "DEVICE_CONNECT"
    DEVICE_DISCONNECT = "DEVICE_DISCONNECT"

    # File activities
    FILE_OPEN = "FILE_OPEN"
    FILE_WRITE = "FILE_WRITE"
    FILE_COPY = "FILE_COPY"
    FILE_DELETE = "FILE_DELETE"

    # Email activities
    EMAIL_SENT = "EMAIL_SENT"
    EMAIL_RECEIVED = "EMAIL_RECEIVED"

    # HTTP activities
    HTTP_REQUEST = "HTTP_REQUEST"

    # Badge activities (if present)
    BADGE_IN = "BADGE_IN"
    BADGE_OUT = "BADGE_OUT"


class RiskLevel(str, Enum):
    """User risk classification levels."""
    BENIGN = "benign"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatScenario(str, Enum):
    """
    CERT r4.2 threat scenario categories.

    These are inferred from the ground truth answer files and CERT documentation:
    - Scenario 1: Data exfiltration (e.g., using removable media)
    - Scenario 2: IT system sabotage / misuse
    - Scenario 3: Fraud (unauthorized access for personal gain)
    - Scenario 4: Espionage (unauthorized access for competitive advantage)
    """
    UNKNOWN = "unknown"
    DATA_EXFIL = "data_exfiltration"
    SABOTAGE = "sabotage"
    FRAUD = "fraud"
    ESPIONAGE = "espionage"


# ─── CERT r4.2 Raw Schemas ───────────────────────────────────────────────────

@dataclass(frozen=True)
class LogonSchema:
    """Schema definition for logon.csv (CERT r4.2)."""
    columns: tuple[str, ...] = (
        "id",
        "date",
        "user",
        "pc",
        "activity",
    )
    dtypes: Dict[str, pl.DataType] = field(default_factory=lambda: {
        "id": pl.Utf8,
        "date": pl.Utf8,          # Parsed to datetime later
        "user": pl.Utf8,
        "pc": pl.Utf8,
        "activity": pl.Categorical,
    })
    expected_activity_values: tuple[str, ...] = ("Logon", "Logoff")
    primary_key: str = "id"


@dataclass(frozen=True)
class DeviceSchema:
    """Schema definition for device.csv (CERT r4.2)."""
    columns: tuple[str, ...] = (
        "id",
        "date",
        "user",
        "pc",
        "activity",
    )
    dtypes: Dict[str, pl.DataType] = field(default_factory=lambda: {
        "id": pl.Utf8,
        "date": pl.Utf8,
        "user": pl.Utf8,
        "pc": pl.Utf8,
        "activity": pl.Categorical,
    })
    expected_activity_values: tuple[str, ...] = ("Connect", "Disconnect")
    primary_key: str = "id"


@dataclass(frozen=True)
class FileSchema:
    """Schema definition for file.csv (CERT r4.2).

    Note: Unlike r5.2, r4.2 file.csv has columns:
    id, date, user, pc, filename, content

    The 'activity' column is NOT present in r4.2 — file operations
    are inferred from the filename path (R:\ = removable media write).
    """
    columns: tuple[str, ...] = (
        "id",
        "date",
        "user",
        "pc",
        "filename",
        "content",
    )
    dtypes: Dict[str, pl.DataType] = field(default_factory=lambda: {
        "id": pl.Utf8,
        "date": pl.Utf8,
        "user": pl.Utf8,
        "pc": pl.Utf8,
        "filename": pl.Utf8,
        "content": pl.Utf8,
    })
    primary_key: str = "id"

    # File path patterns
    removable_drive_prefix: str = "R:\\"   # Removable media mount point


@dataclass(frozen=True)
class EmailSchema:
    """Schema definition for email.csv (CERT r4.2).

    Note: r4.2 has:
    id, date, user, pc, to, cc, bcc, from, size, attachment_count, content

    Unlike r5.2 which uses attachment_count instead of a list of attachments.
    The 'to', 'cc', 'bcc' fields are semicolon-delimited strings.
    """
    columns: tuple[str, ...] = (
        "id",
        "date",
        "user",
        "pc",
        "to",
        "cc",
        "bcc",
        "from",
        "size",
        "attachments",
        "content",
    )
    dtypes: Dict[str, pl.DataType] = field(default_factory=lambda: {
        "id": pl.Utf8,
        "date": pl.Utf8,
        "user": pl.Utf8,
        "pc": pl.Utf8,
        "to": pl.Utf8,
        "cc": pl.Utf8,
        "bcc": pl.Utf8,
        "from": pl.Utf8,
        "size": pl.Int64,
        "attachments": pl.Int64,   # Number of attachments (integer)
        "content": pl.Utf8,
    })
    primary_key: str = "id"

    # Email domain patterns
    internal_domain: str = "dtaa.com"   # Internal corporate domain
    non_employee_domains: tuple[str, ...] = (
        "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
        "aol.com", "icloud.com", "mail.com", "protonmail.com",
    )


@dataclass(frozen=True)
class HttpSchema:
    """Schema definition for http.csv (CERT r4.2)."""
    columns: tuple[str, ...] = (
        "id",
        "date",
        "user",
        "pc",
        "url",
        "content",
    )
    dtypes: Dict[str, pl.DataType] = field(default_factory=lambda: {
        "id": pl.Utf8,
        "date": pl.Utf8,
        "user": pl.Utf8,
        "pc": pl.Utf8,
        "url": pl.Utf8,
        "content": pl.Utf8,
    })
    primary_key: str = "id"


@dataclass(frozen=True)
class LdapSchema:
    """Schema definition for LDAP monthly snapshots."""
    columns: tuple[str, ...] = (
        "employee_name",
        "user_id",
        "email",
        "role",
        "projects",
        "business_unit",
        "functional_unit",
        "department",
        "team",
        "supervisor",
    )
    dtypes: Dict[str, pl.DataType] = field(default_factory=lambda: {
        "employee_name": pl.Utf8,
        "user_id": pl.Utf8,
        "email": pl.Utf8,
        "role": pl.Utf8,
        "projects": pl.Utf8,
        "business_unit": pl.Utf8,
        "functional_unit": pl.Utf8,
        "department": pl.Utf8,
        "team": pl.Utf8,
        "supervisor": pl.Utf8,
    })
    primary_key: str = "user_id"


@dataclass(frozen=True)
class PsychometricSchema:
    """Schema definition for psychometric.csv."""
    columns: tuple[str, ...] = (
        "employee_name",
        "user_id",
        "O",   # Openness
        "C",   # Conscientiousness
        "E",   # Extraversion
        "A",   # Agreeableness
        "N",   # Neuroticism
    )
    dtypes: Dict[str, pl.DataType] = field(default_factory=lambda: {
        "employee_name": pl.Utf8,
        "user_id": pl.Utf8,
        "O": pl.Int64,
        "C": pl.Int64,
        "E": pl.Int64,
        "A": pl.Int64,
        "N": pl.Int64,
    })
    primary_key: str = "user_id"

    # Score ranges (all Big Five traits: 0-50)
    score_min: int = 0
    score_max: int = 50

    # Risk thresholds (research-based correlations with insider risk)
    high_neuroticism_threshold: int = 35
    low_conscientiousness_threshold: int = 25
    low_agreeableness_threshold: int = 25


@dataclass(frozen=True)
class GroundTruthSchema:
    """Schema for ground truth / answer files."""
    columns: tuple[str, ...] = (
        "dataset",
        "scenario",
        "details",
        "user",
        "start",
        "end",
    )
    dtypes: Dict[str, pl.DataType] = field(default_factory=lambda: {
        "dataset": pl.Float64,   # e.g., 4.2, 5.1 (dataset.scenario format)
        "scenario": pl.Int64,
        "details": pl.Utf8,
        "user": pl.Utf8,
        "start": pl.Utf8,
        "end": pl.Utf8,
    })
    primary_key: str = "user"


@dataclass(frozen=True)
class UnifiedEventSchema:
    """
    Canonical schema for all events after normalization.

    This is the unified event format that all log sources are
    transformed into before downstream processing.
    """
    columns: tuple[str, ...] = (
        "event_id",              # Original ID from source file
        "event_hash",            # SHA-256 of (user+timestamp+type) for deduplication
        "timestamp",             # Normalized datetime (UTC-aware)
        "user_id",               # Canonical user ID
        "user_hash",             # Pseudonymized user ID (for joins)
        "pc_id",                 # Canonical PC ID
        "pc_hash",               # Pseudonymized PC ID
        "event_type",            # Unified event type (EventType enum)
        "raw_type",              # Original log source (logon/device/file/email/http)
        "target_object",          # Target entity (PC, filename, URL, etc.)
        "target_details",         # JSON-encoded event-specific metadata
        "temporal_flags",        # JSON-encoded temporal features
        "session_id",            # Associated session ID (if applicable)
        "is_decoy_interaction",   # Whether event involved a decoy file
        "confidence",             # Data quality confidence [0.0, 1.0]
    )
    dtypes: Dict[str, pl.DataType] = field(default_factory=lambda: {
        "event_id": pl.Utf8,
        "event_hash": pl.Utf8,
        "timestamp": pl.Datetime,
        "user_id": pl.Utf8,
        "user_hash": pl.Utf8,
        "pc_id": pl.Utf8,
        "pc_hash": pl.Utf8,
        "event_type": pl.Categorical,
        "raw_type": pl.Categorical,
        "target_object": pl.Utf8,
        "target_details": pl.Utf8,
        "temporal_flags": pl.Utf8,
        "session_id": pl.Utf8,
        "is_decoy_interaction": pl.Boolean,
        "confidence": pl.Float64,
    })


# ─── Schema Registry ───────────────────────────────────────────────────────────

@dataclass
class SchemaRegistry:
    """
    Central registry of all CERT r4.2 schema definitions.

    Provides lookup by filename and validation utilities.
    """

    logon: LogonSchema = field(default_factory=LogonSchema)
    device: DeviceSchema = field(default_factory=DeviceSchema)
    file: FileSchema = field(default_factory=FileSchema)
    email: EmailSchema = field(default_factory=EmailSchema)
    http: HttpSchema = field(default_factory=HttpSchema)
    ldap: LdapSchema = field(default_factory=LdapSchema)
    psychometric: PsychometricSchema = field(default_factory=PsychometricSchema)
    ground_truth: GroundTruthSchema = field(default_factory=GroundTruthSchema)

    def get_schema(self, source: str) -> Any:
        """
        Get schema definition by source name.

        Args:
            source: Source name (logon, device, file, email, http, ldap,
                   psychometric, ground_truth)

        Returns:
            Corresponding schema dataclass
        """
        mapping = {
            "logon": self.logon,
            "device": self.device,
            "file": self.file,
            "email": self.email,
            "http": self.http,
            "ldap": self.ldap,
            "psychometric": self.psychometric,
            "ground_truth": self.ground_truth,
        }
        if source not in mapping:
            raise ValueError(f"Unknown schema source: {source}. "
                             f"Valid sources: {list(mapping.keys())}")
        return mapping[source]

    def get_dtype_mapping(self, source: str) -> Dict[str, pl.DataType]:
        """Get Polars dtype mapping for a given source."""
        return dict(self.get_schema(source).dtypes)

    def get_columns(self, source: str) -> tuple[str, ...]:
        """Get expected column names for a given source."""
        return self.get_schema(source).columns

    def validate_columns(
        self, source: str, actual_columns: list[str]
    ) -> list[str]:
        """
        Validate that actual columns match expected schema.

        Args:
            source: Source name
            actual_columns: Column names from loaded DataFrame

        Returns:
            List of warning messages (empty if all columns match)
        """
        schema = self.get_schema(source)
        warnings = []
        expected = set(schema.columns)
        actual = set(actual_columns)

        missing = expected - actual
        extra = actual - expected

        if missing:
            warnings.append(
                f"[{source}] Missing expected columns: {sorted(missing)}"
            )
        if extra:
            warnings.append(
                f"[{source}] Unexpected extra columns: {sorted(extra)}"
            )

        return warnings


# ─── Global Registry ──────────────────────────────────────────────────────────
CERT_SCHEMAS = SchemaRegistry()

# ─── Lazy imports for heavy modules ───────────────────────────────────────────
# These are imported lazily to avoid circular dependencies and speed up startup
__all__ = [
    # Schemas and enums
    "EventType",
    "RiskLevel",
    "ThreatScenario",
    "LogonSchema",
    "DeviceSchema",
    "FileSchema",
    "EmailSchema",
    "HttpSchema",
    "LdapSchema",
    "PsychometricSchema",
    "GroundTruthSchema",
    "UnifiedEventSchema",
    "SchemaRegistry",
    "CERT_SCHEMAS",
    # Classes (lazy-loaded)
    "CertDatasetLoader",
    "LogNormalizer",
    "PrivacyManager",
    "AuditLogger",
]


def __getattr__(name: str):
    """Lazy-load heavy modules on first access."""
    if name == "CertDatasetLoader":
        from src.cert_dataset.loaders import CertDatasetLoader
        return CertDatasetLoader
    if name == "LogNormalizer":
        from src.cert_dataset.normalizer import LogNormalizer
        return LogNormalizer
    if name == "PrivacyManager":
        from src.cert_dataset.privacy import PrivacyManager
        return PrivacyManager
    if name == "AuditLogger":
        from src.cert_dataset.privacy import AuditLogger
        return AuditLogger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
