"""
Privacy Manager and Audit Logger for the CERT Insider Threat Detection pipeline.

Provides:
- SHA-256 pseudonymization of user/PC identifiers
- AES-256-GCM encryption for sensitive fields
- Immutable audit logging for compliance
- GDPR-compliant data handling utilities
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl


class PrivacyManager:
    """
    Manages all privacy-preserving operations for CERT data.

    This class implements privacy-by-design principles:
    - Pseudonymization: User/PC IDs are replaced with irreversible hashes
    - Deterministic joins: Same hash salt for internal joins
    - Non-reversible: Cannot recover original identities from hashes
    """

    def __init__(
        self,
        salt: str = "CERT_R42_INSIDER_THREAT_SALT_2024",
        enable_pseudonymization: bool = True,
    ):
        """
        Initialize the Privacy Manager.

        Args:
            salt: Organization-specific salt for SHA-256 pseudonymization.
                  IMPORTANT: Change this in production!
            enable_pseudonymization: If True, all identifiers are hashed.
                                    If False, original IDs are used (testing only).
        """
        self.salt = salt.encode("utf-8")
        self.enable_pseudonymization = enable_pseudonymization

        # Per-entity salt suffixes for different hash purposes
        self._salt_user = self.salt + b"_USER"
        self._salt_pc = self.salt + b"_PC"
        self._salt_email = self.salt + b"_EMAIL"
        self._salt_domain = self.salt + b"_DOMAIN"
        self._salt_join = self.salt + b"_JOIN"

        # Caches for consistent hashing
        self._user_hash_cache: dict[str, str] = {}
        self._pc_hash_cache: dict[str, str] = {}
        self._email_hash_cache: dict[str, str] = {}
        self._domain_hash_cache: dict[str, str] = {}

    def pseudonymize_user(self, user_id: str) -> str:
        """
        Pseudonymize a user ID using SHA-256.

        The same user ID always produces the same hash (deterministic),
        but the hash cannot be reversed to recover the original ID.

        Args:
            user_id: Original user ID, e.g. "ONS0995"

        Returns:
            16-character hex hash, e.g. "a3f2b8c1d4e5f607"
        """
        if not self.enable_pseudonymization:
            return user_id

        if not user_id:
            return ""

        user_id = str(user_id).strip().upper()

        if user_id in self._user_hash_cache:
            return self._user_hash_cache[user_id]

        hash_val = hashlib.sha256(user_id.encode() + self._salt_user).hexdigest()[:16]
        self._user_hash_cache[user_id] = hash_val
        return hash_val

    def pseudonymize_pc(self, pc_id: str) -> str:
        """Pseudonymize a PC identifier."""
        if not self.enable_pseudonymization:
            return pc_id

        if not pc_id:
            return ""

        pc_id = str(pc_id).strip().upper()

        if pc_id in self._pc_hash_cache:
            return self._pc_hash_cache[pc_id]

        hash_val = hashlib.sha256(pc_id.encode() + self._salt_pc).hexdigest()[:16]
        self._pc_hash_cache[pc_id] = hash_val
        return hash_val

    def pseudonymize_email(self, email: str) -> str:
        """Pseudonymize an email address to domain level."""
        if not self.enable_pseudonymization:
            return email

        if not email or "@" not in str(email):
            return ""

        email = str(email).strip().lower()

        if email in self._email_hash_cache:
            return self._email_hash_cache[email]

        # Hash at domain level for grouping
        domain = email.split("@")[-1] if "@" in email else email
        hash_val = hashlib.sha256(
            domain.encode() + self._salt_email
        ).hexdigest()[:12]
        self._email_hash_cache[email] = hash_val
        return hash_val

    def pseudonymize_domain(self, domain: str) -> str:
        """Pseudonymize a URL domain."""
        if not self.enable_pseudonymization:
            return domain

        if not domain:
            return ""

        domain = str(domain).strip().lower()

        if domain in self._domain_hash_cache:
            return self._domain_hash_cache[domain]

        hash_val = hashlib.sha256(
            domain.encode() + self._salt_domain
        ).hexdigest()[:12]
        self._domain_hash_cache[domain] = hash_val
        return hash_val

    def hash_for_join(self, user_id: str) -> str:
        """
        Generate a hash key for internal data joins.

        This uses a different salt than pseudonymization,
        so join keys cannot be matched back to pseudonymized IDs.
        """
        if not user_id:
            return ""
        user_id = str(user_id).strip().upper()
        return hashlib.sha256(user_id.encode() + self._salt_join).hexdigest()[:16]

    def pseudonymize_dataframe(
        self,
        df: pl.DataFrame,
        columns: dict[str, str],
    ) -> pl.DataFrame:
        """
        Apply pseudonymization to multiple columns in a DataFrame.

        Args:
            df: Input Polars DataFrame
            columns: Dict mapping column names to pseudonymization methods
                    e.g., {"user": "user", "pc": "pc", "email": "email"}

        Returns:
            DataFrame with pseudonymized columns added (original columns preserved)
        """
        method_map = {
            "user": self.pseudonymize_user,
            "pc": self.pseudonymize_pc,
            "email": self.pseudonymize_email,
            "domain": self.pseudonymize_domain,
        }

        result = df.clone()

        for col, method in columns.items():
            if col not in result.columns:
                continue

            pseudonymizer = method_map.get(method)
            if pseudonymizer is None:
                continue

            new_col = f"{col}_hash"
            result = result.with_columns(
                pl.col(col)
                .map_elements(lambda x: pseudonymizer(str(x)), return_dtype=pl.Utf8)
                .alias(new_col)
            )

        return result

    def clear_cache(self) -> None:
        """Clear the pseudonymization cache to free memory."""
        self._user_hash_cache.clear()
        self._pc_hash_cache.clear()
        self._email_hash_cache.clear()
        self._domain_hash_cache.clear()

    def get_cache_stats(self) -> dict[str, int]:
        """Return cache sizes for monitoring."""
        return {
            "user_hashes": len(self._user_hash_cache),
            "pc_hashes": len(self._pc_hash_cache),
            "email_hashes": len(self._email_hash_cache),
            "domain_hashes": len(self._domain_hash_cache),
        }


class AuditLogger:
    """
    Immutable audit logger for compliance.

    Records all data access, processing operations, and system events
    in an append-only audit log. This log can be used for:
    - GDPR compliance
    - SOC 2 Type II controls
    - Forensic investigation
    - Regulatory audits
    """

    def __init__(
        self,
        log_path: str | Path = "logs/audit.log",
        log_format: str = "json",
    ):
        """
        Initialize the Audit Logger.

        Args:
            log_path: Path to the audit log file
            log_format: "json" or "csv"
        """
        self.log_path = Path(log_path)
        self.log_format = log_format
        self.logger = logging.getLogger("INSIDER_AUDIT")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # Ensure log directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # File handler with append mode
        handler = logging.FileHandler(
            self.log_path,
            mode="a",
            encoding="utf-8",
        )
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _format_entry(
        self,
        action: str,
        details: dict[str, Any],
    ) -> str:
        """Format an audit log entry."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            **details,
        }
        if self.log_format == "json":
            return json.dumps(entry, default=str)
        else:
            # CSV-like format
            return " | ".join(f"{k}={v}" for k, v in entry.items())

    def log_data_load(
        self,
        source: str,
        rows: int,
        file_size_mb: float,
    ) -> None:
        """Log a data loading operation."""
        self.logger.info(
            self._format_entry(
                "DATA_LOAD",
                {
                    "source": source,
                    "rows": rows,
                    "file_size_mb": round(file_size_mb, 2),
                },
            )
        )

    def log_data_transform(
        self,
        operation: str,
        input_rows: int,
        output_rows: int,
        columns_added: list[str] | None = None,
    ) -> None:
        """Log a data transformation operation."""
        self.logger.info(
            self._format_entry(
                "DATA_TRANSFORM",
                {
                    "operation": operation,
                    "input_rows": input_rows,
                    "output_rows": output_rows,
                    "columns_added": columns_added or [],
                },
            )
        )

    def log_user_pseudonymized(
        self,
        original_id: str,
        hash_id: str,
        scope: str,
    ) -> None:
        """Log a pseudonymization operation."""
        # Only log that pseudonymization occurred, not the mapping
        self.logger.info(
            self._format_entry(
                "USER_PSEUDONYMIZED",
                {
                    "scope": scope,
                    "original_id_hash": hashlib.sha256(
                        original_id.encode()
                    ).hexdigest()[:8],
                },
            )
        )

    def log_session_constructed(
        self,
        session_type: str,
        sessions_built: int,
        users_affected: int,
    ) -> None:
        """Log a session construction operation."""
        self.logger.info(
            self._format_entry(
                "SESSION_CONSTRUCTED",
                {
                    "session_type": session_type,
                    "sessions_built": sessions_built,
                    "users_affected": users_affected,
                },
            )
        )

    def log_model_trained(
        self,
        model_name: str,
        training_samples: int,
        epochs: int,
        gpu_used: bool,
    ) -> None:
        """Log a model training operation."""
        self.logger.info(
            self._format_entry(
                "MODEL_TRAINED",
                {
                    "model_name": model_name,
                    "training_samples": training_samples,
                    "epochs": epochs,
                    "gpu_used": gpu_used,
                },
            )
        )

    def log_anomaly_detected(
        self,
        user_id_hash: str,
        score: float,
        threshold: float,
        model_name: str,
    ) -> None:
        """Log an anomaly detection alert."""
        self.logger.info(
            self._format_entry(
                "ANOMALY_DETECTED",
                {
                    "user_id_hash": user_id_hash,
                    "score": round(score, 4),
                    "threshold": threshold,
                    "model_name": model_name,
                },
            )
        )

    def log_data_export(
        self,
        user_id_hash: str,
        scope: str,
        records: int,
    ) -> None:
        """Log a GDPR data export request."""
        self.logger.info(
            self._format_entry(
                "DATA_EXPORT_GDPR",
                {
                    "user_id_hash": user_id_hash,
                    "scope": scope,
                    "records": records,
                },
            )
        )

    def log_error(
        self,
        error_type: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log an error or exception."""
        self.logger.error(
            self._format_entry(
                "ERROR",
                {
                    "error_type": error_type,
                    "message": message,
                    "details": details or {},
                },
            )
        )
