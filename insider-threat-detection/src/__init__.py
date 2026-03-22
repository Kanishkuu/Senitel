"""
CERT Insider Threat Detection — Data Loading and Preprocessing Package

This package provides the data loading and normalization pipeline for the
CERT Insider Threat Dataset r4.2.

Modules:
    cert_dataset  — Schema definitions, loaders, normalizers, privacy tools
    utils        — Configuration, logging, helpers
"""

from src.cert_dataset import (
    CertDatasetLoader,
    LogNormalizer,
    PrivacyManager,
    AuditLogger,
)
from src.utils import (
    PipelineConfig,
    setup_logging,
    get_logger,
)

__version__ = "0.1.0"
__all__ = [
    "CertDatasetLoader",
    "LogNormalizer",
    "PrivacyManager",
    "AuditLogger",
    "PipelineConfig",
    "setup_logging",
    "get_logger",
]
