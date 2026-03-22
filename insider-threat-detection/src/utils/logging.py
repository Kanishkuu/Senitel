"""
Structured logging setup for the CERT Insider Threat Detection pipeline.

Provides JSON-structured logs for production and human-readable logs for
development. All modules use `get_logger(__name__)` for consistency.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from structlog.types import EventDict, Processor

if TYPE_CHECKING:
    from src.utils.config import PipelineConfig


# ─── Structlog Processors ──────────────────────────────────────────────────────

def add_timestamp_logger_name(
    logger: object, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add logger name (from structlog's internal `logger` bind) to output."""
    event_dict["logger"] = event_dict.get("logger", "root")
    return event_dict


def rename_event_key(
    logger: object, method_name: str, event_dict: EventDict
) -> EventDict:
    """Rename 'event' key to 'message' for standard log format compatibility."""
    event_dict.setdefault("message", event_dict.pop("event", ""))
    return event_dict


def add_pipeline_context(
    logger: object, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add pipeline version and stage context to every log entry."""
    event_dict.setdefault("pipeline_version", "0.1.0")
    event_dict.setdefault("dataset_version", "r4.2")
    return event_dict


def add_memory_usage(
    logger: object, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add current memory usage (RSS in MB) to log entries."""
    try:
        import psutil
        process = psutil.Process()
        event_dict["memory_mb"] = round(process.memory_info().rss / 1_048_576, 2)
    except ImportError:
        pass
    return event_dict


# ─── Setup Functions ──────────────────────────────────────────────────────────

def setup_logging(
    level: str = "INFO",
    log_format: str = "json",
    log_dir: Path | str | None = None,
    force_json: bool = False,
) -> None:
    """
    Configure structured logging for the entire pipeline.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_format: "json" for machine-readable, "console" for humans
        log_dir: Directory for log files (created if needed)
        force_json: Force JSON output even when log_format is "console"
                   (useful for CI/CD)
    """
    import logging.config
    import time

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Shared processors for all output modes
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        add_pipeline_context,
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME: True,
                structlog.processors.CallsiteParameter.FUNC_NAME: True,
                structlog.processors.CallsiteParameter.LINENO: True,
            }
        ),
        rename_event_key,
    ]

    if log_format == "json" or force_json:
        shared_processors.append(add_memory_usage)
        renderer = structlog.processors.JSONRenderer()
    else:
        # Human-readable console output
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Standard library configuration
    handlers: list[logging.Handler] = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    handlers.append(console_handler)

    # File handler (rotating)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            log_dir / "pipeline.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(file_handler)

    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=handlers,
        force=True,
    )

    # Suppress noisy third-party loggers
    for noisy_logger in ["urllib3", "requests", "charset_normalizer"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger for the given module name.

    Usage:
        logger = get_logger(__name__)
        logger.info("processing_file", file="logon.csv", rows=1_000_000)

    Args:
        name: Logger name (typically __name__ from the calling module)

    Returns:
        Configured structlog bound logger
    """
    return structlog.get_logger(name)
