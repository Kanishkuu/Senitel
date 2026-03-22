"""
General-purpose helper utilities for the CERT Insider Threat Detection pipeline.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from pathlib import Path


def format_bytes(num_bytes: int | float) -> str:
    """
    Format a byte count as a human-readable string.

    Args:
        num_bytes: Number of bytes

    Returns:
        Formatted string, e.g. "14.53 GB"
    """
    if num_bytes < 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds as a human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string, e.g. "2h 34m 12s"
    """
    if seconds < 0:
        return "0s"

    parts = []
    for label, size in [
        ("d", 86400),
        ("h", 3600),
        ("m", 60),
        ("s", 1),
    ]:
        count = int(seconds // size)
        if count > 0 or (label == "s" and not parts):
            parts.append(f"{count}{label}")
        seconds %= size

    return " ".join(parts[:4])


def compute_memory_usage() -> dict[str, float]:
    """
    Compute current process memory usage.

    Returns:
        Dictionary with memory statistics in MB:
        - rss: Resident Set Size (actual physical memory used)
        - vms: Virtual Memory Size (total virtual address space)
        - percent: Percentage of total system memory
    """
    try:
        process = psutil.Process()
        mem = process.memory_info()
        return {
            "rss_mb": round(mem.rss / 1_048_576, 2),
            "vms_mb": round(mem.vms / 1_048_576, 2),
            "percent": round(process.memory_percent(), 2),
            "available_mb": round(
                psutil.virtual_memory().available / 1_048_576, 2
            ),
        }
    except ImportError:
        return {"rss_mb": 0, "vms_mb": 0, "percent": 0, "available_mb": 0}


def estimate_processing_time(
    rows_processed: int,
    total_rows: int,
    elapsed_seconds: float,
) -> dict[str, float]:
    """
    Estimate total processing time and remaining time.

    Args:
        rows_processed: Number of rows processed so far
        total_rows: Total rows to process
        elapsed_seconds: Seconds elapsed so far

    Returns:
        Dictionary with:
        - rows_per_second: Processing throughput
        - eta_seconds: Estimated seconds remaining
        - total_estimated_seconds: Total estimated time
        - percent_complete: Percentage complete
    """
    if rows_processed <= 0 or elapsed_seconds <= 0:
        return {
            "rows_per_second": 0.0,
            "eta_seconds": 0.0,
            "total_estimated_seconds": 0.0,
            "percent_complete": 0.0,
        }

    rows_per_second = rows_processed / elapsed_seconds
    percent_complete = (rows_processed / total_rows) * 100
    remaining_rows = total_rows - rows_processed
    eta_seconds = remaining_rows / rows_per_second if rows_per_second > 0 else 0
    total_estimated = elapsed_seconds + eta_seconds

    return {
        "rows_per_second": round(rows_per_second, 1),
        "eta_seconds": round(eta_seconds, 1),
        "total_estimated_seconds": round(total_estimated, 1),
        "percent_complete": round(percent_complete, 2),
    }


class ProgressTracker:
    """
    Tracks and reports processing progress with throughput estimation.

    Usage:
        tracker = ProgressTracker(total=1_000_000, description="Loading logon.csv")
        for chunk in chunks:
            process(chunk)
            tracker.update(len(chunk))
        tracker.finish()
    """

    def __init__(
        self,
        total: int,
        description: str = "",
        log_interval: float = 10.0,
    ):
        """
        Initialize progress tracker.

        Args:
            total: Total number of items to process
            description: Human-readable description of the operation
            log_interval: Seconds between progress log messages
        """
        self.total = total
        self.description = description
        self.log_interval = log_interval
        self.processed = 0
        self.start_time = time.monotonic()
        self.last_log_time = self.start_time
        self._finished = False

    def update(self, count: int) -> None:
        """Update progress by adding `count` processed items."""
        if self._finished:
            return
        self.processed += count
        current_time = time.monotonic()
        elapsed = current_time - self.start_time

        if current_time - self.last_log_time >= self.log_interval:
            eta_data = estimate_processing_time(
                self.processed, self.total, elapsed
            )
            pct = eta_data["percent_complete"]
            rate = eta_data["rows_per_second"]
            eta = format_duration(eta_data["eta_seconds"])
            mem = compute_memory_usage()

            print(
                f"  {self.description}: {self.processed:,}/{self.total:,} "
                f"({pct:.1f}%) | {rate:,.0f} rows/s | ETA: {eta} "
                f"| MEM: {mem['rss_mb']:.0f} MB",
                end="\r",
                flush=True,
            )
            self.last_log_time = current_time

    def finish(self) -> None:
        """Mark processing as complete and print final summary."""
        if self._finished:
            return
        self._finished = True
        elapsed = time.monotonic() - self.start_time
        rate = self.processed / elapsed if elapsed > 0 else 0
        print(
            f"  {self.description}: {self.processed:,}/{self.total:,} "
            f"(100.0%) | {rate:,.0f} rows/s | "
            f"took {format_duration(elapsed)}"
        )


def parse_cert_timestamp(ts_str: str) -> datetime | None:
    """
    Parse CERT dataset timestamp string to datetime.

    CERT uses MM/DD/YYYY HH:MM:SS format (12-hour, no AM/PM indicator).
    This means times like 01:00:00 could be 1 AM or 1 PM — we treat them
    as literal hours since the dataset is synthetic and internally consistent.

    Args:
        ts_str: Timestamp string from CERT CSV, e.g. "01/02/2010 07:27:19"

    Returns:
        datetime object or None if parsing fails
    """
    if not ts_str or not isinstance(ts_str, str):
        return None

    ts_str = ts_str.strip()

    # Try primary format: MM/DD/YYYY HH:MM:SS
    for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M"):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue

    # Try with extra spaces or AM/PM
    try:
        # Handle "1/2/2010 7:30:45" (single-digit month/day)
        ts_str_fixed = ts_str.strip()
        parts = ts_str_fixed.split(" ")
        if len(parts) == 2:
            date_parts = parts[0].split("/")
            if len(date_parts) == 3:
                # Normalize to MM/DD/YYYY
                month = date_parts[0].zfill(2)
                day = date_parts[1].zfill(2)
                year = date_parts[2]
                normalized = f"{month}/{day}/{year} {parts[1]}"
                return datetime.strptime(normalized, "%m/%d/%Y %H:%M:%S")
    except (ValueError, IndexError):
        pass

    return None


def validate_user_id(user_id: str | None) -> bool:
    """
    Validate a CERT user ID format.

    CERT user IDs follow the pattern: 3 uppercase letters + 4 digits
    Examples: AAA0001, ONS0995, CSF0929

    Args:
        user_id: User ID string to validate

    Returns:
        True if valid CERT user ID format
    """
    if not user_id or not isinstance(user_id, str):
        return False
    user_id = user_id.strip()
    if len(user_id) != 7:
        return False
    return user_id[:3].isalpha() and user_id[:3].isupper() and user_id[3:].isdigit()


def validate_pc_id(pc_id: str | None) -> bool:
    """
    Validate a CERT PC identifier format.

    CERT PC IDs follow the pattern: PC-XXXX (e.g., PC-0168, PC-8025)

    Args:
        pc_id: PC identifier string to validate

    Returns:
        True if valid CERT PC ID format
    """
    if not pc_id or not isinstance(pc_id, str):
        return False
    pc_id = pc_id.strip()
    if not pc_id.startswith("PC-"):
        return False
    suffix = pc_id[3:]
    return len(suffix) >= 1 and suffix.replace("-", "").isdigit()
