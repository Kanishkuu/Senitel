"""
Sequence Encoder for CERT Insider Threat Detection.

Encodes raw event logs into fixed-length temporal sequences with 16-dimensional
feature vectors per timestep. Produces daily_sequences.parquet matching the
required output schema.

Features per timestep (16 total):
    [0]  event_type_token       - Categorical event type (0-11)
    [1]  pc_id_encoded          - Encoded PC identifier
    [2]  hour_bucket            - Hour of day normalized (0-1)
    [3]  day_of_week            - Day of week normalized (0-1)
    [4]  is_working_hours       - Binary (0.0/1.0)
    [5]  is_weekend            - Binary (0.0/1.0)
    [6]  is_after_hours         - Binary (0.0/1.0)
    [7]  session_progress      - Position in session (0.0-1.0)
    [8]  device_connected       - Binary (0.0/1.0)
    [9]  file_operation_type    - File op encoding (0=none, 1=open, 2=write, 3=copy, 4=delete)
    [10] email_sent             - Binary (0.0/1.0)
    [11] is_removable_access    - Binary (0.0/1.0, R:\ drive)
    [12] url_domain_category    - Encoded domain category
    [13] activity_density       - Events per minute in session window
    [14] temporal_entropy       - Shannon entropy of event type distribution
    [15] session_event_count    - Cumulative event count in session

Output schema: user_id, date, sequence, seq_len, window_id, session_count,
               event_count, label
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import polars as pl
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SEQ_LEN: int = 512
DATA_DIR: Path = Path("C:/Darsh/NCPI/insider-threat-detection/data/normalized")
OUTPUT_DIR: Path = Path("C:/Darsh/NCPI/insider-threat-detection/data/processed")

# Working hours configuration (business hours)
WORK_START_HOUR: int = 8   # 08:00
WORK_END_HOUR: int = 18    # 18:00
AFTER_HOURS_START: int = 22  # 22:00

# Session timeout in minutes (gap between events > this = new session)
SESSION_GAP_MINUTES: int = 30

# Event type token mapping (12 categories, 0-11)
EVENT_TYPE_MAP: dict[str, int] = {
    "Logon": 0,
    "Logoff": 1,
    "Connect": 2,
    "Disconnect": 3,
    "Open": 4,
    "Write": 5,
    "Copy": 6,
    "Delete": 7,
    "Email": 8,
    "HTTP": 9,
    "File": 10,   # Generic file access (non-removable)
    "Other": 11,
}

# Domain category encoding
DOMAIN_CATEGORY_MAP: dict[str, int] = {
    "na": 0,
    "none": 0,
    "": 0,
    "business": 1,
    "webmail": 2,
    "social": 3,
    "streaming": 4,
    "gaming": 5,
    "shopping": 6,
    "news": 7,
    "entertainment": 8,
    "technology": 9,
    "other": 10,
}

# File operation encoding
FILE_OP_MAP: dict[str, int] = {
    "none": 0,
    "open": 1,
    "write": 2,
    "copy": 3,
    "delete": 4,
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    """Tracks state within a user-day session."""
    start_time: Optional[datetime] = None
    events: list[list[float]] = field(default=list)
    device_active: bool = False
    current_pc: Optional[str] = None
    file_op_type: int = 0
    removable_active: bool = False
    email_active: bool = False
    url_category: int = 0
    event_type_counts: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    session_idx: int = 0


@dataclass
class EncoderConfig:
    """Configuration for the sequence encoder."""
    max_seq_len: int = MAX_SEQ_LEN
    work_start_hour: int = WORK_START_HOUR
    work_end_hour: int = WORK_END_HOUR
    after_hours_start: int = AFTER_HOURS_START
    session_gap_minutes: int = SESSION_GAP_MINUTES
    data_dir: Path = DATA_DIR
    output_dir: Path = OUTPUT_DIR


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def compute_shannon_entropy(counts: dict[int, int], total: int) -> float:
    """Compute Shannon entropy of a distribution."""
    if total <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return float(entropy)


def normalize_hour(hour: int) -> float:
    """Normalize hour to [0, 1] range (hour / 23)."""
    return hour / 23.0


def normalize_day_of_week(day: int) -> float:
    """Normalize day of week to [0, 1] range (Monday=0, Sunday=6)."""
    return day / 6.0


def is_working_hours(hour: int, config: EncoderConfig) -> float:
    """Check if hour falls within working hours."""
    return 1.0 if config.work_start_hour <= hour < config.work_end_hour else 0.0


def is_after_hours(hour: int, config: EncoderConfig) -> float:
    """Check if hour is after hours (late night)."""
    return 1.0 if hour >= config.after_hours_start or hour < 6 else 0.0


def is_weekend(day: int) -> float:
    """Check if day is weekend (Saturday=5 or Sunday=6)."""
    return 1.0 if day >= 5 else 0.0


def encode_event_type(event_type: str) -> int:
    """Encode event type string to integer token."""
    return EVENT_TYPE_MAP.get(event_type, EVENT_TYPE_MAP["Other"])


def encode_domain_category(domain: str) -> int:
    """Encode domain category to integer."""
    if domain is None:
        return 0
    domain_lower = str(domain).lower().strip()
    return DOMAIN_CATEGORY_MAP.get(domain_lower, DOMAIN_CATEGORY_MAP["other"])


def encode_file_operation(op_type: str) -> int:
    """Encode file operation type."""
    if op_type is None:
        return 0
    op_lower = str(op_type).lower().strip()
    return FILE_OP_MAP.get(op_lower, 0)


# ---------------------------------------------------------------------------
# Sequence Encoder Class
# ---------------------------------------------------------------------------

class SequenceEncoder:
    """
    Encodes CERT insider threat event logs into temporal sequences.

    Processes raw event logs from multiple sources (logon, device, file, email,
    http) and produces fixed-length sequences with 16-dimensional feature vectors.

    Output DataFrame columns:
        - user_id: str
        - date: date
        - sequence: list[list[float]] (max_seq_len x 16)
        - seq_len: int (actual length before padding)
        - window_id: str (format: {user_id}_{date})
        - session_count: int
        - event_count: int
        - label: int (0=benign, 1=malicious)
    """

    FEATURE_DIM: int = 16

    def __init__(self, config: Optional[EncoderConfig] = None):
        """Initialize encoder with configuration."""
        self.config = config or EncoderConfig()
        self._pc_encoder: dict[str, int] = {}
        self._pc_counter: int = 0
        self._reset()

    def _reset(self) -> None:
        """Reset encoder state for new encoding run."""
        self._pc_encoder = {}
        self._pc_counter = 0

    def _get_pc_id(self, pc: str) -> int:
        """Get or create encoded PC identifier."""
        if pc not in self._pc_encoder:
            self._pc_encoder[pc] = self._pc_counter
            self._pc_counter += 1
        return self._pc_encoder[pc]

    def _build_feature_vector(
        self,
        timestamp: datetime,
        event_type: int,
        pc_id: int,
        session: SessionState,
        config: EncoderConfig,
    ) -> list[float]:
        """Build 16-dimensional feature vector for a single event."""
        hour = timestamp.hour
        day = timestamp.weekday()

        # Session progress (position within session, 0.0 to 1.0)
        if session.start_time is None or len(session.events) == 0:
            session_progress = 0.0
        else:
            elapsed = (timestamp - session.start_time).total_seconds()
            # Estimate session duration or cap at reasonable time (4 hours max)
            session_progress = min(1.0, elapsed / 14400.0)

        # Activity density (events per minute in session)
        elapsed_minutes = max(1.0, (timestamp - session.start_time).total_seconds() / 60.0) if session.start_time else 1.0
        activity_density = len(session.events) / elapsed_minutes

        # Temporal entropy
        total_events = sum(session.event_type_counts.values())
        temporal_entropy = compute_shannon_entropy(dict(session.event_type_counts), total_events)

        # Session event count (1-indexed for first event)
        session_event_count = len(session.events) + 1

        # Build feature vector (16 features as specified)
        features = [
            float(event_type),                    # [0]  event_type_token (0-11)
            float(pc_id),                         # [1]  pc_id_encoded
            normalize_hour(hour),                 # [2]  hour_bucket (0-1)
            normalize_day_of_week(day),           # [3]  day_of_week (0-1)
            is_working_hours(hour, config),       # [4]  is_working_hours (0/1)
            is_weekend(day),                      # [5]  is_weekend (0/1)
            is_after_hours(hour, config),         # [6]  is_after_hours (0/1)
            float(session_progress),              # [7]  session_progress (0-1)
            1.0 if session.device_active else 0.0,  # [8]  device_connected (0/1)
            float(session.file_op_type),          # [9]  file_operation_type (0-4)
            1.0 if session.email_active else 0.0,   # [10] email_sent (0/1)
            1.0 if session.removable_active else 0.0,  # [11] is_removable_access (0/1)
            float(session.url_category),           # [12] url_domain_category
            float(activity_density),              # [13] activity_density
            float(temporal_entropy),              # [14] temporal_entropy
            float(session_event_count),           # [15] session_event_count
        ]

        return features

    def _update_session_state(
        self,
        session: SessionState,
        event_type: str,
        pc: str,
        device_connected: Optional[bool] = None,
        file_op: Optional[str] = None,
        is_removable: Optional[bool] = None,
        email_sent: Optional[bool] = None,
        url_category: Optional[str] = None,
    ) -> None:
        """Update session state based on event attributes."""
        session.current_pc = pc

        if device_connected is not None:
            session.device_active = device_connected

        if file_op is not None and file_op.lower() != "none":
            session.file_op_type = encode_file_operation(file_op)
        else:
            session.file_op_type = 0

        if is_removable is not None:
            session.removable_active = is_removable

        if email_sent is not None:
            session.email_active = email_sent

        if url_category is not None:
            session.url_category = encode_domain_category(url_category)

        # Update event type counts for entropy calculation
        event_token = encode_event_type(event_type)
        session.event_type_counts[event_token] += 1

    def _reset_session_file_state(self, session: SessionState) -> None:
        """Reset file-related session state (file ops are per-event)."""
        session.file_op_type = 0
        session.url_category = 0

    def _reset_session_email_state(self, session: SessionState) -> None:
        """Reset email-related session state after event."""
        session.email_active = False

    def load_data(self) -> dict[str, pl.DataFrame]:
        """Load all input data files using Polars."""
        data_files = {
            "logon": self.config.data_dir / "logon.parquet",
            "device": self.config.data_dir / "device.parquet",
            "file": self.config.data_dir / "file.parquet",
            "email": self.config.data_dir / "email.parquet",
            "http": self.config.data_dir / "http.parquet",
            "ground_truth": self.config.data_dir / "ground_truth.parquet",
        }

        loaded_data = {}

        for name, path in data_files.items():
            if not path.exists():
                raise FileNotFoundError(f"Required data file not found: {path}")

            df = pl.read_parquet(path)
            loaded_data[name] = df

        return loaded_data

    def _detect_timestamp_column(self, df: pl.DataFrame) -> str:
        """Detect the timestamp column name in a DataFrame."""
        candidates = ["timestamp", "time", "datetime", "date", "event_time", "ts"]
        for col in candidates:
            if col in df.columns:
                return col
        # Try to find by dtype
        for col in df.columns:
            if df[col].dtype in [pl.Datetime, pl.Date]:
                return col
        raise ValueError(f"No timestamp column found in DataFrame. Columns: {df.columns}")

    def _detect_user_column(self, df: pl.DataFrame) -> str:
        """Detect the user column name in a DataFrame."""
        candidates = ["user", "user_id", "username", "employee", "employee_id", "id"]
        for col in candidates:
            if col in df.columns:
                return col
        raise ValueError(f"No user column found. Columns: {df.columns}")

    def _detect_pc_column(self, df: pl.DataFrame) -> str:
        """Detect the PC/workstation column name in a DataFrame."""
        candidates = ["pc", "computer", "workstation", "device", "host", "pc_name"]
        for col in candidates:
            if col in df.columns:
                return col
        raise ValueError(f"No PC column found. Columns: {df.columns}")

    def _parse_timestamp(self, df: pl.DataFrame, ts_col: str) -> pl.DataFrame:
        """Parse timestamp column to datetime."""
        dtype = df[ts_col].dtype

        if dtype == pl.Datetime:
            return df

        if dtype == pl.Date:
            return df.with_columns([
                pl.col(ts_col).cast(pl.Datetime).alias(ts_col)
            ])

        # String format
        return df.with_columns([
            pl.col(ts_col).str.to_datetime("%Y-%m-%d %H:%M:%S.%f").alias(ts_col)
        ])

    def _prepare_logon_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Prepare and normalize logon data."""
        ts_col = self._detect_timestamp_column(df)
        user_col = self._detect_user_column(df)
        pc_col = self._detect_pc_column(df)

        # Find activity column
        activity_col = None
        for col in df.columns:
            if col.lower() in ["activity", "action", "event", "type", "logon"]:
                activity_col = col
                break

        df = self._parse_timestamp(df, ts_col)

        result = df.select([
            pl.col(ts_col).alias("timestamp"),
            pl.col(user_col).alias("user"),
            pl.col(pc_col).alias("pc"),
        ])

        if activity_col:
            result = result.with_columns([
                pl.col(activity_col).alias("activity")
            ])
        else:
            result = result.with_columns([
                pl.lit("Logon").alias("activity")
            ])

        return result.with_columns([
            pl.col("timestamp").dt.date().alias("date"),
            pl.col("timestamp").dt.hour().alias("hour"),
            pl.col("timestamp").dt.weekday().alias("day_of_week"),
        ])

    def _prepare_device_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Prepare and normalize device data."""
        ts_col = self._detect_timestamp_column(df)
        user_col = self._detect_user_column(df)
        pc_col = self._detect_pc_column(df)

        activity_col = None
        for col in df.columns:
            if col.lower() in ["activity", "action", "event", "type"]:
                activity_col = col
                break

        df = self._parse_timestamp(df, ts_col)

        result = df.select([
            pl.col(ts_col).alias("timestamp"),
            pl.col(user_col).alias("user"),
            pl.col(pc_col).alias("pc"),
        ])

        if activity_col:
            result = result.with_columns([
                pl.col(activity_col).alias("activity")
            ])
        else:
            result = result.with_columns([
                pl.lit("Connect").alias("activity")
            ])

        return result.with_columns([
            pl.col("timestamp").dt.date().alias("date"),
        ])

    def _prepare_file_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Prepare and normalize file data."""
        ts_col = self._detect_timestamp_column(df)
        user_col = self._detect_user_column(df)
        pc_col = self._detect_pc_column(df)

        # Find operation type column
        op_col = None
        for col in df.columns:
            if col.lower() in ["operation", "operation_type", "op_type", "action", "type"]:
                op_col = col
                break

        # Find removable media column
        removable_col = None
        for col in df.columns:
            if col.lower() in ["removable", "is_removable", "removable_media", "external"]:
                removable_col = col
                break

        df = self._parse_timestamp(df, ts_col)

        result = df.select([
            pl.col(ts_col).alias("timestamp"),
            pl.col(user_col).alias("user"),
            pl.col(pc_col).alias("pc"),
        ])

        if op_col:
            result = result.with_columns([
                pl.col(op_col).alias("operation_type")
            ])
        else:
            result = result.with_columns([
                pl.lit("Open").alias("operation_type")
            ])

        if removable_col:
            result = result.with_columns([
                pl.col(removable_col).cast(pl.Boolean).alias("is_removable")
            ])
        else:
            result = result.with_columns([
                pl.lit(False).alias("is_removable")
            ])

        return result.with_columns([
            pl.col("timestamp").dt.date().alias("date"),
        ])

    def _prepare_email_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Prepare and normalize email data."""
        ts_col = self._detect_timestamp_column(df)
        user_col = self._detect_user_column(df)
        pc_col = self._detect_pc_column(df)

        # Find attachments column
        attach_col = None
        for col in df.columns:
            if col.lower() in ["attachment", "attachments", "has_attachment", "has_attachments"]:
                attach_col = col
                break

        df = self._parse_timestamp(df, ts_col)

        result = df.select([
            pl.col(ts_col).alias("timestamp"),
            pl.col(user_col).alias("user"),
            pl.col(pc_col).alias("pc"),
        ])

        if attach_col:
            result = result.with_columns([
                pl.col(attach_col).cast(pl.Boolean).alias("has_attachments")
            ])
        else:
            result = result.with_columns([
                pl.lit(False).alias("has_attachments")
            ])

        return result.with_columns([
            pl.col("timestamp").dt.date().alias("date"),
            pl.lit("Email").alias("activity"),
        ])

    def _prepare_http_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Prepare and normalize HTTP data."""
        ts_col = self._detect_timestamp_column(df)
        user_col = self._detect_user_column(df)
        pc_col = self._detect_pc_column(df)

        # Find domain/category column
        domain_col = None
        for col in df.columns:
            if col.lower() in ["domain", "url", "domain_category", "category", "domain_cat"]:
                domain_col = col
                break

        df = self._parse_timestamp(df, ts_col)

        result = df.select([
            pl.col(ts_col).alias("timestamp"),
            pl.col(user_col).alias("user"),
            pl.col(pc_col).alias("pc"),
        ])

        if domain_col:
            result = result.with_columns([
                pl.col(domain_col).alias("domain_category")
            ])
        else:
            result = result.with_columns([
                pl.lit("other").alias("domain_category")
            ])

        return result.with_columns([
            pl.col("timestamp").dt.date().alias("date"),
            pl.lit("HTTP").alias("activity"),
        ])

    def _unify_and_sort_events(self, data: dict[str, pl.DataFrame]) -> pl.DataFrame:
        """Unify all event sources into a single sorted dataframe."""
        unified_events = []

        # Process logon events
        try:
            logon = self._prepare_logon_data(data["logon"])
            unified_events.append(
                logon.select([
                    pl.lit("Logon").alias("event_type"),
                    pl.col("user"),
                    pl.col("pc"),
                    pl.col("timestamp"),
                    pl.col("date"),
                    pl.col("activity"),
                ])
            )
        except Exception as e:
            print(f"Warning: Could not process logon data: {e}")

        # Process device events
        try:
            device = self._prepare_device_data(data["device"])
            device_typed = device.with_columns([
                pl.col("activity").map_elements(
                    lambda x: "Connect" if "onnect" in str(x) else "Disconnect",
                    return_dtype=pl.Utf8
                ).alias("event_type")
            ])
            unified_events.append(
                device_typed.select([
                    pl.col("event_type"),
                    pl.col("user"),
                    pl.col("pc"),
                    pl.col("timestamp"),
                    pl.col("date"),
                ])
            )
        except Exception as e:
            print(f"Warning: Could not process device data: {e}")

        # Process file events
        try:
            file_df = self._prepare_file_data(data["file"])
            unified_events.append(
                file_df.select([
                    pl.col("operation_type").alias("event_type"),
                    pl.col("user"),
                    pl.col("pc"),
                    pl.col("timestamp"),
                    pl.col("date"),
                    pl.col("is_removable"),
                    pl.col("operation_type"),
                ])
            )
        except Exception as e:
            print(f"Warning: Could not process file data: {e}")

        # Process email events
        try:
            email = self._prepare_email_data(data["email"])
            unified_events.append(
                email.select([
                    pl.lit("Email").alias("event_type"),
                    pl.col("user"),
                    pl.col("pc"),
                    pl.col("timestamp"),
                    pl.col("date"),
                    pl.col("has_attachments").alias("email_has_attachments"),
                ])
            )
        except Exception as e:
            print(f"Warning: Could not process email data: {e}")

        # Process HTTP events
        try:
            http_df = self._prepare_http_data(data["http"])
            unified_events.append(
                http_df.select([
                    pl.lit("HTTP").alias("event_type"),
                    pl.col("user"),
                    pl.col("pc"),
                    pl.col("timestamp"),
                    pl.col("date"),
                    pl.col("domain_category"),
                ])
            )
        except Exception as e:
            print(f"Warning: Could not process http data: {e}")

        if not unified_events:
            raise ValueError("No event data could be processed")

        # Concatenate all events and sort by timestamp
        combined = pl.concat(unified_events, how="diagonal_relaxed")
        combined = combined.sort("timestamp")

        return combined

    def _get_labels(self, data: dict[str, pl.DataFrame]) -> dict[str, int]:
        """Extract user labels from ground truth data."""
        try:
            gt = data["ground_truth"]
            labels = {}

            user_col = self._detect_user_column(gt)
            label_col = None
            for col in gt.columns:
                if col.lower() in ["malicious", "is_malicious", "label", "threat", "is_threat"]:
                    label_col = col
                    break

            if label_col is None:
                print("Warning: Could not find label column in ground truth")
                return labels

            for row in gt.iter_rows():
                row_dict = dict(zip(gt.columns, row))
                user = str(row_dict[user_col])
                label = int(row_dict[label_col])
                labels[user] = label

            return labels
        except Exception as e:
            print(f"Warning: Could not load ground truth labels: {e}")
            return {}

    def encode(self, data: Optional[dict[str, pl.DataFrame]] = None) -> pl.DataFrame:
        """
        Encode all events into sequences.

        Returns a Polars DataFrame with columns:
            user_id, date, sequence, seq_len, window_id,
            session_count, event_count, label
        """
        self._reset()

        if data is None:
            data = self.load_data()

        # Get labels
        labels = self._get_labels(data)

        # Unify and sort all events
        print("Unifying event sources...")
        all_events = self._unify_and_sort_events(data)

        print(f"Total events to process: {len(all_events)}")

        # Get unique user-date combinations
        user_dates = (
            all_events.select(["user", "date"])
            .unique()
            .sort(["user", "date"])
            .rename({"user": "user_id"})
        )

        print(f"Unique user-date combinations: {len(user_dates)}")

        # Process each user-date
        results = []
        config = self.config

        # Collect all column names for row iteration
        columns = all_events.columns

        # Iterate over user_dates
        for row_idx, row in enumerate(tqdm(user_dates.iter_rows(named=True), total=len(user_dates), desc="Encoding sequences")):
            user_id = str(row["user_id"])
            date = row["date"]

            # Filter events for this user-date using boolean mask
            mask = (all_events["user"] == user_id) & (all_events["date"] == date)
            user_day_events = all_events.filter(mask).sort("timestamp")

            if len(user_day_events) == 0:
                continue

            # Initialize session tracking
            session = SessionState()
            sessions_in_day = 0
            sequence: list[list[float]] = []

            # Track device state per PC for this user-day
            device_state: dict[str, bool] = {}

            # Process each event
            event_columns = user_day_events.columns
            prev_timestamp: Optional[datetime] = None

            for event_row in user_day_events.iter_rows():
                event_dict = dict(zip(event_columns, event_row))

                timestamp: datetime = event_dict.get("timestamp")
                event_type_raw: str = str(event_dict.get("event_type", "Other"))
                pc: str = str(event_dict.get("pc", "Unknown"))
                pc_id: int = self._get_pc_id(pc)

                # Check for session start (Logon event) or gap
                is_logon = "ogon" in event_type_raw.lower()
                is_session_start = False

                if is_logon:
                    is_session_start = True
                elif prev_timestamp is not None:
                    gap = (timestamp - prev_timestamp).total_seconds() / 60
                    if gap > config.session_gap_minutes:
                        is_session_start = True

                if is_session_start:
                    sessions_in_day += 1
                    session = SessionState(start_time=timestamp, session_idx=sessions_in_day)

                prev_timestamp = timestamp

                # Determine device connection state
                device_active = device_state.get(pc, False)
                if "Connect" in event_type_raw:
                    device_state[pc] = True
                    device_active = True
                elif "Disconnect" in event_type_raw:
                    device_state[pc] = False
                    device_active = False

                # File operation details
                file_op = "none"
                is_removable = False
                if "is_removable" in event_dict:
                    is_removable = bool(event_dict.get("is_removable", False))
                if "operation_type" in event_dict:
                    file_op = str(event_dict.get("operation_type", "none"))

                # Email sent
                email_sent = False
                if "Email" in event_type_raw:
                    email_sent = True
                elif "email_has_attachments" in event_dict:
                    email_sent = bool(event_dict.get("email_has_attachments", False))

                # URL category
                url_category = "other"
                if "domain_category" in event_dict:
                    url_category = str(event_dict.get("domain_category", "other"))

                # Update session state
                self._update_session_state(
                    session=session,
                    event_type=event_type_raw,
                    pc=pc,
                    device_connected=device_active,
                    file_op=file_op if file_op != "none" else None,
                    is_removable=is_removable if is_removable else None,
                    email_sent=email_sent if email_sent else None,
                    url_category=url_category if url_category and url_category != "other" else None,
                )

                # Encode event type
                event_type_token = encode_event_type(event_type_raw)

                # Build feature vector
                features = self._build_feature_vector(
                    timestamp=timestamp,
                    event_type=event_type_token,
                    pc_id=pc_id,
                    session=session,
                    config=config,
                )

                # Check sequence length limit
                if len(sequence) < config.max_seq_len:
                    sequence.append(features)
                    session.events.append(features)

                # Reset per-event state after adding
                self._reset_session_file_state(session)
                self._reset_session_email_state(session)

            # Pad sequence if needed to max_seq_len
            seq_len = len(sequence)
            if seq_len < config.max_seq_len:
                padding = [[0.0] * self.FEATURE_DIM for _ in range(config.max_seq_len - seq_len)]
                sequence.extend(padding)

            # Create window ID
            date_str = date.strftime("%Y%m%d") if hasattr(date, "strftime") else str(date).replace("-", "")
            window_id = f"{user_id}_{date_str}"

            # Get label (default to 0/benign if user not in ground truth)
            label = labels.get(user_id, 0)

            results.append({
                "user_id": user_id,
                "date": date,
                "sequence": sequence,  # list[list[float]] for Parquet
                "seq_len": seq_len,
                "window_id": window_id,
                "session_count": sessions_in_day,
                "event_count": seq_len,
                "label": label,
            })

        # Create output DataFrame
        result_df = pl.DataFrame(results)

        return result_df

    def save(self, df: pl.DataFrame, filename: str = "daily_sequences.parquet") -> Path:
        """Save encoded sequences to Parquet file."""
        output_path = self.config.output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.write_parquet(output_path, use_pyarrow=True)
        print(f"Saved sequences to: {output_path}")

        return output_path

    def to_tensors(
        self,
        df: Optional[pl.DataFrame] = None,
        device: Optional[str] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Convert encoded sequences to PyTorch tensors.

        Args:
            df: DataFrame with encoded sequences. If None, loads from default path.
            device: Target device for tensors ('cpu', 'cuda', etc.)

        Returns:
            Dictionary of tensors:
                - sequences: (N, max_seq_len, 16) float32
                - seq_lens: (N,) int64
                - labels: (N,) int64
                - session_counts: (N,) int64
                - event_counts: (N,) int64
                - device: str
        """
        if df is None:
            output_path = self.config.output_dir / "daily_sequences.parquet"
            df = pl.read_parquet(output_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        sequences = []
        seq_lens = []
        labels = []
        session_counts = []
        event_counts = []

        for row in df.iter_rows():
            seq = row[2]  # sequence column (index 2)
            seq_len = row[3]  # seq_len column (index 3)

            # Convert to tensor
            seq_tensor = torch.tensor(seq, dtype=torch.float32)

            sequences.append(seq_tensor)
            seq_lens.append(seq_len)
            labels.append(row[7])  # label column (index 7)
            session_counts.append(row[5])  # session_count column (index 5)
            event_counts.append(row[6])  # event_count column (index 6)

        return {
            "sequences": torch.stack(sequences).to(device),
            "seq_lens": torch.tensor(seq_lens, dtype=torch.long).to(device),
            "labels": torch.tensor(labels, dtype=torch.long).to(device),
            "session_counts": torch.tensor(session_counts, dtype=torch.long).to(device),
            "event_counts": torch.tensor(event_counts, dtype=torch.long).to(device),
            "device": device,
        }

    def get_feature_names(self) -> list[str]:
        """Return the names of the 16 features."""
        return [
            "event_type_token",
            "pc_id_encoded",
            "hour_bucket",
            "day_of_week",
            "is_working_hours",
            "is_weekend",
            "is_after_hours",
            "session_progress",
            "device_connected",
            "file_operation_type",
            "email_sent",
            "is_removable_access",
            "url_domain_category",
            "activity_density",
            "temporal_entropy",
            "session_event_count",
        ]


# ---------------------------------------------------------------------------
# PyTorch Dataset Wrapper
# ---------------------------------------------------------------------------

class SequenceDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for insider threat sequences.

    Provides batches of sequences with corresponding labels and metadata.
    """

    def __init__(
        self,
        df: Optional[pl.DataFrame] = None,
        data_path: Optional[Path] = None,
    ):
        """
        Initialize dataset.

        Args:
            df: Polars DataFrame with encoded sequences
            data_path: Path to Parquet file (alternative to df)
        """
        if df is not None:
            self.df = df
        elif data_path is not None:
            self.df = pl.read_parquet(data_path)
        else:
            # Load default
            default_path = OUTPUT_DIR / "daily_sequences.parquet"
            if default_path.exists():
                self.df = pl.read_parquet(default_path)
            else:
                raise ValueError("No DataFrame or path provided and default not found")

        self.n_samples = len(self.df)
        self.feature_dim = SequenceEncoder.FEATURE_DIM
        self.max_seq_len = self.df["seq_len"].max() if "seq_len" in self.df.columns else MAX_SEQ_LEN

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample."""
        row = self.df.row(idx, named=True)

        sequence = torch.tensor(row["sequence"], dtype=torch.float32)
        seq_len = torch.tensor(row["seq_len"], dtype=torch.long)
        label = torch.tensor(row["label"], dtype=torch.long)

        return {
            "sequence": sequence,
            "seq_len": seq_len,
            "label": label,
            "user_id": row["user_id"],
            "window_id": row["window_id"],
            "session_count": torch.tensor(row["session_count"], dtype=torch.long),
            "event_count": torch.tensor(row["event_count"], dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Custom collate function for DataLoader."""
        return {
            "sequences": torch.stack([b["sequence"] for b in batch]),
            "seq_lens": torch.stack([b["seq_len"] for b in batch]),
            "labels": torch.stack([b["label"] for b in batch]),
            "session_counts": torch.stack([b["session_count"] for b in batch]),
            "event_counts": torch.stack([b["event_count"] for b in batch]),
            "user_ids": [b["user_id"] for b in batch],
            "window_ids": [b["window_id"] for b in batch],
        }


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> pl.DataFrame:
    """Main function to run the encoding pipeline."""
    print("=" * 60)
    print("CERT Insider Threat - Sequence Encoder")
    print("=" * 60)
    print(f"Output feature dimension: {SequenceEncoder.FEATURE_DIM}")
    print(f"Max sequence length: {MAX_SEQ_LEN}")

    # Initialize encoder
    config = EncoderConfig(
        max_seq_len=MAX_SEQ_LEN,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
    )
    encoder = SequenceEncoder(config)

    # Encode sequences
    print("\n[1/3] Loading and processing event data...")
    result_df = encoder.encode()

    print(f"\n[2/3] Encoded {len(result_df)} user-day sequences")
    print(f"       Total events encoded: {result_df['event_count'].sum()}")

    # Show statistics
    print("\nSequence Statistics:")
    print(f"  - Mean sequence length: {result_df['seq_len'].mean():.1f}")
    print(f"  - Max sequence length: {result_df['seq_len'].max()}")
    print(f"  - Min sequence length: {result_df['seq_len'].min()}")
    print(f"  - Unique users: {result_df['user_id'].n_unique()}")
    print(f"  - Total sessions: {result_df['session_count'].sum()}")
    print(f"  - Malicious samples: {result_df['label'].sum()}")
    print(f"  - Benign samples: {(result_df['label'] == 0).sum()}")

    # Show PC encoder stats
    print(f"\nPC Statistics:")
    print(f"  - Unique PCs encoded: {encoder._pc_counter}")

    # Save output
    print(f"\n[3/3] Saving to Parquet...")
    output_path = encoder.save(result_df)

    print("\n" + "=" * 60)
    print("Encoding complete!")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Demonstrate tensor conversion
    print("\n[Bonus] Converting to PyTorch tensors...")
    tensors = encoder.to_tensors(result_df)
    print(f"  - sequences shape: {tensors['sequences'].shape}")
    print(f"  - labels shape: {tensors['labels'].shape}")
    print(f"  - device: {tensors['device']}")

    return result_df


if __name__ == "__main__":
    result = main()
