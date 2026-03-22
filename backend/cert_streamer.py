"""
Real-time CERT r4.2 Event Streamer for Sentinel

Streams events from CERT r4.2 CSV files to the backend.
Uses pandas for maximum compatibility.

Usage:
    python cert_streamer.py                    # Stream from all sources
    python cert_streamer.py --source logon    # Stream from single source
    python cert_streamer.py --sample 1000    # Stream sample of events
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

import pandas as pd
import requests

# Backend configuration
BACKEND_URL = "http://localhost:5000"
STREAM_INTERVAL = 0.5  # seconds between events


def parse_cert_timestamp(date_str: str) -> Optional[datetime]:
    """Parse CERT MM/DD/YYYY HH:MM:SS timestamp."""
    try:
        return datetime.strptime(str(date_str).strip(), "%m/%d/%Y %H:%M:%S")
    except (ValueError, TypeError):
        return None


def is_late_night(dt: datetime) -> bool:
    """Check if event is late night (suspicious hours like 10pm-5am)."""
    # Late night is more suspicious - true off-hours
    return dt.hour >= 22 or dt.hour < 5


def is_removable_media(filename: str) -> bool:
    """Check if filename indicates removable media."""
    return str(filename).lower().startswith("r:\\")


def has_external_email(recipients: str) -> bool:
    """Check if email has external recipients."""
    if not recipients:
        return False
    external_domains = ["@gmail", "@yahoo", "@hotmail", "@outlook", "@mail."]
    for ext in external_domains:
        if ext in str(recipients).lower():
            return True
    return False


class CertEventLoader:
    """Loads CERT r4.2 CSV files."""

    def __init__(self, data_dir: Path, sample_size: Optional[int] = None):
        self.data_dir = Path(data_dir)
        self.sample_size = sample_size

    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load and preprocess a single CSV file."""
        filepath = self.data_dir / filename

        if not filepath.exists():
            print(f"Warning: {filename} not found")
            return pd.DataFrame()

        print(f"Loading {filename}...")

        # Read CSV
        df = pd.read_csv(filepath, low_memory=False)

        # Parse timestamp
        df['timestamp'] = df['date'].apply(parse_cert_timestamp)
        df = df.dropna(subset=['timestamp'])

        # Add temporal features
        df['hour'] = df['timestamp'].apply(lambda x: x.hour)
        df['day_of_week'] = df['timestamp'].apply(lambda x: x.weekday())

        # Sample if requested
        if self.sample_size and len(df) > self.sample_size:
            df = df.sample(n=self.sample_size)

        print(f"  Loaded {len(df):,} rows from {filename}")
        return df

    def load_all(self) -> pd.DataFrame:
        """Load all CERT CSV files and combine chronologically."""
        dfs = []

        # Load logon
        logon_df = self.load_csv("logon.csv")
        if len(logon_df) > 0:
            logon_df = logon_df.copy()
            logon_df['source_type'] = 'logon'
            logon_df['action'] = logon_df['activity'].str.upper()
            dfs.append(logon_df)

        # Load device
        device_df = self.load_csv("device.csv")
        if len(device_df) > 0:
            device_df = device_df.copy()
            device_df['source_type'] = 'device'
            device_df['action'] = device_df['activity'].str.upper()
            dfs.append(device_df)

        # Load file (all files, not just removable)
        file_df = self.load_csv("file.csv")
        if len(file_df) > 0:
            file_df = file_df.copy()
            file_df['source_type'] = 'file'
            file_df['action'] = file_df['filename'].str.split('\\').str[-1]
            dfs.append(file_df)

        # Load email
        email_df = self.load_csv("email.csv")
        if len(email_df) > 0:
            email_df = email_df.copy()
            email_df['source_type'] = 'email'
            email_df['action'] = 'Email'
            email_df['is_external'] = email_df['to'].apply(has_external_email)
            dfs.append(email_df)

        # Load HTTP
        http_df = self.load_csv("http.csv")
        if len(http_df) > 0:
            http_df = http_df.copy()
            http_df['source_type'] = 'http'
            http_df['action'] = http_df['url'].str[:50]
            http_df['domain'] = http_df['url'].str.extract(r'https?://([^/]+)', expand=False)
            dfs.append(http_df)

        if not dfs:
            print("Error: No data files loaded!")
            return pd.DataFrame()

        # Combine all events
        combined = pd.concat(dfs, ignore_index=True)

        # Parse timestamps and sort chronologically
        combined['timestamp'] = combined['date'].apply(parse_cert_timestamp)
        combined = combined.dropna(subset=['timestamp'])
        combined = combined.sort_values('timestamp').reset_index(drop=True)

        print(f"\nTotal events: {len(combined):,}")
        print(f"Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
        print(f"By source: {combined['source_type'].value_counts().to_dict()}")

        return combined

    def get_events(self) -> Generator[dict, None, None]:
        """Generate events one at a time, sorted by timestamp."""
        df = self.load_all()

        for _, row in df.iterrows():
            yield row.to_dict()


class EventFormatter:
    """Formats raw CERT events into Sentinel API format."""

    def __init__(self):
        self.event_counter = 0

    def format_event(self, row: dict) -> dict:
        """Format a raw CERT event into Sentinel event format."""
        self.event_counter += 1

        timestamp = row.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = parse_cert_timestamp(timestamp)

        user = str(row.get('user', 'UNKNOWN')).upper()
        pc = str(row.get('pc', 'PC-0000')).upper()
        source_type = row.get('source_type', 'process')
        action = str(row.get('action', 'Activity'))
        filename = str(row.get('filename', ''))
        url = str(row.get('url', ''))

        # Get hour and day from timestamp for the model
        hour = timestamp.hour
        day_of_week = timestamp.weekday()

        return {
            "id": f"CERT-{timestamp.strftime('%Y%m%d%H%M%S')}-{self.event_counter:06d}",
            "timestamp": timestamp.isoformat(),
            "logType": source_type,
            "userId": user,
            "userName": self._generate_user_name(user),
            "department": self._get_department(user),
            "action": action[:50] if len(action) > 50 else action,
            "details": self._generate_details(source_type, row),
            "sourceIp": f"192.168.{random.randint(1,255)}.{random.randint(1,255)}",
            "resource": self._generate_resource(source_type, row),
            "sessionId": f"SES-{random.randint(100000, 999999)}",
            # Threat scoring will be done by the transformer model
            "threatLevel": "normal",  # Will be overwritten by model
            "riskScore": 0,  # Will be overwritten by model
            "anomalyFactors": [],
            "cert_data": {
                "pc": pc,
                "day_of_week": day_of_week,
                "hour": hour,
                "filename": filename,
                "url": url,
            }
        }

    def _generate_user_name(self, user_id: str) -> str:
        names = ["John", "Sarah", "Michael", "Lisa", "David", "Emma", "James", "Anna", "Robert", "Jennifer"]
        return f"{names[hash(user_id) % len(names)]} {user_id[:3]}"

    def _get_department(self, user_id: str) -> str:
        depts = ["Engineering", "Sales", "HR", "Finance", "IT Security", "Operations", "Marketing"]
        return depts[hash(user_id) % len(depts)]

    def _generate_resource(self, source_type: str, row: dict) -> str:
        if source_type == "file":
            return f"/files/{row.get('filename', 'unknown')}"
        elif source_type == "http":
            url = row.get('url', '')
            domain = url.split("/")[2] if "//" in url else url
            return f"/web/{domain[:30]}"
        elif source_type == "email":
            return "/email"
        elif source_type == "device":
            return f"/device/{row.get('pc', 'unknown')}"
        return "/system"

    def _generate_details(self, source_type: str, row: dict) -> str:
        if source_type == "logon":
            return f"User {row.get('user', '')} {row.get('action', '')} on {row.get('pc', '')}"
        elif source_type == "file":
            return f"File access: {row.get('filename', 'unknown')}"
        elif source_type == "email":
            return f"Email sent"
        elif source_type == "http":
            return f"HTTP: {str(row.get('url', ''))[:40]}"
        elif source_type == "device":
            return f"Device {row.get('action', '')}"
        return "Activity recorded"


def stream_events(
    data_dir: Path,
    backend_url: str,
    interval: float,
    sample_size: Optional[int] = None,
    max_events: Optional[int] = None,
):
    """Stream events to the backend."""
    print("=" * 60)
    print("CERT r4.2 Real-Time Event Streamer")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Backend URL: {backend_url}")
    print(f"Stream interval: {interval}s")
    if sample_size:
        print(f"Sample size: {sample_size:,} events per file")
    print("=" * 60)

    loader = CertEventLoader(data_dir, sample_size=sample_size)
    formatter = EventFormatter()

    event_count = 0
    threat_count = 0

    print("\nStarting stream... (Press Ctrl+C to stop)\n")

    try:
        for raw_event in loader.get_events():
            event = formatter.format_event(raw_event)

            if event["threatLevel"] != "normal":
                threat_count += 1

            # Send to backend
            try:
                response = requests.post(
                    f"{backend_url}/api/stream/event",
                    json=event,
                    headers={"Content-Type": "application/json"},
                    timeout=5
                )

                if response.status_code == 200:
                    status = "THREAT" if event["threatLevel"] != "normal" else "OK"
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"[{status:6}] {event['userId']} - {event['action'][:30]}"
                    )
                else:
                    print(f"Error: HTTP {response.status_code}")

            except requests.exceptions.ConnectionError:
                print(f"Backend not available at {backend_url}, waiting...")
                time.sleep(5)
            except Exception as e:
                print(f"Error: {e}")

            event_count += 1

            if max_events and event_count >= max_events:
                print(f"\nReached max events limit ({max_events})")
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nStream stopped by user")

    print("\n" + "=" * 60)
    print("STREAM SUMMARY")
    print("=" * 60)
    print(f"Total events: {event_count:,}")
    print(f"Threat events: {threat_count:,} ({100*threat_count/max(event_count,1):.1f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Stream CERT r4.2 events to Sentinel")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/r4.2/r4.2",
        help="Path to CERT r4.2 data directory"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=BACKEND_URL,
        help="Backend API URL"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=STREAM_INTERVAL,
        help="Seconds between events"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size per CSV file"
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum events to stream"
    )

    args = parser.parse_args()

    # Get project root
    if hasattr(__import__('sys'), 'executable'):
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
    else:
        PROJECT_ROOT = Path.cwd()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    stream_events(
        data_dir=data_dir,
        backend_url=args.backend,
        interval=args.interval,
        sample_size=args.sample,
        max_events=args.max_events,
    )


if __name__ == "__main__":
    main()
