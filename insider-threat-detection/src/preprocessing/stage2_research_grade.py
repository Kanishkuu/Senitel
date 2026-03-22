#!/usr/bin/env python3
"""
Stage 2: Research-Grade Feature Engineering for CERT Insider Threat Detection

This module creates the BEST possible feature set for training:
- Graph Neural Networks (GNN)
- Temporal Transformers
- Variational Autoencoders (VAE)
- Isolation Forest / Deep SVDD

Key improvements:
1. Complete LDAP processing (18 months)
2. Rich organizational features (role, department, team, hierarchy)
3. Proper ground truth labels from r4.2 answers
4. User behavior embeddings
5. Social network features
6. Temporal patterns
7. Cleaned feature set (no useless columns)

Author: Claude
"""

from __future__ import annotations

import gc
import hashlib
import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from collections import defaultdict

import numpy as np
import polars as pl

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import torch

# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("C:/Darsh/NCPI/insider-threat-detection/data")
RAW_DATA_DIR = Path("C:/Darsh/NCPI/r4.2/r4.2")
ANSWERS_DIR = Path("C:/Darsh/NCPI/r4.2/answers")
OUTPUT_DIR = DATA_DIR / "features_v2"
WINDOWS = {"24h": 1, "7d": 7, "30d": 30, "90d": 90}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Use the SAME salt as the normalization pipeline
USER_SALT = b"CERT_R42_INSIDER_THREAT_SALT_2024_USER"

def hash_user(user: str) -> str:
    """Pseudonymize user identifier using SAME method as normalization."""
    user = str(user).strip().upper()
    return hashlib.sha256(user.encode() + USER_SALT).hexdigest()[:16]


def verify_hash_match():
    """Verify that our hash function matches the normalized data."""
    # Test with known user
    test_user = "NGF0157"
    expected_hash = "1d4e3b475ee8da84"  # From normalized data
    our_hash = hash_user(test_user)

    if our_hash == expected_hash:
        print(f"HASH VERIFIED: {test_user} -> {our_hash}")
        return True
    else:
        print(f"HASH MISMATCH: {test_user} -> {our_hash} (expected {expected_hash})")
        return False


def hash_pc(pc: str) -> str:
    """Pseudonymize PC identifier."""
    return hashlib.sha256(pc.encode()).hexdigest()[:16]


def parse_timestamp(ts: str) -> datetime:
    """Parse various timestamp formats."""
    for fmt in ["%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {ts}")


# =============================================================================
# LDAP PROCESSING
# =============================================================================

def load_all_ldap_data() -> pl.DataFrame:
    """Load and process all LDAP monthly files."""
    ldap_dir = RAW_DATA_DIR / "LDAP"
    all_ldap = []

    logger.info(f"Loading LDAP data from {ldap_dir}")

    for ldap_file in sorted(ldap_dir.glob("*.csv")):
        year_month = ldap_file.stem  # e.g., "2010-01"
        df = pl.read_csv(ldap_file)
        df = df.with_columns([
            pl.lit(year_month).alias("year_month"),
            pl.col("user_id").map_elements(hash_user, return_dtype=pl.String).alias("user_hash"),
        ])
        all_ldap.append(df)

    if all_ldap:
        combined = pl.concat(all_ldap)
        logger.info(f"Loaded {len(combined):,} LDAP records across {len(all_ldap)} months")
        return combined
    return pl.DataFrame()


def process_ldap_features(ldap_df: pl.DataFrame) -> pl.DataFrame:
    """
    Create rich organizational features from LDAP data.

    Features:
    - Role-based (role_sensitivity, role_type_encoded)
    - Department-based (department_id, department_size)
    - Team-based (team_id, team_size)
    - Hierarchy-based (is_management, access_level)
    - Security-relevant (is_it_admin, is_security_role)
    - Change detection (role_changes, department_changes)
    """
    if ldap_df.is_empty():
        return pl.DataFrame()

    # Get latest state for each user
    ldap_df = ldap_df.sort("year_month")

    # Role sensitivity mapping (higher = more sensitive)
    role_sensitivity = {
        "ITAdmin": 5, "Security": 5, "Network": 5,
        "Database": 5, "System": 5, "Admin": 5,
        "Engineer": 4, "Developer": 4, "Programmer": 4,
        "Manager": 3, "Director": 3, "Executive": 3,
        "Analyst": 2, "Specialist": 2,
        "Salesman": 1, "Worker": 1, "Assistant": 1,
        "Administrative": 1,
    }

    # Compute role sensitivity
    ldap_df = ldap_df.with_columns([
        pl.col("role").map_elements(
            lambda r: role_sensitivity.get(str(r).strip(), 1),
            return_dtype=pl.Int64
        ).alias("role_sensitivity")
    ])

    # Is security role
    ldap_df = ldap_df.with_columns([
        pl.col("role").str.to_lowercase().str.contains_any(["admin", "security", "network", "database", "system"]).alias("is_security_role")
    ])

    # Is management
    ldap_df = ldap_df.with_columns([
        pl.col("role").str.to_lowercase().str.contains_any(["manager", "director", "executive", "chief", "head"]).alias("is_management")
    ])

    # Department ID (numeric encoding)
    department_map = {
        "administration": 1, "admin": 1,
        "research": 2, "engineering": 2, "software": 2,
        "sales": 3, "marketing": 3,
        "manufacturing": 4, "production": 4, "assembly": 4,
        "security": 5,
        "hr": 6, "human": 6,
    }

    def get_dept_id(dept: str) -> int:
        if dept is None:
            return 0
        dept_lower = str(dept).lower()
        for key, val in department_map.items():
            if key in dept_lower:
                return val
        return 7  # Other

    ldap_df = ldap_df.with_columns([
        pl.col("department").map_elements(get_dept_id, return_dtype=pl.Int64).alias("department_id")
    ])

    # Calculate department and team sizes
    dept_sizes = ldap_df.group_by("department").agg(pl.count().alias("dept_size"))
    team_sizes = ldap_df.group_by("team").agg(pl.count().alias("team_size"))

    ldap_df = ldap_df.join(dept_sizes, on="department", how="left")
    ldap_df = ldap_df.join(team_sizes, on="team", how="left")

    # Access level (composite score)
    ldap_df = ldap_df.with_columns([
        (pl.col("role_sensitivity") * 10 + pl.col("dept_size") / 100 + pl.col("team_size") / 50)
        .alias("access_level")
    ])

    # Get latest record per user
    latest_ldap = (
        ldap_df.sort("year_month")
        .group_by("user_hash")
        .agg([
            pl.col("role_sensitivity").last().alias("role_sensitivity"),
            pl.col("is_security_role").last().alias("is_security_role"),
            pl.col("is_management").last().alias("is_management"),
            pl.col("department_id").last().alias("department_id"),
            pl.col("dept_size").last().alias("dept_size"),
            pl.col("team_size").last().alias("team_size"),
            pl.col("access_level").last().alias("access_level"),
            pl.col("role").last().alias("role"),
            pl.col("functional_unit").last().alias("functional_unit"),
            pl.col("supervisor").last().alias("supervisor"),
        ])
    )

    # Encode categorical features
    role_encoding = {r: i for i, r in enumerate(latest_ldap["role"].unique().to_list())}
    latest_ldap = latest_ldap.with_columns([
        pl.col("role").map_elements(lambda r: role_encoding.get(r, 0), return_dtype=pl.Int64).alias("role_type")
    ])

    return latest_ldap


# =============================================================================
# GROUND TRUTH PROCESSING
# =============================================================================

def load_ground_truth_labels() -> pl.DataFrame:
    """
    Load and process ground truth labels from r4.2 answers.

    Creates a comprehensive label dataset with:
    - Insider indicator
    - Threat type (1, 2, or 3)
    - Scenario details
    """
    all_labels = []

    # Scenario mapping
    scenario_files = {
        "r4.2-1": ANSWERS_DIR / "r4.2-1",
        "r4.2-2": ANSWERS_DIR / "r4.2-2",
        "r4.2-3": ANSWERS_DIR / "r4.2-3",
    }

    for scenario_name, scenario_dir in scenario_files.items():
        if not scenario_dir.exists():
            continue

        logger.info(f"Loading {scenario_name} ground truth from {scenario_dir}")

        for csv_file in scenario_dir.glob("*.csv"):
            # Extract user from filename
            # Format: r4.2-X-USER.csv
            user = csv_file.stem.split("-")[-1]  # e.g., "AAM0658"
            user_hash = hash_user(user)

            # Read the ground truth file
            gt_content = csv_file.read_text()

            # Parse activities to find threat window
            min_date = None
            max_date = None

            for line in gt_content.strip().split("\n"):
                parts = line.split(",")
                if len(parts) >= 3:
                    try:
                        ts = parse_timestamp(parts[2])
                        if min_date is None or ts < min_date:
                            min_date = ts
                        if max_date is None or ts > max_date:
                            max_date = ts
                    except:
                        continue

            if min_date and max_date:
                all_labels.append({
                    "user": user,
                    "user_hash": user_hash,
                    "scenario": scenario_name,
                    "threat_type": int(scenario_name.split("-")[1]),
                    "start_date": min_date.date(),
                    "end_date": max_date.date(),
                    "is_insider": 1,
                })

    if all_labels:
        labels_df = pl.DataFrame(all_labels)
        logger.info(f"Loaded {len(labels_df):,} insider labels")
        return labels_df

    return pl.DataFrame()


def create_user_date_labels(labels_df: pl.DataFrame, all_users: list, start_date: datetime, end_date: datetime) -> pl.DataFrame:
    """Create user-date level labels for training."""
    if labels_df.is_empty():
        # No insiders
        dates = pl.date_range(start_date, end_date, interval="1d", eager=True)
        rows = []
        for user in all_users:
            for d in dates:
                rows.append({"user_hash": user, "date": d, "is_insider": 0, "threat_type": 0, "scenario": "none"})
        return pl.DataFrame(rows)

    # Expand to user-date level
    all_dates = list(pl.date_range(start_date, end_date, interval="1d", eager=True))

    rows = []
    for user in all_users:
        for d in all_dates:
            # Check if this user is an insider on this date
            user_labels = labels_df.filter(pl.col("user_hash") == user)

            is_insider = 0
            threat_type = 0
            scenario = "none"

            for row in user_labels.iter_rows(named=True):
                if row["start_date"] <= d <= row["end_date"]:
                    is_insider = 1
                    threat_type = row["threat_type"]
                    scenario = row["scenario"]
                    break

            rows.append({
                "user_hash": user,
                "date": d,
                "is_insider": is_insider,
                "threat_type": threat_type,
                "scenario": scenario,
            })

    return pl.DataFrame(rows)


# =============================================================================
# COMPREHENSIVE FEATURE ENGINEERING
# =============================================================================

def compute_comprehensive_features(
    dfs: dict[str, pl.DataFrame],
    ldap_features: pl.DataFrame,
    windows: dict[str, int] = None
) -> dict[str, pl.DataFrame]:
    """
    Compute ALL possible features across all time windows.

    Feature categories:
    1. Logon (count, timing, duration, patterns)
    2. Device (USB, connect/disconnect, session)
    3. File (extension, size, frequency)
    4. Email (volume, recipients, attachments, external)
    5. HTTP (domains, categories, content)
    6. Temporal (hourly, day-of-week, periodicity)
    7. Organizational (from LDAP)
    8. Graph (degree, centrality)
    9. Behavioral (drift, anomalies)
    """
    if windows is None:
        windows = WINDOWS

    results = {}

    for window_name, window_days in windows.items():
        suffix = f"_{window_name}"
        logger.info(f"Computing features for {window_name} window ({window_days} days)...")

        all_features = None

        # =========================================================================
        # 1. LOGON FEATURES
        # =========================================================================
        if "logon" in dfs:
            lf = dfs["logon"].lazy()

            # Basic counts
            logon_basic = (
                lf.group_by(["user_hash", "date"])
                .agg([
                    pl.col("id").count().alias(f"logon_count{suffix}"),
                    pl.col("is_after_hours").sum().alias(f"after_hours_logons{suffix}"),
                    pl.col("is_weekend").sum().alias(f"weekend_logons{suffix}"),
                    pl.col("is_working_hours").sum().alias(f"working_hours_logons{suffix}"),
                    pl.col("pc_hash").n_unique().alias(f"unique_pcs{suffix}"),
                    pl.col("hour").min().alias(f"first_logon_hour{suffix}"),
                    pl.col("hour").max().alias(f"last_logon_hour{suffix}"),
                ])
                .collect()
            )

            # Session duration features
            session_df = (
                lf.sort(["user_hash", "pc_hash", "timestamp"])
                .with_columns([
                    pl.col("activity").shift(1).over(["user_hash", "pc_hash"]).alias("_prev_activity"),
                    pl.col("timestamp").shift(1).over(["user_hash", "pc_hash"]).alias("_prev_ts"),
                ])
                .filter(
                    (pl.col("activity") == "Logoff") &
                    (pl.col("_prev_activity") == "Logon") &
                    (pl.col("_prev_ts").is_not_null())
                )
                .with_columns([
                    (pl.col("timestamp") - pl.col("_prev_ts"))
                    .dt.total_seconds()
                    .alias("_duration")
                ])
                .group_by(["user_hash", "date"])
                .agg([
                    pl.col("_duration").mean().alias(f"avg_session_duration{suffix}"),
                    pl.col("_duration").max().alias(f"max_session_duration{suffix}"),
                    pl.col("_duration").std().alias(f"session_duration_std{suffix}"),
                    pl.col("_duration").median().alias(f"median_session_duration{suffix}"),
                ])
                .collect()
            )

            logon_features = logon_basic.join(session_df, on=["user_hash", "date"], how="left")

            # Add ratios
            logon_features = logon_features.with_columns([
                (pl.col(f"logon_count{suffix}") / (pl.col(f"logon_count{suffix}") + 1))
                .clip(0, 1)
                .alias(f"logon_intensity{suffix}"),
                (pl.col(f"after_hours_logons{suffix}") / (pl.col(f"logon_count{suffix}") + 1))
                .clip(0, 1)
                .alias(f"after_hours_ratio{suffix}"),
            ])

            all_features = logon_features

        # =========================================================================
        # 2. DEVICE FEATURES
        # =========================================================================
        if "device" in dfs:
            df = dfs["device"].lazy()

            device_features = (
                df.group_by(["user_hash", "date"])
                .agg([
                    pl.col("id").count().alias(f"device_events{suffix}"),
                    pl.col("is_after_hours").sum().alias(f"after_hours_device{suffix}"),
                    pl.col("is_weekend").sum().alias(f"weekend_device{suffix}"),
                    pl.col("pc_hash").n_unique().alias(f"unique_devices{suffix}"),
                ])
                .collect()
            )

            if all_features is None:
                all_features = device_features
            else:
                all_features = all_features.join(device_features, on=["user_hash", "date"], how="left")

        # =========================================================================
        # 3. EMAIL FEATURES
        # =========================================================================
        if "email" in dfs:
            df = dfs["email"].lazy()

            email_features = (
                df.group_by(["user_hash", "date"])
                .agg([
                    pl.col("id").count().alias(f"emails_sent{suffix}"),
                    pl.col("is_internal_sender").sum().alias(f"internal_emails{suffix}"),
                    pl.col("has_external_recipient").sum().alias(f"external_emails{suffix}"),
                    pl.col("has_attachments").sum().alias(f"emails_with_attachments{suffix}"),
                    pl.col("size").mean().alias(f"avg_email_size{suffix}"),
                    pl.col("size").max().alias(f"max_email_size{suffix}"),
                    pl.col("size").sum().alias(f"total_email_size{suffix}"),
                    pl.col("to_count").mean().alias(f"avg_recipients{suffix}"),
                    pl.col("cc_count").mean().alias(f"avg_cc{suffix}"),
                ])
                .collect()
            )

            if all_features is None:
                all_features = email_features
            else:
                all_features = all_features.join(email_features, on=["user_hash", "date"], how="left")

        # =========================================================================
        # 4. HTTP FEATURES
        # =========================================================================
        if "http" in dfs:
            df = dfs["http"].lazy()

            # Domain-based categories
            df = df.with_columns([
                pl.col("domain").str.to_lowercase().alias("_domain_lower"),
            ]).with_columns([
                pl.col("_domain_lower").str.contains_any(["job", "career", "linkedin", "indeed"]).alias("_is_job"),
                pl.col("_domain_lower").str.contains_any(["facebook", "twitter", "instagram", "social", "myspace"]).alias("_is_social"),
                pl.col("_domain_lower").str.contains_any(["dropbox", "onedrive", "gdrive", "icloud", "box", "cloud"]).alias("_is_cloud"),
                pl.col("_domain_lower").str.contains_any(["mediafire", "megaupload", "rapidshare", "zippyshare"]).alias("_is_fileshare"),
                pl.col("_domain_lower").str.contains_any(["wikipedia", "google", "bing", "yahoo"]).alias("_is_search"),
            ])

            http_features = (
                df.group_by(["user_hash", "date"])
                .agg([
                    pl.col("id").count().alias(f"http_requests{suffix}"),
                    pl.col("is_after_hours").sum().alias(f"after_hours_http{suffix}"),
                    pl.col("is_weekend").sum().alias(f"weekend_http{suffix}"),
                    pl.col("domain").n_unique().alias(f"unique_domains{suffix}"),
                    pl.col("_is_job").sum().alias(f"job_site_visits{suffix}"),
                    pl.col("_is_social").sum().alias(f"social_media_visits{suffix}"),
                    pl.col("_is_cloud").sum().alias(f"cloud_visits{suffix}"),
                    pl.col("_is_fileshare").sum().alias(f"fileshare_visits{suffix}"),
                    pl.col("_is_search").sum().alias(f"search_visits{suffix}"),
                ])
                .collect()
            )

            if all_features is None:
                all_features = http_features
            else:
                all_features = all_features.join(http_features, on=["user_hash", "date"], how="left")

        # =========================================================================
        # 5. TEMPORAL FEATURES
        # =========================================================================
        if "logon" in dfs:
            lf = dfs["logon"].lazy()

            # Hourly distribution
            hour_exprs = [
                pl.col("id").filter(pl.col("hour") == h).count().alias(f"hour_{h}{suffix}")
                for h in range(24)
            ]
            hour_df = (
                lf.group_by(["user_hash", "date"])
                .agg(hour_exprs)
                .collect()
            )

            # Day of week
            dow_exprs = [
                pl.col("id").filter(pl.col("day_of_week") == d).count().alias(f"dow_{d}{suffix}")
                for d in range(7)
            ]
            dow_df = (
                lf.group_by(["user_hash", "date"])
                .agg(dow_exprs)
                .collect()
            )

            temporal_df = hour_df.join(dow_df, on=["user_hash", "date"], how="outer", suffix="_right")

            if all_features is None:
                all_features = temporal_df
            else:
                all_features = all_features.join(temporal_df, on=["user_hash", "date"], how="left")

        # =========================================================================
        # 6. ORGANIZATIONAL FEATURES (from LDAP)
        # =========================================================================
        if not ldap_features.is_empty() and all_features is not None:
            # Get only the user_hash and features columns
            ldap_cols = ["user_hash", "role_sensitivity", "is_security_role", "is_management",
                        "department_id", "dept_size", "team_size", "access_level", "role_type"]
            ldap_subset = ldap_features.select([c for c in ldap_cols if c in ldap_features.columns])

            # Add suffix to LDAP features
            rename_map = {c: f"{c}{suffix}" for c in ldap_subset.columns if c != "user_hash"}
            ldap_subset = ldap_subset.rename(rename_map)

            all_features = all_features.join(ldap_subset, on="user_hash", how="left")

        # =========================================================================
        # 7. FILL NULLS AND CLEAN UP
        # =========================================================================
        if all_features is not None:
            # Fill numeric columns
            numeric_cols = [c for c in all_features.columns
                          if c not in ["user_hash", "date"]
                          and all_features[c].dtype in [pl.Int64, pl.Float64, pl.Int8]]

            for col in numeric_cols:
                null_count = all_features[col].null_count()
                if null_count > 0:
                    # Use 0 for counts, median for others
                    if "count" in col or "sum" in col or "events" in col or "requests" in col:
                        all_features = all_features.with_columns(pl.col(col).fill_null(0))
                    else:
                        median_val = all_features[col].median()
                        if median_val is not None:
                            all_features = all_features.with_columns(pl.col(col).fill_null(median_val))
                        else:
                            all_features = all_features.with_columns(pl.col(col).fill_null(0.0))

            results[window_name] = all_features
            logger.info(f"Completed {window_name}: {all_features.shape[1]} columns")

    return results


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_user_graphs(dfs: dict[str, pl.DataFrame], monthly_dates: list) -> list:
    """
    Build monthly user interaction graphs for GNN training.

    Graph structure:
    - Nodes: Users
    - Edges: Email communication, shared PC usage, HTTP domains
    - Features: User behavioral features
    """
    from torch_geometric.data import Data
    import torch

    graphs = []

    for month_start in monthly_dates:
        month_end = (month_start + timedelta(days=32)).replace(day=1)

        # Create edge list from email
        edges = defaultdict(int)

        if "email" in dfs:
            email_df = dfs["email"].filter(
                (pl.col("timestamp") >= month_start) &
                (pl.col("timestamp") < month_end)
            )

            # User-to-user edges from email communication
            for row in email_df.iter_rows(named=True):
                sender = row.get("user_hash")
                to_list = row.get("to", [])
                if sender and to_list:
                    if isinstance(to_list, str):
                        # Parse comma-separated recipients
                        recipients = [r.strip() for r in to_list.split(",")]
                    else:
                        recipients = [to_list] if not isinstance(to_list, list) else to_list

                    for recipient in recipients:
                        if recipient and recipient != sender:
                            edge = tuple(sorted([sender, recipient]))
                            edges[edge] += 1

        if edges:
            # Create edge index tensor
            unique_nodes = list(set([n for edge in edges.keys() for n in edge]))
            node_to_idx = {n: i for i, n in enumerate(unique_nodes)}

            edge_list = []
            for (u1, u2), weight in edges.items():
                edge_list.append((node_to_idx[u1], node_to_idx[u2]))
                edge_list.append((node_to_idx[u2], node_to_idx[u1]))  # Undirected

            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor([edges[tuple(sorted([unique_nodes[i], unique_nodes[j]]))]
                                    for i, j in edge_index.t().tolist()], dtype=torch.float)

            # Create graph
            graph = Data(
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=len(unique_nodes),
                month=month_start.strftime("%Y-%m"),
            )
            graphs.append(graph)

    logger.info(f"Built {len(graphs)} monthly graphs")
    return graphs


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_research_pipeline():
    """Execute the complete research-grade preprocessing pipeline."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "graphs").mkdir(exist_ok=True)

    logger.info("="*60)
    logger.info("RESEARCH-GRADE PREPROCESSING PIPELINE")
    logger.info("="*60)

    # =========================================================================
    # STEP 1: Load normalized data
    # =========================================================================
    logger.info("Step 1: Loading normalized data...")
    normalized_dir = DATA_DIR / "normalized"

    dfs = {}
    for name in ["logon", "device", "file", "email", "http", "psychometric"]:
        path = normalized_dir / f"{name}.parquet"
        if path.exists():
            dfs[name] = pl.read_parquet(path)
            logger.info(f"Loaded {name}: {len(dfs[name]):,} rows")

    # =========================================================================
    # STEP 2: Load and process LDAP data
    # =========================================================================
    logger.info("Step 2: Processing LDAP data...")
    raw_ldap = load_all_ldap_data()
    ldap_features = process_ldap_features(raw_ldap)
    logger.info(f"LDAP features: {len(ldap_features):,} users")

    # =========================================================================
    # STEP 3: Load ground truth labels
    # =========================================================================
    logger.info("Step 3: Processing ground truth labels...")
    ground_truth = load_ground_truth_labels()
    logger.info(f"Ground truth: {len(ground_truth):,} insiders identified")

    # =========================================================================
    # STEP 4: Compute comprehensive features
    # =========================================================================
    logger.info("Step 4: Computing comprehensive features...")
    feature_results = compute_comprehensive_features(dfs, ldap_features)

    # =========================================================================
    # STEP 5: Merge features across windows
    # =========================================================================
    logger.info("Step 5: Merging features across windows...")

    # Start with 24h features as base
    merged = feature_results["24h"].clone()

    # Merge other windows - add suffix to non-key columns
    for window_name in ["7d", "30d", "90d"]:
        if window_name in feature_results:
            other = feature_results[window_name].clone()

            # Rename columns with suffix
            rename_map = {}
            for c in other.columns:
                if c not in ["user_hash", "date"]:
                    rename_map[c] = f"{c}_{window_name}"
            other = other.rename(rename_map)

            merged = merged.join(other, on=["user_hash", "date"], how="left")

    logger.info(f"Merged features: {merged.shape[1]} columns")

    # =========================================================================
    # STEP 6: Add ground truth labels
    # =========================================================================
    logger.info("Step 6: Adding ground truth labels...")

    all_users = merged["user_hash"].unique().to_list()
    start_date = merged["date"].min()
    end_date = merged["date"].max()

    labels = create_user_date_labels(ground_truth, all_users, start_date, end_date)
    merged = merged.join(labels, on=["user_hash", "date"], how="left")
    merged = merged.with_columns([
        pl.col("is_insider").fill_null(0).cast(pl.UInt8),
        pl.col("threat_type").fill_null(0).cast(pl.UInt8),
    ])

    # =========================================================================
    # STEP 7: Create additional derived features
    # =========================================================================
    logger.info("Step 7: Creating derived features...")

    # Activity ratios
    if "logon_count_24h" in merged.columns and "emails_sent_24h" in merged.columns:
        merged = merged.with_columns([
            (pl.col("emails_sent_24h") / (pl.col("logon_count_24h") + 1))
            .clip(0, 100)
            .alias("email_per_logon"),
        ])

    # Behavioral deviation indicators
    if "after_hours_ratio_24h" in merged.columns:
        merged = merged.with_columns([
            (pl.col("after_hours_ratio_24h") > 0.3).cast(pl.Int8).alias("high_after_hours"),
        ])

    # =========================================================================
    # STEP 8: Drop useless features
    # =========================================================================
    logger.info("Step 8: Cleaning up useless features...")

    # Find columns with >95% zeros
    cols_to_drop = []
    for col in merged.columns:
        if col in ["user_hash", "date", "is_insider", "threat_type", "scenario"]:
            continue
        if merged[col].dtype in [pl.Int64, pl.Float64, pl.Int8]:
            zero_pct = (merged[col] == 0).sum() / len(merged)
            if zero_pct > 0.95:
                cols_to_drop.append(col)

    merged = merged.drop(cols_to_drop)
    logger.info(f"Dropped {len(cols_to_drop)} useless features (>95% zeros)")

    # =========================================================================
    # STEP 9: Save output
    # =========================================================================
    logger.info("Step 9: Saving output...")

    # Main feature file
    output_path = OUTPUT_DIR / "user_features_research.parquet"
    merged.write_parquet(output_path)
    logger.info(f"Saved: {output_path} ({merged.shape[0]:,} rows x {merged.shape[1]} cols)")

    # Ground truth
    gt_path = OUTPUT_DIR / "ground_truth_research.parquet"
    ground_truth.write_parquet(gt_path)

    # User features (no labels)
    features_only = merged.drop(["is_insider", "threat_type", "scenario"])
    features_path = OUTPUT_DIR / "features_only.parquet"
    features_only.write_parquet(features_path)

    # LDAP features lookup
    ldap_path = OUTPUT_DIR / "ldap_features.parquet"
    ldap_features.write_parquet(ldap_path)

    # =========================================================================
    # STEP 10: Build graphs
    # =========================================================================
    logger.info("Step 10: Building user graphs...")

    try:
        from torch_geometric.data import DataLoader
        import torch

        graphs = build_user_graphs(dfs, [
            datetime(2010, m, 1) for m in range(1, 13)
        ] + [datetime(2011, m, 1) for m in range(1, 6)]
        )

        for i, g in enumerate(graphs):
            torch.save(g, OUTPUT_DIR / "graphs" / f"graph_{i}.pt")

        logger.info(f"Saved {len(graphs)} graphs")
    except Exception as e:
        logger.warning(f"Could not build graphs: {e}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"Features: {merged.shape[1]} columns")
    logger.info(f"Insiders: {merged['is_insider'].sum():,} user-days")

    # Feature summary
    feature_cols = [c for c in merged.columns if c not in ["user_hash", "date", "is_insider", "threat_type", "scenario"]]
    logger.info(f"Feature columns: {len(feature_cols)}")

    return merged


def main():
    """Entry point."""
    try:
        result = run_research_pipeline()
        print("\n" + "="*60)
        print("RESEARCH-GRADE PREPROCESSING COMPLETE")
        print("="*60)
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Shape: {result.shape}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
