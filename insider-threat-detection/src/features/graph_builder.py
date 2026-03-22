"""
Graph Construction Module for CERT Insider Threat Detection
===========================================================

Builds heterogeneous temporal graphs from normalized CERT dataset.
Outputs PyTorch Geometric HeteroData with monthly snapshots.

Node Types:
- user: ~1000 nodes, features: (85,) = behavioral_stats[64] + psychometric[5] + ldap[8] + graph_stats[8]
- pc: ~1100 nodes, features: (16,) = access_stats[12] + type[4] (shared/dedicated)
- domain: N nodes, features: (32,) = contact_count[1] + category[31] (one-hot)

Edge Types:
- ("user", "used_pc", "pc"): weight (logon count), recency, after_hours_ratio, weekend_ratio, duration_mean
- ("user", "sent_email", "domain"): email_count, total_size, attachment_size, has_external_contact, after_hours_ratio
- ("user", "browsed", "domain"): request_count, is_sensitive_category, after_hours_ratio
- ("user", "device_on", "pc"): connect_count, duration
- ("user", "shared_pc_with", "user"): co-access count of shared PCs

Monthly snapshots: graph_YYYY_MM.pt format

Author: Claude
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class GraphBuilder:
    """
    Builds heterogeneous temporal graphs from CERT insider threat dataset.

    Processes normalized parquet files and constructs PyTorch Geometric
    HeteroData objects with monthly snapshots.
    """

    # Domain categories from CERT dataset
    DOMAIN_CATEGORIES = [
        'advertising', 'blog', 'business', 'cdn', 'chat', 'climate',
        'cloudprovider', 'commercial', 'cryptocurrency', 'dating',
        'download', 'drugs', 'education', 'entertainment', 'file_transfer',
        'financial', 'forum', 'gambling', 'government', 'hacking',
        'health', 'humor', 'infrastructure', 'job_search', 'legal',
        'mail', 'military', 'museum', 'news', 'nudity', 'online_storage',
        'podcast', 'politics', 'porn', 'press', 'proxy', 'radio',
        'reigion', 'search_engine', 'sex', 'shopping', 'social_network',
        'software_update', 'sports', 'stock', 'streaming', 'tech',
        'tor', 'translation', 'trash', 'video_conferencing', 'vpn',
        'weapons', 'web_application', 'webmail', 'webpage', 'unknown'
    ]

    def __init__(
        self,
        data_dir: str = "C:/Darsh/NCPI/insider-threat-detection/data/normalized",
        output_dir: str = "C:/Darsh/NCPI/insider-threat-detection/data/graphs",
        device: str = "cpu"
    ):
        """
        Initialize the GraphBuilder.

        Args:
            data_dir: Directory containing normalized parquet files
            output_dir: Directory to save graph snapshots
            device: Device for PyTorch tensors ('cpu' or 'cuda')
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cache for loaded data
        self._logon_df: Optional[pd.DataFrame] = None
        self._device_df: Optional[pd.DataFrame] = None
        self._email_df: Optional[pd.DataFrame] = None
        self._http_df: Optional[pd.DataFrame] = None
        self._psychometric_df: Optional[pd.DataFrame] = None
        self._ldap_df: Optional[pd.DataFrame] = None

        # Mappings for node indexing
        self._user_to_idx: Dict[str, int] = {}
        self._pc_to_idx: Dict[str, int] = {}
        self._domain_to_idx: Dict[str, int] = {}

        # Feature dimensions
        self.user_feature_dim = 85   # behavioral[64] + psychometric[5] + ldap[8] + graph[8]
        self.pc_feature_dim = 16     # access_stats[12] + type[4]
        self.domain_feature_dim = 32 # contact_count[1] + category[31]

    # =========================================================================
    # Data Loading Methods
    # =========================================================================

    def load_all_data(self) -> None:
        """Load all normalized parquet files into memory."""
        print("Loading normalized data files...")

        # Load logon data
        logon_path = self.data_dir / "logon.parquet"
        if logon_path.exists():
            self._logon_df = pd.read_parquet(logon_path)
            self._logon_df['timestamp'] = pd.to_datetime(self._logon_df['timestamp'])
            print(f"  - Loaded logon.parquet: {len(self._logon_df):,} records")
        else:
            raise FileNotFoundError(f"Missing: {logon_path}")

        # Load device data
        device_path = self.data_dir / "device.parquet"
        if device_path.exists():
            self._device_df = pd.read_parquet(device_path)
            self._device_df['timestamp'] = pd.to_datetime(self._device_df['timestamp'])
            print(f"  - Loaded device.parquet: {len(self._device_df):,} records")
        else:
            # Device data might be optional
            self._device_df = pd.DataFrame()
            print("  - device.parquet not found, skipping device connections")

        # Load email data
        email_path = self.data_dir / "email.parquet"
        if email_path.exists():
            self._email_df = pd.read_parquet(email_path)
            self._email_df['timestamp'] = pd.to_datetime(self._email_df['timestamp'])
            print(f"  - Loaded email.parquet: {len(self._email_df):,} records")
        else:
            raise FileNotFoundError(f"Missing: {email_path}")

        # Load HTTP data
        http_path = self.data_dir / "http.parquet"
        if http_path.exists():
            self._http_df = pd.read_parquet(http_path)
            self._http_df['timestamp'] = pd.to_datetime(self._http_df['timestamp'])
            print(f"  - Loaded http.parquet: {len(self._http_df):,} records")
        else:
            raise FileNotFoundError(f"Missing: {http_path}")

        # Load psychometric data
        psychometric_path = self.data_dir / "psychometric.parquet"
        if psychometric_path.exists():
            self._psychometric_df = pd.read_parquet(psychometric_path)
            print(f"  - Loaded psychometric.parquet: {len(self._psychometric_df):,} records")
        else:
            raise FileNotFoundError(f"Missing: {psychometric_path}")

        # Load LDAP data
        ldap_path = self.data_dir / "ldap.parquet"
        if ldap_path.exists():
            self._ldap_df = pd.read_parquet(ldap_path)
            print(f"  - Loaded ldap.parquet: {len(self._ldap_df):,} records")
        else:
            raise FileNotFoundError(f"Missing: {ldap_path}")

        print("Data loading complete.\n")

    # =========================================================================
    # Node Index Mapping Methods
    # =========================================================================

    def build_node_mappings(self, snapshot_date: Optional[pd.Timestamp] = None) -> None:
        """
        Build mappings from node IDs to consecutive indices.

        Args:
            snapshot_date: If provided, only include nodes active up to this date.
        """
        print("Building node index mappings...")

        # Filter data by date if snapshot_date is provided
        logon_df = self._logon_df
        device_df = self._device_df
        email_df = self._email_df
        http_df = self._http_df

        if snapshot_date is not None:
            logon_df = logon_df[logon_df['timestamp'] <= snapshot_date]
            if len(device_df) > 0:
                device_df = device_df[device_df['timestamp'] <= snapshot_date]
            email_df = email_df[email_df['timestamp'] <= snapshot_date]
            http_df = http_df[http_df['timestamp'] <= snapshot_date]

        # Build user mapping from all data sources
        all_users = set()
        all_users.update(logon_df['user'].dropna().unique())
        if len(device_df) > 0:
            all_users.update(device_df['user'].dropna().unique())
        all_users.update(email_df['user'].dropna().unique())
        all_users.update(http_df['user'].dropna().unique())
        all_users.update(self._psychometric_df['user'].dropna().unique())
        all_users.update(self._ldap_df['user'].dropna().unique())

        self._user_to_idx = {user: idx for idx, user in enumerate(sorted(all_users))}
        print(f"  - Users: {len(self._user_to_idx):,}")

        # Build PC mapping from logon and device data
        all_pcs = set()
        all_pcs.update(logon_df['pc'].dropna().unique())
        if len(device_df) > 0:
            all_pcs.update(device_df['pc'].dropna().unique())

        self._pc_to_idx = {pc: idx for idx, pc in enumerate(sorted(all_pcs))}
        print(f"  - PCs: {len(self._pc_to_idx):,}")

        # Build domain mapping from email and HTTP data
        all_domains = set()
        all_domains.update(email_df['sender_domain'].dropna().unique())
        all_domains.update(http_df['domain'].dropna().unique())

        self._domain_to_idx = {domain: idx for idx, domain in enumerate(sorted(all_domains))}
        print(f"  - Domains: {len(self._domain_to_idx):,}")

    # =========================================================================
    # User Feature Computation Methods
    # =========================================================================

    def compute_user_behavioral_stats(
        self,
        df: pd.DataFrame,
        snapshot_date: pd.Timestamp,
        window_days: int = 30
    ) -> pd.DataFrame:
        """
        Compute behavioral statistics for users from activity data.

        Args:
            df: Activity dataframe with user, pc, timestamp columns
            snapshot_date: Reference date for the snapshot
            window_days: Number of days to look back

        Returns:
            DataFrame with user behavioral statistics (64 features)
        """
        start_date = snapshot_date - pd.Timedelta(days=window_days)
        window_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= snapshot_date)]

        if len(window_df) == 0:
            return pd.DataFrame()

        # Group by user
        grouped = window_df.groupby('user')

        stats_list = []
        for user, group in grouped:
            pc_counts = group['pc'].value_counts()
            n_unique_pcs = len(pc_counts)
            n_total_events = len(group)

            # Compute time-based features
            timestamps = group['timestamp']

            # Hourly distribution features (24 bins)
            hours = timestamps.dt.hour
            hour_hist, _ = np.histogram(hours, bins=24, range=(0, 24))
            hour_hist = hour_hist / (hour_hist.sum() + 1e-8)

            # Day of week distribution (7 bins)
            dow = timestamps.dt.dayofweek
            dow_hist, _ = np.histogram(dow, bins=7, range=(0, 7))
            dow_hist = dow_hist / (dow_hist.sum() + 1e-8)

            # After hours ratio (events between 6pm-8am)
            after_hours = ((hours >= 18) | (hours < 8)).mean()

            # Weekend ratio
            is_weekend = group.get('is_weekend', pd.Series([False] * len(group)))
            if 'is_weekend' in group.columns:
                weekend_ratio = group['is_weekend'].mean()
            else:
                weekend_ratio = (dow >= 5).mean()

            # Event frequency
            days_active = (timestamps.dt.date.nunique())
            events_per_day = n_total_events / (days_active + 1e-8)

            # PC diversity (entropy of PC usage)
            pc_probs = pc_counts / pc_counts.sum()
            pc_entropy = -(pc_probs * np.log(pc_probs + 1e-8)).sum()

            # Time between events statistics
            sorted_times = timestamps.sort_values()
            if len(sorted_times) > 1:
                time_diffs = sorted_times.diff().dt.total_seconds().dropna()
                inter_event_mean = time_diffs.mean() if len(time_diffs) > 0 else 0
                inter_event_std = time_diffs.std() if len(time_diffs) > 1 else 0
                inter_event_median = time_diffs.median() if len(time_diffs) > 0 else 0
            else:
                inter_event_mean = inter_event_std = inter_event_median = 0

            # Session statistics
            session_lengths = group.groupby(
                group['timestamp'].dt.date
            ).size()
            session_mean = session_lengths.mean() if len(session_lengths) > 0 else 0
            session_std = session_lengths.std() if len(session_lengths) > 1 else 0

            # Combine all features (64 total)
            features = np.concatenate([
                hour_hist,                           # 24 features
                dow_hist,                            # 7 features
                [after_hours, weekend_ratio,         # 2 features
                 n_unique_pcs, n_total_events,       # 2 features
                 events_per_day, pc_entropy,         # 2 features
                 inter_event_mean, inter_event_std, inter_event_median,  # 3 features
                 session_mean, session_std,          # 2 features
                 days_active,                        # 1 feature
                 pc_counts.max() / (n_total_events + 1e-8),  # PC concentration
                 (pc_counts > 1).sum(),              # Number of repeat PCs
                 (timestamps.dt.hour >= 0).sum() / (n_total_events + 1e-8),  # morning ratio
                 (timestamps.dt.hour >= 12).sum() / (n_total_events + 1e-8),  # afternoon ratio
                 (timestamps.dt.hour >= 18).sum() / (n_total_events + 1e-8),  # evening ratio
                 ],
            ])

            # Pad to 64 features if needed
            if len(features) < 64:
                features = np.pad(features, (0, 64 - len(features)))
            elif len(features) > 64:
                features = features[:64]

            stats_list.append({
                'user': user,
                'behavioral_features': features
            })

        return pd.DataFrame(stats_list)

    def compute_user_psychometric_features(self) -> pd.DataFrame:
        """
        Compute psychometric features for users.

        Returns:
            DataFrame with user psychometric features (5 features)
        """
        if self._psychometric_df is None or len(self._psychometric_df) == 0:
            # Return empty dataframe with correct columns
            return pd.DataFrame(columns=['user', 'psychometric_features'])

        features = []
        for _, row in self._psychometric_df.iterrows():
            psychometric = np.array([
                row['O'],   # Openness
                row['C'],   # Conscientiousness
                row['E'],   # Extraversion
                row['A'],   # Agreeableness
                row['N'],   # Neuroticism
            ], dtype=np.float32)

            features.append({
                'user': row['user'],
                'psychometric_features': psychometric
            })

        return pd.DataFrame(features)

    def compute_user_ldap_features(self) -> pd.DataFrame:
        """
        Compute LDAP-based features for users.

        Returns:
            DataFrame with user LDAP features (8 features)
        """
        if self._ldap_df is None or len(self._ldap_df) == 0:
            return pd.DataFrame(columns=['user', 'ldap_features'])

        features = []
        for _, row in self._ldap_df.iterrows():
            ldap = np.array([
                row['role_sensitivity'] if 'role_sensitivity' in row else 0.0,
                row['is_it_admin'] if 'is_it_admin' in row else 0.0,
                row['is_manager'] if 'is_manager' in row else 0.0,
                # Binary indicators for various roles
                1.0 if row.get('role_sensitivity', 0) >= 3 else 0.0,
                1.0 if row.get('role_sensitivity', 0) >= 4 else 0.0,
                1.0 if row.get('role_sensitivity', 0) >= 5 else 0.0,
                row.get('is_it_admin', 0.0) * row.get('role_sensitivity', 0) / 5.0,
                row.get('is_manager', 0.0) * row.get('role_sensitivity', 0) / 5.0,
            ], dtype=np.float32)

            features.append({
                'user': row['user'],
                'ldap_features': ldap
            })

        return pd.DataFrame(features)

    def compute_user_graph_stats(
        self,
        snapshot_date: pd.Timestamp,
        window_days: int = 30
    ) -> pd.DataFrame:
        """
        Compute graph-derived statistics for users.

        Args:
            snapshot_date: Reference date for the snapshot
            window_days: Number of days to look back

        Returns:
            DataFrame with user graph statistics (8 features)
        """
        start_date = snapshot_date - pd.Timedelta(days=window_days)

        # Get logon data in window
        logon_window = self._logon_df[
            (self._logon_df['timestamp'] >= start_date) &
            (self._logon_df['timestamp'] <= snapshot_date)
        ]

        # Get email data in window
        email_window = self._email_df[
            (self._email_df['timestamp'] >= start_date) &
            (self._email_df['timestamp'] <= snapshot_date)
        ]

        # Get HTTP data in window
        http_window = self._http_df[
            (self._http_df['timestamp'] >= start_date) &
            (self._http_df['timestamp'] <= snapshot_date)
        ]

        # Compute per-user statistics
        all_users = set(logon_window['user'].dropna().unique())
        all_users.update(email_window['user'].dropna().unique())
        all_users.update(http_window['user'].dropna().unique())

        stats_list = []
        for user in all_users:
            # Activity counts
            n_logons = len(logon_window[logon_window['user'] == user])
            n_emails = len(email_window[email_window['user'] == user])
            n_http = len(http_window[http_window['user'] == user])
            total_activity = n_logons + n_emails + n_http

            # Activity diversity (how many different activity types)
            activity_types = sum([
                n_logons > 0,
                n_emails > 0,
                n_http > 0
            ])

            # Unique domains contacted
            unique_domains = http_window[http_window['user'] == user]['domain'].nunique()

            # Unique PCs used
            unique_pcs = logon_window[logon_window['user'] == user]['pc'].nunique()

            # External email ratio
            if n_emails > 0:
                external_emails = email_window[
                    (email_window['user'] == user) &
                    (email_window['has_external_recipient'] == True)
                ]
                external_ratio = len(external_emails) / n_emails
            else:
                external_ratio = 0.0

            # Sensitive browsing ratio
            if n_http > 0:
                sensitive_browsing = http_window[
                    (http_window['user'] == user) &
                    (http_window.get('is_sensitive_category', pd.Series([False] * len(http_window))) == True)
                ]
                sensitive_ratio = len(sensitive_browsing) / n_http
            else:
                sensitive_ratio = 0.0

            # Recent activity (activity in last 7 days)
            recent_start = snapshot_date - pd.Timedelta(days=7)
            recent_logons = len(logon_window[
                (logon_window['user'] == user) &
                (logon_window['timestamp'] >= recent_start)
            ])
            recent_ratio = recent_logons / (n_logons + 1e-8)

            graph_stats = np.array([
                np.log1p(total_activity),     # log(1 + total_activity)
                activity_types / 3.0,          # normalized diversity
                np.log1p(unique_domains),      # log(1 + unique_domains)
                np.log1p(unique_pcs),          # log(1 + unique_pcs)
                external_ratio,                # external contact ratio
                sensitive_ratio,               # sensitive browsing ratio
                recent_ratio,                  # recent activity ratio
                n_logons / (total_activity + 1e-8),  # logon fraction
            ], dtype=np.float32)

            stats_list.append({
                'user': user,
                'graph_stats': graph_stats
            })

        return pd.DataFrame(stats_list)

    def build_user_features(
        self,
        snapshot_date: pd.Timestamp,
        window_days: int = 30
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build complete user feature matrix.

        Args:
            snapshot_date: Reference date for the snapshot
            window_days: Number of days to look back for behavioral features

        Returns:
            Tuple of (feature_matrix, user_list) where feature_matrix is (N, 85)
        """
        # Compute all feature components
        behavioral_stats = self.compute_user_behavioral_stats(
            self._logon_df, snapshot_date, window_days
        )
        psychometric_features = self.compute_user_psychometric_features()
        ldap_features = self.compute_user_ldap_features()
        graph_stats = self.compute_user_graph_stats(snapshot_date, window_days)

        # Merge all features by user
        user_features_df = pd.DataFrame({'user': list(self._user_to_idx.keys())})

        user_features_df = user_features_df.merge(
            behavioral_stats, on='user', how='left'
        )
        user_features_df = user_features_df.merge(
            psychometric_features, on='user', how='left'
        )
        user_features_df = user_features_df.merge(
            ldap_features, on='user', how='left'
        )
        user_features_df = user_features_df.merge(
            graph_stats, on='user', how='left'
        )

        # Handle missing values with zeros
        for col in ['behavioral_features', 'psychometric_features', 'ldap_features', 'graph_stats']:
            user_features_df[col] = user_features_df[col].apply(
                lambda x: x if isinstance(x, np.ndarray) and len(x) > 0
                          else np.zeros(64 if 'behavioral' in col else (5 if 'psychometric' in col else (8 if 'ldap' in col else 8)))
            )

        # Concatenate all features
        feature_list = []
        user_list = []
        for _, row in user_features_df.iterrows():
            combined = np.concatenate([
                row['behavioral_features'],  # 64
                row['psychometric_features'], # 5
                row['ldap_features'],          # 8
                row['graph_stats'],            # 8
            ])  # Total: 85

            feature_list.append(combined)
            user_list.append(row['user'])

        return np.array(feature_list, dtype=np.float32), user_list

    # =========================================================================
    # PC Feature Computation Methods
    # =========================================================================

    def compute_pc_features(
        self,
        snapshot_date: pd.Timestamp,
        window_days: int = 30
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build PC node features.

        Args:
            snapshot_date: Reference date for the snapshot
            window_days: Number of days to look back

        Returns:
            Tuple of (feature_matrix, pc_list) where feature_matrix is (N, 16)
        """
        start_date = snapshot_date - pd.Timedelta(days=window_days)

        # Get logon data in window
        logon_window = self._logon_df[
            (self._logon_df['timestamp'] >= start_date) &
            (self._logon_df['timestamp'] <= snapshot_date)
        ]

        feature_list = []
        pc_list = []

        for pc in self._pc_to_idx.keys():
            pc_logons = logon_window[logon_window['pc'] == pc]

            if len(pc_logons) == 0:
                # No activity - zero features
                access_stats = np.zeros(12, dtype=np.float32)
                pc_type = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # unknown
            else:
                n_logons = len(pc_logons)
                n_unique_users = pc_logons['user'].nunique()

                # Access statistics (12 features)
                after_hours = pc_logons.get('is_after_hours', pd.Series([False] * len(pc_logons)))
                if 'is_after_hours' in pc_logons.columns:
                    after_hours_ratio = pc_logons['is_after_hours'].mean()
                else:
                    hours = pc_logons['timestamp'].dt.hour
                    after_hours_ratio = ((hours >= 18) | (hours < 8)).mean()

                if 'is_weekend' in pc_logons.columns:
                    weekend_ratio = pc_logons['is_weekend'].mean()
                else:
                    dow = pc_logons['timestamp'].dt.dayofweek
                    weekend_ratio = (dow >= 5).mean()

                # Time between logons
                sorted_times = pc_logons['timestamp'].sort_values()
                if len(sorted_times) > 1:
                    time_diffs = sorted_times.diff().dt.total_seconds().dropna()
                    inter_logon_mean = time_diffs.mean() if len(time_diffs) > 0 else 0
                    inter_logon_std = time_diffs.std() if len(time_diffs) > 1 else 0
                else:
                    inter_logon_mean = inter_logon_std = 0

                # Active days
                active_days = pc_logons['timestamp'].dt.date.nunique()

                # Hourly distribution features (5 bins)
                hours = pc_logons['timestamp'].dt.hour
                morning = ((hours >= 6) & (hours < 12)).mean()
                afternoon = ((hours >= 12) & (hours < 18)).mean()
                evening = ((hours >= 18) & (hours < 24)).mean()
                night = ((hours >= 0) & (hours < 6)).mean()

                access_stats = np.array([
                    np.log1p(n_logons),          # log(1 + logon_count)
                    np.log1p(n_unique_users),    # log(1 + unique_users)
                    after_hours_ratio,           # after hours ratio
                    weekend_ratio,               # weekend ratio
                    inter_logon_mean / 3600,     # mean inter-logon time (hours)
                    inter_logon_std / 3600,       # std inter-logon time (hours)
                    active_days / window_days,   # fraction of active days
                    morning,                      # morning ratio
                    afternoon,                    # afternoon ratio
                    evening,                      # evening ratio
                    night,                        # night ratio
                    n_unique_users / (n_logons + 1e-8),  # user per logon ratio
                ], dtype=np.float32)

                # PC type classification (4 features)
                # Shared: more than 1 unique user
                # Dedicated: 1 unique user
                is_shared = 1.0 if n_unique_users > 1 else 0.0
                is_dedicated = 1.0 if n_unique_users == 1 else 0.0
                is_high_traffic = 1.0 if n_logons > np.percentile(logon_window.groupby('pc').size(), 75) else 0.0
                is_low_traffic = 1.0 if n_logons < np.percentile(logon_window.groupby('pc').size(), 25) else 0.0

                pc_type = np.array([is_shared, is_dedicated, is_high_traffic, is_low_traffic], dtype=np.float32)

            combined = np.concatenate([access_stats, pc_type])  # 12 + 4 = 16
            feature_list.append(combined)
            pc_list.append(pc)

        return np.array(feature_list, dtype=np.float32), pc_list

    # =========================================================================
    # Domain Feature Computation Methods
    # =========================================================================

    def compute_domain_features(
        self,
        snapshot_date: pd.Timestamp,
        window_days: int = 30
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build Domain node features.

        Args:
            snapshot_date: Reference date for the snapshot
            window_days: Number of days to look back

        Returns:
            Tuple of (feature_matrix, domain_list) where feature_matrix is (N, 32)
        """
        start_date = snapshot_date - pd.Timedelta(days=window_days)

        # Get email and HTTP data in window
        email_window = self._email_df[
            (self._email_df['timestamp'] >= start_date) &
            (self._email_df['timestamp'] <= snapshot_date)
        ]

        http_window = self._http_df[
            (self._http_df['timestamp'] >= start_date) &
            (self._http_df['timestamp'] <= snapshot_date)
        ]

        feature_list = []
        domain_list = []

        for domain in self._domain_to_idx.keys():
            # Count unique contacts (users who sent to or browsed this domain)
            email_users = set(email_window[email_window['sender_domain'] == domain]['user'].unique())
            http_users = set(http_window[http_window['domain'] == domain]['user'].unique())
            contact_count = len(email_users | http_users)

            # Determine category from HTTP data
            domain_http = http_window[http_window['domain'] == domain]
            if len(domain_http) > 0 and 'domain_category' in domain_http.columns:
                category = domain_http['domain_category'].mode()
                if len(category) > 0:
                    category = category[0]
                else:
                    category = 'unknown'
            else:
                category = 'unknown'

            # One-hot encode category (31 categories)
            category_idx = self.DOMAIN_CATEGORIES.index(category) if category in self.DOMAIN_CATEGORIES else -1
            category_onehot = np.zeros(31, dtype=np.float32)
            if category_idx >= 0 and category_idx < 31:
                category_onehot[category_idx] = 1.0

            # Combined features (1 + 31 = 32)
            contact_count_norm = np.log1p(contact_count) / 10.0  # normalized
            domain_features = np.concatenate([[contact_count_norm], category_onehot])

            feature_list.append(domain_features)
            domain_list.append(domain)

        return np.array(feature_list, dtype=np.float32), domain_list

    # =========================================================================
    # Edge Construction Methods
    # =========================================================================

    def build_used_pc_edges(
        self,
        snapshot_date: pd.Timestamp,
        window_days: int = 30
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build edges from user to PC based on logon activity.

        Args:
            snapshot_date: Reference date for the snapshot
            window_days: Number of days to look back

        Returns:
            Tuple of (edge_index, edge_attr, edge_weight)
        """
        start_date = snapshot_date - pd.Timedelta(days=window_days)

        logon_window = self._logon_df[
            (self._logon_df['timestamp'] >= start_date) &
            (self._logon_df['timestamp'] <= snapshot_date)
        ]

        # Aggregate by (user, pc) pairs
        agg_data = []

        for (user, pc), group in logon_window.groupby(['user', 'pc']):
            if user not in self._user_to_idx or pc not in self._pc_to_idx:
                continue

            n_logons = len(group)

            # After hours ratio
            if 'is_after_hours' in group.columns:
                after_hours_ratio = group['is_after_hours'].mean()
            else:
                hours = group['timestamp'].dt.hour
                after_hours_ratio = ((hours >= 18) | (hours < 8)).mean()

            # Weekend ratio
            if 'is_weekend' in group.columns:
                weekend_ratio = group['is_weekend'].mean()
            else:
                dow = group['timestamp'].dt.dayofweek
                weekend_ratio = (dow >= 5).mean()

            # Recency (days since last logon)
            last_logon = group['timestamp'].max()
            recency = (snapshot_date - last_logon).total_seconds() / 86400  # in days

            # Duration stats (if available)
            duration_mean = 0.0
            if 'duration' in group.columns:
                duration_mean = group['duration'].mean()

            agg_data.append({
                'user': user,
                'pc': pc,
                'logon_count': n_logons,
                'after_hours_ratio': after_hours_ratio,
                'weekend_ratio': weekend_ratio,
                'recency': recency,
                'duration_mean': duration_mean,
            })

        agg_df = pd.DataFrame(agg_data)

        if len(agg_df) == 0:
            return np.array([[], []], dtype=np.int64), np.zeros((0, 5), dtype=np.float32), np.zeros(0, dtype=np.float32)

        # Build edge index
        src = [self._user_to_idx[u] for u in agg_df['user']]
        dst = [self._pc_to_idx[p] for p in agg_df['pc']]
        edge_index = np.array([src, dst], dtype=np.int64)

        # Build edge attributes: [logon_count, recency, after_hours_ratio, weekend_ratio, duration_mean]
        edge_attr = np.column_stack([
            np.log1p(agg_df['logon_count'].values),  # log-transformed count
            np.clip(agg_df['recency'].values, 0, window_days) / window_days,  # normalized recency
            agg_df['after_hours_ratio'].values,
            agg_df['weekend_ratio'].values,
            np.log1p(agg_df['duration_mean'].values),
        ]).astype(np.float32)

        # Edge weights based on logon count
        edge_weight = agg_df['logon_count'].values.astype(np.float32)

        return edge_index, edge_attr, edge_weight

    def build_email_edges(
        self,
        snapshot_date: pd.Timestamp,
        window_days: int = 30
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build edges from user to domain based on email activity.

        Args:
            snapshot_date: Reference date for the snapshot
            window_days: Number of days to look back

        Returns:
            Tuple of (edge_index, edge_attr, edge_weight)
        """
        start_date = snapshot_date - pd.Timedelta(days=window_days)

        email_window = self._email_df[
            (self._email_df['timestamp'] >= start_date) &
            (self._email_df['timestamp'] <= snapshot_date)
        ]

        # Aggregate by (user, domain) pairs
        agg_data = []

        for (user, domain), group in email_window.groupby(['user', 'sender_domain']):
            if user not in self._user_to_idx or domain not in self._domain_to_idx:
                continue

            n_emails = len(group)
            total_size = group['size'].sum()
            attachment_size = group['attachments'].fillna(0).sum()
            has_external = group['has_external_recipient'].any()

            # After hours ratio
            hours = group['timestamp'].dt.hour
            after_hours_ratio = ((hours >= 18) | (hours < 8)).mean()

            agg_data.append({
                'user': user,
                'domain': domain,
                'email_count': n_emails,
                'total_size': total_size,
                'attachment_size': attachment_size,
                'has_external_contact': 1.0 if has_external else 0.0,
                'after_hours_ratio': after_hours_ratio,
            })

        agg_df = pd.DataFrame(agg_data)

        if len(agg_df) == 0:
            return np.array([[], []], dtype=np.int64), np.zeros((0, 5), dtype=np.float32), np.zeros(0, dtype=np.float32)

        # Build edge index
        src = [self._user_to_idx[u] for u in agg_df['user']]
        dst = [self._domain_to_idx[d] for d in agg_df['domain']]
        edge_index = np.array([src, dst], dtype=np.int64)

        # Build edge attributes
        edge_attr = np.column_stack([
            np.log1p(agg_df['email_count'].values),
            np.log1p(agg_df['total_size'].values) / 20.0,  # normalized
            np.log1p(agg_df['attachment_size'].values) / 15.0,  # normalized
            agg_df['has_external_contact'].values,
            agg_df['after_hours_ratio'].values,
        ]).astype(np.float32)

        # Edge weights based on email count
        edge_weight = agg_df['email_count'].values.astype(np.float32)

        return edge_index, edge_attr, edge_weight

    def build_browsed_edges(
        self,
        snapshot_date: pd.Timestamp,
        window_days: int = 30
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build edges from user to domain based on HTTP browsing activity.

        Args:
            snapshot_date: Reference date for the snapshot
            window_days: Number of days to look back

        Returns:
            Tuple of (edge_index, edge_attr, edge_weight)
        """
        start_date = snapshot_date - pd.Timedelta(days=window_days)

        http_window = self._http_df[
            (self._http_df['timestamp'] >= start_date) &
            (self._http_df['timestamp'] <= snapshot_date)
        ]

        # Aggregate by (user, domain) pairs
        agg_data = []

        for (user, domain), group in http_window.groupby(['user', 'domain']):
            if user not in self._user_to_idx or domain not in self._domain_to_idx:
                continue

            n_requests = len(group)

            # Sensitive category ratio
            if 'is_sensitive_category' in group.columns:
                is_sensitive = group['is_sensitive_category'].mean()
            else:
                is_sensitive = 0.0

            # After hours ratio
            hours = group['timestamp'].dt.hour
            after_hours_ratio = ((hours >= 18) | (hours < 8)).mean()

            agg_data.append({
                'user': user,
                'domain': domain,
                'request_count': n_requests,
                'is_sensitive_category': is_sensitive,
                'after_hours_ratio': after_hours_ratio,
            })

        agg_df = pd.DataFrame(agg_data)

        if len(agg_df) == 0:
            return np.array([[], []], dtype=np.int64), np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)

        # Build edge index
        src = [self._user_to_idx[u] for u in agg_df['user']]
        dst = [self._domain_to_idx[d] for d in agg_df['domain']]
        edge_index = np.array([src, dst], dtype=np.int64)

        # Build edge attributes
        edge_attr = np.column_stack([
            np.log1p(agg_df['request_count'].values),
            agg_df['is_sensitive_category'].values,
            agg_df['after_hours_ratio'].values,
        ]).astype(np.float32)

        # Edge weights based on request count
        edge_weight = agg_df['request_count'].values.astype(np.float32)

        return edge_index, edge_attr, edge_weight

    def build_device_on_edges(
        self,
        snapshot_date: pd.Timestamp,
        window_days: int = 30
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build edges from user to PC based on device connection activity.

        Args:
            snapshot_date: Reference date for the snapshot
            window_days: Number of days to look back

        Returns:
            Tuple of (edge_index, edge_attr, edge_weight)
        """
        if self._device_df is None or len(self._device_df) == 0:
            # Return empty edges if device data not available
            return np.array([[], []], dtype=np.int64), np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)

        start_date = snapshot_date - pd.Timedelta(days=window_days)

        device_window = self._device_df[
            (self._device_df['timestamp'] >= start_date) &
            (self._device_df['timestamp'] <= snapshot_date)
        ]

        # Aggregate by (user, pc) pairs
        agg_data = []

        for (user, pc), group in device_window.groupby(['user', 'pc']):
            if user not in self._user_to_idx or pc not in self._pc_to_idx:
                continue

            n_connects = len(group)
            duration = group.get('duration', pd.Series([0] * len(group))).sum()

            agg_data.append({
                'user': user,
                'pc': pc,
                'connect_count': n_connects,
                'duration': duration,
            })

        agg_df = pd.DataFrame(agg_data)

        if len(agg_df) == 0:
            return np.array([[], []], dtype=np.int64), np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)

        # Build edge index
        src = [self._user_to_idx[u] for u in agg_df['user']]
        dst = [self._pc_to_idx[p] for p in agg_df['pc']]
        edge_index = np.array([src, dst], dtype=np.int64)

        # Build edge attributes
        edge_attr = np.column_stack([
            np.log1p(agg_df['connect_count'].values),
            np.log1p(agg_df['duration'].values) / 10.0,
        ]).astype(np.float32)

        # Edge weights based on connect count
        edge_weight = agg_df['connect_count'].values.astype(np.float32)

        return edge_index, edge_attr, edge_weight

    def build_shared_pc_edges(
        self,
        snapshot_date: pd.Timestamp,
        window_days: int = 30
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build edges between users who accessed shared PCs.

        Args:
            snapshot_date: Reference date for the snapshot
            window_days: Number of days to look back

        Returns:
            Tuple of (edge_index, edge_attr, edge_weight)
        """
        start_date = snapshot_date - pd.Timedelta(days=window_days)

        logon_window = self._logon_df[
            (self._logon_df['timestamp'] >= start_date) &
            (self._logon_df['timestamp'] <= snapshot_date)
        ]

        # Find shared PCs (PCs with more than one unique user)
        pc_user_counts = logon_window.groupby('pc')['user'].nunique()
        shared_pcs = pc_user_counts[pc_user_counts > 1].index.tolist()

        # Build co-access matrix
        co_access_counts = {}

        for pc in shared_pcs:
            pc_users = logon_window[logon_window['pc'] == pc]['user'].unique()
            pc_users = [u for u in pc_users if u in self._user_to_idx]

            # Count co-access between all pairs
            for i, u1 in enumerate(pc_users):
                for u2 in pc_users[i+1:]:
                    pair = tuple(sorted([u1, u2]))
                    co_access_counts[pair] = co_access_counts.get(pair, 0) + 1

        # Build edges
        edges = []
        for (u1, u2), count in co_access_counts.items():
            if u1 in self._user_to_idx and u2 in self._user_to_idx:
                edges.append({
                    'user1': u1,
                    'user2': u2,
                    'co_access_count': count,
                })

        if len(edges) == 0:
            return np.array([[], []], dtype=np.int64), np.zeros((0, 1), dtype=np.float32), np.zeros(0, dtype=np.float32)

        edge_df = pd.DataFrame(edges)

        # Build edge index (bidirectional: user1 -> user2 and user2 -> user1)
        src1 = [self._user_to_idx[u] for u in edge_df['user1']]
        dst1 = [self._user_to_idx[u] for u in edge_df['user2']]
        src2 = [self._user_to_idx[u] for u in edge_df['user2']]
        dst2 = [self._user_to_idx[u] for u in edge_df['user1']]

        edge_index = np.array([src1 + src2, dst1 + dst2], dtype=np.int64)

        # Edge attributes (co-access count)
        counts = np.concatenate([edge_df['co_access_count'].values, edge_df['co_access_count'].values])
        edge_attr = np.log1p(counts).reshape(-1, 1).astype(np.float32)

        # Edge weights
        edge_weight = np.concatenate([edge_df['co_access_count'].values, edge_df['co_access_count'].values]).astype(np.float32)

        return edge_index, edge_attr, edge_weight

    # =========================================================================
    # Full Graph Construction
    # =========================================================================

    def build_snapshot(
        self,
        snapshot_date: pd.Timestamp,
        window_days: int = 30
    ) -> HeteroData:
        """
        Build a complete graph snapshot for a given date.

        Args:
            snapshot_date: Reference date for the snapshot
            window_days: Number of days to look back for features

        Returns:
            HeteroData object with all nodes and edges
        """
        print(f"\nBuilding snapshot for {snapshot_date.strftime('%Y-%m')}...")

        # Rebuild node mappings for this snapshot
        self.build_node_mappings(snapshot_date)

        # Create HeteroData object
        data = HeteroData()

        # Set metadata
        data.snapshot_date = snapshot_date.strftime('%Y-%m-%d')
        data.window_days = window_days

        # Build user features
        print("  - Building user features...")
        user_features, user_list = self.build_user_features(snapshot_date, window_days)
        data['user'].x = torch.tensor(user_features, dtype=torch.float32, device=self.device)

        # Build PC features
        print("  - Building PC features...")
        pc_features, pc_list = self.compute_pc_features(snapshot_date, window_days)
        data['pc'].x = torch.tensor(pc_features, dtype=torch.float32, device=self.device)

        # Build domain features
        print("  - Building domain features...")
        domain_features, domain_list = self.compute_domain_features(snapshot_date, window_days)
        data['domain'].x = torch.tensor(domain_features, dtype=torch.float32, device=self.device)

        # Build edges
        print("  - Building 'used_pc' edges...")
        used_pc_index, used_pc_attr, used_pc_weight = self.build_used_pc_edges(snapshot_date, window_days)
        data['user', 'used_pc', 'pc'].edge_index = torch.tensor(used_pc_index, dtype=torch.long, device=self.device)
        data['user', 'used_pc', 'pc'].edge_attr = torch.tensor(used_pc_attr, dtype=torch.float32, device=self.device)
        data['user', 'used_pc', 'pc'].edge_weight = torch.tensor(used_pc_weight, dtype=torch.float32, device=self.device)

        print("  - Building 'sent_email' edges...")
        email_index, email_attr, email_weight = self.build_email_edges(snapshot_date, window_days)
        data['user', 'sent_email', 'domain'].edge_index = torch.tensor(email_index, dtype=torch.long, device=self.device)
        data['user', 'sent_email', 'domain'].edge_attr = torch.tensor(email_attr, dtype=torch.float32, device=self.device)
        data['user', 'sent_email', 'domain'].edge_weight = torch.tensor(email_weight, dtype=torch.float32, device=self.device)

        print("  - Building 'browsed' edges...")
        browsed_index, browsed_attr, browsed_weight = self.build_browsed_edges(snapshot_date, window_days)
        data['user', 'browsed', 'domain'].edge_index = torch.tensor(browsed_index, dtype=torch.long, device=self.device)
        data['user', 'browsed', 'domain'].edge_attr = torch.tensor(browsed_attr, dtype=torch.float32, device=self.device)
        data['user', 'browsed', 'domain'].edge_weight = torch.tensor(browsed_weight, dtype=torch.float32, device=self.device)

        print("  - Building 'device_on' edges...")
        device_index, device_attr, device_weight = self.build_device_on_edges(snapshot_date, window_days)
        data['user', 'device_on', 'pc'].edge_index = torch.tensor(device_index, dtype=torch.long, device=self.device)
        data['user', 'device_on', 'pc'].edge_attr = torch.tensor(device_attr, dtype=torch.float32, device=self.device)
        data['user', 'device_on', 'pc'].edge_weight = torch.tensor(device_weight, dtype=torch.float32, device=self.device)

        print("  - Building 'shared_pc_with' edges...")
        shared_index, shared_attr, shared_weight = self.build_shared_pc_edges(snapshot_date, window_days)
        data['user', 'shared_pc_with', 'user'].edge_index = torch.tensor(shared_index, dtype=torch.long, device=self.device)
        data['user', 'shared_pc_with', 'user'].edge_attr = torch.tensor(shared_attr, dtype=torch.float32, device=self.device)
        data['user', 'shared_pc_with', 'user'].edge_weight = torch.tensor(shared_weight, dtype=torch.float32, device=self.device)

        # Print summary
        print(f"\n  Snapshot Summary:")
        print(f"    - Users: {data['user'].x.shape[0]:,}")
        print(f"    - PCs: {data['pc'].x.shape[0]:,}")
        print(f"    - Domains: {data['domain'].x.shape[0]:,}")
        print(f"    - used_pc edges: {data['user', 'used_pc', 'pc'].edge_index.shape[1]:,}")
        print(f"    - sent_email edges: {data['user', 'sent_email', 'domain'].edge_index.shape[1]:,}")
        print(f"    - browsed edges: {data['user', 'browsed', 'domain'].edge_index.shape[1]:,}")
        print(f"    - device_on edges: {data['user', 'device_on', 'pc'].edge_index.shape[1]:,}")
        print(f"    - shared_pc_with edges: {data['user', 'shared_pc_with', 'user'].edge_index.shape[1]:,}")

        return data

    def generate_monthly_snapshots(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        window_days: int = 30,
        overwrite: bool = False
    ) -> List[str]:
        """
        Generate monthly snapshots for the entire dataset period.

        Args:
            start_date: Start date string (YYYY-MM-DD), defaults to first month in data
            end_date: End date string (YYYY-MM-DD), defaults to last month in data
            window_days: Number of days to look back for features
            overwrite: Whether to overwrite existing snapshots

        Returns:
            List of generated snapshot file paths
        """
        # Load data if not already loaded
        if self._logon_df is None:
            self.load_all_data()

        # Determine date range
        if start_date is None:
            start_date = self._logon_df['timestamp'].min()
        else:
            start_date = pd.to_datetime(start_date)

        if end_date is None:
            end_date = self._logon_df['timestamp'].max()
        else:
            end_date = pd.to_datetime(end_date)

        # Generate list of month-end dates
        months = pd.date_range(
            start=start_date + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1),
            end=end_date,
            freq='M'
        )

        generated_files = []

        print(f"\nGenerating {len(months)} monthly snapshots...")
        print(f"Date range: {months[0].strftime('%Y-%m')} to {months[-1].strftime('%Y-%m')}")

        for snapshot_date in tqdm(months, desc="Building snapshots"):
            year = snapshot_date.year
            month = snapshot_date.month
            filename = f"graph_{year}_{month:02d}.pt"
            filepath = self.output_dir / filename

            # Skip if exists and not overwriting
            if filepath.exists() and not overwrite:
                print(f"\nSkipping existing: {filename}")
                generated_files.append(str(filepath))
                continue

            # Build snapshot
            data = self.build_snapshot(snapshot_date, window_days)

            # Save snapshot
            torch.save(data, filepath)
            generated_files.append(str(filepath))
            print(f"  Saved: {filename}")

        print(f"\nGenerated {len(generated_files)} snapshots.")
        return generated_files

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def load_snapshot(self, filepath: str) -> HeteroData:
        """
        Load a saved graph snapshot.

        Args:
            filepath: Path to the .pt file

        Returns:
            HeteroData object
        """
        return torch.load(filepath, map_location=self.device)

    def get_snapshot_info(self, filepath: str) -> Dict[str, Any]:
        """
        Get information about a saved snapshot.

        Args:
            filepath: Path to the .pt file

        Returns:
            Dictionary with snapshot information
        """
        data = self.load_snapshot(filepath)

        info = {
            'file': os.path.basename(filepath),
            'date': data.snapshot_date,
            'window_days': data.window_days,
            'num_users': data['user'].x.shape[0],
            'num_pcs': data['pc'].x.shape[0],
            'num_domains': data['domain'].x.shape[0],
            'user_feature_dim': data['user'].x.shape[1],
            'pc_feature_dim': data['pc'].x.shape[1],
            'domain_feature_dim': data['domain'].x.shape[1],
            'num_used_pc_edges': data['user', 'used_pc', 'pc'].edge_index.shape[1],
            'num_email_edges': data['user', 'sent_email', 'domain'].edge_index.shape[1],
            'num_browsed_edges': data['user', 'browsed', 'domain'].edge_index.shape[1],
            'num_device_edges': data['user', 'device_on', 'pc'].edge_index.shape[1],
            'num_shared_edges': data['user', 'shared_pc_with', 'user'].edge_index.shape[1],
        }

        return info

    def list_snapshots(self) -> List[str]:
        """
        List all saved snapshots in the output directory.

        Returns:
            List of snapshot file paths
        """
        return sorted([str(f) for f in self.output_dir.glob("graph_*.pt")])


# =============================================================================
# Config Class & Alias
# =============================================================================

from dataclasses import dataclass

@dataclass
class GraphConfig:
    """Configuration for graph builder."""
    data_dir: str = "C:/Darsh/NCPI/insider-threat-detection/data/normalized"
    output_dir: str = "C:/Darsh/NCPI/insider-threat-detection/data/graphs"
    device: str = "cpu"


# Alias for backward compatibility
BehavioralGraphBuilder = GraphBuilder


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main function to generate all monthly snapshots."""
    print("=" * 60)
    print("CERT Insider Threat Detection - Graph Builder")
    print("=" * 60)

    # Initialize builder
    builder = GraphBuilder(
        data_dir="C:/Darsh/NCPI/insider-threat-detection/data/normalized",
        output_dir="C:/Darsh/NCPI/insider-threat-detection/data/graphs",
        device="cpu"
    )

    # Generate all monthly snapshots
    snapshots = builder.generate_monthly_snapshots(
        window_days=30,
        overwrite=False
    )

    print("\n" + "=" * 60)
    print("Snapshot Generation Complete!")
    print("=" * 60)

    # Print summary of all snapshots
    print("\nGenerated Snapshots:")
    for snapshot_path in snapshots:
        info = builder.get_snapshot_info(snapshot_path)
        print(f"\n  {info['file']}:")
        print(f"    Date: {info['date']}")
        print(f"    Nodes: {info['num_users']:,} users, {info['num_pcs']:,} PCs, {info['num_domains']:,} domains")
        print(f"    Edges: {info['num_used_pc_edges']:,} used_pc, "
              f"{info['num_email_edges']:,} email, {info['num_browsed_edges']:,} browsed, "
              f"{info['num_device_edges']:,} device, {info['num_shared_edges']:,} shared")

    return snapshots


if __name__ == "__main__":
    main()
