"""Feature engineering package for CERT Insider Threat Detection."""

# Statistical features module (functions)
from src.features.statistical import (
    compute_logon_features,
    compute_session_features,
    compute_device_features,
    compute_file_features,
    compute_email_features,
    compute_http_features,
    compute_temporal_features,
    compute_drift_features,
    compute_rolling_features,
    merge_psychometric_features,
    merge_ldap_features,
    compute_all_window_features,
    create_user_date_ground_truth,
)

# Sequence encoder module
from src.features.sequence_encoder import SequenceEncoder, SequenceDataset

# Graph builder module
from src.features.graph_builder import GraphBuilder

__all__ = [
    # Statistical features
    "compute_logon_features",
    "compute_session_features",
    "compute_device_features",
    "compute_file_features",
    "compute_email_features",
    "compute_http_features",
    "compute_temporal_features",
    "compute_drift_features",
    "compute_rolling_features",
    "merge_psychometric_features",
    "merge_ldap_features",
    "compute_all_window_features",
    "create_user_date_ground_truth",
    # Sequence encoder
    "SequenceEncoder",
    "SequenceDataset",
    # Graph builder
    "GraphBuilder",
]
