"""
Detection models for insider threat detection.

Models: VAE, Isolation Forest, LSTM, Ensemble
Optimized for: 16 GB RAM + NVIDIA RTX 4060 (8 GB VRAM)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class VAEAnomalyDetector(nn.Module):
    """
    Variational Autoencoder for anomaly detection.

    Trains on normal behavior, flags high reconstruction error as anomalous.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @staticmethod
    def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor,
                 logvar: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """Beta-VAE loss for anomaly detection."""
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_loss


class LSTMSequenceClassifier(nn.Module):
    """
    LSTM for sequence-based insider threat detection.

    Processes variable-length event sequences.
    """

    def __init__(
        self,
        input_dim: int = 5,  # event_type + 4 temporal features
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * direction_factor, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
            mask: (batch, seq_len) - 1 for real, 0 for padding
        """
        # Pack if mask provided
        if mask is not None:
            lengths = mask.sum(dim=1).long().clamp(min=1)
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        output, (h_n, _) = self.lstm(x)

        if mask is not None:
            output, _ = pad_packed_sequence(output, batch_first=True)

        # Use last hidden state
        if self.bidirectional:
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            h = torch.cat([h_forward, h_backward], dim=1)
        else:
            h = h_n[-1]

        return self.fc(h).squeeze(-1)


class TransformerClassifier(nn.Module):
    """
    Transformer for sequence-based insider threat detection.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape

        x = self.embedding(x)
        x = x + self.pos_embedding[:, :seq_len, :]

        # Attention mask for padding
        if mask is not None:
            key_padding_mask = (mask == 0)
        else:
            key_padding_mask = None

        x = self.transformer(x, key_padding_mask=key_padding_mask)

        # Use CLS token (first position) or mean pooling
        x = x[:, 0, :]  # CLS token

        return self.classifier(x).squeeze(-1)


class EnsembleAnomalyScorer:
    """
    Ensemble of multiple anomaly detectors.

    Combines scores from:
    - VAE (reconstruction error)
    - Isolation Forest (anomaly score)
    - LSTM/Transformer (sequence prediction)
    """

    def __init__(
        self,
        vae_model: VAEAnomalyDetector | None = None,
        iso_forest_model: Any | None = None,
        sequence_model: LSTMSequenceClassifier | TransformerClassifier | None = None,
        weights: dict[str, float] | None = None,
    ):
        self.vae = vae_model
        self.iso_forest = iso_forest_model
        self.sequence_model = sequence_model

        # Default weights (can be tuned)
        self.weights = weights or {
            "vae": 0.3,
            "iso_forest": 0.3,
            "sequence": 0.4,
        }

    def score_vae(self, x: torch.Tensor) -> np.ndarray:
        """Get VAE reconstruction error as anomaly score."""
        if self.vae is None:
            return np.zeros(len(x))

        self.vae.eval()
        with torch.no_grad():
            recon, mu, logvar = self.vae(x)
            errors = torch.mean((recon - x) ** 2, dim=1).numpy()

        return errors

    def score_iso_forest(self, x: np.ndarray) -> np.ndarray:
        """Get Isolation Forest anomaly scores."""
        if self.iso_forest is None:
            return np.zeros(len(x))

        # decision_function returns negative for anomalies
        scores = self.iso_forest.decision_function(x)
        # Flip so higher = more anomalous
        return -scores

    def score_sequence(self, seq: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
        """Get sequence model anomaly scores."""
        if self.sequence_model is None:
            return np.zeros(len(seq))

        self.sequence_model.eval()
        with torch.no_grad():
            scores = self.sequence_model(seq, mask).numpy()

        # Higher prediction = more likely anomalous
        return scores

    def score_ensemble(
        self,
        tabular: np.ndarray | torch.Tensor,
        sequences: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> np.ndarray:
        """
        Compute ensemble anomaly scores.

        Args:
            tabular: (num_users, feature_dim) feature matrix
            sequences: Tuple of (sequences, masks) if available

        Returns:
            (num_users,) anomaly scores (higher = more anomalous)
        """
        num_users = len(tabular)
        scores = np.zeros(num_users)

        # VAE score
        if self.vae is not None:
            if isinstance(tabular, np.ndarray):
                tabular_t = torch.tensor(tabular, dtype=torch.float32)
            else:
                tabular_t = tabular
            vae_scores = self.score_vae(tabular_t)
            scores += self.weights.get("vae", 0) * vae_scores

        # Isolation Forest score
        if self.iso_forest is not None:
            if isinstance(tabular, torch.Tensor):
                tabular_np = tabular.numpy()
            else:
                tabular_np = tabular
            iso_scores = self.score_iso_forest(tabular_np)
            scores += self.weights.get("iso_forest", 0) * iso_scores

        # Sequence model score
        if self.sequence_model is not None and sequences is not None:
            seq, mask = sequences
            seq_scores = self.score_sequence(seq, mask)
            scores += self.weights.get("sequence", 0) * seq_scores

        # Normalize to [0, 1]
        if scores.max() > 0:
            scores = (scores - scores.min()) / (scores.max() - scores.min())

        return scores


def train_vae(
    model: VAEAnomalyDetector,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    beta: float = 1.0,
) -> list[float]:
    """Train VAE on normal behavior data."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            x = batch["features"].to(device)
            optimizer.zero_grad()

            recon, mu, logvar = model(x)
            loss = VAEAnomalyDetector.vae_loss(recon, x, mu, logvar, beta)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  VAE Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return losses


def train_sequence_model(
    model: LSTMSequenceClassifier | TransformerClassifier,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
) -> list[float]:
    """Train sequence classifier."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    losses = []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            seq = batch["sequences"].to(device)
            mask = batch["masks"].to(device)
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(device)

            optimizer.zero_grad()
            preds = model(seq, mask)

            if labels is not None:
                loss = criterion(preds, labels.float())
            else:
                # Self-supervised: train on normal, flag deviations
                # For now, just compute prediction variance as pseudo-loss
                loss = 1 - preds.mean()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Sequence Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return losses


def save_model(model: nn.Module, path: Path) -> None:
    """Save PyTorch model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  Model saved: {path}")


def load_model(model: nn.Module, path: Path) -> nn.Module:
    """Load PyTorch model."""
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model
