"""
Masked Autoencoder (MAE) Framework for Continuous Time-Series and Flow Vectors.
Learns deep foundational representations by reconstructing masked temporal sequences.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from spodeian_ml.ssl.masking import SequenceMasker


class MaskedAutoencoder(nn.Module):
    """
    Encoder-Decoder Masked Autoencoder.
    The Encoder is frozen and exported to Candle for production inference,
    while the Decoder is strictly an offline training artifact.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 2,
        mask_ratio: float = 0.25,
        mask_value: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.masker = SequenceMasker(mask_ratio=mask_ratio, mask_value=mask_value)

        # 1. Base Encoder: Maps inputs to latent space
        encoder_layers = []
        curr = input_dim
        for _ in range(num_layers):
            encoder_layers.append(nn.Linear(curr, hidden_dim))
            encoder_layers.append(nn.LayerNorm(hidden_dim))
            encoder_layers.append(nn.GELU())
            curr = hidden_dim
        encoder_layers.append(nn.Linear(curr, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # 2. Reconstruction Decoder: Rebuilds original features from latent vectors
        decoder_layers = []
        curr = latent_dim
        for _ in range(num_layers):
            decoder_layers.append(nn.Linear(curr, hidden_dim))
            decoder_layers.append(nn.LayerNorm(hidden_dim))
            decoder_layers.append(nn.GELU())
            curr = hidden_dim
        decoder_layers.append(nn.Linear(curr, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts latent embeddings for downstream heads or inference."""
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstructs original feature space from latent vectors."""
        return self.decoder(latent)

    def forward(self, x: torch.Tensor, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (reconstructed, masked_input, latent).
        """
        masked_x, mask_bool = self.masker(x, seed=seed)
        latent = self.encode(masked_x)
        reconstructed = self.decode(latent)
        return reconstructed, masked_x, latent

    def compute_ssl_loss(self, x: torch.Tensor, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes reconstruction MSE loss between clean original target and reconstructed output.
        Returns (loss, reconstructed).
        """
        reconstructed, _, _ = self.forward(x, seed=seed)
        loss = nn.functional.mse_loss(reconstructed, x)
        return loss, reconstructed
