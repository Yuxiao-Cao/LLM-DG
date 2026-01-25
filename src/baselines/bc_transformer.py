"""
Behavioral Cloning Transformer baseline.

Maps kinematic state [d_ego, v_ego, d_opp, v_opp] to continuous acceleration
output in [-3, 3] m/s^2 using an enhanced Transformer encoder.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer.

    Adds sinusoidal position information to input embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 10):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, : x.size(1), :]


class BCTransformer(nn.Module):
    """
    Enhanced Transformer encoder policy for behavioral cloning.

    The input state [d_ego, v_ego, d_opp, v_opp] is treated as 2 tokens:
    - Token 0: [d_ego, v_ego] (ego vehicle state)
    - Token 1: [d_opp, v_opp] (opponent vehicle state)

    Architecture:
        Input features -> Linear projection -> LayerNorm ->
        Positional encoding ->
        Transformer encoder (x4 layers) -> LayerNorm ->
        Mean pooling -> MLP head -> Tanh scaling

    Args:
        feature_dim: Dimension per token (default 2 for [distance, velocity])
        d_model: Transformer embedding dimension
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout probability
        output_bound: Maximum absolute output value (default 3.0)
    """

    def __init__(
        self,
        feature_dim: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_bound: float = 3.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.output_bound = output_bound

        # Input projection: (feature_dim) -> (d_model)
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=10)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN architecture for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output norm
        self.output_norm = nn.LayerNorm(d_model)

        # Enhanced output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for transformer
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 4) where columns are
               [d_ego, v_ego, d_opp, v_opp]

        Returns:
            Output tensor of shape (batch_size, 1) with values in [-output_bound, output_bound]
        """
        batch_size = x.size(0)

        # Reshape input into 2 tokens: (batch_size, 2, 2)
        # Token 0: ego state [d_ego, v_ego]
        # Token 1: opp state [d_opp, v_opp]
        tokens = x.view(batch_size, 2, 2)

        # Project to d_model dimensions: (batch_size, 2, d_model)
        token_embeddings = self.input_projection(tokens) * math.sqrt(self.d_model)
        token_embeddings = self.input_norm(token_embeddings)

        # Add positional encoding
        token_embeddings = self.pos_encoding(token_embeddings)

        # Apply transformer encoder
        encoded = self.transformer_encoder(
            token_embeddings
        )  # (batch_size, 2, d_model)

        # Apply output norm
        encoded = self.output_norm(encoded)

        # Global average pooling over tokens
        pooled = encoded.mean(dim=1)  # (batch_size, d_model)

        # Output projection
        out = self.output_head(pooled)  # (batch_size, 1)

        # Scale to [-3, 3] using tanh
        out = 3.0 * torch.tanh(out)
        # Clamp to exact bounds
        out = torch.clamp(out, -self.output_bound, self.output_bound)

        return out

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prediction method for inference.

        Args:
            x: Input tensor of shape (batch_size, 4) or (4,)

        Returns:
            Acceleration values in [-3, 3]
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return self.forward(x).squeeze()


def create_transformer_model(
    feature_dim: int = 2,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 4,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    output_bound: float = 3.0,
    **kwargs,
) -> BCTransformer:
    """
    Factory function to create enhanced BC-Transformer model.

    Args:
        feature_dim: Features per token (default 2)
        d_model: Transformer embedding dimension (default 128)
        nhead: Number of attention heads (default 4)
        num_layers: Number of transformer layers (default 4)
        dim_feedforward: Feedforward dimension (default 256)
        dropout: Dropout probability (default 0.1)
        output_bound: Output bound (default 3.0)

    Returns:
        BCTransformer model instance
    """
    return BCTransformer(
        feature_dim=feature_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        output_bound=output_bound,
    )
