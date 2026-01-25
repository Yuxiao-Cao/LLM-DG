"""
Behavioral Cloning MLP baseline.

Maps kinematic state [d_ego, v_ego, d_opp, v_opp] to continuous acceleration
output in [-3, 3] m/s^2.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCMLP(nn.Module):
    """
    Enhanced Multi-Layer Perceptron policy for behavioral cloning.

    Architecture:
        Input (4) -> Hidden(256) -> ReLU -> Dropout ->
        Hidden(256) -> ReLU -> Dropout ->
        Hidden(128) -> ReLU -> Dropout ->
        Hidden(128) -> ReLU -> Dropout ->
        Hidden(64) -> ReLU -> Output(1)
        Output is clamped to [-3, 3] via tanh scaling

    Args:
        input_dim: Dimension of input state (default 4 for [d_ego, v_ego, d_opp, v_opp])
        hidden_dims: List of hidden layer dimensions
        output_bound: Maximum absolute output value (default 3.0 for acceleration range)
        dropout: Dropout probability for regularization
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dims: list = None,
        output_bound: float = 3.0,
        dropout: float = 0.1,
    ):
        if hidden_dims is None:
            hidden_dims = [256, 256, 128, 128, 64]

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_bound = output_bound
        self.dropout = dropout

        # Build hidden layers with dropout
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            # Add dropout after each hidden layer except the last
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Kaiming initialization for ReLU."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Kaiming initialization for ReLU layers
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, 1) with values in [-output_bound, output_bound]
        """
        # Get raw output
        out = self.network(x)
        # Scale to [-3, 3] using tanh - using a larger scaling factor first for better gradients
        out = 3.0 * torch.tanh(out)
        # Then clamp to exact bounds
        out = torch.clamp(out, -self.output_bound, self.output_bound)
        return out

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prediction method for inference.

        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,)

        Returns:
            Acceleration values in [-3, 3]
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return self.forward(x).squeeze()


def create_mlp_model(
    input_dim: int = 4,
    hidden_dims: list = None,
    output_bound: float = 3.0,
    dropout: float = 0.1,
    **kwargs,
) -> BCMLP:
    """
    Factory function to create BC-MLP model with enhanced architecture.

    Args:
        input_dim: Input dimension
        hidden_dims: Hidden layer dimensions (default [256, 256, 128, 128, 64])
        output_bound: Output bound (default 3.0)
        dropout: Dropout probability (default 0.1)

    Returns:
        BCMLP model instance
    """
    if hidden_dims is None:
        hidden_dims = [256, 256, 128, 128, 64]

    return BCMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_bound=output_bound,
        dropout=dropout,
    )
