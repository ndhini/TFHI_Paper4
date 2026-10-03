import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hypergraph_utils import hypergraph_propagation
from models.hypergraph_attention import HypergraphAttention


class HGNNLayer(nn.Module):
    """
    Research Grade Hypergraph Neural Network Layer

    Pipeline

        Features
            │
            ▼
    Hypergraph Attention
            │
            ▼
    Hypergraph Propagation
            │
            ▼
    Linear Projection
            │
            ▼
    LayerNorm
            │
            ▼
    GELU
            │
            ▼
    Dropout
            │
            ▼
    Residual Connection
    """

    def __init__(
        self,
        in_features=256,
        out_features=256,
        num_heads=4,
        dropout=0.30
    ):

        super().__init__()

        # ---------------------------------
        # Hypergraph Attention
        # ---------------------------------
        self.attention = HypergraphAttention(
            feature_dim=in_features,
            num_heads=num_heads,
            dropout=dropout
        )

        # ---------------------------------
        # Linear Projection
        # ---------------------------------
        self.linear1 = nn.Linear(
            in_features,
            out_features
        )

        self.linear2 = nn.Linear(
            out_features,
            out_features
        )

        # ---------------------------------
        # Normalization
        # ---------------------------------
        self.norm1 = nn.LayerNorm(
            out_features
        )

        self.norm2 = nn.LayerNorm(
            out_features
        )

        # ---------------------------------
        # Activation
        # ---------------------------------
        self.activation = nn.GELU()

        # ---------------------------------
        # Dropout
        # ---------------------------------
        self.dropout = nn.Dropout(
            dropout
        )

        # ---------------------------------
        # Residual
        # ---------------------------------
        if in_features != out_features:

            self.residual = nn.Linear(
                in_features,
                out_features
            )

        else:

            self.residual = nn.Identity()

    def forward(self, X, H):

        residual = self.residual(X)

        # ---------------------------------
        # Hypergraph Attention
        # ---------------------------------
        X = self.attention(X)

        # ---------------------------------
        # Hypergraph Propagation
        # ---------------------------------
        X = hypergraph_propagation(
            X,
            H
        )

        # ---------------------------------
        # Projection 1
        # ---------------------------------
        X = self.linear1(X)

        X = self.norm1(X)

        X = self.activation(X)

        X = self.dropout(X)

        # ---------------------------------
        # Projection 2
        # ---------------------------------
        X = self.linear2(X)

        X = self.norm2(X)

        X = self.activation(X)

        X = self.dropout(X)

        # ---------------------------------
        # Residual Connection
        # ---------------------------------
        X = X + residual

        return X