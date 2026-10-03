import torch
import torch.nn as nn
import torch.nn.functional as F


class HypergraphAttention(nn.Module):
    """
    Hypergraph Multi-Head Attention

    Learns importance weights for every node
    before hypergraph propagation.

    Input:
        features : (N, F)

    Output:
        attended_features : (N, F)
    """

    def __init__(
        self,
        feature_dim=256,
        num_heads=4,
        dropout=0.2
    ):

        super().__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads

        assert feature_dim % num_heads == 0

        self.head_dim = feature_dim // num_heads

        self.query = nn.Linear(feature_dim, feature_dim)

        self.key = nn.Linear(feature_dim, feature_dim)

        self.value = nn.Linear(feature_dim, feature_dim)

        self.output = nn.Linear(feature_dim, feature_dim)

        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features):

        residual = features

        N = features.size(0)

        Q = self.query(features)
        K = self.key(features)
        V = self.value(features)

        Q = Q.view(N, self.num_heads, self.head_dim).transpose(0, 1)
        K = K.view(N, self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(N, self.num_heads, self.head_dim).transpose(0, 1)

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        scores = scores / (self.head_dim ** 0.5)

        attention = F.softmax(
            scores,
            dim=-1
        )

        attention = self.dropout(attention)

        out = torch.matmul(
            attention,
            V
        )

        out = out.transpose(0, 1).contiguous()

        out = out.view(
            N,
            self.feature_dim
        )

        out = self.output(out)

        out = self.dropout(out)

        out = out + residual

        out = self.norm(out)

        return out