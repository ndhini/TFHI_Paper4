import torch
import torch.nn as nn


class TrustAggregator(nn.Module):
    """
    Trust Aggregator V2

    Learns trust scores for features,
    refines them, and preserves information
    using residual learning.
    """

    def __init__(self, feature_dim=256):

        super().__init__()

        self.trust_network = nn.Sequential(

            nn.Linear(feature_dim, 128),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(128, feature_dim),

            nn.Sigmoid()

        )

        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features):

        # ----------------------------
        # Learn Trust Score
        # ----------------------------
        trust = self.trust_network(features)

        # ----------------------------
        # Trust-weighted Features
        # ----------------------------
        refined = features * trust

        # ----------------------------
        # Residual Connection
        # ----------------------------
        refined = refined + features

        # ----------------------------
        # Layer Normalization
        # ----------------------------
        refined = self.norm(refined)

        return refined