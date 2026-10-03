import torch
import torch.nn as nn


class AdaptiveEdgeWeight(nn.Module):
    """
    Fast Adaptive Edge Weight Learning

    Vectorized implementation.
    """

    def __init__(self, feature_dim=256):

        super().__init__()

        self.edge_network = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, features, H):

        device = features.device

        weighted_H = H.float().clone()

        # ---------------------------------
        # Find all existing edges
        # ---------------------------------
        edge_index = torch.nonzero(H > 0, as_tuple=False)

        if edge_index.numel() == 0:
            return weighted_H

        src = edge_index[:, 0]
        dst = edge_index[:, 1]

        # ---------------------------------
        # Create all feature pairs at once
        # ---------------------------------
        pair_features = torch.cat(
            (
                features[src],
                features[dst]
            ),
            dim=1
        )

        # ---------------------------------
        # Predict all edge weights together
        # ---------------------------------
        weights = self.edge_network(pair_features).squeeze()

        # ---------------------------------
        # Update Hypergraph
        # ---------------------------------
        weighted_H[src, dst] = weights

        return weighted_H