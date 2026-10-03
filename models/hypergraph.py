import torch
import torch.nn as nn
import torch.nn.functional as F


class HypergraphBuilder(nn.Module):
    """
    Adaptive Hybrid Connectivity Matrix (AHCM)
    Hypergraph Builder

    Combines:
    - Cosine Similarity
    - Gaussian Similarity
    """

    def __init__(self, top_k=5, alpha=0.7, sigma=1.0):
        super().__init__()

        self.top_k = top_k
        self.alpha = alpha
        self.sigma = sigma

    def forward(self, features):
        """
        Args:
            features : Tensor [Batch, Feature]

        Returns:
            H : Weighted Hypergraph Incidence Matrix
        """

        ############################################################
        # Normalize Features
        ############################################################
        x = F.normalize(features, dim=1)

        ############################################################
        # Cosine Similarity
        ############################################################
        cosine_sim = torch.mm(x, x.t())

        ############################################################
        # Euclidean Distance
        ############################################################
        distance = torch.cdist(x, x, p=2)

        ############################################################
        # Gaussian Similarity
        ############################################################
        gaussian_sim = torch.exp(
            -(distance ** 2) / (2 * (self.sigma ** 2))
        )

        ############################################################
        # Adaptive Hybrid Connectivity Matrix (AHCM)
        ############################################################
        sim = (
            self.alpha * cosine_sim
            + (1 - self.alpha) * gaussian_sim
        )

        ############################################################
        # Top-K Neighbors
        ############################################################
        _, indices = torch.topk(
            sim,
            self.top_k,
            dim=1
        )

        N = features.size(0)

        ############################################################
        # Hypergraph Incidence Matrix
        ############################################################
        H = torch.zeros(
            (N, N),
            device=features.device,
            dtype=features.dtype
        )

        ############################################################
        # Row and Column Indices
        ############################################################
        rows = indices.reshape(-1)

        cols = (
            torch.arange(
                N,
                device=features.device
            )
            .repeat_interleave(self.top_k)
        )

        ############################################################
        # Adaptive Edge Weights
        ############################################################
        weights = sim.gather(
            1,
            indices
        ).reshape(-1)

        ############################################################
        # Construct Weighted Hypergraph
        ############################################################
        H[rows, cols] = weights

        return H