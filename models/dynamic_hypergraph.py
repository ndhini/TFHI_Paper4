import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicHypergraph(nn.Module):
    """
    Adaptive Hybrid Dynamic Hypergraph Construction (AHCM)

    Features:
    - Cosine Similarity
    - Gaussian Similarity
    - Adaptive Hybrid Connectivity Matrix (AHCM)
    - Top-K Neighbor Selection
    - Fully Vectorized
    """

    def __init__(
        self,
        max_neighbors=8,
        similarity_threshold=0.70,
        alpha=0.7,
        sigma=1.0
    ):

        super().__init__()

        self.max_neighbors = max_neighbors
        self.threshold = similarity_threshold
        self.alpha = alpha
        self.sigma = sigma

    def forward(self, features):

        ############################################################
        # Normalize Features
        ############################################################
        features = F.normalize(features, dim=1)

        ############################################################
        # Cosine Similarity
        ############################################################
        cosine_similarity = torch.matmul(
            features,
            features.t()
        )

        ############################################################
        # Euclidean Distance
        ############################################################
        distance = torch.cdist(
            features,
            features,
            p=2
        )

        ############################################################
        # Gaussian Similarity
        ############################################################
        gaussian_similarity = torch.exp(
            -(distance ** 2) /
            (2 * (self.sigma ** 2))
        )

        ############################################################
        # Adaptive Hybrid Connectivity Matrix (AHCM)
        ############################################################
        similarity = (
            self.alpha * cosine_similarity
            +
            (1 - self.alpha) * gaussian_similarity
        )

        ############################################################
        # Remove Self Connections
        ############################################################
        similarity.fill_diagonal_(0)

        ############################################################
        # Top-K Neighbor Selection
        ############################################################
        values, indices = torch.topk(
            similarity,
            k=self.max_neighbors,
            dim=1
        )

        ############################################################
        # Hypergraph Construction
        ############################################################
        H = torch.zeros_like(similarity)

        H.scatter_(
            1,
            indices,
            (values > self.threshold).float()
        )

        return H