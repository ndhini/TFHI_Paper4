import torch
import torch.nn as nn


class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive Feature Fusion (AFF)

    Learns how much information should be retained from
    the original encoder features and the HGNN-enhanced
    features using a learnable gating mechanism.
    """

    def __init__(self, feature_dim=256):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, original_feature, enhanced_feature):
        """
        Parameters
        ----------
        original_feature : Tensor
            Encoder output
            Shape: [Batch, Feature]

        enhanced_feature : Tensor
            HGNN output
            Shape: [Batch, Feature]

        Returns
        -------
        Tensor
            Adaptively fused feature representation
        """

        # Concatenate both feature representations
        fusion_input = torch.cat(
            (original_feature, enhanced_feature),
            dim=1
        )

        # Learn adaptive fusion weights
        alpha = self.gate(fusion_input)

        # Adaptive Feature Fusion
        output = (
            alpha * enhanced_feature
            + (1.0 - alpha) * original_feature
        )

        return output