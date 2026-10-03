import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """
    Converts a feature vector into multiple tokens.

    Input:
        (Batch, 256)

    Output:
        (Batch, 8, 32)
    """

    def __init__(
        self,
        feature_dim=256,
        num_tokens=8
    ):

        super().__init__()

        assert feature_dim % num_tokens == 0, \
            "feature_dim must be divisible by num_tokens"

        self.num_tokens = num_tokens
        self.token_dim = feature_dim // num_tokens

        self.projection = nn.Linear(
            feature_dim,
            feature_dim
        )

        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x):

        # Project features
        x = self.projection(x)

        x = self.norm(x)

        # (Batch,256)
        batch_size = x.size(0)

        # (Batch,8,32)
        tokens = x.view(
            batch_size,
            self.num_tokens,
            self.token_dim
        )

        return tokens