import torch
import torch.nn as nn


class SEAttention(nn.Module):
    """
    Squeeze-and-Excitation Attention

    Learns channel-wise importance and
    reweights feature maps adaptively.
    """

    def __init__(self, channels, reduction=16):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(

            nn.Linear(
                channels,
                channels // reduction,
                bias=False
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                channels // reduction,
                channels,
                bias=False
            ),

            nn.Sigmoid()

        )

    def forward(self, x):

        batch_size, channels, _ = x.size()

        weights = self.pool(x).view(batch_size, channels)

        weights = self.fc(weights)

        weights = weights.view(batch_size, channels, 1)

        return x * weights