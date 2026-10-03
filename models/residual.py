import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual CNN Block

    Input:
        (Batch, C, Length)

    Output:
        (Batch, C, Length)

    Architecture:

        Conv1D
            ↓
        BatchNorm
            ↓
        GELU
            ↓
        Conv1D
            ↓
        BatchNorm
            ↓
        Skip Connection
            ↓
        GELU
    """

    def __init__(
        self,
        channels,
        kernel_size=3,
        dropout=0.20
    ):

        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.bn1 = nn.BatchNorm1d(channels)

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.bn2 = nn.BatchNorm1d(channels)

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        identity = x

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.activation(out)

        out = self.dropout(out)

        out = self.conv2(out)

        out = self.bn2(out)

        out = out + identity

        out = self.activation(out)

        return out