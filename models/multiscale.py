import torch
import torch.nn as nn


class MultiScaleBlock(nn.Module):
    """
    Enhanced Multi-Scale Feature Extraction

    Parallel convolutions with
    residual learning.
    """

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        # -----------------------------
        # Branch 1
        # -----------------------------
        self.branch3 = nn.Sequential(

            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm1d(out_channels),

            nn.GELU()

        )

        # -----------------------------
        # Branch 2
        # -----------------------------
        self.branch5 = nn.Sequential(

            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=5,
                padding=2,
                bias=False
            ),

            nn.BatchNorm1d(out_channels),

            nn.GELU()

        )

        # -----------------------------
        # Branch 3
        # -----------------------------
        self.branch7 = nn.Sequential(

            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=7,
                padding=3,
                bias=False
            ),

            nn.BatchNorm1d(out_channels),

            nn.GELU()

        )

        # -----------------------------
        # Feature Fusion
        # -----------------------------
        self.fusion = nn.Sequential(

            nn.Conv1d(
                out_channels * 3,
                out_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm1d(out_channels),

            nn.GELU(),

            nn.Dropout(0.10)

        )

        # -----------------------------
        # Residual Projection
        # -----------------------------
        self.shortcut = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):

        identity = self.shortcut(x)

        b1 = self.branch3(x)

        b2 = self.branch5(x)

        b3 = self.branch7(x)

        out = torch.cat(
            [b1, b2, b3],
            dim=1
        )

        out = self.fusion(out)

        out = out + identity

        return out