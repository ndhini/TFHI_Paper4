import torch
import torch.nn as nn

from models.multiscale import MultiScaleBlock
from models.residual import ResidualBlock
from models.se_attention import SEAttention


class SignalEncoder(nn.Module):
    """
    Advanced Signal Encoder

    Architecture

    IQ Signal
        ↓
    MultiScale CNN
        ↓
    Residual Block
        ↓
    SE Attention
        ↓
    Residual Block
        ↓
    Global Average Pooling
        ↓
    Fully Connected
        ↓
    256-D Feature Embedding
    """

    def __init__(self, embedding_dim=256):

        super().__init__()

        # -----------------------------------
        # Initial Feature Extraction
        # -----------------------------------
        self.stem = nn.Sequential(

            nn.Conv1d(
                in_channels=2,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm1d(64),

            nn.GELU()

        )

        # -----------------------------------
        # Multi-scale Learning
        # -----------------------------------
        self.multiscale = MultiScaleBlock(
            in_channels=64,
            out_channels=128
        )

        # -----------------------------------
        # Residual Block 1
        # -----------------------------------
        self.residual1 = ResidualBlock(
            channels=128
        )

        # -----------------------------------
        # Channel Attention
        # -----------------------------------
        self.se = SEAttention(
            channels=128
        )

        # -----------------------------------
        # Feature Expansion
        # -----------------------------------
        self.expand = nn.Sequential(

            nn.Conv1d(
                128,
                256,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm1d(256),

            nn.GELU()

        )

        # -----------------------------------
        # Residual Block 2
        # -----------------------------------
        self.residual2 = ResidualBlock(
            channels=256
        )

        # -----------------------------------
        # Global Pooling
        # -----------------------------------
        self.pool = nn.AdaptiveAvgPool1d(1)

        # -----------------------------------
        # Embedding
        # -----------------------------------
        self.embedding = nn.Sequential(

            nn.Linear(
                256,
                embedding_dim
            ),

            nn.BatchNorm1d(
                embedding_dim
            ),

            nn.GELU(),

            nn.Dropout(
                0.30
            )

        )

    def forward(self, x):

        # Initial convolution
        x = self.stem(x)

        # Multi-scale feature extraction
        x = self.multiscale(x)

        # Residual learning
        x = self.residual1(x)

        # Channel attention
        x = self.se(x)

        # Feature expansion
        x = self.expand(x)

        # Deep residual learning
        x = self.residual2(x)

        # Global pooling
        x = self.pool(x)

        x = x.squeeze(-1)

        # Final embedding
        x = self.embedding(x)

        return x