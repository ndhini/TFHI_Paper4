import torch
import torch.nn as nn


class FeatureTransformer(nn.Module):
    """
    Enhanced Transformer Feature Extractor

    Input:
        (Batch, 8, 32)

    Output:
        (Batch, 256)
    """

    def __init__(
        self,
        token_dim=32,
        num_tokens=8,
        num_heads=4,
        num_layers=2,
        dropout=0.10
    ):

        super().__init__()

        # -------------------------------------------------
        # Learnable Positional Embedding
        # -------------------------------------------------
        self.position_embedding = nn.Parameter(
            torch.randn(1, num_tokens, token_dim)
        )

        # -------------------------------------------------
        # Transformer Encoder
        # -------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # -------------------------------------------------
        # Layer Normalization
        # -------------------------------------------------
        self.norm = nn.LayerNorm(token_dim)

        # -------------------------------------------------
        # Output Projection
        # -------------------------------------------------
        self.output_projection = nn.Sequential(

            nn.Linear(token_dim * num_tokens, 256),

            nn.GELU(),

            nn.Dropout(0.20)

        )

        # -------------------------------------------------
        # Residual Projection
        # -------------------------------------------------
        self.shortcut = nn.Linear(
            token_dim * num_tokens,
            256
        )

    def forward(self, tokens):

        # ------------------------------------------
        # Add Positional Encoding
        # ------------------------------------------
        x = tokens + self.position_embedding

        # ------------------------------------------
        # Transformer
        # ------------------------------------------
        x = self.transformer(x)

        # ------------------------------------------
        # LayerNorm
        # ------------------------------------------
        x = self.norm(x)

        # ------------------------------------------
        # Flatten
        # ------------------------------------------
        batch_size = x.size(0)

        x = x.reshape(batch_size, -1)

        # ------------------------------------------
        # Residual Connection
        # ------------------------------------------
        identity = self.shortcut(x)

        x = self.output_projection(x)

        x = x + identity

        return x