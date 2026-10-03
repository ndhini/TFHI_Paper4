import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyLoss(nn.Module):
    """
    Signal Consistency Self-Supervised Loss

    This loss encourages the encoder to produce similar
    feature representations for:
        - Original Signal
        - Augmented Signal

    Loss = Mean Squared Error between normalized embeddings.
    """

    def __init__(self):
        super().__init__()

    def forward(self, z1, z2):

        # ---------------------------------------
        # Normalize embeddings
        # ---------------------------------------
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # ---------------------------------------
        # Consistency Loss
        # ---------------------------------------
        loss = F.mse_loss(z1, z2)

        return loss