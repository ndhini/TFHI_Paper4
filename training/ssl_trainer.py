import torch
import torch.nn as nn

from models.ssl_augment import SignalAugmentation
from training.ssl_loss import ConsistencyLoss


class SSLTrainer(nn.Module):
    """
    Signal Consistency Self-Supervised Trainer

    Input:
        x : (Batch,2,128)

    Output:
        ssl_loss
        original_features
        augmented_features
    """

    def __init__(self, encoder):

        super().__init__()

        self.encoder = encoder

        self.augment = SignalAugmentation()

        self.ssl_loss = ConsistencyLoss()

    def forward(self, x):

        # ----------------------------
        # Original Signal
        # ----------------------------
        z1 = self.encoder(x)

        # ----------------------------
        # Augmented Signal
        # ----------------------------
        x_aug = self.augment(x)

        z2 = self.encoder(x_aug)

        # ----------------------------
        # Self-Supervised Loss
        # ----------------------------
        loss = self.ssl_loss(z1, z2)

        return loss, z1, z2