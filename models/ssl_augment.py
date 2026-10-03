import torch
import torch.nn as nn


class SignalAugmentation(nn.Module):
    """
    Signal-Specific Data Augmentation
    ---------------------------------
    Augmentations:
        1. Additive Gaussian Noise
        2. Amplitude Scaling
        3. Phase Rotation

    Input:
        (Batch, 2, 128)

    Output:
        (Batch, 2, 128)
    """

    def __init__(
        self,
        noise_std=0.02,
        amp_range=(0.9, 1.1),
        phase_range=(-10, 10)
    ):
        super().__init__()

        self.noise_std = noise_std
        self.amp_range = amp_range
        self.phase_range = phase_range

    def forward(self, x):

        x_aug = x.clone()

        # ---------------------------------------
        # 1. Gaussian Noise
        # ---------------------------------------
        noise = torch.randn_like(x_aug) * self.noise_std
        x_aug = x_aug + noise

        # ---------------------------------------
        # 2. Amplitude Scaling
        # ---------------------------------------
        scale = torch.empty(
            x_aug.size(0),
            1,
            1,
            device=x.device
        ).uniform_(
            self.amp_range[0],
            self.amp_range[1]
        )

        x_aug = x_aug * scale

        # ---------------------------------------
        # 3. Phase Rotation
        # ---------------------------------------
        phase = torch.empty(
            x_aug.size(0),
            device=x.device
        ).uniform_(
            self.phase_range[0],
            self.phase_range[1]
        )

        phase = torch.deg2rad(phase)

        cos = torch.cos(phase).view(-1, 1)
        sin = torch.sin(phase).view(-1, 1)

        I = x_aug[:, 0, :]
        Q = x_aug[:, 1, :]

        new_I = I * cos - Q * sin
        new_Q = I * sin + Q * cos

        x_aug[:, 0, :] = new_I
        x_aug[:, 1, :] = new_Q

        return x_aug