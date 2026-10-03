import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        similarity = torch.matmul(z1, z2.T)

        similarity = similarity / self.temperature

        labels = torch.arange(z1.size(0)).to(z1.device)

        loss = F.cross_entropy(similarity, labels)

        return loss