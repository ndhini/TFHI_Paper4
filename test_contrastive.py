import torch

from training.contrastive import ContrastiveLoss

loss_fn = ContrastiveLoss()

x1 = torch.randn(16,256)

x2 = torch.randn(16,256)

loss = loss_fn(x1,x2)

print("Contrastive Loss =", loss.item())