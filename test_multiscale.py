import torch

from models.multiscale import MultiScaleTemporalEncoder

model = MultiScaleTemporalEncoder()

x = torch.randn(8,2,128)

y = model(x)

print()

print("Input :",x.shape)

print("Output:",y.shape)