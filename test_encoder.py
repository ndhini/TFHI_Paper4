import torch
from models.encoder import SignalEncoder

model = SignalEncoder()

x = torch.randn(8, 2, 128)

y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)