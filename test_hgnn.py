import torch
from models.hgnn import HGNNLayer

X = torch.randn(16,256)

H = torch.eye(16)

model = HGNNLayer(256,256)

Y = model(X,H)

print("Input :",X.shape)
print("Output:",Y.shape)