import torch
from models.transformer import FeatureTransformer

x = torch.randn(16, 256)

model = FeatureTransformer()

y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)