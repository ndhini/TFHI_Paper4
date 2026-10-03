import torch
from models.hypergraph import HypergraphBuilder

x = torch.randn(16, 256)

builder = HypergraphBuilder(top_k=4)

H = builder(x)

print("Feature Shape :", x.shape)
print("Incidence Matrix :", H.shape)