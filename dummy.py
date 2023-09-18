import torch

x = torch.load("x.pt")

x = x.repeat(1, 1, 1, 3, 1, 3)

print(x)
