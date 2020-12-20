import torch
a = torch.tensor([[2,3,4],[3,4,5]])
b = torch.tensor([[7,8,9],[1,2,3]])

c = torch.stack([a,b], dim=0)
print(c.shape)