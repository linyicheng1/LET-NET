import torch

a = []
for i in range(8):
    m = torch.zeros([10 ** i, 1], device=f'cuda:{i}')
    a.append(m)

c = 0
