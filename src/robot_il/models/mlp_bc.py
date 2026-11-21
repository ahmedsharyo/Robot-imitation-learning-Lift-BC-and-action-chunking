import torch.nn as nn


class MLPBC(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=(256, 256)):
        super().__init__()
        dims = [obs_dim, *hidden, action_dim]
        layers = []
        for i in range(len(dims)-2): layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, obs): return self.net(obs)
