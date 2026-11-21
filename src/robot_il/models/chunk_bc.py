import torch.nn as nn


class ChunkBC(nn.Module):
    def __init__(self, obs_dim, action_dim, horizon, hidden=(256, 256)):
        super().__init__()
        if horizon < 1: raise ValueError("horizon must be positive")
        self.horizon, self.action_dim = horizon, action_dim
        dims = [obs_dim, *hidden, horizon * action_dim]
        layers = []
        for i in range(len(dims)-2): layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, obs): return self.net(obs).view(*obs.shape[:-1], self.horizon, self.action_dim)
