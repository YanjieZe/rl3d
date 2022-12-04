import math
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability"""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function, see appendix C from https://arxiv.org/pdf/1812.05905.pdf"""
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):
    def __init__(self, obs_shape=None, out_dim=None):
        super().__init__()
        self.out_shape = obs_shape
        self.out_dim = out_dim

    def forward(self, x):
        return x


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

        
        
class EfficientActor(nn.Module):
    def __init__(self, out_dim, projection_dim, action_shape, hidden_dim, log_std_min, log_std_max, state_shape=None, hidden_dim_state=None):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max



        self.trunk = nn.Sequential(nn.Linear(out_dim, projection_dim),
								   nn.LayerNorm(projection_dim), nn.Tanh())

        self.layers = nn.Sequential( 
            nn.Linear(projection_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        if state_shape:
            self.state_encoder = nn.Sequential(nn.Linear(state_shape[0], hidden_dim_state),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(hidden_dim_state, projection_dim),
                                                nn.LayerNorm(projection_dim), nn.Tanh())
        else:
            self.state_encoder = None

        self.apply(orthogonal_init)

    def forward(self, x, state, compute_pi=True, compute_log_pi=True):
        try:
            x = self.trunk(x)
        except:
            import pdb; pdb.set_trace()
        if self.state_encoder:
            x = x + self.state_encoder(state)

        mu, log_std = self.layers(x).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std


class QFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(orthogonal_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        return self.trunk(torch.cat([obs, action], dim=1))


class EfficientCritic(nn.Module):
    def __init__(self, out_dim, projection_dim, action_shape, hidden_dim,  state_shape=None, hidden_dim_state=None):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(out_dim, projection_dim),
                                        nn.LayerNorm(projection_dim), nn.Tanh())
        if state_shape:
            self.state_encoder = nn.Sequential(nn.Linear(state_shape[0], hidden_dim_state),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(hidden_dim_state, projection_dim),
                                                nn.LayerNorm(projection_dim), nn.Tanh())
        else:
            self.state_encoder = None

        self.Q1 = nn.Sequential(
            nn.Linear(projection_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        self.Q2 = nn.Sequential(
            nn.Linear(projection_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        self.apply(orthogonal_init)

    def forward(self, obs, state, action):
        obs = self.projection(obs)

        if self.state_encoder:
            obs = obs + self.state_encoder(state)

        h = torch.cat([obs, action], dim=-1)
        return self.Q1(h), self.Q2(h)


