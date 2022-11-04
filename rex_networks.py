import torch
from torch import nn

class ResLin(nn.Module):
    def __init__(self, in_n, out_n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_n, out_n),
            nn.LeakyReLU(0.1)
        )
    def forward(self, x):
        x = x + self.net(x)
        return x

class Agent_net(nn.Module):
    '''
    Estimates the best action which would have maximised q value for a given state
    '''
    def __init__(self, n_obs: int, n_act: int, dev="cuda") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 800),
            nn.LeakyReLU(0.1),
            # ResLin(400, 400),
            nn.Linear(800, 800),
            nn.LeakyReLU(0.1),
            nn.Linear(800, 800),
            nn.LeakyReLU(0.1),
            nn.Linear(800, n_act),
            nn.Tanh()
        )
        self.dev = dev

    def forward(self, x):
        '''
        x : observation
        '''
        x = self.net(x)
        mask = torch.ones(x.shape, device=self.dev)
        mask[:, 0::3] = 0
        out = x * mask
        return out


class Critic_net(nn.Module):
    '''
    Estimates the qvalue given in input space and the action performed on this state
    '''
    def __init__(self, n_obs: int, n_act: int, dev="cuda") -> None:
        super().__init__()
        self.obs_head = nn.Sequential(
            nn.Linear(n_obs, 800),
            nn.LeakyReLU()
        )
        self.net = nn.Sequential(
            nn.Linear(800+n_act, 800),
            nn.LeakyReLU(0.1),
            nn.Linear(800, 800),
            nn.LeakyReLU(0.1),
            nn.Linear(800, 800),
            nn.LeakyReLU(0.1),
            # ResLin(800, 800),
            nn.Linear(800, 1),
        )
        self.dev = dev

    def forward(self, obs, act):
        obs_enc = self.obs_head(obs)
        out = torch.cat([obs_enc, act], dim=1)
        out = self.net(out)
        return out

