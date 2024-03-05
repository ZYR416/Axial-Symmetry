import torch
import torch.nn as nn
from torch.distributions import Normal


class Controller:
    def __init__(self, path, prefix, s_dim=2, a_dim=1, hidden=32):
        self.actor = AxiDeterministicActor(s_dim, a_dim, hidden) if prefix == 'XY' \
            else DeterministicActor(s_dim, a_dim, hidden)
        self.actor.load_state_dict(torch.load(path + '/' + prefix + 'Actor.pth'))

    def get_action(self, s):
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float)
            mean, std = self.actor(s)
            # dist = Normal(mean, std)
            # a = dist.sample()
            a = mean.numpy() # torch.clamp(a, -1, 1).numpy()
            return a.item()

class DeterministicActor(nn.Module):
    def __init__(self, s_dim, a_dim, hidden):
        super(DeterministicActor, self).__init__()
        self.feature = nn.Sequential(nn.Linear(s_dim, hidden, bias=False),
                                     nn.Tanh(),
                                     nn.Linear(hidden, hidden, bias=False),
                                     nn.Tanh())
        self.mean = nn.Sequential(nn.Linear(hidden, a_dim, bias=False),
                                  nn.Tanh())
        # self.std = nn.Sequential(nn.Linear(hidden, a_dim),
        #                          nn.Softplus())
        self.log_std = nn.Parameter(torch.ones(size=[a_dim]), requires_grad=True)

    def forward(self, s):
        feature = self.feature(s)
        mean = self.mean(feature)
        std = self.log_std.exp()
        return mean, std


class AxiDeterministicActor(nn.Module):
    def __init__(self, s_dim, a_dim, hidden):
        super(AxiDeterministicActor, self).__init__()
        self.feature_h = nn.Sequential(nn.Linear(2, int(hidden / 2)),
                                       nn.Tanh() )

        self.feature_ev = nn.Sequential(nn.Linear(4, int(hidden / 2), bias=False),
                                        nn.Tanh(), )

        self.mean = nn.Sequential(nn.Linear(hidden, hidden, bias=False),
                                  nn.Tanh(),
                                  nn.Linear(hidden, hidden, bias=False),
                                  nn.Tanh(),
                                  nn.Linear(hidden, 1, bias=False),
                                  nn.Tanh())

        self.log_std = nn.Parameter(torch.ones(size=[a_dim]), requires_grad=True)

    def forward(self, s):
        s_h = s[0: 2]
        s_ev = s[2:]
        feature_h = self.feature_h(s_h)
        feature_ev = self.feature_ev(s_ev)
        mean = self.mean(torch.cat([feature_ev, torch.multiply(feature_h, feature_ev)], dim=-1))
        std = self.log_std.exp()
        return mean, std
