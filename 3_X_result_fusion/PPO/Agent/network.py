import torch
import torch.nn as nn


def init_linear(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.01)


class Abs(nn.Module):
    def __init__(self):
        super(Abs, self).__init__()

    def forward(self, x):
        return torch.abs(torch.tanh(x))


class Actor(nn.Module):
    def __init__(self, s_dim,  a_dim, hidden):
        super(Actor, self).__init__()
        self.feature_h = nn.Sequential(nn.Linear(2, hidden, bias=False),
                                       nn.Tanh(),
                                       nn.Linear(hidden, hidden, bias=False),
                                       nn.Tanh(),
                                       nn.Linear(hidden, 1, bias=False),
                                       nn.Tanh())
        self.feature_ev = nn.Sequential(nn.Linear(4, hidden, bias=False),
                                        nn.Tanh(),
                                        nn.Linear(hidden, hidden, bias=False),
                                        nn.Tanh(),
                                        nn.Linear(hidden, 1, bias=False),
                                        nn.Tanh())
        self.log_std = nn.Parameter(torch.ones(size=[a_dim]), requires_grad=True)
        init_linear(self)

    def forward(self, s_h, s_ev):
        feature_h = self.feature_h(s_h)
        feature_ev = self.feature_ev(s_ev)
        mean = torch.multiply(feature_h, feature_ev)
        std = self.log_std.exp()
        return mean, std


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, hidden):
        super(Critic, self).__init__()
        self.feature_h_1 = nn.Sequential(nn.Linear(2, hidden, bias=False),
                                         nn.Tanh(),
                                         nn.Linear(hidden, hidden, bias=False),
                                         nn.Tanh(),
                                         nn.Linear(hidden,1, bias=False),
                                         nn.Tanh())
        self.feature_eva_1 = nn.Sequential(nn.Linear(4, hidden, bias=False),
                                          nn.Tanh(),
                                          nn.Linear(hidden, hidden, bias=False),
                                          nn.Tanh(),
                                          nn.Linear(hidden, 1, bias=False),
                                          nn.Tanh())
        init_linear(self)

    def forward(self, s_h, s_ev):
        feature_h_1 = self.feature_h_1(s_h)
        feature_eva_1 = self.feature_eva_1(s_ev)
        q1 = torch.multiply(feature_h_1, feature_eva_1)
        return q1

#