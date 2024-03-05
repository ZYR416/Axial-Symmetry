import torch
import torch.nn as nn


def init_linear(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.1)


class Abs(nn.Module):
    def __init__(self):
        super(Abs, self).__init__()

    def forward(self, x):
        return torch.abs(torch.tanh(x))


class DeterministicActor(nn.Module):
    def __init__(self, s_dim,  a_dim, hidden):
        super(DeterministicActor, self).__init__()
        self.feature_h = nn.Sequential(nn.Linear(2, hidden, bias=False),
                                               nn.Tanh(),
                                               nn.Linear(hidden, hidden, bias=False),
                                               nn.Tanh(),
                                               nn.Linear(hidden, 1, bias=False))
        self.feature_ev = nn.Sequential(nn.Linear(4, hidden, bias=False),
                                                nn.Tanh(),
                                                nn.Linear(hidden, hidden, bias=False),
                                                nn.Tanh(),
                                                nn.Linear(hidden, 1, bias=False))
        init_linear(self)

    def forward(self, s_h, s_ev):
        feature_h = self.feature_h(s_h)
        feature_ev = self.feature_ev(s_ev)
        a = torch.multiply(feature_h, feature_ev)
        return a


class DoubleCritic(nn.Module):
    def __init__(self, s_dim, a_dim, hidden):
        super(DoubleCritic, self).__init__()
        self.feature_h_1 = nn.Sequential(nn.Linear(2, hidden, bias=False),
                                         nn.Tanh(),
                                         nn.Linear(hidden, hidden, bias=False),
                                         nn.Tanh(),
                                         nn.Linear(hidden, 1, bias=False))



        self.feature_eva_1 = nn.Sequential(nn.Linear(5, hidden, bias=False),
                                           Abs(),
                                           nn.Linear(hidden, hidden, bias=False),
                                           Abs(),
                                           nn.Linear(hidden, 1, bias=False))

        init_linear(self)

    def forward(self, s_h, s_ev, a):
        s_eva = torch.cat([s_ev, a], dim=-1)
        feature_h_1 = self.feature_h_1(s_h)
        feature_eva_1 = self.feature_eva_1(s_eva)
        q1 = torch.multiply(feature_h_1, feature_eva_1)

        return q1

    def Q1(self, s_h, s_ev, a):
        s_eva = torch.cat([s_ev, a], dim=-1)
        feature_h_1 = self.feature_h_1(s_h)
        feature_eva_1 = self.feature_eva_1(s_eva)
        q1 = torch.multiply(feature_h_1, feature_eva_1)
        return q1