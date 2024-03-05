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
        self.feature_h = nn.Sequential(nn.Linear(2, int(hidden/2), bias=False),
                                       nn.Tanh(),)

        self.feature_ev = nn.Sequential(nn.Linear(4, int(hidden/2), bias=False),
                                        nn.Tanh(),)

        self.actor = nn.Sequential(nn.Linear(hidden, hidden, bias=False),
                                   nn.Tanh(),
                                   nn.Linear(hidden, hidden, bias=False),
                                   nn.Tanh(),
                                   nn.Linear(hidden, 1, bias=False),
                                   nn.Tanh())
        init_linear(self)

    def forward(self, s_h, s_ev):
        feature_h = self.feature_h(s_h)
        feature_ev = self.feature_ev(s_ev)
        a = self.actor(torch.cat([feature_ev, torch.multiply(feature_h, feature_ev)], dim=-1))
        return a


class DoubleCritic(nn.Module):
    def __init__(self, s_dim, a_dim, hidden):
        super(DoubleCritic, self).__init__()
        self.feature_h_1 = nn.Sequential(nn.Linear(2, int(hidden/2), bias=False),
                                         nn.Tanh())

        self.feature_eva_1 = nn.Sequential(nn.Linear(5, int(hidden/2), bias=False),
                                           nn.Tanh())

        self.q1 = nn.Sequential(nn.Linear(hidden, hidden, bias=False),
                                Abs(),
                                nn.Linear(hidden, hidden, bias=False),
                                Abs(),
                                nn.Linear(hidden, 1, bias=False),)


        self.feature_h_2 = nn.Sequential(nn.Linear(2, int(hidden/2), bias=False),
                                         nn.Tanh(),)

        self.feature_eva_2 = nn.Sequential(nn.Linear(5, int(hidden/2), bias=False),
                                           Abs(),)

        self.q2 = nn.Sequential(nn.Linear(hidden, hidden, bias=False),
                                Abs(),
                                nn.Linear(hidden, hidden, bias=False),
                                Abs(),
                                nn.Linear(hidden, 1, bias=False))
        init_linear(self)

    def forward(self, s_h, s_ev, a):
        s_eva = torch.cat([s_ev, a], dim=-1)
        feature_h_1 = self.feature_h_1(s_h)
        feature_eva_1 = self.feature_eva_1(s_eva)
        q1 = self.q1(torch.cat([feature_eva_1, torch.multiply(feature_h_1, feature_eva_1)], dim=-1))

        return q1

    def Q1(self, s_h, s_ev, a):
        s_eva = torch.cat([s_ev, a], dim=-1)
        feature_h_1 = self.feature_h_1(s_h)
        feature_eva_1 = self.feature_eva_1(s_eva)
        q1 = self.q1(torch.cat([feature_eva_1, torch.multiply(feature_h_1, feature_eva_1)], dim=-1))
        return q1


