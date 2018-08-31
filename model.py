
import torch
import numpy as np
import torch.nn as nn

def fanin_init(size, fanin=None):
    """
    weight initializer known from https://arxiv.org/abs/1502.01852
    :param size:
    :param fanin:
    :return:
    """
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, h1=400, h2=300, eps=0.03):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(h1 + action_dim, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(h2, 1)
        self.fc3.weight.data.uniform_(-eps, eps)

        self.relu = nn.ReLU()

    def forward(self, state, action):
        """
        return critic Q(s,a)
        :param state: state [n, state_dim] (n is batch_size)
        :param action: action [n, action_dim]
        :return: Q(s,a) [n, 1]
        """

        s1 = self.relu(self.fc1(state))
        x = torch.cat((s1, action), dim=1)

        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, h1=400, h2=300, eps=0.03):
        """
        :param state_dim: int
        :param action_dim: int
        :param action_lim: Used to limit action space in [-action_lim,action_lim]
        :return:
        """
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(h1, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(h2, action_dim)
        self.fc3.weight.data.uniform_(-eps, eps)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        """
        return actor policy function Pi(s)
        :param state: state [n, state_dim]
        :return: action [n, action_dim]
        """
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        action = self.tanh(self.fc3(x)) # tanh limit (-1, 1)
        return action
