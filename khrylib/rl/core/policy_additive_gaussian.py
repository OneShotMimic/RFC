import torch.nn as nn
from khrylib.rl.core.distributions import MixtureOfGaussian
from khrylib.rl.core.policy import Policy
from khrylib.utils.math import *


class GatingFunction(nn.Module):
    def __init__(self, state_dim, goal_dim, num_primitives=8):
        super(GatingFunction, self).__init__()
        self.sn1 = nn.Linear(state_dim, 512)
        self.sn2 = nn.Linear(512, 256)
        self.gn1 = nn.Linear(goal_dim, 512)
        self.gn2 = nn.Linear(512, 256)
        self.bottleneck = nn.Linear(512, 256)
        self.out = nn.Linear(256, num_primitives)

    def forward(self, state, goal):
        s = self.sn1(state).relu()
        s = self.sn2(s).relu()
        g = self.gn1(goal).relu()
        g = self.gn2(g).relu()
        h = torch.cat((s, g), axis=1)
        h = self.bottleneck(h).relu()
        return self.out(h).sigmoid()

class AdditivePolicyGaussian(Policy):
    def __init__(self, net, action_dim, net_out_dim=None, log_std=0, fix_std=False, num_primitives=8, goal_dim=39):
        super().__init__()
        self.type = 'gaussian'
        self.net = net
        if net_out_dim is None:
            net_out_dim = net.out_dim
        # TODO: Define gating function
        self.gating_function = GatingFunction(state_dim=net.state_dim, 
                                              goal_dim=goal_dim, num_primitives=num_primitives)

        # Define primitives
        self.num_primitives = num_primitives
        self.action_dim = action_dim
        self.action_mean = nn.Linear(net_out_dim, action_dim * self.num_primitives)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.1)

        # [n_primitives, n_action]
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim * self.num_primitives) * log_std, 
                                           requires_grad=not fix_std).view(self.num_primitives, action_dim)

    def set_goal(self,g):
        self.g = g

    def forward(self, x):
        latent = self.net(x)
        w = self.gating_function(x,self.g) # [batch_size, n_primitives]
        action_mean = self.action_mean(latent).view(-1, self.num_primitives, self.action_dim) # [batch_size, n_primitives, action_dim]
        action_std = torch.exp(self.action_log_std.expand_as(action_mean)) #[batch_size, n_primitives, action_dim]
        return MixtureOfGaussian(action_mean, action_std, probs=torch.softmax(w, dim=1))

    def get_fim(self, x):
        """Currently not support TRPO"""
        raise NotImplementedError


