import torch.nn as nn
from khrylib.rl.core.distributions import MixtureOfGaussian
from khrylib.rl.core.policy import Policy
from khrylib.utils.math import *


class GatingFunction(nn.Module):
    def __init__(self, state_dim, goal_dim, num_primitives=5):
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
        axis = 1 if state.dim() == 2 else 0
        h = torch.cat((s, g), axis=axis)
        h = self.bottleneck(h).relu()
        return self.out(h).softmax(dim=1)

class AdditivePolicyGaussian(Policy):
    def __init__(self, net, action_dim, net_out_dim=None, log_std=0, fix_std=False, num_primitives=8, goal_dim=39, summarize_w=False):
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
        self.cum_w = []
        self.summarize_w = summarize_w

    def set_goal(self,g):
        if not torch.is_tensor(g):
            g = torch.from_numpy(g)
        self.g = g

    def forward(self, x):
        latent = self.net(x)
        if x.dim() != 1:
            goal = self.g.repeat(x.shape[0]).view(x.shape[0],-1).to(latent.device)
        else:
            goal = self.g.to(latent.device)
        w = self.gating_function(x,goal) # [batch_size, n_primitives]
        if self.summarize_w:
            self.cum_w.append(w[0].cpu().detach().numpy())
        action_mean = self.action_mean(latent).view(-1, self.num_primitives, self.action_dim) # [batch_size, n_primitives, action_dim]
        action_std = torch.exp(self.action_log_std.expand_as(action_mean)).to(latent.device) #[batch_size, n_primitives, action_dim]
        return MixtureOfGaussian(action_mean, action_std, probs=torch.softmax(w, dim=1))

    def forward_with_beta(self,x,w):
        latent = self.net(x)
        if w.dim() == 1:
            w = w.view(1,-1)
        if self.summarize_w:
            self.cum_w.append(w)
        action_mean = self.action_mean(latent).view(-1, self.num_primitives, self.action_dim) # [batch_size, n_primitives, action_dim]
        action_std = torch.exp(self.action_log_std.expand_as(action_mean)).to(latent.device) #[batch_size, n_primitives, action_dim]
        return MixtureOfGaussian(action_mean, action_std, probs=torch.softmax(w, dim=1))

    def get_fim(self, x):
        """Currently not support TRPO"""
        raise NotImplementedError

    def summary_w(self):
        sum_w = np,


