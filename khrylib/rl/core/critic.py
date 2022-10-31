import torch.nn as nn
import torch


class Value(nn.Module):
    def __init__(self, net, net_out_dim=None):
        super().__init__()
        self.net = net
        if net_out_dim is None:
            net_out_dim = net.out_dim
        self.value_head = nn.Linear(net_out_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.net(x)
        value = self.value_head(x)
        return value

class GoalValue(nn.Module):
    def __init__(self,net, net_out_dim=None):
        super().__init__()
        self.net = net
        if net_out_dim is None:
            net_out_dim = net.out_dim
        self.value_head = nn.Linear(net_out_dim,1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)
        self.goal = None

    def set_goal(self,goal):
        if not torch.is_tensor(goal):
            goal = torch.from_numpy(goal)
        self.goal = goal

    def forward(self,x):
        if x.dim() >= 1:
            g = self.goal.repeat(x.shape[0]).view(x.shape[0],-1)
        xg = torch.hstack([x,g.to(x.device)])
        h = self.net(xg)
        value = self.value_head(h)
        return value
        