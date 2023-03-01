import torch.nn as nn
from khrylib.rl.core.distributions import DiagGaussian
from khrylib.rl.core.policy import Policy
from khrylib.utils.math import *

class GatingFunction(nn.Module):
    def __init__(self, state_dim, goal_dim, num_primitives=6):
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
        h = torch.cat((s,g),axis=axis)
        h = self.bottleneck(h).relu()
        return self.out(h).sigmoid()

class SplitPolicyGaussian(Policy):
    def __init__(self, net, action_dim, net_out_dim=None, log_std=0, fix_std=False, num_primitives=6, goal_dim=39, summarize_w=False):
        super().__init__()
        self.type="gaussian"
        self.net = net
        if net_out_dim is None:
            net_out_dim = net.out_dim

        self.gating_function = GatingFunction(state_dim=net.state_dim,
                                              goal_dim = goal_dim, num_primitives=num_primitives)
        self.num_primitives = num_primitives
        self.action_dim = action_dim
        self.num_primitives_lb = int(self.num_primitives//2)
        self.num_primitives_ub = self.num_primitives - self.num_primitives_lb
        self.action_mean_lb = nn.Linear(net_out_dim, 14 * self.num_primitives_lb)
        self.action_mean_ub = nn.Linear(net_out_dim, (action_dim-14)*self.num_primitives_ub)
        self.action_mean_lb.weight.data.mul_(0.1)
        self.action_mean_ub.weight.data.mul_(0.1)
        self.action_mean_lb.bias.data.mul_(0.1)
        self.action_mean_ub.bias.data.mul_(0.1)

        # [n_primitives, n_action]
        self.action_log_std_lb = nn.Parameter(torch.ones(1, 14 * self.num_primitives_lb)*log_std,
                                           requires_grad= not fix_std).view(self.num_primitives_lb, 14)
        self.action_log_std_ub = nn.Parameter(torch.ones(1, (action_dim-14) * self.num_primitives_ub)*log_std,
                                           requires_grad= not fix_std).view(self.num_primitives_ub, action_dim - 14)
        self.summarize_w = summarize_w
        self.cum_w = []

    def set_goal(self, g):
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
        actions_means_lb = self.action_mean_lb(latent).view(-1, self.num_primitives_lb, 14)
        actions_means_ub = self.action_mean_ub(latent).view(-1, self.num_primitives_ub, self.action_dim-14)
        action_std_lb = torch.exp(self.action_log_std_lb.expand_as(actions_means_lb)).to(latent.device)
        action_std_ub = torch.exp(self.action_log_std_ub.expand_as(actions_means_ub)).to(latent.device)
        weighted_inv_std_lb = w[:,:self.num_primitives_lb].unsqueeze(2) / action_std_lb
        weighted_inv_std_ub = w[:,self.num_primitives_lb:].unsqueeze(2) / action_std_ub
        action_mean_lb = 1/(weighted_inv_std_lb.sum(dim=1)) * (weighted_inv_std_lb * actions_means_lb).sum(dim=1)
        action_mean_ub = 1/(weighted_inv_std_ub.sum(dim=1)) * (weighted_inv_std_ub * actions_means_ub).sum(dim=1)
        action_std_lb = 1/(weighted_inv_std_lb.sum(dim=1))
        action_std_ub = 1/(weighted_inv_std_ub.sum(dim=1))
        action_std = torch.cat([action_std_lb, action_std_ub], dim=1)
        action_mean = torch.cat([action_mean_lb, action_mean_ub], dim=1)
        return DiagGaussian(action_mean, action_std)
    
    def forward_with_beta(self, x, w):
        latent = self.net(x)
        if w.dim() == 1:
            w = w.view(1,-1)
        if self.summarize_w:
            self.cum_w.append(w)
        actions_means_lb = self.action_mean_lb(latent).view(-1, self.num_primitives_lb, 14)
        actions_means_ub = self.action_mean_ub(latent).view(-1, self.num_primitives_ub, self.action_dim-14)
        action_std_lb = torch.exp(self.action_log_std_lb.expand_as(actions_means_lb)).to(latent.device)
        action_std_ub = torch.exp(self.action_log_std_ub.expand_as(actions_means_ub)).to(latent.device)
        weighted_inv_std_lb = w[:,:self.num_primitives_lb].unsqueeze(2) / action_std_lb
        weighted_inv_std_ub = w[:,self.num_primitives_lb:].unsqueeze(2) / action_std_ub
        action_mean_lb = 1/(weighted_inv_std_lb.sum(dim=1)) * (weighted_inv_std_lb * actions_means_lb).sum(dim=1)
        action_mean_ub = 1/(weighted_inv_std_ub.sum(dim=1)) * (weighted_inv_std_ub * actions_means_ub).sum(dim=1)
        action_mean = torch.cat([action_mean_lb, action_mean_ub], dim=1)
        return action_mean.squeeze()

    def get_fim(self,x):
        return NotImplementedError
    
    def summary_w(self):
        sum_w = np.asarray(self.cum_w).sum(axis=0)
        sum_w /= np.linalg.norm(sum_w)
        return sum_w
        


        

