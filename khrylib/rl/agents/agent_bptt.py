from argparse import _ActionStr
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_

from khrylib.rl.core import LoggerRL
from khrylib.utils.torch import *
from khrylib.rl.utils import torch_utils as tu
import math
import time
import os

class AgentBPTT:
    '''
    Agent for trajectory optimization via BPTT.
    Agent should be paired with a feedback policy network,
    The output of policy network should be torch.tensor that could propagate gradient
    '''
    def __init__(self, env, policy_net, dtype, 
                 device = 'cpu',gamma = 0.99, custom_reward = None,
                 end_reward = True, mean_action = False, render=False, 
                 traj_length = None, beta_dim=10, clip_gradient = True,
                 grad_norm = None):
        self.env = env
        self.policy_net = policy_net
        self.clip_gradient = clip_gradient
        self.grad_norm = grad_norm
        self.policy_net.eval()
        assert(traj_length is not None)
        self.traj_length = traj_length
        self.beta_dim = beta_dim

        self.dtype = dtype
        self.device = device
        self.gamma = gamma
        self.custom_reward = custom_reward
        self.end_reward = end_reward
        self.mean_action = mean_action
        self.render = render
        self.logger_cls = LoggerRL

        # initialize parameters related to optimization
        self.beta = nn.Embedding(self.traj_length, beta_dim)
        self.optim = torch.optim.Adam(self.action_embedding.parameters())
        self.losses = []
    
    # Sample a trajectory with loss computed
    # Need to aware method that DiffRL take to make gradient more stable
    def sample(self)->torch.Tensor:
        state = self.env.reset()
        loss = 0
        for i in range(self.traj_length):
            # TODO: (Eric) Action may not directly feedin state need to see how to transform to observations
            action = self.policy_net(state, self.beta(torch.tensor(i).to(self.device)))
            state, rew, _, info = self.env.step(torch.tanh(action))
            if self.running_state is not None:
                next_state = self.running_state(next_state)
            if self.custom_reward is not None and i < self.traj_length - 1:
                rew += self.custom_reward(self.env, state, action, info)
            else:
                rew += self.end_reward(self.env, state, action, info)
            loss -= rew * self.gamma ** i
        self.loss = loss

    def update_params(self):
        self.optim.zero_grad()
        self.loss.backward()
        # Clip the gradient if necessary
        with torch.no_grad():
            grad_norm_before_clip = tu.grad_norm(self.beta.parameters())
            if self.clip_grad:
                clip_grad_norm_(self.beta.parameters(), self.grad_norm)
            if torch.isnan(grad_norm_before_clip):
                print("========== NaN gradient detected! ============")
                for params in self.beta.parameters():
                    params.grad.zero_()
        self.optim.step()
        self.losses.append(float(self.loss))

                


        




            
        