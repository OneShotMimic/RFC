import torch
import torch.nn as nn
import numpy as np

class AccNet(nn.Module):
    def __init__(self, state_dim, act_dim, out_dim,
                 max_acc,
                 min_acc):
        super(AccNet, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_acc = max_acc if torch.is_tensor(max_acc) else torch.as_tensor(max_acc)
        self.min_acc = min_acc if torch.is_tensor(min_acc) else torch.as_tensor(min_acc)
        self.fc1 = nn.Linear(state_dim+act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, out_dim)

    def forward(self, state, action):
        h = self.fc1(torch.hstack([state, action]))
        h = self.fc2(h)
        a = self.out(h).clamp(self.min_acc, self.max_acc)
        return a

class NNWorld:
    def __init__(self, 
                 q_dim, 
                 dq_dim,
                 min_acc, 
                 max_acc,
                 dt=0.0015,
                 device="cpu"):
        self.q_dim = q_dim
        self.dq_dim = dq_dim
        self.state_dim = q_dim + dq_dim
        self.dt = dt
        self.device = torch.device(device)
        self.acc_net = AccNet(self.state_dim, dq_dim, 
                              dq_dim, max_acc, min_acc).to(self.device)
        self.q = torch.zeros(self.q_dim).to(self.device)
        self.dq = torch.zeros(self.dq_dim).to(self.device)
        self.action = torch.zeros(self.dq_dim).to(self.device)

    def getState(self):
        state = torch.hstack([self.q, self.dq]).cpu().detach.numpy()
        return state.copy()

    def setState(self, state):
        state = state if torch.is_tensor(state) else torch.as_tensor(state)
        state = state.to(self.device)
        self.q = state[:self.q_dim]
        self.dq = state[self.q_dim:self.q_dim+self.dq_dim]

    def setAction(self, act):
        action = act if torch.is_tensor(act) else torch.as_tensor(act)
        action = action.to(self.device)
        self.action = action

    def step(self):
        self.dq = self.dq + self.acc_net(torch.hstack([self.q, self.dq]), self.action) * self.dt
        self.q = self.q + self.dq * self.dt

    # Assume action will not change locally
    @staticmethod
    def timestep(world, state, action):
        world.setState(state)
        world.setAction(action)
        world.step()
        return torch.hstack([world.q, world.dq])
