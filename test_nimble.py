import torch
import torch.nn as nn
import nimblephysics as nimble
from functools import partial

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4,2,dtype=torch.double)

    def forward(self,x):
        return self.fc1(x).sigmoid()

class CustomAutoDiff:
    def __init__(self, sim, window_size=2):
        self.model = sim
        self.window_size = window_size
        self.us = []
        self.xs = []
        self.jacobians_u = []
        self.jacobians_x = []
        self.results = []

    def forward(self,u,x):
        self.us.append(u)
        self.xs.append(x)
        with torch.no_grad():
            res = self.model(u,x).clone() # x should be auto-regressive
        res.requires_grad_(True)
        self.results.append(res)
        with torch.no_grad():
            jac = torch.autograd.functional.jacobian(self.model,(u,x))
        self.jacobians_u.append(jac[0])
        self.jacobians_x.append(jac[1])
        return res

    def backprop_window(self,index):
        gradient = torch.zeros_like(self.xs[index])
        for i in range(min(index+self.window_size,len(self.results))-1,index-1,-1):
            gradient = torch.matmul(gradient,self.jacobians_x[i]) + self.results[i].grad
        gradient = torch.matmul(gradient, self.jacobians_u[index])
        return gradient

    def backprop(self):
        """
        Assume self.results already have gradients
        """
        actions_grad = torch.zeros(len(self.us), len(self.us[0]))
        for i in range(len(self.us)):
            actions_grad[i] = self.backprop_window(i)
        actions = torch.vstack(self.us)
        actions.backward(actions_grad)

world = nimble.simulation.World()
world.setGravity([0, -9.81, 0])
world.setTimeStep(0.01)

box = nimble.dynamics.Skeleton()
boxJoint, boxBody = box.createTranslationalJoint2DAndBodyNodePair()
boxShape = boxBody.createShapeNode(nimble.dynamics.BoxShape([.1, .1, .1]))
boxVisual = boxShape.createVisualAspect()
boxVisual.setColor([0.5, 0.5, 0.5])
world.addSkeleton(box)

initial_position: torch.Tensor = torch.tensor([3.0, 0.0])
initial_velocity: torch.Tensor = torch.zeros((world.getNumDofs()))
state = torch.cat((initial_position, initial_velocity), 0).double()
mlp = MLP()
optim = torch.optim.Adam(mlp.parameters())
def sim(u,x):
    return nimble.timestep(world, x, u)

autodiff = CustomAutoDiff(sim=sim, window_size=3)

optim.zero_grad()
loss = 0
for i in range(10):
    state = autodiff.forward(mlp(state.detach()), state)
    loss += state.norm()
loss.backward()
actions_grad = autodiff.backprop()
print(mlp.fc1.bias.grad)
print(mlp.fc1.weight.grad)
optim.step()
# for i in range(len(autodiff.us)):
#     print(autodiff.us[i].grad)