import argparse
import os
import sys
import pickle
import torch
import pytorch_kinematics as pk
sys.path.append(os.getcwd())
from khrylib.utils import *
from khrylib.rl.core.policy_mcp_gaussian import MCPPolicyGaussian
from khrylib.rl.core.policy_additive_gaussian import AdditivePolicyGaussian
from khrylib.motion_parser.models import MotionParser
from khrylib.models.mlp import MLP
from motion_imitation.envs.humanoid_im_nimble import HumanoidNimbleEnv
from motion_imitation.utils.config import Config

policy_dict = {"multiplicative":MCPPolicyGaussian,
               "additive":AdditivePolicyGaussian}

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
            res, done = self.model(u, x, True)
        res.requires_grad_(True)
        self.results.append(res)
        jac = torch.autograd.functional.jacobian(self.model,(u,x))
        self.jacobians_u.append(jac[0])
        self.jacobians_x.append(jac[1])
        return res,done

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

    def reset(self):
        self.us = []
        self.xs = []
        self.jacobian_u = []
        self.jacobian_x = []
        self.results = []

# TODO: Should follow test_nimble when propagating gradient
# TODO: Should implement MPI based batch training
# TODO: Need to freeze part of policy
# TODO: Implement early termination for cyclic motion

def loss_function(state, epos, env):
    # Convert epos to nimble format
    loss = (state[:len(state)//2]-epos).norm()
    return loss
    
def train(env, policy, motion_parser, lossfn, optim, horizon=50, iters=100, H=3):
    autodiff = CustomAutoDiff(sim=env.diff_step, window_size=H)
    for iter in range(iters):
        env.reset()
        optim.zero_grad()
        autodiff.reset()
        exp_traj = env.get_expert_pos_indices(range(horizon))
        motion_parser.parse_data(exp_traj)
        state = env.getState() # Should use exp traj for motion parser input
        loss = 0
        for i in range(horizon):
            beta = motion_parser.get_beta(state, i)
            action = policy.forward_with_beta(state, beta)
            state, done = autodiff.forward(action, state)
            loss += lossfn(state, exp_traj[i], env)
            if done:
                break
            print("Step:",i, state)
        loss.backward()
        autodiff.backprop()
        optim.step()
        print("Iters:", iter, "Loss:", loss)

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default=None)
parser.add_argument("--num_iters", default=100, type=int)
parser.add_argument("--policy_iter", default=None, type=str)
parser.add_argument("--policy", default="multiplicative", type=str)
args = parser.parse_args()

cfg = Config(args.cfg, False, create_dirs=False)
cfg.env_start_first = True
logger = create_logger(os.path.join(cfg.log_dir, 'log_eval.txt'))

"""make and seed env"""
dtype = torch.float64
torch.set_default_dtype(dtype)
torch.manual_seed(cfg.seed)
env = HumanoidNimbleEnv(cfg, disable_nimble_visualizer=True)
env.seed(cfg.seed)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

"""load learner policy"""
# NOTE: Goal dimension does not matter since we are not using the goal
policy_net = policy_dict[args.policy](MLP(state_dim, cfg.policy_hsize, cfg.policy_htype), action_dim, log_std=cfg.log_std, fix_std=cfg.fix_std)
if args.policy_iter is not None:
    cp_path = f"{cfg.model_dir}/iter_{args.policy_iter}.p"
    logger.info(f"loading model from checkpoint {cp_path}")
    model_cp = pickle.load(open(cp_path, "rb"))
    policy_net.load_state_dict(model_cp["policy_dict"])
    running_state = model_cp['running_state']

"""Initialize motion parser?"""
motion_parser = MotionParser(32, policy_net.num_primitives, device="cpu") # 26 + 6
optimizer = torch.optim.Adam(motion_parser.parameters())

train(env, policy_net, motion_parser, loss_function, optimizer)
