import argparse
import os
import sys
import pickle
import torch
from mujoco_py import MujocoException
import pytorch_kinematics as pk
sys.path.append(os.getcwd())
from khrylib.utils import *
from khrylib.rl.core.policy_mcp_gaussian import MCPPolicyGaussian
from khrylib.rl.core.policy_additive_gaussian import AdditivePolicyGaussian
from khrylib.motion_parser.models import MotionParser, DummyMotionParser
from khrylib.models.mlp import MLP
from motion_imitation.envs.humanoid_im_nimble import HumanoidNimbleEnv
from motion_imitation.utils.config import Config

policy_dict = {"multiplicative":MCPPolicyGaussian,
               "additive":AdditivePolicyGaussian}

weight_rec = 0.5
weight_direct = 0.5

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

    def pop_last(self):
        self.us.pop()
        self.xs.pop()

# TODO: Should follow test_nimble when propagating gradient
# TODO: Should implement MPI based batch training
# TODO: Need to freeze part of policy
# TODO: Implement early termination for cyclic motion

def loss_function(state, epos):
    # Convert epos to nimble format
    loss = (state[:6]-epos[:6]).norm()
    return loss

def direct_loss(action, epos):
    #print("Error:",action - epos[6:])
    loss = (action - epos[6:]).norm()
    return loss

def train(env, policy, motion_parser, lossfn, optim, args):
    autodiff = CustomAutoDiff(sim=env.diff_step, window_size=args.horizon)
    best_loss = np.inf 
    best_iter_id = 0
    for iter in range(args.num_iters):
        env.reset()
        optim.zero_grad()
        autodiff.reset()
        #NOTICE: OUTPUT is already nimble state and in tensor format
        exp_traj = env.get_expert_pos_indices(range(args.steps))
        motion_parser.parse_data(exp_traj)
        state = env.getState() # Should use exp traj for motion parser input
        loss = 0
        for i in range(args.steps):
            beta = motion_parser.get_beta(state, i)
            action = policy.forward_with_beta(env.get_diff_obs(state), beta)
            try:
                state, done = autodiff.forward(action, state) # state should be nimble state
            except MujocoException:
                # POP last entry
                autodiff.pop_last()
                break
            loss += lossfn(state, exp_traj[i]) * weight_rec
            loss += direct_loss(action, exp_traj[i]) * weight_direct
            if done and not args.fix_step:
                break
            # print("Step:",i)
        loss.backward(retain_graph=True)
        autodiff.backprop()
        optim.step()
        if iter % args.save_freq == 0 and iter !=0:
            if float(loss) < best_loss:
                print("loss", loss, "best_loss", best_loss)
                best_loss = float(loss)
                best_iter_id = iter
            print(f"Model: {iter} is saved, best iter {best_iter_id}")
            torch.save(motion_parser.state_dict(), f"models_mp/{args.save_name}_{iter//args.save_freq}.pth")
        print("Iters:", iter, "Loss:", loss)

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default=None)
parser.add_argument("--num_iters", default=100, type=int)
parser.add_argument("--policy_iter", default=None, type=int)
parser.add_argument("--policy", default="multiplicative", type=str)
parser.add_argument("--exp_name", default=None, required=True)
parser.add_argument("--save_name", default=None, required=True, type=str)
parser.add_argument("--expert_id",default=0, type=int)
parser.add_argument("--horizon", type=int, default=2)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--save_freq",type=int, default=5)
parser.add_argument("--fix_step", action="store_true", default=False)
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
env.switch_expert(args.expert_id)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

"""load learner policy"""
# NOTE: Goal dimension does not matter since we are not using the goal
policy_net = policy_dict[args.policy](MLP(state_dim, cfg.policy_hsize, cfg.policy_htype), 
                                      action_dim, log_std=cfg.log_std, fix_std=cfg.fix_std, goal_dim=39)
if args.policy_iter is not None:
    cp_path = f"{cfg.model_dir}/iter_{args.exp_name}_{args.policy_iter:04}.p"
    logger.info(f"loading model from checkpoint {cp_path}")
    model_cp = pickle.load(open(cp_path, "rb"))
    policy_net.load_state_dict(model_cp["policy_dict"])
    running_state = model_cp['running_state']

"""Initialize motion parser?"""
# motion_parser = MotionParser(32, 26, device="cpu") # 26 + 6
# motion_parser = DummyMotionParser(38,policy_net.num_primitives,args.steps)
motion_parser = MotionParser(38, 76, policy_net.num_primitives, sa_keep_percent=1/2)

optimizer = torch.optim.Adam(motion_parser.parameters(),lr=1e-5)

train(env, policy_net, motion_parser, loss_function, optimizer, args)
