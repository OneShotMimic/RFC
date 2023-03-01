import argparse
import os
import sys
import pickle
import torch
from functools import partial
from mujoco_py import MujocoException
import nevergrad as ng
sys.path.append(os.getcwd())
from khrylib.utils import *
from khrylib.rl.core.policy_mcp_gaussian import MCPPolicyGaussian
from khrylib.rl.core.policy_additive_gaussian import AdditivePolicyGaussian
from khrylib.rl.core.policy_split import SplitPolicyGaussian
from khrylib.motion_parser.models import MotionParser, DummyMotionParser
from khrylib.models.mlp import MLP
from motion_imitation.envs.humanoid_im import HumanoidEnv
from motion_imitation.utils.config import Config
from motion_imitation.reward_function import reward_func

policy_dict = {"multiplicative":MCPPolicyGaussian,
               "additive":AdditivePolicyGaussian,
               "split":SplitPolicyGaussian}

weight_rec = 0.5
weight_direct = 0.5


def get_traj(betas, env, policy, reward_fn, args):
    betas = (torch.from_numpy(betas).view(args.steps,-1) + 5) / 10.0
    state = env.reset()
    loss = 0
    for i in range(args.steps):
        beta = betas[i]
        action = policy.forward_with_beta(torch.from_numpy(state), beta).detach().numpy()
        state, reward, done, info = env.step(action)
        reward += reward_fn(env, state, action, info)[0]
        loss -= reward
        if done and not args.fix_step:
            break
    return loss

def train(env, policy, optim, reward_fn, args):
    obj_fn = partial(get_traj, env=env, policy=policy, args=args, reward_fn=reward_fn)
    recommendation = optim.minimize(obj_fn)
    return torch.from_numpy(recommendation.value[0][0]).view(args.steps, -1)

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default=None)
parser.add_argument("--num_iters", default=3000, type=int)
parser.add_argument("--policy_iter", default=None, type=int)
parser.add_argument("--policy", default="multiplicative", type=str)
parser.add_argument("--exp_name", default=None, required=True)
parser.add_argument("--save_name", default=None, required=True, type=str)
parser.add_argument("--expert_id",default=0, type=int)
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
env = HumanoidEnv(cfg)
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
motion_parser = DummyMotionParser(38, policy_net.num_primitives, args.steps)
#motion_parser = MotionParser(38, 76, policy_net.num_primitives, sa_keep_percent=1/2)

instrum = ng.p.Instrumentation(ng.p.Array(shape=(args.steps * policy_net.num_primitives,)).set_bounds(lower=-5, upper=5))
optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=args.num_iters)
prog_bar = ng.callbacks.ProgressBar()
optimizer.register_callback("tell", prog_bar)
reward_fn = reward_func[cfg.reward_id]
betas = train(env, policy_net, optimizer, reward_fn, args)
motion_parser.load_betas(betas)
torch.save(motion_parser.state_dict(), f"models_mp/{args.save_name}.pth")
