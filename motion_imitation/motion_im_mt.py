import argparse
import os
import sys
import pickle
import time
sys.path.append(os.getcwd())

from khrylib.utils import *
from khrylib.rl.core.policy_mcp_gaussian import MCPPolicyGaussian
from khrylib.rl.core.policy_additive_gaussian import AdditivePolicyGaussian
from khrylib.rl.core.critic import GoalValue
from khrylib.rl.agents import AgentPPO
from khrylib.models.mlp import MLP
from torch.utils.tensorboard import SummaryWriter
from motion_imitation.envs.humanoid_im import HumanoidEnv
from motion_imitation.envs.humanoid_im_nimble import HumanoidNimbleEnv
from motion_imitation.utils.config import Config
from motion_imitation.reward_function import reward_func

policy_dict = {"multiplicative":MCPPolicyGaussian,
               "additive":AdditivePolicyGaussian}

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None)
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--num_threads', type=int, default=16)
parser.add_argument('--gpu_index', type=int, default=0)
parser.add_argument('--iter', type=int, default=0)
parser.add_argument('--policy',type=str, default="multiplicative")
parser.add_argument('--show_noise', action='store_true', default=False)
parser.add_argument('--change_goal_freq',type=int, default=5)
parser.add_argument('--simulator',type=str, default="mujoco")
parser.add_argument('--exp_name', type=str, default=None, required=True)
args = parser.parse_args()
if args.render:
    args.num_threads = 1


cfg = Config(args.cfg, args.test, create_dirs=not (args.render or args.iter > 0))

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
tb_logger = SummaryWriter(cfg.tb_dir) if not args.render else None
logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'), file_handle=not args.render)

"""environment"""
if args.simulator == "mujoco":
    env = HumanoidEnv(cfg)
elif args.simulator == "nimble":
    env = HumanoidNimbleEnv(cfg)
env.seed(cfg.seed)
actuators = env.model.actuator_names
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
running_state = ZFilter((state_dim,), clip=5)

"""define actor and critic"""
#policy_net = PolicyGaussian(MLP(state_dim, cfg.policy_hsize, cfg.policy_htype), action_dim, log_std=cfg.log_std, fix_std=cfg.fix_std)
policy_net = policy_dict[args.policy](MLP(state_dim, cfg.policy_hsize, cfg.policy_htype), 
                                      action_dim, log_std=cfg.log_std, fix_std=cfg.fix_std, goal_dim=33)
value_net = GoalValue(MLP(state_dim+33, cfg.value_hsize, cfg.value_htype)) # 39 is goal dim
policy_net.set_goal(env.get_goal())
value_net.set_goal(env.get_goal())
if args.iter > 0:
    cp_path = f"{cfg.model_dir}/iter_{args.exp_name}_{args.iter:04}.p"
    logger.info('loading model from checkpoint: %s' % cp_path)
    model_cp = pickle.load(open(cp_path, "rb"))
    policy_net.load_state_dict(model_cp['policy_dict'])
    value_net.load_state_dict(model_cp['value_dict'])
    running_state = model_cp['running_state']
to_device(device, policy_net, value_net)

if cfg.policy_optimizer == 'Adam':
    optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=cfg.policy_lr, weight_decay=cfg.policy_weightdecay)
else:
    optimizer_policy = torch.optim.SGD(policy_net.parameters(), lr=cfg.policy_lr, momentum=cfg.policy_momentum, weight_decay=cfg.policy_weightdecay)
if cfg.value_optimizer == 'Adam':
    optimizer_value = torch.optim.Adam(value_net.parameters(), lr=cfg.value_lr, weight_decay=cfg.value_weightdecay)
else:
    optimizer_value = torch.optim.SGD(value_net.parameters(), lr=cfg.value_lr, momentum=cfg.value_momentum, weight_decay=cfg.value_weightdecay)

# reward functions
expert_reward = reward_func[cfg.reward_id]

"""create agent"""
agent = AgentPPO(env=env, dtype=dtype, device=device, running_state=running_state,
                 custom_reward=expert_reward, mean_action=args.render and not args.show_noise,
                 render=args.render, num_threads=args.num_threads,
                 policy_net=policy_net, value_net=value_net,
                 optimizer_policy=optimizer_policy, optimizer_value=optimizer_value, opt_num_epochs=cfg.num_optim_epoch,
                 gamma=cfg.gamma, tau=cfg.tau, clip_epsilon=cfg.clip_epsilon,
                 policy_grad_clip=[(policy_net.parameters(), 40)], end_reward=cfg.end_reward,
                 use_mini_batch=cfg.mini_batch_size < cfg.min_batch_size, mini_batch_size=cfg.mini_batch_size)


def pre_iter_update(i_iter):
    cfg.update_adaptive_params(i_iter)
    agent.set_noise_rate(cfg.adp_noise_rate)
    set_optimizer_lr(optimizer_policy, cfg.adp_policy_lr)
    if cfg.fix_std:
        policy_net.action_log_std.fill_(cfg.adp_log_std)
    return


def main_loop():

    if args.render:
        pre_iter_update(args.iter)
        agent.sample(1e8)
    else:
        for i_iter in range(args.iter, cfg.max_iter_num):
            """generate multiple trajectories that reach the minimum batch_size"""
            pre_iter_update(i_iter)
            batch, log = agent.sample(cfg.min_batch_size)
            if cfg.end_reward:
                agent.env.end_reward = log.avg_c_reward * cfg.gamma / (1 - cfg.gamma)   

            """update networks"""
            t0 = time.time()
            agent.update_params(batch)
            t1 = time.time()

            """logging"""
            c_info = log.avg_c_info
            logger.info(
                '{}\tT_sample {:.2f}\tT_update {:.2f}\tETA {}\texpert_R_avg {:.4f} {}'
                '\texpert_R_range ({:.4f}, {:.4f})\teps_len {:.2f}'
                .format(i_iter, log.sample_time, t1 - t0, get_eta_str(i_iter, cfg.max_iter_num, t1 - t0 + log.sample_time), log.avg_c_reward,
                        np.array2string(c_info, formatter={'all': lambda x: '%.4f' % x}, separator=','),
                        log.min_c_reward, log.max_c_reward, log.avg_episode_len))

            tb_logger.add_scalar('total_reward', log.avg_c_reward, i_iter)
            tb_logger.add_scalar('episode_len', log.avg_episode_reward, i_iter)
            for i in range(c_info.shape[0]):
                tb_logger.add_scalar('reward_%d' % i, c_info[i], i_iter)
                tb_logger.add_scalar('eps_reward_%d' % i, log.avg_episode_c_info[i], i_iter)

            """ Update goal from time to time """
            if (i_iter+1) % args.change_goal_freq == 0:
                agent.env.switch_expert()
                goal = agent.env.get_goal()
                agent.policy_net.set_goal(goal)
                agent.value_net.set_goal(goal)

            """ Save model from time to time """
            if cfg.save_model_interval > 0 and (i_iter+1) % cfg.save_model_interval == 0:
                tb_logger.flush()
                with to_cpu(policy_net, value_net):
                    cp_path = f"{cfg.model_dir}/iter_{args.exp_name}_{i_iter+1:04}.p"
                    model_cp = {'policy_dict': policy_net.state_dict(),
                                'value_dict': value_net.state_dict(),
                                'running_state': running_state}
                    pickle.dump(model_cp, open(cp_path, 'wb'))

            """clean up gpu memory"""
            torch.cuda.empty_cache()

        logger.info('training done!')


main_loop()
