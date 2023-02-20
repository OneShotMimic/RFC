import argparse
import os
import sys
import pickle
import time
import subprocess
import shutil
sys.path.append(os.getcwd())

from khrylib.utils import *
from khrylib.rl.utils.visualizer import Visualizer
from khrylib.rl.core.policy_gaussian import PolicyGaussian
from khrylib.rl.core.critic import Value
from khrylib.models.mlp import MLP
from motion_imitation.envs.humanoid_im import HumanoidEnv
from motion_imitation.utils.config import Config

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None)
parser.add_argument('--vis_model_file', default='mocap_v2_vis')
parser.add_argument('--iter', type=int, default=-1)
parser.add_argument('--focus', action='store_true', default=True)
parser.add_argument('--hide_expert', action='store_true', default=False)
parser.add_argument('--preview', action='store_true', default=False)
parser.add_argument('--record', action='store_true', default=False)
parser.add_argument('--record_expert', action='store_true', default=False)
parser.add_argument('--azimuth', type=float, default=45)
parser.add_argument('--video_dir', default='out/videos/motion_im')
parser.add_argument('--simulator', type=str, default="mujoco")
parser.add_argument('--exp_name', type=str, default=None, required=True)
parser.add_argument('--render_nimble',action="store_true", default=False)
args = parser.parse_args()
cfg = Config(args.cfg, False, create_dirs=False)
cfg.env_start_first = True
logger = create_logger(os.path.join(cfg.log_dir, 'log_eval.txt'))

"""make and seed env"""
dtype = torch.float64
torch.set_default_dtype(dtype)
torch.manual_seed(cfg.seed)
torch.set_grad_enabled(False)
if args.simulator == "mujoco":
    env = HumanoidEnv(cfg)
elif args.simulator == "nimble":
    raise NotImplementedError
    #env = HumanoidNimbleEnv(cfg, disable_nimble_visualizer = not args.render_nimble)
env.seed(cfg.seed)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

"""load learner policy"""
policy_net = PolicyGaussian(MLP(state_dim, cfg.policy_hsize, cfg.policy_htype), action_dim, log_std=cfg.log_std, fix_std=cfg.fix_std)
value_net = Value(MLP(state_dim, cfg.value_hsize, cfg.value_htype))
cp_path = f"{cfg.model_dir}/iter_{args.exp_name}_{args.iter:04}.p"
logger.info('loading model from checkpoint: %s' % cp_path)
model_cp = pickle.load(open(cp_path, "rb"))
policy_net.load_state_dict(model_cp['policy_dict'])
value_net.load_state_dict(model_cp['value_dict'])
running_state = model_cp['running_state']


class MyVisulizer(Visualizer):

    def __init__(self, vis_file, actions = None):
        super().__init__(vis_file, actions)
        if args.simulator == "nimble":
            ngeom = len(env.mj_model.geom_rgba) - 1
        else:
            ngeom = len(env.model.geom_rgba) - 1
        self.env_vis.model.geom_rgba[ngeom + 21: ngeom * 2 + 21] = np.array([0.7, 0.0, 0.0, 1])
        self.env_vis.viewer.cam.lookat[2] = 1.0
        self.env_vis.viewer.cam.azimuth = args.azimuth
        self.env_vis.viewer.cam.elevation = -8.0
        self.env_vis.viewer.cam.distance = 5.0
        self.T = 12

    def data_generator(self):
        action_saved = False
        while True:
            poses = {'pred': [], 'gt': []}
            state = env.reset()
            if running_state is not None:
                state = running_state(state, update=False)
            actions = []
            for t in range(200):
                epos = env.get_expert_attr('qpos', env.get_expert_index(t)).copy()
                if env.expert['meta']['cyclic']:
                    init_pos = env.expert['init_pos']
                    cycle_h = env.expert['cycle_relheading']
                    cycle_pos = env.expert['cycle_pos']
                    epos[:3] = quat_mul_vec(cycle_h, epos[:3] - init_pos) + cycle_pos
                    epos[3:7] = quaternion_multiply(cycle_h, epos[3:7])
                poses['gt'].append(epos)
                poses['pred'].append(env.data.qpos.copy()) # Should have no difference
                state_var = tensor(state, dtype=dtype).unsqueeze(0)
                if self.stored_actions is not None:
                    action = self.stored_actions[t]
                    print("Use stored action")
                else:
                    action = policy_net.select_action(state_var, mean_action=True)[0].cpu().numpy()
                actions.append(action)
                next_state, reward, done, _ = env.step(action)
                if args.render_nimble:
                    env.render_nimble()
                print("t:",t)
                if running_state is not None:
                    next_state = running_state(next_state, update=False)
                if done:
                    pass
                state = next_state
            if action_saved == False and self.stored_actions is None:
                np.save("actions_.npy", actions)
            poses['gt'] = np.vstack(poses['gt'])
            poses['pred'] = np.vstack(poses['pred'])
            self.num_fr = poses['pred'].shape[0]
            yield poses

    def update_pose(self):
        model = env.mj_model if args.simulator == "nimble" else env.model
        print("env_vis:", self.env_vis.data.qpos.shape)
        self.env_vis.data.qpos[:model.nq] = self.data['pred'][self.fr]
        self.env_vis.data.qpos[model.nq:] = self.data['gt'][self.fr]
        self.env_vis.data.qpos[model.nq] += 1.0
        if args.record_expert:
            self.env_vis.data.qpos[:model.nq] = self.data['gt'][self.fr]
        if args.hide_expert:
            self.env_vis.data.qpos[model.nq + 2] = 100.0
        if args.focus:
            self.env_vis.viewer.cam.lookat[:2] = self.env_vis.data.qpos[:2]
        self.env_vis.sim_forward()

    def record_video(self):
        frame_dir = f'{args.video_dir}/frames'
        if os.path.exists(frame_dir):
            shutil.rmtree(frame_dir)
        os.makedirs(frame_dir, exist_ok=True)
        for fr in range(self.num_fr):
            self.fr = fr
            self.update_pose()
            for _ in range(20):
                self.render()
            if not args.preview:
                t0 = time.time()
                save_screen_shots(self.env_vis.viewer.window, f'{frame_dir}/%04d.png' % fr)
                print('%d/%d, %.3f' % (fr, self.num_fr, time.time() - t0))

        if not args.preview:
            out_name = f'{args.video_dir}/{args.cfg}_{"expert" if args.record_expert else args.iter}.mp4'
            cmd = ['/usr/local/bin/ffmpeg', '-y', '-r', '30', '-f', 'image2', '-start_number', '0',
                '-i', f'{frame_dir}/%04d.png', '-vcodec', 'libx264', '-crf', '5', '-pix_fmt', 'yuv420p', out_name]
            subprocess.call(cmd)



vis = MyVisulizer(f'{args.vis_model_file}.xml')#, actions = np.load("actions_.npy"))

if args.record:
    vis.record_video()
else:
    vis.show_animation()
