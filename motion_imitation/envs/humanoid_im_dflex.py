import os
import sys
from turtle import heading
sys.path.append(os.getcwd())

from khrylib.rl.envs.common import dflex_env
from gym import spaces
from khrylib.utils import *
from khrylib.rl.utils import torch_utils as tu
from khrylib.utils import load_utils as lu
from motion_imitation.utils.tools import get_expert_dflex
import dflex as df
try:
    from pxr import Usd
except:
    print("No pxr package")
import pickle
import time

class HumanoidDFlexEnv(dflex_env.DflexEnv):

    def __init__(self, cfg, **kwargs):
        dflex_env.DflexEnv.__init__(self, cfg.mujoco_model_file, 15, **kwargs)
        self.vf_dim = 6
        self.cfg = cfg
        self.set_cam_first = set()
        # env specific
        self.end_reward = 0.0
        self.start_ind = 0
        self.bquat = self.get_body_quat()
        self.prev_bquat = None
        self.set_model_params()
        self.expert = None
        st = time.time()
        self.load_expert()
        print(f"Take {time.time() - st} to load expert")

    def build_model(self, fullpath):
        """
        Should build dflex model based on fullpath mjcf.
        return:
        model, integrator(sim), state, dt
        """
        builder = df.sim.ModelBuilder()
        dt = 1.0 / 60.0
        
        ground = True

        # TODO: Need closer inspection to see whether it is correct number or not.
        start_rot = df.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)

        # The world directional convension are hard-coded?

        start_pos = []

        # TODO: Need closer inspection to see whether it is correct number or not.
        start_height = 1.35

        asset_folder = os.path.join(os.path.dirname(__file__), '../../khrylib/assets/mujoco_models')

        # Load MJCF model
        lu.parse_mjcf(os.path.join(asset_folder, "mocap_v2_dflex.xml"), builder,
        stiffness = 0.0,
        damping=0.0,
        contact_ke = 2.e+4,
        contact_kd = 5.e+3,
        contact_kf = 1.e+3,
        contact_mu = 0.75,
        limit_ke = 1.e+3,
        limit_kd = 1.e+1,
        armature = 0.007,
        load_stiffness = True,
        load_armature=True)

        start_pos_z = 0.0
        start_pos.append([0.0, start_height, start_pos_z])

        builder.joint_q[:3] = start_pos[-1]
        builder.joint_q[3:7] = start_rot
        
        model = builder.finalize(self.device)
        model.ground = ground
        model.gravity = torch.tensor((0.0, -9.81, 0.0), dtype = torch.float32, device = self.device)
        self.body_mass = builder.body_mass
        
        sim = df.sim.SemiImplicitIntegrator()
        state = model.state() # Should be initial state
        model.collide(state)
        return model, sim, state, dt

    def load_expert(self):
        expert_qpos, expert_meta = pickle.load(open(self.cfg.expert_traj_file, "rb"))
        self.expert = get_expert_dflex(expert_qpos, expert_meta, self)

    def set_model_params(self):
        pass

    def get_obs(self):
        obs = self.get_full_obs()
        return obs

    def get_full_obs(self) -> np.ndarray:
        state = self.state
        qpos = state.joint_q.clone().cpu().numpy()
        qvel = state.joint_qd.clone().cpu().numpy()
        # transform velocity
        qvel[:3] = transform_vec(qvel[:3], qpos[3:7], self.cfg.obs_coord).ravel()
        obs = []
        # pos
        if self.cfg.obs_heading:
            obs.append(np.array([get_heading(qpos[3:7])]))
        if self.cfg.root_deheading:
            qpos[3:7] = de_heading(qpos[3:7])
        obs.append(qpos[2:])
        # vel
        if self.cfg.obs_vel == 'root':
            obs.append(qvel[:6])
        elif self.cfg.obs_vel == 'full':
            obs.append(qvel)
        # phase
        if self.cfg.obs_phase:
            phase = self.get_phase()
            obs.append(np.array([phase]))
        obs = np.concatenate(obs).squeeze()
        return obs # Should be a 1-D vector.

    # Get the position of end effectors
    # By choosing different transforms, one can choose to express it in local or global coordinate
    # This may be problematic, we need to fingerout whether using local coordinate or global one.
    def get_ee_pos(self, transform)->np.ndarray:
        with torch.no_grad():
            ee_name = ['lfoot', 'rfoot', 'lwrist', 'rwrist', 'head']
            ee_pos = []
            qpos = self.state.joint_q
            root_pos = qpos[:3].clone().cpu().numpy()
            root_q = qpos[3:7].clone().cpu().numpy()
            fk_input = dict(zip(self.joint_names[1:], qpos[7:]))
            ret_dict = self.kinematic_chain.forward_kinematics(fk_input)
            for name in ee_name:
                bone_vec = ret_dict[name].get_matrix()[:,:3,3].cpu().squeeze().numpy()
                if transform is not None:
                    bone_vec = bone_vec - root_pos
                    bone_vec = transform_vec(bone_vec, root_q, transform)
                ee_pos.append(bone_vec)
        return np.concatenate(ee_pos)

    # Idea: Stable PD controller need to computed expected acceleration..
    # TODO: Need to implement this for SPD controller
    def compute_desired_accel(self)->torch.Tensor:
        """
        Output should be a 1-D torch.tensor, 
        which has the dimension of velocity
        """
        q_accel = torch.zeros(size = self.state.joint_qd.shape, device = self.device).float()
        return q_accel

    def compute_torque(self, ctrl)->torch.Tensor:
        """
        Computed control torque using PD controller from reference pose.
        This function should be differentiable.
        Output should be a 1-D torch.tensor.
        """
        cfg = self.cfg
        dt = self.dt
        ctrl_joint = ctrl[:self.ndof] * cfg.a_scale # a_scale: 32
        qpos = self.state.joint_q.clone().to(self.device)
        qvel = self.state.joint_qd.clone().to(self.device)
        base_pos = cfg.a_ref
        target_pos = torch.from_numpy(base_pos + ctrl_joint).to(self.device)

        k_p = torch.zeros(len(qvel), device = self.device)
        k_d = torch.zeros(len(qvel), device = self.device)
        k_p[6:] = torch.from_numpy(cfg.jkp).to(self.device)
        k_d[6:] = torch.from_numpy(cfg.jkd).to(self.device)
        qpos_err = torch.cat((torch.zeros(6, device = self.device), qpos[7:] + qvel[6:]*dt - target_pos))
        qvel_err = qvel
        q_accel = self.compute_desired_accel()
        qvel_err += q_accel * dt
        torque = -k_p[6:] * qpos_err[6:] - k_d[6:] * qvel_err[6:]
        return torque

    """ RFC-Implicit """
    def rfc_implicit(self, vf)->torch.Tensor:
        vf *= self.cfg.residual_force_scale
        hq = get_heading_q(self.state.joint_q[3:7])
        residual_force = vf[:3]
        if torch.is_tensor(residual_force):
            residual_force = residual_force.cpu().numpy()
        vf[:3] = quat_mul_vec(hq, residual_force)
        # 6 Dof residual force, basically means entire root wrench is overriden
        # Is it possible to apply root force in dflex
        self.state.joint_act[:vf.shape[0]] = torch.from_numpy(vf).to(self.device)

    def do_simulation(self, action:torch.Tensor, n_frames:int):
        with torch.no_grad():
            t0 = time.time()
            cfg = self.cfg
            
            ctrl = action.squeeze()
            torque = self.compute_torque(ctrl)
            torque = torch.clip(torque, torch.from_numpy(-cfg.torque_lim).to(self.device), torch.from_numpy(cfg.torque_lim).to(self.device))
            # TODO: May need to extend actuatable joint including root
            self.state.joint_act[6:] = torque

            """ Residual Force Control (RFC) """
            if cfg.residual_force:
                vf = ctrl[-self.vf_dim:].copy()
                if cfg.residual_force_mode == 'implicit':
                    self.rfc_implicit(vf)
                else:
                    raise NotImplementedError
            self.state.joint_q = self.state.joint_q.contiguous()
            self.state.joint_qd = self.state.joint_qd.contiguous()
            self.state.joint_act = self.state.joint_act.contiguous()
            self.state = self.sim.forward(self.model, self.state, self.dt, n_frames, self.MM_caching_frequency)

    # Need to implement carefully
    def step(self, a:torch.Tensor):
        cfg = self.cfg
        # record prev state
        self.prev_qpos = self.state.joint_q.clone().cpu().numpy()
        self.prev_qvel = self.state.joint_qd.clone().cpu().numpy()
        self.prev_bquat = self.bquat.copy() # It is numpy
        # do simulation
        self.do_simulation(a, self.frame_skip)
        self.cur_t += 1
        self.bquat = self.get_body_quat()
        self.update_expert()
        # get obs
        head_pos = self.get_body_com('head')
        reward = 1.0
        if cfg.env_term_body == 'head':
            fail = self.expert is not None and head_pos[2] < self.expert['head_height_lb'] - 0.1
        else:
            fail = self.expert is not None and self.state.joint_q[2] < self.expert['height_lb'] - 0.1
        cyclic = self.expert['meta']['cyclic']
        end =  (cyclic and self.cur_t >= cfg.env_episode_len) or (not cyclic and self.cur_t + self.start_ind >= self.expert['len'] + cfg.env_expert_trail_steps)
        done = fail or end
        obs = self.get_obs()
        return obs, reward, done, {'fail': fail, 'end': end}

    def reset_model(self):
        cfg = self.cfg
        if self.expert is not None:
            ind = 0 if self.cfg.env_start_first else self.np_random.randint(self.expert['len'])
            self.start_ind = ind
            init_pose = self.expert['qpos'][ind, :].copy()
            init_vel = self.expert['qvel'][ind, :].copy()
            init_pose[7:] += self.np_random.normal(loc=0.0, scale=cfg.env_init_noise, size=len(self.state.joint_q[0]) - 7)
            self.set_state(init_pose, init_vel)
            self.bquat = self.get_body_quat()
            self.update_expert()
        else:
            init_pose = self.state.joint_q[0]
            init_pose[2] += 1.0
            self.set_state(init_pose, self.state.joint_qd[0])
        return self.get_obs()

    def viewer_setup(self, mode='human'):
        
        if self.viewer is not None:
            self.stage = Usd.Stage.CreateNew("outputs/" + "Humanoid_" + str(self.num_envs) + ".usd")

            self.renderer = df.render.UsdRenderer(self.model, self.stage)
            self.renderer.draw_points = True
            self.renderer.draw_springs = True
            self.renderer.draw_shapes = True
            self.render_time = 0.0

    def update_expert(self):
        expert = self.expert
        if expert['meta']['cyclic']:
            if self.cur_t == 0:
                expert['cycle_relheading'] = np.array([1, 0, 0, 0])
                expert['cycle_pos'] = expert['init_pos'].copy()
            elif self.get_expert_index(self.cur_t) == 0:
                expert['cycle_relheading'] = quaternion_multiply(get_heading_q(self.state.joint_q[3:7].squeeze().numpy()),
                                                              quaternion_inverse(expert['init_heading']))
                expert['cycle_pos'] = np.concatenate((self.state.joint_q[:2].squeeze().numpy(), expert['init_pos'][[2]]))


    def get_phase(self):
        ind = self.get_expert_index(self.cur_t)
        return ind / self.expert['len']

    def get_expert_index(self, t):
        return (self.start_ind + t) % self.expert['len'] \
                if self.expert['meta']['cyclic'] else min(self.start_ind + t, self.expert['len'] - 1)

    def get_expert_offset(self, t):
        if self.expert['meta']['cyclic']:
            n = (self.start_ind + t) // self.expert['len']
            offset = self.expert['meta']['cycle_offset'] * n
        else:
            offset = np.zeros(2)
        return offset

    def get_expert_attr(self, attr, ind):
        return self.expert[attr][ind, :]

    def get_bodyaddr(self):
        return {'root': (0, 7), 
                'lfemur': (7, 10), 
                'ltibia': (10, 11), 
                'lfoot': (11, 14), 
                'rfemur': (14, 17), 
                'rtibia': (17, 18), 
                'rfoot': (18, 21), 
                'upperback': (21, 24), 
                'lowerneck': (24, 27), 
                'lclavicle': (27, 29), 
                'lhumerus': (29, 32), 
                'lradius': (32, 33), 
                'rclavicle': (33, 35), 
                'rhumerus': (35, 38), 
                'rradius': (38, 39)}



