import os
import sys
sys.path.append(os.getcwd())

from khrylib.rl.envs.common import dflex_env
from gym import spaces
from khrylib.utils import *
from khrylib.rl.utils import torch_utils as tu
from khrylib.utils import load_utils as lu
from motion_imitation.utils.tools import get_expert_dflex

from khrylib.utils.transformation import quaternion_from_euler

import dflex as df
try:
    from pxr import Usd
except:
    print("No pxr package")
import pickle
import time

class HumanoidDFlexEnv(dflex_env.DflexEnv):

    def __init__(self, cfg, **kwargs):
        dflex_env.DflexEnv.__init__(self, cfg.mujoco_model_file, 48, **kwargs)
        self.vf_dim = 6
        self.cfg = cfg
        self.set_cam_first = set()
        # env specific
        self.end_reward = 0.0
        self.start_ind = 0
        self.body_qposaddr = self.get_bodyaddr()
        self.bquat = self.get_body_quat()
        self.prev_bquat = None
        self.set_model_params()
        self.expert = None
        st = time.time()
        self.load_expert()
        print(f"Take {time.time() - st} to load expert")
        #input("Press Enter to Continue")

    def build_model(self, fullpath):
        """
        Should build dflex model based on fullpath mjcf.
        return:
        model, integrator(sim), state, dt
        """
        builder = df.sim.ModelBuilder()
        dt = 1.0 / (60.0 * self.frame_skip)
        
        ground = True

        # TODO: Need closer inspection to see whether it is correct number or not.
        start_rot = df.quat_from_axis_angle((1.0, 0.0, 0.0), 0)

        # The world directional convension are hard-coded?

        start_pos = []

        # TODO: Need closer inspection to see whether it is correct number or not.

        asset_folder = os.path.join(os.path.dirname(__file__), '../../khrylib/assets/mujoco_models')

        # Load MJCF model
        lu.parse_mjcf(os.path.join(asset_folder, "rfc_humanoid.xml"), builder,
        stiffness = 0.0,
        damping=0.1,
        contact_ke = 2.e+4,
        contact_kd = 5.e+3,
        contact_kf = 1.e+3,
        contact_mu = 0.75,
        limit_ke = 1.e+3,
        limit_kd = 1.e+1,
        armature = 0.007,
        load_stiffness = True,
        load_armature=True)

        start_pos_z = 3.0
        start_pos.append([0.0, 0.0, start_pos_z])

        builder.joint_q[:3] = start_pos[-1]
        builder.joint_q[3:7] = start_rot
        print("Builder q:", builder.joint_q, type(self.device), self.device)
        
        model = builder.finalize(self.device)
        model.ground = ground
        model.gravity = torch.tensor((0.0, 0, -9.81), dtype = torch.float32, device = torch.device(self.device))
        self.body_mass = builder.body_mass
        
        sim = df.sim.SemiImplicitIntegrator()
        state = model.state() # Should be initial state
        # Seems to be an invalid number...
        print("Start q:", state.joint_q)
        #input()
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
        self.set_mj_state(self.state.joint_q.detach().numpy().copy(),
                       self.state.joint_qd.detach().numpy().copy())
        data = self.data
        ee_name = ['lfoot', 'rfoot', 'lwrist', 'rwrist', 'head']
        ee_pos = []
        root_pos = data.qpos[:3]
        root_q = data.qpos[3:7].copy()
        for name in ee_name:
            bone_id = self.mj_model._body_name2id[name]
            bone_vec = self.data.body_xpos[bone_id]
            # In heading mode the vector is only
            if transform is not None: # transform is a string, not a transformation matrix
                bone_vec = bone_vec - root_pos # Each joint's relative position to root
                bone_vec = transform_vec(bone_vec, root_q, transform) # Express different bone's location in local frame
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
        # with torch.no_grad():
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
        # self.state.joint_q = self.state.joint_q
        # self.state.joint_qd = self.state.joint_qd
        #self.state.joint_act = torch.zeros_like(self.state.joint_act)
        # print("Action:",self.state.joint_act, self.state.joint_act.shape)
        # input()
        # print(self.state.joint_q, self.state.joint_qd)
        # input()
        self.state = self.sim.forward(self.model, self.state, 1.0/60.0, 48, self.MM_caching_frequency)
        #print("q:",self.state.joint_q)

    # Need to implement carefully
    def step(self, a:torch.Tensor):
        cfg = self.cfg
        # record prev state
        self.prev_qpos = self.state.joint_q.clone().cpu().numpy()
        self.prev_qvel = self.state.joint_qd.clone().cpu().numpy()
        self.prev_bquat = self.bquat.copy() # It is numpy
        # do simulation
        self.do_simulation(a, self.frame_skip)
        self.set_mj_state(self.state.joint_q.detach().numpy().copy(),
                          self.state.joint_qd.detach().numpy().copy())
        self.cur_t += 1
        self.bquat = self.get_body_quat()
        self.update_expert()
        # get obs
        head_pos = self.get_body_com('head')
        #print("Head Pos:", head_pos)
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

    # This function is problematic
    def reset_model(self):
        cfg = self.cfg
        if self.expert is not None:
            ind = 0 if self.cfg.env_start_first else self.np_random.randint(self.expert['len'])
            self.start_ind = ind
            init_pose = self.expert['qpos'][ind, :].copy()
            init_pose[2] += 1.5
            init_vel = self.expert['qvel'][ind, :].copy()
            init_pose[7:] += self.np_random.normal(loc=0.0, scale=cfg.env_init_noise, size=len(self.state.joint_q) - 7)
            self.set_state(torch.from_numpy(init_pose), torch.from_numpy(init_vel))
            self.bquat = self.get_body_quat()
            self.update_expert()
        else: # Why?
            self.set_state(self.init_qpos, self.init_qvel)
        return self.get_obs()

    def viewer_setup(self, mode):
        self.mj_viewer.cam.trackbodyid = 1
        self.mj_viewer.cam.lookat[:2] = self.data.qpos[:2]
        if mode not in self.set_cam_first:
            self.mj_viewer.video_fps = 33
            self.mj_viewer.frame_skip = self.frame_skip
            self.mj_viewer.cam.distance = self.model.stat.extent * 1.2
            self.mj_viewer.cam.elevation = -20
            self.mj_viewer.cam.azimuth = 45
            self.set_cam_first.add(mode)

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

    def get_body_quat(self)->np.ndarray:
        self.set_mj_state(self.state.joint_q.detach().numpy().copy(), 
                       self.state.joint_qd.detach().numpy().copy())
        qpos = self.data.qpos.copy()
        body_quat = [qpos[3:7]]
        for body in self.mj_model.body_names[1:]:
            if body == 'root' or not body in self.body_qposaddr:
                continue
            start, end = self.body_qposaddr[body]
            euler = np.zeros(3)
            euler[:end - start] = qpos[start:end]
            quat = quaternion_from_euler(euler[0], euler[1], euler[2])
            body_quat.append(quat)
        body_quat = np.concatenate(body_quat)
        return body_quat

    def get_com(self)->np.ndarray:
        self.set_mj_state(self.state.joint_q.detach().numpy().copy(),
                       self.state.joint_qd.detach().numpy().copy())
        return self.data.subtree_com[0,:].copy()

    # All body attached to flexible joints
    # Dicard all fixed joint and fixed bodies.
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



