import os
import sys
import nimblephysics as nimble
sys.path.append(os.getcwd())

from khrylib.rl.envs.common import nimble_env
from gym import spaces
from khrylib.utils import *
from khrylib.utils.transformation import quaternion_from_euler
from motion_imitation.utils.tools import get_expert_nimble
import pytorch_kinematics as pk
from mujoco_py import functions as mjf
import pickle
import time
from scipy.linalg import cho_solve, cho_factor


class HumanoidNimbleEnv(nimble_env.NimbleEnv):

    def __init__(self, cfg, disable_nimble_visualizer=False):
        nimble_env.NimbleEnv.__init__(self, cfg.mujoco_model_file, 15, disable_nimble_visualizer)
        self.cfg = cfg
        self.set_cam_first = set()
        # env specific
        self.end_reward = 0.0
        self.start_ind = 0
        self.body_qposaddr = get_body_qposaddr(self.mj_model)
        self.bquat = self.get_body_quat()
        self.prev_bquat = None
        self.expert = None
        print(self.mj_model.joint_names)
        print("Number of Joints:", len(self.mj_model.joint_names))
        ts = time.time()
        self.load_expert()
        #input("Press Enter to Continue")
        print(f"Take {time.time()-ts} to load expert")
        input("Press Enter to Continue")
        self.set_spaces()

    def load_expert(self):
        expert_qposes = []
        expert_metas = []
        for expert_traj_file in self.cfg.expert_traj_files:
            expert_qpos, expert_meta = pickle.load(open(expert_traj_file, "rb"))
            expert_qposes.append(expert_qpos)
            expert_metas.append(expert_meta)
        self.experts = []
        for i in range(len(expert_qposes)):
            self.experts.append(get_expert_nimble(expert_qposes[i], expert_metas[i], self))
        self.expert = self.experts[0]
        self.expert_id = 0

    def set_spaces(self):
        cfg = self.cfg
        self.ndof = self.mj_model.actuator_ctrlrange.shape[0]
        self.vf_dim = 0
        if cfg.residual_force:
            if cfg.residual_force_mode == 'implicit':
                self.vf_dim = 6
            else:
                if cfg.residual_force_bodies == 'all':
                    self.vf_bodies = self.mj_model.body_names[1:]
                else:
                    self.vf_bodies = cfg.residual_force_bodies
                self.body_vf_dim = 6 + cfg.residual_force_torque * 3
                self.vf_dim = self.body_vf_dim * len(self.vf_bodies)
        self.action_dim = self.ndof + self.vf_dim
        self.action_space = spaces.Box(low=-np.ones(self.action_dim), high=np.ones(self.action_dim), dtype=np.float32)
        self.obs_dim = self.get_obs().size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def get_obs(self):
        data = self.data
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
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
        obs = np.concatenate(obs)
        return obs

    def get_diff_obs(self, state:torch.Tensor)->torch.Tensor:
        self.sync_mujoco_with_state(state)
        obs = torch.as_tensor(self.get_obs())
        return obs

    def get_ee_pos(self, transform):

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

    def get_body_quat(self):
        self.sync_mujoco()
        qpos = self.data.qpos.copy()
        body_quat = [qpos[3:7]]
        for body in self.mj_model.body_names[1:]:
            if body == 'root' or ((not body in self.body_qposaddr) and (body not in ['lfoot','rfoot'])):
                continue
            euler = np.zeros(3)
            if not body in ['lfoot','rfoot']:
                start, end = self.body_qposaddr[body]
                euler[:end - start] = qpos[start:end]
            quat = quaternion_from_euler(euler[0], euler[1], euler[2])
            body_quat.append(quat)
        body_quat = np.concatenate(body_quat)
        return body_quat

    def get_com(self):
        self.sync_mujoco()
        return self.data.subtree_com[0, :].copy()

    # Stable-PD control
    def compute_desired_accel(self, qpos_err, qvel_err, k_p, k_d):
        dt = self.mj_model.opt.timestep
        nv = self.mj_model.nv
        M = np.zeros(nv * nv)
        mjf.mj_fullM(self.mj_model, M, self.data.qM)
        M.resize(self.mj_model.nv, self.mj_model.nv)
        C = self.data.qfrc_bias.copy()
        #print("M:\n",M)
        #print("C:\n",C[6:])
        K_p = np.diag(k_p)
        K_d = np.diag(k_d)
        q_accel = cho_solve(cho_factor(M + K_d*dt, overwrite_a=True, check_finite=False),
                            -C[:, None] - K_p.dot(qpos_err[:, None]) - K_d.dot(qvel_err[:, None]), overwrite_b=True, check_finite=False)
        return q_accel.squeeze()

    def compute_desired_accel_nimble(self, qpos_err, qvel_err, k_p, k_d, state):
        dt = self.world.getTimeStep()
        # Get M matrix from nimble
        self.world.setState(state.detach().numpy())
        M = torch.from_numpy(self.world.getMassMatrix())
        C = torch.from_numpy(self.world.getCoriolisAndGravityAndExternalForces())
        #print("M:\n",M)
        #print("C:\n",C[6:][self.n2m])
        K_p = torch.diag(k_p)
        K_d = torch.diag(k_d)
        q_accel = torch.inverse(M+K_d*dt).matmul(-C-K_p.matmul(qpos_err)-K_d.matmul(qvel_err))
        return q_accel.squeeze()

    def compute_torque(self, ctrl):
        cfg = self.cfg
        dt = self.mj_model.opt.timestep
        ctrl_joint = ctrl[:self.ndof] * cfg.a_scale[self.mj_nonfoot]
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        base_pos = cfg.a_ref[self.mj_nonfoot]
        target_pos = base_pos + ctrl_joint
        #print("Target pos:", target_pos)
        k_p = np.zeros(qvel.shape[0])
        k_d = np.zeros(qvel.shape[0])
        # Dimension of cfg.jkp should equals to actuated joint number.
        KP = cfg.jkp[self.mj_nonfoot]
        KD = cfg.jkd[self.mj_nonfoot]
        k_p[6:] = KP
        k_d[6:] = KD
        qpos_err = np.concatenate((np.zeros(6), qpos[7:] + qvel[6:]*dt - target_pos))
        #print("Pos Err:",qpos_err)
        qvel_err = qvel
        q_accel = self.compute_desired_accel(qpos_err, qvel_err, k_p, k_d)
        qvel_err += q_accel * dt
        torque = -KP * qpos_err[6:] - KD * qvel_err[6:]
        return torque # Should be in mujoco frame

    def compute_torque_diff(self, ctrl: torch.Tensor, state: torch.Tensor)->torch.Tensor:
        """
        ctrl: reference pos in mujoco
        state: nimble state
        """
        cfg = self.cfg
        dt = self.world.getTimeStep()
        ctrl_joint = ctrl[:self.ndof][self.m2n] * torch.from_numpy(cfg.a_scale[self.mj_nonfoot][self.m2n])
        qpos = state[:len(state)//2]
        qvel = state[len(state)//2:]
        base_pos = torch.from_numpy(cfg.a_ref[self.mj_nonfoot][self.m2n])
        target_pos = base_pos + ctrl_joint # Are they have the same pos base?
        #print("Target pos:", target_pos[self.n2m])
        k_p = torch.zeros(qvel.shape[0])
        k_d = torch.zeros(qvel.shape[0])
        KP = torch.from_numpy(cfg.jkp[self.mj_nonfoot][self.m2n])
        KD = torch.from_numpy(cfg.jkd[self.mj_nonfoot][self.m2n])
        k_p[6:] = KP
        k_d[6:] = KD
        qpos_err = torch.cat([torch.zeros(6), qpos[6:]+qvel[6:]*dt-target_pos])
        #print("Pos Err:",qpos_err[6:][self.n2m])
        qvel_err = qvel
        q_accel = self.compute_desired_accel_nimble(qpos_err, qvel_err, k_p, k_d, state)
        qvel_err = q_accel * dt + qvel_err
        torque = -KP * qpos_err[6:] - KD * qvel_err[6:]
        return torque

    """ RFC-Implicit """
    def rfc_implicit(self, vf):
        vf *= self.cfg.residual_force_scale
        hq = get_heading_q(self.data.qpos[3:7])
        vf[:3] = quat_mul_vec(hq, vf[:3])
        self.current_control_force[:vf.shape[0]] = vf[[3,4,5,0,1,2]] * 0.1 # Set the root force

    def rfc_implicit_diff(self, vf, state):
        vf *= torch.from_numpy(self.cfg.residual_force_scale)
        hq = torch.tensor([0., 0., state[5]]) # yaw
        vf[:3] = pk.euler_angle_to_matrix(hq,convention="XYZ").matmul(vf[:3])
        return vf[[3,4,5,0,1,2]] * 0.1

    def do_simulation(self, action, n_frames):
        t0 = time.time()
        cfg = self.cfg
        for i in range(n_frames):
            ctrl = action
            assert(cfg.action_type!='torque')
            torque = self.compute_torque_diff(torch.as_tensor(ctrl), torch.as_tensor(self.world.getState())).numpy()
            torque = np.clip(torque, -cfg.torque_lim[self.mj_nonfoot][self.m2n], cfg.torque_lim[self.mj_nonfoot][self.m2n])
            self.current_control_force[6:] = torque * 0.2

            """ Residual Force Control (RFC) """
            if cfg.residual_force:
                vf = ctrl[-self.vf_dim:].copy()
                if cfg.residual_force_mode == 'implicit':
                    self.rfc_implicit(vf)
                else:
                    raise NotImplementedError
            #print("Torque:\n",torque[self.m2n]*0.2)
            #print("State:\n", self.world.getState())
            self.world.setControlForces(self.current_control_force)
            self.world.step()
            self.sync_mujoco()
            #input("Press Enter to continue")
        # self.gui.displayState(torch.from_numpy(self.world.getState()))
       
        if self.viewer is not None:
            self.viewer.sim_time = time.time() - t0

    def diff_step(self, action:torch.Tensor, state:torch.Tensor,return_done=False)->torch.Tensor:
        for _ in range(self.frame_skip):
            ctrl = action
            assert(self.cfg.action_type!='torque')
            torque = self.compute_torque_diff(ctrl, state)
            torque = torque.clamp(torch.from_numpy(-self.cfg.torque_lim[self.mj_nonfoot][self.m2n]),
                                  torch.from_numpy(self.cfg.torque_lim[self.mj_nonfoot][self.m2n]))*0.2
            #print("Torque:\n",torque)
            #print("State:\n", state.numpy())
            if self.cfg.residual_force:
                assert(self.cfg.residual_force_mode == 'implicit')
                vf = self.rfc_implicit_diff(ctrl[-self.vf_dim:],state)
            else:
                vf = torch.zeros(6)
            act = torch.cat([vf, torque])
            state = nimble.timestep(self.world, state, act)
            #print(state)
            self.sync_mujoco_with_state(state)
            #input("Press Enter to continue")
        head_pos = self.get_body_com("head")
        if self.cfg.env_term_body == "head":
            fail = self.expert is not None and head_pos[2] < self.expert['head_height_lb'] - 0.1
        else:
            fail = self.expert is not None and self.data.qpos[2] < self.expert['height_lb'] - 0.1
        if return_done:
            return state, fail
        else:
            return state
            
    def step(self, a, record_data = False):
        cfg = self.cfg
        # record prev state keep using mujoco
        self.prev_qpos = self.data.qpos.copy()
        self.prev_qvel = self.data.qvel.copy()
        self.prev_bquat = self.bquat.copy()
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
            fail = self.expert is not None and self.data.qpos[2] < self.expert['height_lb'] - 0.1
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
            #print("Shapes:", init_pose.shape, init_vel.shape)
            init_pose[7:] += self.np_random.normal(loc=0.0, scale=cfg.env_init_noise, size=self.mj_model.nq - 7)
            self.set_state(init_pose, init_vel)
            self.bquat = self.get_body_quat()
            self.update_expert()
        else:
            init_pose = self.data.qpos
            init_pose[2] += 1.0
            self.set_state(init_pose, self.data.qvel)
        return self.get_obs()

    def viewer_setup(self, mode):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.lookat[:2] = self.data.qpos[:2]
        if mode not in self.set_cam_first:
            self.viewer.video_fps = 33
            self.viewer.frame_skip = self.frame_skip
            self.viewer.cam.distance = self.mj_model.stat.extent * 1.2
            self.viewer.cam.elevation = -20
            self.viewer.cam.azimuth = 45
            self.set_cam_first.add(mode)

    def update_expert(self):
        expert = self.expert
        if expert['meta']['cyclic']:
            if self.cur_t == 0:
                expert['cycle_relheading'] = np.array([1, 0, 0, 0])
                expert['cycle_pos'] = expert['init_pos'].copy()
            elif self.get_expert_index(self.cur_t) == 0:
                expert['cycle_relheading'] = quaternion_multiply(get_heading_q(self.data.qpos[3:7]),
                                                              quaternion_inverse(expert['init_heading']))
                expert['cycle_pos'] = np.concatenate((self.data.qpos[:2], expert['init_pos'][[2]]))


    def get_phase(self):
        ind = self.get_expert_index(self.cur_t)
        return ind / self.expert['len']

    def get_expert_index(self, t):
        return (self.start_ind + t) % self.expert['len'] \
                if self.expert['meta']['cyclic'] else min(self.start_ind + t, self.expert['len'] - 1)

    def get_expert_pos_indices(self,ts):
        poses = []
        for t in ts:
            qpos = self.get_expert_attr("qpos",self.get_expert_index(t))
            # TODO: Verify whether it is correct
            if self.expert['meta']['cyclic']:
                init_pos = self.expert['init_pos']
                cycle_h = self.expert['cycle_relheading']
                cycle_pos = self.expert['cycle_pos']
                qpos[:3] = quat_mul_vec(cycle_h, qpos[:3] - init_pos) + cycle_pos
                qpos[3:7] = quaternion_multiply(cycle_h, qpos[3:7])
            nimble_qpos = self.to_nimble_state(pos=qpos)
            poses.append(nimble_qpos)
        return torch.as_tensor(np.vstack(poses))
        
    def get_expert_offset(self, t):
        if self.expert['meta']['cyclic']:
            n = (self.start_ind + t) // self.expert['len']
            offset = self.expert['meta']['cycle_offset'] * n
        else:
            offset = np.zeros(2)
        return offset

    def get_expert_attr(self, attr, ind):
        return self.expert[attr][ind, :]

    def getState(self):
        return torch.from_numpy(self.world.getState())

    def get_goal(self):
        return self.get_expert_attr("qpos", self.expert['len']-1)

    def switch_expert(self, idx=None):
        if idx is not None:
            self.expert = self.experts[idx]
            self.expert_id = idx
        else:
            self.expert_id = (self.expert_id + 1)%len(self.experts)
            self.expert = self.experts[self.expert_id]
