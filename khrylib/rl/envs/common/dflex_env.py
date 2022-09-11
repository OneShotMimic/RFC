import os
import sys
import copy
import numpy as np
import torch
import dflex as df
from gym import spaces
from gym.utils import seeding
import pytorch_kinematics as pk
from pathlib import Path

DEFAULT_SIZE = 500

class DflexEnv:
    def __init__(self, fullpath, frame_skip, **kwargs):
        fullpath = "mocap_v2_dflex.xml"
        if not  os.path.exists(fullpath):
            fullpath = os.path.join(Path(__file__).parent.parent.parent.parent, 'assets/mujoco_models', os.path.basename(fullpath))
            if not os.path.exists(fullpath):
                raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        # Should create observation space as well as action space
        # Different environment may have different initialization method
        self.initialize_properties(**kwargs)
        # Need something similar to init_sim
        # model is model and sim is integrator
        # This build model function should be implemented in subclass not this class
        # Should also include self.dt
        self.model, self.sim, self.state, self.dt = self.build_model(fullpath) # state is initial state
        self.kinematic_chain = pk.build_chain_from_mjcf(open(fullpath).read())
        self.kinematic_chain.to(dtype=torch.float32, device = self.device)
        self.joint_names = self.kinematic_chain.get_joint_parameter_names()
        self.body_names = None

        self.viewer = None
        self._viewers = {}

        self.obs_dim = None
        self.action_space = None
        self.observation_space = None

        self.np_random = None

        # Initialize spaces
        self.observation_space = spaces.Box(np.ones(self.num_observations) * -np.inf, np.ones(self.num_observations)*np.Inf)
        self.action_space = spaces.Box(-np.ones(self.num_actions), np.ones(self.num_actions))
        # Parameters related to simulation
        self.cur_t = 0
        self.num_frames = 0
        self.num_agents = 1
        self.meta_data = {
            "render.modes": ['human', 'rgb_array'],
            "video.frames_per_second": int(np.round(1.0/self.dt))
        }
        # Need to get initial pose from simulator
        # Should come from builder
        # we should keep in mind that the major difference is DFlex is a batch of simulators
        self.init_qpos = self.state.joint_q.cpu().clone()
        self.init_qvel = self.state.joint_qd.cpu().clone()

        self.prev_qpos = None
        self.prev_qvel = None

        self.seed()

    # Which have similar function to set space
    def initialize_properties(self, **kwargs):
        self.no_grad = kwargs["no_grad"]
        self.device = kwargs["device"]
        df.config.no_grad = kwargs["no_grad"]

        self.num_environments = 1
        assert(self.num_environments == 1)
        self.MM_caching_frequency = kwargs["MM_caching_frequency"]

        self.num_observations = kwargs["num_obs"]
        self.ndof = 32
        self.num_actions = kwargs["num_act"] # Should be 38

        # TODO: (Eric) Need a method to get real action bound
        # Buffer fo torch 
        self.obs_buf = torch.zeros((self.num_environments, self.num_observations), device = self.device, requires_grad=False).float()
        self.rew_buf = torch.zeros(self.num_environments, device = self.device, requires_grad=False).float()
        self.reset_buf = torch.zeros(self.num_environments, device=self.device, dtype=torch.long, requires_grad=False)
        self.termination_buf = torch.zeros(self.num_environments, device=self.device, dtype=torch.long, requires_grad=False)
        self.progress_buf = torch.zeros(self.num_environments, device=self.device, dtype=torch.long, requires_grad=False)
        self.actions = torch.zeros((self.num_environments, self.num_actions), device = self.device, requires_grad=False).float()
        self.extras = {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        raise NotImplementedError

    def reset_model(self):
        raise NotImplementedError

    def viewer_setup(self, mode):
        pass

    def reset(self):
        envs_id = torch.arange(self.num_environments, dtype=torch.long, device=self.device)
        self.cur_t = 0
        self.state.joint_q = self.init_qpos.clone()
        self.state.joint_qd = self.init_qvel.clone()
        self.progress_buf[envs_id] = 0
        self.get_obs()
        return self.obs_buf

    def set_state(self, qpos, qvel):
        assert qpos.shape == self.init_qpos[0].shape
        assert qvel.shape == self.init_qvel[0].shape
        self.state.joint_q[0] = qpos
        self.state.joint_qd[0] = qvel

    def denormalize_action(self, action):
        neutral_action = (self.act_high+self.act_low) / 2
        scale = (self.act_high - self.act_low)/2
        action += 1
        return scale * action + neutral_action


    # Here assume there is no external force applied on root joint
    def do_simulation(self, ctrl, n_frames):
        # Set same action and integration for n_frames
        # Assume substep is equivalant to frame skip
        raise DeprecationWarning
        action = ctrl.clamp(-1,1)
        self.state.joint_act[6:] = action
        self.state = self.sim.forward(self.model, self.state, self.dt, n_frames, self.MM_caching_frequency)

    # TODO: (Eric) Implement render related features
    def render(self, mode="human", width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        pass

    def close(self):
        pass

    def _get_viewer(self, mode):
        pass

    def set_custom_key_callback(self, key_func):
        pass

    def get_body_com(self, body_name)->np.ndarray:
        # TODO: (Eric) need to ensure some order between joint angle and names
        joint_angle = self.state.joint_q
        fk_input = dict(zip(self.joint_names[1:], joint_angle[7:]))
        return self.kinematic_chain.forward_kinematics(fk_input)[body_name].get_matrix()[:,:3, 3].detach().cpu().numpy()

    def state_vector(self)->np.ndarray:
        return np.hstack([self.state.joint_q, self.state.joint_qd])

    # only express the orientation of a vector in world
    def vec_body2world(self, body_name, vec)->np.ndarray:
        joint_angle = self.state.joint_q.clone()
        fk_input = dict(zip(self.joint_names[1:], joint_angle[7:]))
        rot = self.kinematic_chain.forward_kinematics(fk_input)[body_name].get_matrix().detach().cpu().numpy()[:,:3,:3]
        vec_world = (rot @ vec[:,None]).ravel()
        return vec_world

    def pos_body2world(self, body_name, pos):
        return self.get_body_com(body_name) + self.vec_body2world(body_name, pos)

    # TODO: Matrix multiplication is suspecious..
    # Notice that since we treat free joint as fixed joint, even root body's rotation need to rotated again.
    def get_body_quat(self)->np.ndarray:
        with torch.no_grad():
            joint_angle = copy.deepcopy(self.state.joint_q).to(self.device)
            root_rot = pk.quaternion_to_matrix(joint_angle[3:7])
            fk_input = dict(zip(self.joint_names[1:], joint_angle[7:])) # Need to exclude fixed joint.
            ret_dict = self.kinematic_chain.forward_kinematics(fk_input)
            if self.body_names is None:
                body_names = list(ret_dict.keys())
                self.body_names = self.extract_proper_names(body_names)
            quat_list = []
            for bodyname in self.body_names:
                if bodyname not in self.get_bodyaddr():
                    continue
                rot = ret_dict[bodyname].get_matrix()[:,:3,:3].squeeze()
                quat = pk.matrix_to_quaternion(root_rot @ rot).detach().cpu().numpy()
                quat_list.append(quat)
        return np.concatenate(quat_list)

    def get_com(self)->np.ndarray:
        with torch.no_grad():
            qpos = self.state.joint_q
            body_mass = self.body_mass
            root_pos = qpos[:3].cpu().numpy()
            root_quat = qpos[3:7]
            fk_input = dict(zip(self.joint_names[1:], qpos[7:]))
            ret_dict = self.kinematic_chain.forward_kinematics(fk_input)
            if self.body_names is None:
                body_names = list(ret_dict.keys())
                self.body_names = self.extract_proper_names(body_names)
            total_mass = sum(body_mass)
            weighted_pos = np.zeros(3)
            for i, bodyname in enumerate(self.body_names):
                rel_pos = ret_dict[bodyname].get_matrix()[:,:3,3].squeeze()
                abs_pos = pk.quaternion_apply(root_quat, rel_pos).detach().cpu().numpy() + root_pos
                weighted_pos += abs_pos * body_mass[i]
            weighted_pos /= total_mass
        return weighted_pos

    def extract_proper_names(self,body_names):
        proper_name = []
        for name in body_names:
            if name.endswith("child"):
                continue
            else:
                proper_name.append(name)
        return proper_name

    def get_bodyaddr(self):
        raise NotImplementedError



            


    
