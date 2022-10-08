import os
import mujoco_py
import copy
import numpy as np
import torch
import dflex as df
from gym import spaces
from gym.utils import seeding
from pathlib import Path
from khrylib.rl.envs.common.mjviewer import MjViewer

DEFAULT_SIZE = 500

class DflexEnv:
    def __init__(self, fullpath, frame_skip, **kwargs):
        fullpath = "rfc_humanoid.xml"
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
        self.body_names = None

        self.viewer = None

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

        # Initialize mujoco sim
        self.initialize_mujoco_sim("/home/ericcsr/oneshot/RFC/khrylib/assets/mujoco_models/mocap_v2.xml")
        self.prev_qpos = None
        self.prev_qvel = None

        self.seed()

    def initialize_mujoco_sim(self, fullpath):
        self.mj_model = mujoco_py.load_model_from_path(fullpath)
        self.mj_sim = mujoco_py.MjSim(self.mj_model)
        self.data = self.mj_sim.data
        self.mj_viewer = None
        self._mj_viewers = {}
        self.init_qpos_mj = self.init_qpos.detach().numpy()
        self.init_qvel_mj = self.init_qvel.detach().numpy()


    # Which have similar function to set space
    def initialize_properties(self, **kwargs):
        self.no_grad = kwargs["no_grad"]
        df.config.no_grad = kwargs["no_grad"]
        self.device = kwargs["device"]
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

    def reset_mujoco(self):
        self.mj_sim.reset()
        ob = self.reset_model() # Unimplemented
        old_viewer = self.mj_viewer
        for mode, v in self._mj_viewers.items():
            self.mj_viewer = v
            self.viewer_setup(mode)
        self.mj_viewer = old_viewer
        return ob

    def reset(self):
        envs_id = torch.arange(self.num_environments, dtype=torch.long, device=self.device)
        self.cur_t = 0
        #self.reset_model()
        self.state.joint_q[:] = self.init_qpos.clone()
        self.state.joint_qd[:] = self.init_qvel.clone()
        self.progress_buf[envs_id] = 0
        self.get_obs()
        mj_obs = self.reset_mujoco()
        return self.obs_buf.squeeze()

    def set_state(self, qpos, qvel):
        assert qpos.shape == self.init_qpos.shape
        assert qvel.shape == self.init_qvel.shape
        qpos_dflex = qpos.clone()
        qpos_dflex[[3,4,5,6]] = qpos_dflex[[4,5,6,3]]
        self.state.joint_q[:] = qpos_dflex.clone()
        self.state.joint_qd[:] = qvel.clone()
        if torch.is_tensor(qpos):
            qpos = qpos.detach().numpy().copy()
        if torch.is_tensor(qvel):
            qvel = qvel.detach().numpy().copy()
        self.set_mj_state(qpos, qvel)

    def set_mj_state(self, qpos, qvel):
        '''
        set mujoco sim state from numpy arrays
        '''
        old_state = self.mj_sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.mj_sim.set_state(new_state)
        self.mj_sim.forward()
        self.data = self.mj_sim.data

    def denormalize_action(self, action):
        raise DeprecationWarning
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
        if mode == 'image':
            self._get_viewer(mode).render(width, height)
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            return data[::-1, :, [2, 1, 0]]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def close(self):
        if self.mj_viewer is not None:
            self.mj_viewer = None
            self._mj_viewers = {}

    def _get_viewer(self, mode):
        self.mj_viewer = self._mj_viewers.get(mode)
        if self.mj_viewer is None:
            if mode == 'human':
                self.mj_viewer = MjViewer(self.sim)
            elif mode == 'image':
                self.mj_viewer = mujoco_py.MjRenderContextOffscreen(self.mj_sim, 0)
            self._mj_viewers[mode] = self.mj_viewer
        self.viewer_setup(mode)
        return self.mj_viewer

    def set_custom_key_callback(self, key_func):
        self._get_viewer('human').custom_key_callback = key_func

    def get_body_com(self, body_name)->np.ndarray:
        self.set_mj_state(self.state.joint_q.detach().numpy().copy(), 
                          self.state.joint_qd.detach().numpy().copy())
        return self.data.get_body_xpos(body_name)
        
    def state_vector(self)->np.ndarray:
        return np.hstack([self.state.joint_q.detach().numpy(), 
                          self.state.joint_qd.detach().numpy()])

    # only express the orientation of a vector in world
    def vec_body2world(self, body_name, vec)->np.ndarray:
        self.set_mj_state(self.state.joint_q.detach().numpy().copy(), 
                          self.state.joint_qd.detach().numpy().copy())
        body_xmat = self.data.get_body_xmat(body_name)
        vec_world = (body_xmat @ vec[:,None]).ravel()
        return vec_world

    def pos_body2world(self, body_name, pos):
        self.set_mj_state(self.state.joint_q.detach().numpy().copy(), 
                          self.state.joint_qd.detach().numpy().copy())
        body_xpos = self.data.get_body_xpos(body_name)
        body_xmat = self.data.get_body_xmat(body_name)
        pos_world = (body_xmat @ pos[:, None]).ravel() + body_xpos
        return pos_world

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



            


    
