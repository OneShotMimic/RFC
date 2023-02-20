from gym import spaces
from gym.utils import seeding
from khrylib.utils import mujoco
import numpy as np
from os import path
from pathlib import Path
import mujoco_py
import nimblephysics as nimble
import pytorch_kinematics as pk
import torch
from khrylib.rl.envs.common.mjviewer import MjViewer

DEFAULT_SIZE = 500
PORT_NUM = 8090

mujoco_joints = ['lfemur_z', 'lfemur_y', 'lfemur_x', 'ltibia_x',
                 'lfoot_z', 'lfoot_y', 'lfoot_x', 'rfemur_z', 
                 'rfemur_y', 'rfemur_x', 'rtibia_x', 'rfoot_z', 
                 'rfoot_y', 'rfoot_x', 'upperback_z', 'upperback_y', 
                 'upperback_x', 'lowerneck_z', 'lowerneck_y', 
                 'lowerneck_x', 'lclavicle_z', 'lclavicle_y', 
                 'lhumerus_z', 'lhumerus_y', 'lhumerus_x', 'lradius_x', 
                 'rclavicle_z', 'rclavicle_y', 'rhumerus_z', 'rhumerus_y', 
                 'rhumerus_x', 'rradius_x']

mujoco_joints_no_foot = ['lfemur_z', 'lfemur_y', 'lfemur_x', 'ltibia_x', 'rfemur_z', 
                 'rfemur_y', 'rfemur_x', 'rtibia_x', 'upperback_z', 'upperback_y', 
                 'upperback_x', 'lowerneck_z', 'lowerneck_y', 
                 'lowerneck_x', 'lclavicle_z', 'lclavicle_y', 
                 'lhumerus_z', 'lhumerus_y', 'lhumerus_x', 'lradius_x', 
                 'rclavicle_z', 'rclavicle_y', 'rhumerus_z', 'rhumerus_y', 
                 'rhumerus_x', 'rradius_x']

nimble_joints = ['upperback_z','upperback_y','upperback_x','lowerneck_z',
                 'lowerneck_y','lowerneck_x','lclavicle_z','lclavicle_y',
                 'lhumerus_z','lhumerus_y','lhumerus_x','lradius_x',
                 'rclavicle_z','rclavicle_y','rhumerus_z','rhumerus_y',
                 'rhumerus_x','rradius_x','lfemur_z','lfemur_y','lfemur_x',
                 'ltibia_x','lfoot_z','lfoot_y','lfoot_x',
                 'rfemur_z','rfemur_y','rfemur_x','rtibia_x',
                 'rfoot_z','rfoot_y','rfoot_x']

#FOOT_JOINTS = ["lfoot_x", "lfoot_y", "lfoot_z", "rfoot_x", "rfoot_y", "rfoot_z"]
FOOT_JOINTS = []

class NimbleEnv:
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, fullpath, frame_skip, disable_nimble_visualizer=False):
        if not path.exists(fullpath):
            # try the default assets path
            fullpath = path.join(Path(__file__).parent.parent.parent.parent, 'assets/mujoco_models', path.basename(fullpath))
            if not path.exists(fullpath):
                raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        # load mujoco_model
        self.mj_model = mujoco_py.load_model_from_path(fullpath)
        self.mj_sim = mujoco_py.MjSim(self.mj_model)
        # load nimble model
        print("Mujoco Joint:",self.mj_model.joint_names)
        self.world = nimble.simulation.World()
        self.robot = self.world.loadSkeleton(path.join(Path(__file__).parent.parent.parent.parent, 'assets/mujoco_models/rfc_model', path.basename("humanoid_root.urdf")))
        for i in range(self.robot.getNumJoints()):
            print(self.robot.getJoint(i).getName())
            if i > 6: # Except root joint
                self.robot.getJoint(i).setDampingCoefficient(0, 0.1)
                #self.robot.getJoint(i).setSpringStiffness(0, 5)
                #self.robot.getJoint(i).setPositionLimitEnforced(True)
        # Set Body Node friction
        self.robot.getBodyNode("lfoot").setFrictionCoeff(20.0)
        self.robot.getBodyNode("rfoot").setFrictionCoeff(20.0)
        self.robot.getBodyNode("thorax").setFrictionCoeff(20.0)
        self.robot.getBodyNode("lowerneck").setFrictionCoeff(20.0)
        self.robot.getBodyNode("lwrist").setFrictionCoeff(20.0)
        self.robot.getBodyNode("rwrist").setFrictionCoeff(20.0)

        self.floor = self.world.loadSkeleton(path.join(Path(__file__).parent.parent.parent.parent, 'assets/mujoco_models/rfc_model', path.basename("humanoid_floor.urdf")))
        self.floor.getBodyNode("floor").setFrictionCoeff(20.0)
        self.world.setTimeStep(self.mj_model.opt.timestep)
        self.world.setGravity([0, 0, -4])
        print("TimeStep:", self.mj_model.opt.timestep)
        #print("Gravity:", self.mj_model.opt.gravity, self.world.getGravity())
        self.setup_joint_mapping()
        self.data = self.mj_sim.data
        self.viewer = None
        self._viewers = {}
        self.obs_dim = None
        self.action_space = None
        self.observation_space = None
        self.np_random = None
        self.cur_t = 0  # number of steps taken
        self.current_control_force = np.zeros(len(self.n2m)+6)
        self.mj_dirty_bit = False

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        # For nimble rendering
        self.disable_nimble_visualizer = disable_nimble_visualizer
        if not disable_nimble_visualizer:
            self.gui = nimble.NimbleGUI(self.world)
            self.gui.serve(PORT_NUM)

        self.init_qpos = self.mj_sim.data.qpos.ravel().copy()
        self.init_qvel = self.mj_sim.data.qvel.ravel().copy()
        self.init_state = self.to_nimble_state(self.init_qpos, self.init_qvel)
        self.prev_qpos = None
        self.prev_qvel = None
        self.seed()

    def setup_joint_mapping(self):
        #self.m2n_foot = [mujoco_joints.index(nimble_joints[i]) for i in range(len(nimble_joints))]
        self.m2n = [mujoco_joints.index(nimble_joints[i]) for i in range(len(nimble_joints))]
        self.n2m = [nimble_joints.index(mujoco_joints[i]) for i in range(len(mujoco_joints))]
        self.mj_nonfoot = []
        for i in range(len(mujoco_joints)):
            if mujoco_joints[i] not in FOOT_JOINTS:
                self.mj_nonfoot.append(i)

    def to_nimble_state(self, pos=None, vel=None):
        if pos is not None:
            quat = pos[3:7]
            rot = pk.matrix_to_euler_angles(pk.quaternion_to_matrix(torch.from_numpy(quat)),convention="XYZ").numpy()
            pose = pos[:3].copy()
            pose[2] += 0.2
            nimble_pos = np.hstack([pose, rot, pos[7:][self.m2n]])
        if vel is not None:
            nimble_vel = np.hstack([vel[:3], vel[3:6], vel[6:][self.m2n]])
        if pos is not None and vel is not None:
            state = np.hstack([nimble_pos, nimble_vel])
        elif pos is not None:
            state = nimble_pos
        elif vel is not None:
            state = nimble_vel
        else:
            print("Cannot be both None")
            assert(False)
        return state

    def to_mujoco_state(self, state):
        if not torch.is_tensor(state):
            state = torch.from_numpy(state)
        pos = state[:int(len(state)/2)]
        vel = state[int(len(state)/2):]
        # Seems to be ambiguous
        rot = pk.matrix_to_quaternion(pk.euler_angles_to_matrix(pos[3:6], convention="XYZ")).numpy()
        joint_pos = np.zeros(len(self.n2m))
        joint_pos = pos[6:][self.n2m]
        mujoco_pos = np.hstack([pos[:3], rot, joint_pos])
        joint_vel = np.zeros(len(self.n2m))
        joint_vel = vel[6:][self.n2m]
        mujoco_vel = np.hstack([vel[:3], vel[3:6], joint_vel])
        return mujoco_pos, mujoco_vel
        
    def set_spaces(self):
        observation, _reward, done, _info = self.step(np.zeros(self.mj_model.nu))
        self.obs_dim = observation.size
        bounds = self.mj_model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------
    def step(self, action):
        """
        Step the environment forward.
        """
        raise NotImplementedError

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self, mode):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def reset(self):
        self.mj_sim.reset()
        self.cur_t = 0
        ob = self.reset_model() # TODO: Should also reset nimble as well as nimble GUI
        old_viewer = self.viewer
        for mode, v in self._viewers.items():
            self.viewer = v
            self.viewer_setup(mode)
        self.viewer = old_viewer
        return ob

    def set_state(self, qpos, qvel):
        """
        Assume qpos and qvel are mujoco state with foot
        """
        # qpos = np.hstack([qpos[:7], qpos[7:][self.mj_nonfoot]])
        # qvel = np.hstack([qvel[:6], qpos[6:][self.mj_nonfoot]])

        assert qpos.shape == (self.mj_model.nq,) and qvel.shape == (self.mj_model.nv,)
        self.world.setState(self.to_nimble_state(qpos, qvel))
        self.set_mujoco_state(qpos, qvel)
        self.mj_dirty_bit = False

    def set_mujoco_state(self, qpos, qvel):
        old_state = self.mj_sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.mj_sim.set_state(new_state)
        self.mj_sim.forward()

    @property
    def dt(self):
        return self.mj_model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.world.setControlForces(ctrl)
        for _ in range(n_frames):
            self.world.step()
        self.sync_mujoco()

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        if mode == 'image':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it, and the image format is BGR for OpenCV
            return data[::-1, :, [2, 1, 0]]
        elif mode == 'human':
            if not self.disable_nimble_visualizer:
                self.gui.displayState(torch.from_numpy(self.world.getState()))
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = MjViewer(self.mj_sim)
            elif mode == 'image':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.mj_sim, 0)
            self._viewers[mode] = self.viewer
        self.viewer_setup(mode)
        return self.viewer

    def set_custom_key_callback(self, key_func):
        self._get_viewer('human').custom_key_callback = key_func

    def get_body_com(self, body_name):
        self.sync_mujoco()
        return self.data.get_body_xpos(body_name)

    def sync_mujoco(self):
        mj_pos, mj_vel = self.to_mujoco_state(self.world.getState())
        self.set_mujoco_state(mj_pos, mj_vel)

    def sync_mujoco_with_state(self, state):
        if torch.is_tensor(state):
            state = state.cpu().detach().numpy()
        mj_pos, mj_vel = self.to_mujoco_state(state)
        self.set_mujoco_state(mj_pos, mj_vel)

    def state_vector(self):
        return np.concatenate([
            self.mj_sim.data.qpos.flat,
            self.mj_sim.data.qvel.flat
        ])

    def vec_body2world(self, body_name, vec):
        self.sync_mujoco()
        body_xmat = self.data.get_body_xmat(body_name)
        vec_world = (body_xmat @ vec[:, None]).ravel()
        return vec_world

    def pos_body2world(self, body_name, pos):
        self.sync_mujoco()
        body_xpos = self.data.get_body_xpos(body_name)
        body_xmat = self.data.get_body_xmat(body_name)
        pos_world = (body_xmat @ pos[:, None]).ravel() + body_xpos
        return pos_world

    def get_goal(self):
        return self.get_expert_attr('qpos',self.expert['len']-1)

    def render_nimble(self):
        self.gui.displayState(torch.from_numpy(self.world.getState()))
        input("Press Enter to continue")


