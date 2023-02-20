import os
import inspect
import numpy as np
import torch
import pytorch_kinematics as pk


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

MUJOCO_JOINTS = ['lfemur_z', 'lfemur_y', 'lfemur_x', 'ltibia_x',
                 'lfoot_z', 'lfoot_y', 'lfoot_x', 'rfemur_z', 
                 'rfemur_y', 'rfemur_x', 'rtibia_x', 'rfoot_z', 
                 'rfoot_y', 'rfoot_x', 'upperback_z', 'upperback_y', 
                 'upperback_x', 'lowerneck_z', 'lowerneck_y', 
                 'lowerneck_x', 'lclavicle_z', 'lclavicle_y', 
                 'lhumerus_z', 'lhumerus_y', 'lhumerus_x', 'lradius_x', 
                 'rclavicle_z', 'rclavicle_y', 'rhumerus_z', 'rhumerus_y', 
                 'rhumerus_x', 'rradius_x']

MUJOCO_JOINTS_NO_FOOT = ['lfemur_z', 'lfemur_y', 'lfemur_x', 'ltibia_x', 'rfemur_z', 
                 'rfemur_y', 'rfemur_x', 'rtibia_x', 'upperback_z', 'upperback_y', 
                 'upperback_x', 'lowerneck_z', 'lowerneck_y', 
                 'lowerneck_x', 'lclavicle_z', 'lclavicle_y', 
                 'lhumerus_z', 'lhumerus_y', 'lhumerus_x', 'lradius_x', 
                 'rclavicle_z', 'rclavicle_y', 'rhumerus_z', 'rhumerus_y', 
                 'rhumerus_x', 'rradius_x']

NIMBLE_JOINTS = ['upperback_z','upperback_y','upperback_x','lowerneck_z',
                 'lowerneck_y','lowerneck_x','lclavicle_z','lclavicle_y',
                 'lhumerus_z','lhumerus_y','lhumerus_x','lradius_x',
                 'rclavicle_z','rclavicle_y','rhumerus_z','rhumerus_y',
                 'rhumerus_x','rradius_x','lfemur_z','lfemur_y','lfemur_x',
                 'ltibia_x','rfemur_z','rfemur_y',
                 'rfemur_x','rtibia_x']

FOOT_JOINTS = ["lfoot_x", "lfoot_y", "lfoot_z", "rfoot_x", "rfoot_y", "rfoot_z"]

class DataCollector:
    def __init__(self,q_dim, dq_dim, action_dim, dt,max_buffer_size=100000):
        self.state_before = np.zeros((max_buffer_size, q_dim+dq_dim))
        self.acc_after =  np.zeros((max_buffer_size, dq_dim))
        self.action = np.zeros((max_buffer_size, action_dim))
        self.record_ptr = 0
        self.dt = dt
        self.action_dim = action_dim
        self.q_dim = q_dim
        self.dq_dim = dq_dim
        self.max_buffer_size = max_buffer_size
        self.setup_joint_mapping()

    def record(self, pos_before, vel_before, pos_after, vel_after, action):
        state_before = self.to_nimble_state(pos_before,vel_before) 
        state_after = self.to_nimble_state(pos_after, vel_after)
        qacc = (state_after[self.q_dim:self.q_dim+self.dq_dim] - state_before[self.q_dim:self.q_dim+self.dq_dim]) / self.dt
        action = action.numpy() if torch.is_tensor(action) else action
        self.state_before[self.record_ptr] = state_before
        self.acc_after[self.record_ptr] = qacc
        self.action[self.record_ptr] = action
        self.record_ptr = (self.record_ptr+1) % self.max_buffer_size

    def save_data(self, filename):
        filedir = os.path.join(currentdir, f"../../data/human_model/{filename}.npz")
        np.savez(filedir, state_before=self.state_before, action=self.action,
                 acc_after=self.acc_after)

    def setup_joint_mapping(self):
        self.m2n = [MUJOCO_JOINTS_NO_FOOT.index(NIMBLE_JOINTS[i]) for i in range(len(NIMBLE_JOINTS))]
        self.n2m = [NIMBLE_JOINTS.index(MUJOCO_JOINTS_NO_FOOT[i]) for i in range(len(MUJOCO_JOINTS_NO_FOOT))]
        self.mj_nonfoot = []
        for i in range(len(MUJOCO_JOINTS)):
            if MUJOCO_JOINTS[i] not in FOOT_JOINTS:
                self.mj_nonfoot.append(i)

    def to_nimble_state(self, pos=None, vel=None):
        if pos is not None:
            quat = pos[3:7]
            rot = pk.matrix_to_euler_angles(pk.quaternion_to_matrix(torch.from_numpy(quat)),convention="XYZ").numpy()
            pose = pos[:3].copy()
            pose[2] += 0.1
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