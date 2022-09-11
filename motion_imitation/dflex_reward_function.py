from khrylib.utils import *

# Should be quite identical to original version
def world_rfc_implicit_reward(env, state, action, info):
    cfg = env.cfg
    ws = cfg.reward_weights
    w_p, w_v, w_e, w_c, w_vf = ws.get('w_p', 0.6), ws.get('w_v', 0.1), ws.get('w_e', 0.2), ws.get('w_c', 0.1), ws.get('w_vf', 0.0)
    k_p, k_v, k_e, k_c, k_vf = ws.get('k_p', 2), ws.get('k_v', 0.005), ws.get('k_e', 20), ws.get('k_c', 1000), ws.get('k_vf', 1)
    # By default measure the distance between state via L-2 Norm
    v_ord = ws.get("v_ord", 2)

    # data from env state
    t = env.cur_t
    ind = env.get_expert_index(t)
    prev_bquat = env.prev_bquat

    cur_ee = env.get_ee_pos(None)
    cur_bquat = env.get_body_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    cur_com = env.get_com()

    # Prepare data from expert reference trajectory
    e_qpos = env.get_expert_attr("qpos", ind)
    e_rpos = e_qpos[:3]
    e_ee = env.get_expert_attr("ee_wpos", ind).copy()
    e_com = env.get_expert_attr("com", ind).copy()
    e_bquat  =env.get_expert_attr("bquat", ind)
    e_bangvel = env.get_expert_attr("bangvel", ind)
    expert  = env.expert
    # For Cyclic movement
    if expert['meta']['cyclic']:
        init_pos = expert['init_pos']
        cycle_h = expert['cycle_relheading']
        cycle_pos = expert['cycle_pos']
        orig_rpos = e_rpos.copy()
        e_rpos = quat_mul_vec(cycle_h, e_rpos - init_pos) + cycle_pos
        e_com = quat_mul_vec(cycle_h, e_com - orig_rpos) + e_rpos
        for i in range(e_ee.shape[0] // 3):
            e_ee[3*i:3*i+3] = quat_mul_vec(cycle_h, e_ee[3*i:3*i+3] - orig_rpos) + e_rpos

    if not expert['meta']['cyclic'] and env.start_ind + t >= expert["len"]:
        e_bangvel = np.zeros_like(e_bangvel) # Reset the expert reference angular velocity

    # Pose reward 
    pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat, e_bquat))
    pose_diff[i:] *= cfg.b_diffw # Scale differently
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))

    # velocity reward
    vel_dist = np.linalg.norm(cur_bangvel - e_bangvel, ord = v_ord)
    vel_reward = math.exp(-k_v * (vel_dist ** 2))

    # End-effector reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))

    # COM reward
    com_dist = np.linalg.norm(cur_com - e_com)
    com_reward = math.exp(-k_c * (com_dist ** 2))

    # residual force reward
    # encourage the residual force to be as small as possible
    if w_vf > 0.0:
        vf = action[-env.vf_dim:]
        vf_reward = math.exp(-k_vf * (np.linalg.norm(vf) ** 2))
    else:
        vf_reward = 0.0

    reward = w_p * pose_reward + w_v * vel_reward + w_e * ee_reward + w_c * com_reward + w_vf * vf_reward
    reward /= w_p + w_v + w_e + w_c + w_vf
    return reward, np.array([pose_reward, vel_reward, ee_reward, com_reward, vf_reward])

def local_rfc_implicit_reward(env, state, action, info):
    cfg = env.cfg
    ws = cfg.reward_weights
    w_p, w_v, w_e, w_rp, w_rv, w_vf = ws.get('w_p', 0.5), ws.get('w_v', 0.0), ws.get('w_e', 0.2), ws.get('w_rp', 0.1), ws.get('w_rv', 0.1), ws.get('w_vf', 0.1)
    k_p, k_v, k_e, k_vf = ws.get('k_p', 2), ws.get('k_v', 0.005), ws.get('k_e', 20), ws.get('k_vf', 1)
    k_rh, k_rq, k_rl, k_ra = ws.get('k_rh', 300), ws.get('k_rq', 300), ws.get('k_rl', 5.0), ws.get('k_ra', 0.5)
    v_ord = ws.get('v_ord', 2)

    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    prev_bquat = env.prev_bquat
    prev_qpos = env.prev_qpos

    # Current data
    cur_qpos = env.state.joint_q.clone().cpu().numpy() 
    cur_qvel = get_qvel_fd_new(prev_qpos, cur_qpos, env.dt, cfg.obs_coord)
    cur_rlinv_local = cur_qvel[:3]
    cur_rangv = cur_qvel[3:6]
    cur_rq_rmh = de_heading(cur_qpos[3:7])
    cur_ee = env.get_ee_pos(cfg.obs_coord)
    cur_bquat = env.get_body_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)

    # Expert
    e_qpos = env.get_expert_attr('qpos', ind)
    e_rlinv_local = env.get_expert_attr('rlinv_local', ind)
    e_rangv = env.get_expert_attr('rangv', ind)
    e_rq_rmh = env.get_expert_attr('rq_rmh', ind)
    e_ee = env.get_expert_attr('ee_pos', ind)
    e_bquat = env.get_expert_attr('bquat', ind)
    e_bangvel = env.get_expert_attr('bangvel', ind)

    # Pose reward
    # It omit the orientation of root body ()
    pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat[4:], e_bquat[4:]))    # ignore root
    pose_diff *= cfg.b_diffw
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))

    # Velocity reward
    vel_dist = np.linalg.norm(cur_bangvel[3:] - e_bangvel[3:], ord=v_ord)  # ignore root
    vel_reward = math.exp(-k_v * (vel_dist ** 2))

    # EE reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))

    # root position reward compute relative pose reward + obs orientation reward seperately
    root_height_dist = cur_qpos[2] - e_qpos[2]
    root_quat_dist = multi_quat_norm(multi_quat_diff(cur_rq_rmh, e_rq_rmh))[0]
    root_pose_reward = math.exp(-k_rh * (root_height_dist ** 2) - k_rq * (root_quat_dist ** 2))

    # root velocity and angular velocity
    root_linv_dist = np.linalg.norm(cur_rlinv_local - e_rlinv_local)
    root_angv_dist = np.linalg.norm(cur_rangv - e_rangv)
    root_vel_reward = math.exp(-k_rl * (root_linv_dist ** 2) - k_ra * (root_angv_dist ** 2))

    # residual force reward
    action = action.squeeze()
    if w_vf > 0.0:
        vf = action[-env.vf_dim]
        vf_reward = math.exp(-k_vf * (np.linalg.norm(vf) ** 2))
    else:
        vf_reward = 0.0

    reward = w_p * pose_reward + w_v * vel_reward + w_e * ee_reward + w_rp * root_pose_reward + w_rv * root_vel_reward + w_vf * vf_reward
    reward /= w_p + w_v + w_e + w_rp + w_rv + w_vf
    return reward, np.array([pose_reward, vel_reward, ee_reward, root_pose_reward, root_vel_reward, vf_reward])

dflex_reward_func = {"local_rfc_implicit": local_rfc_implicit_reward,
               "world_rfc_implicit": world_rfc_implicit_reward}