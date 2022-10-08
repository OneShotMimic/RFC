import torch
import numpy as np

def estimate_obs_dim(cfg):
    dofs = 32
    base_dim = dofs + 5
    if cfg.obs_heading:
        base_dim += 4

    if cfg.obs_vel == "root":
        base_dim += 6
    elif cfg.obs_vel == 'full':
        base_dim += dofs + 6

    if cfg.obs_phase:
        base_dim += 1
    return base_dim