import numpy as np

def uniform(depth, k):
    h, w = depth.shape[:2]
    depth_flat = depth.flatten()
    valid_mask = np.where(depth_flat > 0)[0]
    valid_depth = depth_flat[valid_mask]
    valid_indices = np.random.choice(valid_mask, size=k, replace=True)
    valid_points = np.zeros((k, 2))
    valid_points[:, 1] = np.floor_divide(valid_indices, w)
    valid_points[:, 0] = np.mod(valid_indices, w)
    return valid_points