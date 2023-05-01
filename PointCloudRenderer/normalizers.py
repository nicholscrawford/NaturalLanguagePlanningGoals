import numpy as np
import torch

def normalize_rgb_zero_centered(colors):
    colors = (2*colors)/255 - 1
    return colors

def denormalize_rgb_zero_centered(colors):
    # Scale colors back to the range [0, 1]
    colors = (colors + 1) / 2
    colors = colors * 255
    colors = torch.clamp(colors, 0, 255)
    return colors

def normalize_rgb(colors):
    colors = colors/255
    return colors

def denormalize_rgb(colors):
    colors = colors * 255
    colors = torch.clamp(colors, 0, 255)
    return colors

def normalize_coords(coords, mean, std):
    norm_coords = (coords - mean) / std
    return norm_coords

def normalize_coords_local_mean(coords):
    if type(coords) == np.ndarray:
        mean = np.mean(coords, axis=(0,1))
        std = np.std(coords, axis=(0,1))

        norm_coords = (coords - mean) / std
        return norm_coords
    elif type(coords) == torch.Tensor:
        mean = torch.mean(coords, axis=(0,1))
        std = torch.std(coords, axis=(0,1))

        norm_coords = (coords - mean) / std
        return norm_coords
    else:
        raise ValueError(f"Type {type(coords)} not recognized.")
    
