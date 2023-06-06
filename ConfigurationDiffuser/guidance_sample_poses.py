import argparse
import os
import pickle
import random
import time
import clip
from CLIPEmbedder.clip_embedder import CLIPEmbedder
from Data.basic_writerdatasets import CLIPEmbedderDataset

import torch
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from ConfigurationDiffuser.train_simplenetwork import (NoiseSchedule,
                                                       load_model, sampling, guidance_sampling)
from Data.basic_writerdatasets import DiffusionDataset
from StructDiffusion.rotation_continuity import \
    compute_rotation_matrix_from_ortho6d

if __name__ == "__main__":
    torch.set_default_dtype(torch.double)

    parser = argparse.ArgumentParser(description="Run a simple model")
    parser.add_argument("--config_file", help='config yaml file',
                        default='ConfigurationDiffuser/Config/sampling_example.yaml',
                        type=str)
    args = parser.parse_args()

    assert os.path.exists(args.config_file), "Cannot find config yaml file at {}".format(args.config_file)

    cfg = OmegaConf.load(args.config_file)

    if not os.path.exists(cfg.poses_dir):
        os.makedirs(cfg.poses_dir)
    if not os.path.exists(cfg.pointclouds_dir):
        os.makedirs(cfg.pointclouds_dir)

    if len(os.listdir(os.path.join(cfg.pointclouds_dir, "1"))) < 10:
        print("Must have the show_gen_poses script running! It's needed to get the point clouds to pass into the model.")
        exit(0)

    clip_model, preprocess = clip.load(cfg.clip_model, device=cfg.device)

    clip_embedder = CLIPEmbedder.load_from_checkpoint(cfg.embedder_checkpoint_path, clip_model = clip_model)


    valid_dataset = DiffusionDataset(cfg.device, ds_root=cfg.pointclouds_dir, clear_cache=True)
    data_cfg = cfg.dataset
    data_iter = {}
    data_iter["train"] = None
    data_iter["valid"] = DataLoader(valid_dataset, batch_size=data_cfg.batch_size, shuffle=False,
                                    # collate_fn=SemanticArrangementDataset.collate_fn,
                                    pin_memory=data_cfg.pin_memory, num_workers=data_cfg.num_workers)

    (model_cfg, model, noise_schedule, optimizer, scheduler, epoch) = load_model(cfg.model_dir)

    poses = guidance_sampling(model_cfg, model, data_iter, noise_schedule, model_cfg.device, cfg.sampling, clip_model, clip_embedder)
    goalpose = poses[-1]
    xyzs = goalpose[:,:,:3]
    flattened_ortho6d = goalpose[:,:,3:].reshape(-1, 6)
    flattened_rmats = compute_rotation_matrix_from_ortho6d(flattened_ortho6d)
    rmats = flattened_rmats.reshape(goalpose.shape[0],goalpose.shape[1], 3, 3)
    
    # k = random.randint(0, 16)
    # print(f"{k}th element in batch.")
    # for i in range(goalpose.shape[1]):
    #     print(f"XYZ: {xyzs[k][i]} ROTATION MATRIX: {rmats[k][i]}")

    with open(os.path.join(cfg.poses_dir, "poses.pickle"), 'wb') as file:
        pickle.dump((xyzs, rmats), file)
