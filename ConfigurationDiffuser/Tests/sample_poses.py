import argparse
import os
import pickle
import random
import time

import torch
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from ConfigurationDiffuser.train_simplenetwork import (NoiseSchedule,
                                                       load_model, sampling)
from Data.basic_writerdatasets import DiffusionDataset
from StructDiffusion.rotation_continuity import \
    compute_rotation_matrix_from_ortho6d

if __name__ == "__main__":
    torch.set_default_dtype(torch.double)
    
    parser = argparse.ArgumentParser(description="Run a simple model")
    parser.add_argument("--config_file", help='config yaml file',
                        default='ConfigurationDiffuser/Config/structfusion_example.yaml',
                        type=str)
    args = parser.parse_args()

    assert os.path.exists(args.config_file), "Cannot find config yaml file at {}".format(args.config_file)

    os.environ["DATETIME"] = time.strftime("%Y%m%d-%H%M%S")
    cfg = OmegaConf.load(args.config_file)

    if not os.path.exists(cfg.experiment_dir):
        os.makedirs(cfg.experiment_dir)

    OmegaConf.save(cfg, os.path.join(cfg.experiment_dir, "config.yaml"))

    train_dataset = DiffusionDataset(cfg.device) #YOOO NO PARAMS 
    valid_dataset = DiffusionDataset(cfg.device,  max_size=16)
    data_cfg = cfg.dataset
 

    data_iter = {}
    data_iter["train"] = DataLoader(train_dataset, batch_size=data_cfg.batch_size, shuffle=True,
                                    # collate_fn=SemanticArrangementDataset.collate_fn,
                                    pin_memory=data_cfg.pin_memory, num_workers=data_cfg.num_workers)
    data_iter["valid"] = DataLoader(valid_dataset, batch_size=data_cfg.batch_size, shuffle=False,
                                    # collate_fn=SemanticArrangementDataset.collate_fn,
                                    pin_memory=data_cfg.pin_memory, num_workers=data_cfg.num_workers)

    (cfg, model, noise_schedule, optimizer, scheduler, epoch) = load_model("/home/nicholscrawfordtaylor/Experiments/NLPGoals/experiments/20230531-133943/model")

    poses = sampling(cfg, model, data_iter, noise_schedule, cfg.device)
    goalpose = poses[-1]
    xyzs = goalpose[:,:,:3]
    rmats = compute_rotation_matrix_from_ortho6d(goalpose[:,:,3:].reshape(-1, 6)).reshape(goalpose.shape[0],goalpose.shape[1], 3, 3)
    k = random.randint(0, 16)
    print(f"{k}th element in batch.")
    for i in range(goalpose.shape[1]):
        print(f"XYZ: {xyzs[k][i]} ROTATION MATRIX: {rmats[k][i]}")

    with open("/home/nicholscrawfordtaylor/Experiments/NLPGoals/experiments/20230531-133943/rots.pickle", 'wb') as file:
        pickle.dump((xyzs[k], rmats[k]), file)
