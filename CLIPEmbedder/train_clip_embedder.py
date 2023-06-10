import argparse
import os
import time

import clip
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from CLIPEmbedder.clip_embedder import CLIPEmbedder
from Data.basic_writerdatasets_st import CLIPEmbedderDataset

if __name__ == "__main__":
    torch.set_default_dtype(torch.float)
    
    parser = argparse.ArgumentParser(description="Run a simple model")
    parser.add_argument("--config_file", help='config yaml file',
                        default='CLIPEmbedder/Config/config.yaml',
                        type=str)
    args = parser.parse_args()

    assert os.path.exists(args.config_file), "Cannot find config yaml file at {}".format(args.config_file)

    os.environ["DATETIME"] = time.strftime("%Y_%m_%d-%H:%M:%S")
    cfg = OmegaConf.load(args.config_file)

    if not os.path.exists(cfg.experiment_dir):
        os.makedirs(cfg.experiment_dir)

    OmegaConf.save(cfg, os.path.join(cfg.experiment_dir, "config.yaml"))

    model, preprocess = clip.load(cfg.clip_model, device=cfg.device)
    
    print("Loading training set.")
    train_loader = DataLoader(CLIPEmbedderDataset(preprocess = preprocess, device = cfg.device, ds_roots = cfg.dataset.train_dirs), batch_size=cfg.dataset.batch_size)
    print("Loading validation set.")
    val_loader = DataLoader(CLIPEmbedderDataset(preprocess = preprocess, device = cfg.device, ds_roots = cfg.dataset.valid_dirs), batch_size=cfg.dataset.batch_size)
    
    # model
    clipembedder = CLIPEmbedder(clip_model = model)

    # train model
    trainer = pl.Trainer(default_root_dir=cfg.experiment_dir, max_epochs=cfg.training.max_epochs)
    trainer.fit(model=clipembedder, train_dataloaders=train_loader, val_dataloaders=val_loader)
