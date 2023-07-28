import argparse
import os
import time

import clip
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from CLIPEmbedder.clip_embedder_pcf import CLIPEmbedder
from Data.pcf_dataset import CLIPEmbedderDataset

if __name__ == "__main__":
    torch.set_default_dtype(torch.float)
    torch.multiprocessing.set_start_method('spawn')
    
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
    train_ds = CLIPEmbedderDataset(preprocess = preprocess, device = cfg.device, ds_roots = cfg.dataset.train_dirs)
    train_loader = DataLoader(train_ds, batch_size=cfg.dataset.batch_size, num_workers=cfg.dataset.num_workers, collate_fn=train_ds.collate_fn)
    print("Loading validation set.")
    val_ds = CLIPEmbedderDataset(preprocess = preprocess, device = cfg.device, ds_roots = cfg.dataset.valid_dirs)
    val_loader = DataLoader(val_ds, batch_size=cfg.dataset.batch_size, num_workers=cfg.dataset.num_workers, collate_fn=val_ds.collate_fn)
    
    # model
    clipembedder = CLIPEmbedder(clip_model = model)

    # train model
    trainer = pl.Trainer(default_root_dir=cfg.experiment_dir, max_epochs=cfg.training.max_epochs)
    trainer.fit(model=clipembedder, train_dataloaders=train_loader, val_dataloaders=val_loader)
