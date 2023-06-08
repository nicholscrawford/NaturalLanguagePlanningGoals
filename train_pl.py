from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
import argparse
import os
import time

from ConfigurationDiffuser.configuration_diffuser_pl import SimpleTransformerDiffuser

if __name__ == "__main__":
    torch.set_default_dtype(torch.double)
    
    parser = argparse.ArgumentParser(description="Run a simple model")
    parser.add_argument("--config_file", help='config yaml file',
                        default='ConfigurationDiffuser/Config/structfusion_pl.yaml',
                        type=str)
    args = parser.parse_args()

    assert os.path.exists(args.config_file), "Cannot find config yaml file at {}".format(args.config_file)

    os.environ["DATETIME"] = time.strftime("%Y_%m_%d-%H:%M:%S")
    cfg = OmegaConf.load(args.config_file)

    if not os.path.exists(cfg.experiment_dir):
        os.makedirs(cfg.experiment_dir)

    OmegaConf.save(cfg, os.path.join(cfg.experiment_dir, "config.yaml"))

    # Initialize the model
    model = SimpleTransformerDiffuser(cfg)

    # Initialize the PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        default_root_dir=cfg.experiment_dir,
        log_every_n_steps=8
    )

    # Train the model
    trainer.fit(model)#, train_dataloader, val_dataloader)

    # Test the model
    # trainer.test(model, test_dataloader)