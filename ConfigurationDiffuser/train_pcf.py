import torch
if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        print("Failed to set multiprocessing start. May leak semaphores.")
        pass

from omegaconf import OmegaConf
import pytorch_lightning as pl
import argparse
import os
import time
from datetime import datetime

from pytorch_lightning.strategies.ddp import DDPStrategy
from lightning.pytorch.profilers import PyTorchProfiler

from ConfigurationDiffuser.configuration_diffuser_pcf import SimpleTransformerDiffuser
def get_experiment_datetime(checkpoint_path):
    # Extract the experiment datetime from the checkpoint path
    experiment_datetime = checkpoint_path.split("/")[6]
    return datetime.strptime(experiment_datetime, "%Y_%m_%d-%H:%M:%S")


if __name__ == "__main__":
    torch.set_default_dtype(torch.double)

    parser = argparse.ArgumentParser(description="Run a simple model")
    parser.add_argument("--config_file", help='config yaml file',
                        default='ConfigurationDiffuser/Config/structfusion_pl.yaml',
                        type=str)
    parser.add_argument("--load_recent", action="store_true", help="Load most recent checkpoint and start from there.")
    args = parser.parse_args()

    assert os.path.exists(args.config_file), "Cannot find config yaml file at {}".format(args.config_file)

    os.environ["DATETIME"] = time.strftime("%Y_%m_%d-%H:%M:%S")
    cfg = OmegaConf.load(args.config_file)

    if not os.path.exists(cfg.experiment_dir):
        os.makedirs(cfg.experiment_dir)

    OmegaConf.save(cfg, os.path.join(cfg.experiment_dir, "config.yaml"))

    # Initialize the model
    if True: #not args.load_recent:
        model = SimpleTransformerDiffuser(cfg)
    else:
        checkpoint_dir = "/home/nicholscrawfordtaylor/Experiments/NLPGoals/diffusion_experiments"
        checkpoint_files = []
        for root, dirs, files in os.walk(checkpoint_dir):
            for file in files:
                if file.endswith(".ckpt"):
                    checkpoint_files.append(os.path.join(root, file))
        most_recent_checkpoint = max(checkpoint_files, key=get_experiment_datetime)
        print("Most recent checkpoint:", most_recent_checkpoint)
        model = SimpleTransformerDiffuser.load_from_checkpoint(most_recent_checkpoint)

    # Initialize the PyTorch Lightning trainer
    #profiler = PyTorchProfiler(dirpath="/home/nicholscrawfordtaylor/code/NaturalLanguagePlanningGoals", filename="perf_logs_psthr")
    profiler = None#PyTorchProfiler(profile_memory=True)# with_flops=True, profile_memory=True)
    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=1, #cfg.training.max_epochs,
        default_root_dir=cfg.experiment_dir,
        log_every_n_steps=8,
        profiler=profiler
    )
    
    # Train the model
    trainer.fit(model)#, train_dataloader, val_dataloader)

    # Test the model
    # trainer.test(model, test_dataloader)