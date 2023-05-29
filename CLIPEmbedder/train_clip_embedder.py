from CLIPEmbedder.clip_embedder import CLIPEmbedder
from Data.ycb_datasets import CLIPEmbedderDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

train_loader = DataLoader(CLIPEmbedderDataset(ds_root="/home/nichols/Data/may22/", clear_cache=False))

# model
autoencoder = CLIPEmbedder()

# train model
trainer = pl.Trainer()
trainer.fit(model=autoencoder, train_dataloaders=train_loader)