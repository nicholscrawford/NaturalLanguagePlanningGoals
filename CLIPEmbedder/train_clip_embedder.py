from CLIPEmbedder.clip_embedder import CLIPEmbedder
from Data.basic_writerdatasets import CLIPEmbedderDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import clip

torch.set_default_dtype(torch.float)
model, preprocess = clip.load("ViT-B/32", device="cuda")

train_loader = DataLoader(CLIPEmbedderDataset(preprocess = preprocess, device = 'cuda'), batch_size=16)

# model
clipembedder = CLIPEmbedder(clip_model = model)#.to(torch.double)

# train model
trainer = pl.Trainer()
trainer.fit(model=clipembedder, train_dataloaders=train_loader)