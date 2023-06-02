from CLIPEmbedder.clip_embedder import CLIPEmbedder
from Data.basic_writerdatasets import CLIPEmbedderDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import clip
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

"""
An example of gradient descent for backward universal guidance from https://arxiv.org/pdf/2302.07121.pdf
"""

torch.set_default_dtype(torch.float)
model, preprocess = clip.load("ViT-B/32", device="cuda")

train_loader = DataLoader(CLIPEmbedderDataset(preprocess = preprocess, device = 'cuda'), batch_size=16)

clipembedder = CLIPEmbedder.load_from_checkpoint('/home/nicholscrawfordtaylor/code/NaturalLanguagePlanningGoals/lightning_logs/version_28/checkpoints/epoch=10-step=3443.ckpt', clip_model = model)

x, y = train_loader.__iter__().__next__()

delta_x = torch.zeros_like(x[1])
delta_x.requires_grad = True
x[1].requires_grad = True #This is required to get the grad for x. Maybe also don't go all the way back to x1, but to the diffvar form.

labels = ["objects in a line", "objects in a circle", "objects in stacks"]
batch_labels = [random.choice(labels) for _ in range(16)]
text = clip.tokenize(batch_labels).to("cuda")
text_features = model.encode_text(text).to(torch.float)
text_features.detach_()

m = 10
for idx in range(m):

    image_features = clipembedder((x[0], x[1] + delta_x)).to('cuda')

    loss = F.mse_loss(image_features, text_features)
    print(loss)
    loss.backward()
    with torch.no_grad():
        delta_x = delta_x - delta_x.grad 
    delta_x.requires_grad = True


print(delta_x)