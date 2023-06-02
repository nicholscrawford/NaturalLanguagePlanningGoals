from CLIPEmbedder.clip_embedder import CLIPEmbedder
from Data.basic_writerdatasets import CLIPEmbedderDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import clip
import torch.nn as nn

import numpy as np

torch.set_default_dtype(torch.float)
model, preprocess = clip.load("ViT-B/32", device="cuda")

train_loader = DataLoader(CLIPEmbedderDataset(preprocess = preprocess, device = 'cuda'), batch_size=16)

clipembedder = CLIPEmbedder.load_from_checkpoint('/home/nicholscrawfordtaylor/code/NaturalLanguagePlanningGoals/lightning_logs/version_28/checkpoints/epoch=10-step=3443.ckpt', clip_model = model)

x, y = train_loader.__iter__().__next__()
labels = ["a diagram", "a dog", "a cat", "objects in a line", "objects in a circle", "objects in stacks"]
text = clip.tokenize(labels).to("cuda")
text_features = model.encode_text(text)

image_features = clipembedder(x).to(torch.half).to("cuda")

# normalized features
image_features = image_features / image_features.norm(dim=1, keepdim=True)
text_features = text_features / text_features.norm(dim=1, keepdim=True)

# cosine similarity as logits
logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).to("cuda")
logit_scale = logit_scale.exp()
logits_per_image = logit_scale * image_features @ text_features.t()
logits_per_text = logits_per_image.t()

probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()

print("Label probs:")

for problist in probs:
    for prob, label in zip(problist.tolist(), labels):
        print(label, prob)
    print()
