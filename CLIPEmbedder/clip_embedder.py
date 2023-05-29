from Data.ycb_datasets import CLIPEmbedderDataset
import clip
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange

from ConfigurationDiffuser.pointcloud_encoder import PointcloudEncoder

class PoseEncoder(nn.Module):
    def __init__(self):
        super(PoseEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = rearrange(x, "batch num_objects h w -> batch num_objects (h w)")
        return self.mlp(x)

class CLIPEmbedder(pl.LightningModule):
    def __init__(self):#self, input_dim, out_dim, num_heads, num_layers):
        super(CLIPEmbedder, self).__init__()

        self.pointcloud_encoder = PointcloudEncoder()
        self.pose_encoder = PoseEncoder()

        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip, self.clip_preprocess = clip.load("ViT-B/32", device="cuda")



        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model= 128, #input_dim,
                nhead=8, #num_heads,
                dim_feedforward=2 * 128, #input_dim,
                dropout=0.1
            ),
            num_layers=4#num_layers
        )

        self.fc = nn.Linear(128, 512)

    def forward(self, x):
        points, tfs = x
        enc_pcs = self.pointcloud_encoder(points)
        enc_tfs = self.pose_encoder(tfs)
        # x: (batch_size, sequence_length, input_dim)
        
        x = x.permute(1, 0, 2)  # (sequence_length, batch_size, input_dim)
        x = self.transformer(x)
        x = torch.mean(x, dim=0)  # Global average pooling over the sequence length
        x = self.fc(x)  # (batch_size, embed_dim)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        
        image = self.clip_preprocess(y.squeeze()).unsqueeze(0).to(self.device)
        loss = F.mse_loss(y_pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        self.log('test_loss', loss)
