from Data.datasets import CLIPEmbedderDataset
import clip

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ConfigurationDiffuser.point_transformer_v2m2_base import Encoder as PointCloudEncoder

class PointCloudTransformer(pl.LightningModule):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers):
        super(PointCloudTransformer, self).__init__()

        encoder = PointCloudEncoder(
            depth=4,  # Depth of the encoder, specifying the number of blocks
            in_channels=6,  # Number of input channels
            embed_channels=128,  # Number of channels in the embedded representation
            groups=8,  # Number of groups for group-wise point-wise transformation
            grid_size=(16, 16),  # Size of the grid for grid-wise transformation
            neighbours=16,  # Number of neighboring points considered for attention computation
            qkv_bias=True,  # Boolean indicating whether to include biases for query, key, and value in attention computation
            pe_multiplier=False,  # Boolean indicating whether to use a positional encoding multiplier
            pe_bias=True,  # Boolean indicating whether to include a bias term in the positional encoding
            attn_drop_rate=0.1,  # Dropout rate applied to the attention scores
            drop_path_rate=0.2,  # Dropout rate applied to the residual connections
            enable_checkpoint=False,  # Boolean indicating whether to enable checkpointing for memory optimization
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=2 * input_dim,
                dropout=0.1
            ),
            num_layers=num_layers
        )

        self.fc = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
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
