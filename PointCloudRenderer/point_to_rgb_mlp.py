import lightning

import torch
from torch import nn
import pytorch_lightning as pl

from PointCloudRenderer.normalizers import normalize_coords, normalize_coords_local_mean, normalize_rgb, normalize_rgb_zero_centered, denormalize_rgb_zero_centered, denormalize_rgb


class MLPPointToRGBModule(pl.LightningModule):
    def __init__(self, k=10):
        super().__init__()
        self.k = k
        self.fc1 = nn.Linear(6*k, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):

        # Extract point coordinates and colors
        coords = x[:, :, :3]
        colors = x[:, :, 3:]

        colors = normalize_rgb_zero_centered(colors)
        coords = normalize_coords_local_mean(coords)

        # Recombine normalized params and reshape input for transformer
        x = torch.cat((coords, colors), dim = 2).view(-1, 6*self.k)
        

        x = x.view(-1, 6*self.k)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        # De-normalize RGB color
        x = denormalize_rgb_zero_centered(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y, _, _ = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _, _ = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y, _, _ = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

