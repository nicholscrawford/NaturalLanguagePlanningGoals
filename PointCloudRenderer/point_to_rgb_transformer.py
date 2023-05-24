import torch
from torch import nn
import pytorch_lightning as pl
import math
from PointCloudRenderer.normalizers import normalize_coords, normalize_coords_local_mean, normalize_rgb, normalize_rgb_zero_centered, denormalize_rgb_zero_centered, denormalize_rgb

class TransformerPointsToRGBModule(pl.LightningModule):
    def __init__(self, k: int, nhead: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.save_hyperparameters()
        
        self.k = k
        self.d_model = 6

        # Define transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Define linear layer to output RGB color
        self.fc = nn.Linear(self.d_model, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, k, 6)
        Returns:
            output: (batch_size, 3)
        """
        # Extract point coordinates and colors
        coords = x[:, :, :3]
        colors = x[:, :, 3:]

        colors = normalize_rgb_zero_centered(colors)
        coords = normalize_coords_local_mean(coords)

        # Recombine normalized params and reshape input for transformer
        x = torch.cat((coords, colors), dim = 2).transpose(0, 1)  # (k, batch_size, d_model)
        
        # Apply transformer layers
        x = self.transformer_encoder(x)
        
        # Average output over all points
        x = x.mean(dim=0)  # (batch_size, d_model)
        #x = x.max(dim=0)[0]  # (batch_size, d_model)
        
        # Predict RGB color
        x = self.fc(x)  # (batch_size, 3)
        
        # De-normalize RGB color
        x = denormalize_rgb_zero_centered(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y, _, _, _ = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _, _, _ = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y, _, _, _ = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
