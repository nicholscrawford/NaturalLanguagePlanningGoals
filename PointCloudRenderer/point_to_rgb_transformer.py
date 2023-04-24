import torch
from torch import nn
import pytorch_lightning as pl

class TransformerPointToRGBModule(pl.LightningModule):
    def __init__(self, k=10, num_heads=4, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.k = k
        self.pos_enc = nn.Parameter(torch.zeros((1, k, hidden_size)))
        self.point_emb = nn.Linear(3, hidden_size)
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        pos_enc = self.pos_enc.repeat(x.shape[0], 1, 1)
        x = self.point_emb(x) + pos_enc
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        x = self.transformer(x, x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, d_model)
        x = self.fc(x.mean(dim=1))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
