import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDiffusionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, diffusion_steps):
        super(TransformerDiffusionModule, self).__init__()

        self.diffusion_steps = diffusion_steps

        # Encoder
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads),
            num_layers
        )

        # Decoder
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        
        # Switch so sequence length is the [0] dimension, batch size is [1]
        x = x.transpose(0, 1)

        # Diffusion steps
        x = self.transformer_encoder(x)

        # Switch so sequence length is the [1] dimension, batch size is [0]
        x = x.transpose(0, 1)

        # Decoder
        x = self.decoder(x)
        x = F.relu(x)

        return x
    
