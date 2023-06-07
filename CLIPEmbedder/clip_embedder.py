from Data.ycb_datasets import CLIPEmbedderDataset
import clip
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange

from StructDiffusion.point_transformer import PointTransformerEncoderSmall
from ConfigurationDiffuser.train_simplenetwork import get_diffusion_variables
from ConfigurationDiffuser.configuration_diffuser import EncoderMLP
class PoseEncoder(nn.Module):
    def __init__(self):
        super(PoseEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
    
    def forward(self, x):
        #x = rearrange(x, "batch num_objects h w -> batch num_objects (h w)")
        return self.mlp(x)

class CLIPEmbedder(pl.LightningModule):
    def __init__(self, clip_model, ignore_rgb = True):#self, input_dim, out_dim, num_heads, num_layers):
        super(CLIPEmbedder, self).__init__()
        self.clip_model = clip_model
        self.ignore_rgb = ignore_rgb
        if ignore_rgb:
            self.object_encoder = PointTransformerEncoderSmall(output_dim=256, input_dim=3, mean_center=True)
        else:
            self.object_encoder = PointTransformerEncoderSmall(output_dim=256, input_dim=6, mean_center=True)

        self.pose_encoder = PoseEncoder()




        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model= 112, #input_dim,
                nhead=8, #num_heads,
                dim_feedforward=2 * 112, #input_dim,
                dropout=0.1
            ),
            num_layers=4#num_layers
        )

        self.fc = nn.Linear(112, 512)
        self.mlp = EncoderMLP(256, 80, uses_pt=True)

        self.fc.to(torch.double)
        self.object_encoder.to(torch.double)
        self.mlp.to(torch.double)
        self.pose_encoder.to(torch.double)
        self.transformer.to(torch.double)

    def encode_pc(self, xyzs, rgbs, batch_size, num_objects):
        if self.ignore_rgb:
            center_xyz, x = self.object_encoder(xyzs, None)
        else:
            center_xyz, x = self.object_encoder(xyzs, rgbs)
        obj_pc_embed = self.mlp(x, center_xyz)
        obj_pc_embed = obj_pc_embed.reshape(batch_size, num_objects, -1)
        return obj_pc_embed

    def forward(self, x):
        points, tfs = x
        tfs = get_diffusion_variables(tfs)
        if self.ignore_rgb:
            xyzs = points[:,:,:, :3]
        batch_size, num_target_objects, num_pts, _ = xyzs.shape

        #########################
        xyzs = xyzs.reshape(batch_size * num_target_objects, num_pts, -1)
        obj_pc_embed = self.encode_pc(xyzs, None, batch_size, num_target_objects)

        enc_tfs = self.pose_encoder(tfs)
        # x: (batch_size, sequence_length, input_dim)
        obj_embedding_sequence = torch.cat((obj_pc_embed, enc_tfs), dim = 2)

        obj_embedding_sequence = rearrange(obj_embedding_sequence, "batch_size seq_len enc_dim -> seq_len batch_size enc_dim")

        x = self.transformer(obj_embedding_sequence)
        x = torch.mean(x, dim=0)  # Global average pooling over the sequence length
        x = self.fc(x)  # (batch_size, embed_dim)
        return x

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        
        y = y.to(torch.float)
        y = self.clip_model.encode_image(y).to(torch.double) #Messing about with y's dtype shouldn't matter too much, since we don't care about backprop in that direction.
        y.detach_()  # Detach y from the computation graph
        loss = F.mse_loss(y_pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        
        y = y.to(torch.float)
        y = self.clip_model.encode_image(y).to(torch.double) #Messing about with y's dtype shouldn't matter too much, since we don't care about backprop in that direction.
        y.detach_()  # Detach y from the computation graph
        loss = F.mse_loss(y_pred, y)
        self.log('valid_loss', loss)
        return loss

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_pred = self(x)
    #     loss = F.mse_loss(y_pred, y)
    #     self.log('test_loss', loss)
