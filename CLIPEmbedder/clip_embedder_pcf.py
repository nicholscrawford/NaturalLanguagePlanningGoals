
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange
import torch.nn.utils.rnn as rnn_utils

from ConfigurationDiffuser.train_simplenetwork import get_diffusion_variables
from ConfigurationDiffuser.configuration_diffuser import EncoderMLP
from pointconvformer_wrapper import PointConvFormerEncoder

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
    def __init__(self, clip_model):#self, input_dim, out_dim, num_heads, num_layers):
        super(CLIPEmbedder, self).__init__()
        self.clip_model = clip_model
        self.point_encoder = PointConvFormerEncoder(
            in_dim=3,
            out_dim=256,
            pool='max',
            hack=True
        )

        self.pose_encoder = PoseEncoder()

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model= 256 + 32, #input_dim,
                nhead=8, #num_heads,
                dim_feedforward=2 * (256 + 32), #input_dim,
                dropout=0.1
            ),
            num_layers=4#num_layers
        )

        self.fc = nn.Linear(256 + 32, 512)
        self.mlp = EncoderMLP(256, 80, uses_pt=True)

        self.fc.to(torch.double)
        self.point_encoder.to(torch.double)
        self.mlp.to(torch.double)
        self.pose_encoder.to(torch.double)
        self.transformer.to(torch.double)

    def create_batch(self, object_encodings, pose_encodings, post_encoder_batch_idxs):
        max_seq_len = post_encoder_batch_idxs.bincount().max()
        batch_size = post_encoder_batch_idxs.bincount().shape[0]
        feature_length = object_encodings.shape[1] + pose_encodings.shape[2]
        pose_encodings_feature_length = pose_encodings.shape[2]
        padded_sequences = torch.empty((max_seq_len, batch_size, feature_length), device="cuda", dtype=torch.double)
        src_key_padding_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool, device="cuda")
        
        for batch_idx in range(post_encoder_batch_idxs.max() + 1):
            seq_len = (post_encoder_batch_idxs == batch_idx).sum()
            padded_sequences[:seq_len, batch_idx, :-pose_encodings_feature_length] = object_encodings[post_encoder_batch_idxs == batch_idx]
            padded_sequences[:seq_len, batch_idx, -pose_encodings_feature_length:] = pose_encodings[batch_idx, :, :]
            src_key_padding_mask[batch_idx, seq_len:] = True

        
        return padded_sequences, src_key_padding_mask 
    
    def forward(self, x):
        pc_rgbs, pc_xyzs, pc_norms, encoder_batch_idxs, post_encoder_batch_idxs, tfs = x
        tfs = get_diffusion_variables(tfs)

        #########################
    
        object_encodings = self.point_encoder(pc_xyzs, pc_rgbs, encoder_batch_idxs, norms=pc_norms)

        pose_encodings = self.pose_encoder(tfs)
        # x: (batch_size, sequence_length, input_dim)
        # obj_embedding_sequence = torch.cat((object_encodings, enc_tfs), dim = 2)
        padded_sequences, mask = self.create_batch(object_encodings, pose_encodings, post_encoder_batch_idxs)
        # obj_embedding_sequence = rearrange(obj_embedding_sequence, "batch_size seq_len enc_dim -> seq_len batch_size enc_dim")

        x = self.transformer(padded_sequences, src_key_padding_mask=mask)
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
