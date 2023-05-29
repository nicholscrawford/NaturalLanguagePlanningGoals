from ConfigurationDiffuser.point_transformer_v2m2_base import Encoder, batch2offset, offset2batch
import torch
from torch import nn


# coords = torch.zeros((256, 3)).to("cuda")
# feats = torch.zeros((256, 3)).to("cuda")
# offset = torch.tensor([128, 256]).to("cuda")
# enc = Encoder(4, 3, 128, 8, grid_size=0.01).to("cuda")
# output = enc((coords, feats, offset))
# print(f"Blocked points {[_.shape for _ in output[0]]}")
# print(f"Blocked points {batch2offset(output[1])}")


class PointcloudEncoder(nn.Module):
    def __init__(self):
        super(PointcloudEncoder, self).__init__()        
        
        self.enc = Encoder(
            depth=4,
            in_channels=3,  # Number of input features per point (x, y, z, r, g, b)
            embed_channels=32,  # Desired dimensionality of the embedded features
            groups=8,
            grid_size=0.01, # Dont think this is right
            # neighbours=16,
            # qkv_bias=True,
            # pe_multiplier=False,
            # pe_bias=True,
            # attn_drop_rate=0.1,
            # drop_path_rate=0.2,
            # enable_checkpoint=False,
        )
        
    def forward(self, x):
        (point_list, offset) = x
        if point_list.shape[0] == 1:
            point_list = point_list.squeeze()
            offset = offset.squeeze().int()
            coords = point_list[:, :3]
            colors = point_list[:, 3:]
            
            (_, embedded_features, _), _batch = self.enc((coords, colors, offset))

        return embedded_features