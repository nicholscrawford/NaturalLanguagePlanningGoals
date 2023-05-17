from ConfigurationDiffuser.point_transformer_v2m2_base import Encoder
import torch

#coord, feat, offset = points

# Define the Encoder parameters
depth = 4
in_channels = 3
embed_channels = 64
groups = 8
grid_size = 16
neighbours = 16
qkv_bias = True
pe_multiplier = False
pe_bias = True
attn_drop_rate = 0.1
drop_path_rate = 0.2
enable_checkpoint = False

# Instantiate the Encoder
encoder = Encoder(
    depth=depth,
    in_channels=in_channels,
    embed_channels=embed_channels,
    groups=groups,
    grid_size=grid_size,
    neighbours=neighbours,
    qkv_bias=qkv_bias,
    pe_multiplier=pe_multiplier,
    pe_bias=pe_bias,
    attn_drop_rate=attn_drop_rate,
    drop_path_rate=drop_path_rate,
    enable_checkpoint=enable_checkpoint
)

# Generate random point cloud data
batch_size = 2
num_points = 100
coord = torch.randn(batch_size, num_points, 3)  # Shape: (batch_size, num_points, 3)
feat = torch.randn(batch_size, num_points, in_channels)  # Shape: (batch_size, num_points, in_channels)
offset = torch.tensor([0, num_points])  # Shape: (2,)

# Create a tuple with coord, feat, and offset
points = (coord, feat, offset)

# Pass the point cloud data through the Encoder
output, cluster = encoder(points)

# Print the output and cluster shapes
print("Output shape:", output.shape)
print("Cluster shape:", cluster.shape)

