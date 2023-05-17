from ConfigurationDiffuser.configuration_diffuser import TransformerDiffusionModule
import torch

# Instantiate the model
input_dim = 6
hidden_dim = 256
num_layers = 4
num_heads = 8
diffusion_steps = 10

model = TransformerDiffusionModule(input_dim, hidden_dim, num_layers, num_heads, diffusion_steps)

# Generate random input
batch_size = 2
sequence_length = 10

input_data = torch.randn(batch_size, sequence_length, input_dim)  # Reshape input tensor

print(f"Input data shape: {input_data.shape}")
# Pass input through the model
output = model(input_data)

# Print the output
print(f"Output data shape: {output.shape}")
