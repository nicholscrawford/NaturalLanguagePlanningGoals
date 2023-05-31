from ConfigurationDiffuser.configuration_diffuser import TransformerDiffuser, SimpleTransformerDiffuser
import torch

# Instantiate the model
input_dim = 6
hidden_dim = 256
num_layers = 4
num_heads = 8
diffusion_steps = 10

model = TransformerDiffuser( 
                            num_attention_heads= 8,
                            encoder_hidden_dim= 512,
                            encoder_dropout= 0.0,
                            encoder_activation= "relu",
                            encoder_num_layers= 8,
                            structure_dropout= 0,
                            object_dropout= 0,
                            ignore_rgb= True,
                            )

batch_size = 20
num_objs = 5
num_points = 512
# Generate random input
t = torch.randint(0, 200, (batch_size,))
pointcloud_xyzs = torch.randn((batch_size, num_objs, num_points, 3))
object_locations = torch.randn((batch_size, num_objs, 9))
struct_xyztheta_inputs = torch.randn((batch_size, 1, 9))
position_index = torch.tensor([[0, 1, 2, 3, 4, 5] for _ in range(batch_size)])
struct_position_index = torch.tensor([[6] for _ in range(batch_size)])
start_token = torch.zeros((batch_size, 1)).int()

# Pass input through the model
struct_xyztheta_outputs, obj_xyztheta_outputs = model(t, pointcloud_xyzs, object_locations, struct_xyztheta_inputs,
                position_index, struct_position_index, start_token)

# Print the output
print(f"struct_xyztheta_outputs shape: {struct_xyztheta_outputs.shape}")
print(f"obj_xyztheta_outputs shape: {obj_xyztheta_outputs.shape}")


model = SimpleTransformerDiffuser( 
                            num_attention_heads= 8,
                            encoder_hidden_dim= 512,
                            encoder_dropout= 0.0,
                            encoder_activation= "relu",
                            encoder_num_layers= 8,
                            structure_dropout= 0,
                            object_dropout= 0,
                            ignore_rgb= True,
                            )

batch_size = 20
num_objs = 5
num_points = 512
# Generate random input
t = torch.randint(0, 200, (batch_size,))
pointcloud_xyzs = torch.randn((batch_size, num_objs, num_points, 3))
object_locations = torch.randn((batch_size, num_objs, 9))
position_index = torch.tensor([[0, 1, 2, 3, 4] for _ in range(batch_size)])

# Pass input through the model
obj_xyztheta_outputs = model(t, pointcloud_xyzs, object_locations,
                position_index)

# Print the output
print(f"obj_xyztheta_outputs shape: {obj_xyztheta_outputs.shape}")
