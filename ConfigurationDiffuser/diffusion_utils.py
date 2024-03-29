import torch
import torch.nn.functional as F
import math
from StructDiffusion.rotation_continuity import compute_rotation_matrix_from_ortho6d

########################################################################################################################
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


########################################################################################################################


class NoiseSchedule:

    def __init__(self, timesteps=200):

        self.timesteps = timesteps

        # define beta schedule
        self.betas = linear_beta_schedule(timesteps=timesteps)
        # self.betas = cosine_beta_schedule(timesteps=timesteps)

        # define alphas
        self.alphas = 1. - self.betas
        # alphas_cumprod: alpha bar
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # For universal guidance eq. 9
        self.sqrt_alphas_cumprod_over_one_minus_alphas_cumprod = torch.sqrt(self.alphas_cumprod / (1-self.alphas_cumprod))

        self.sqrt_one_minus_alphas_cumprod_prev = torch.sqrt(1.0 - self.alphas_cumprod_prev)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.alphas_prev = F.pad(self.alphas[:-1], (1, 0), value=1.0)
        self.sqrt_one_minus_alphas = torch.sqrt(1. - self.alphas)
        self.sqrt_alpha_over_one_minus_alphas = torch.sqrt(self.alphas / ( 1 - self.alphas))
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alpha_over_alpha_prev = torch.sqrt(self.alphas / self.alphas_prev)
        self.sqrt_one_minus_alpha_over_alpha_prev = torch.sqrt((1-self.alphas) / self.alphas_prev)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu()) 
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

########################################################################################################################
def get_diffusion_variables(obj_xyztheta_inputs):

    # important: we need to get the first two columns, not first two rows
    # array([[ 3,  4,  5],
    #   [ 6,  7,  8],
    #   [ 9, 10, 11]])
    # xyz_6d_idxs = [0, 1, 2, 3, 6, 9, 4, 7, 10]
    xyz_6d_idxs = [3, 7, 11, 0, 4, 8, 1, 5, 9]
    # print(batch_data["obj_xyztheta_inputs"].shape)
    # print(batch_data["struct_xyztheta_inputs"].shape)
    B, N, _, _ = obj_xyztheta_inputs.shape
    #obj_xyztheta_inputs_flat = obj_xyztheta_inputs.reshape((B,N,16))
    # only get the first and second columns of rotation
    obj_xyztheta_inputs = obj_xyztheta_inputs.reshape((B,N,16))[:, :, xyz_6d_idxs]  # B, N, 9
    # struct_xyztheta_inputs = struct_xyztheta_inputs[:, :, xyz_6d_idxs]  # B, 1, 9

    # x = torch.cat([struct_xyztheta_inputs, obj_xyztheta_inputs], dim=1)  # B, 1 + N, 9
    x = obj_xyztheta_inputs
    # print(x.shape)
    x[:, :, :3] *= 10
    return x

# forward diffusion (using the nice property)
def q_sample(x_start, t, noise_schedule, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(noise_schedule.sqrt_alphas_cumprod, t, x_start.shape)
    # print("sqrt_alphas_cumprod_t", sqrt_alphas_cumprod_t)
    sqrt_one_minus_alphas_cumprod_t = extract(
        noise_schedule.sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    # print("sqrt_one_minus_alphas_cumprod_t", sqrt_one_minus_alphas_cumprod_t)
    # print("noise", noise)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise