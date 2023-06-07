import math
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch3d.transforms as tra3d
import torch.optim as optim
import time
import clip
import os
import argparse
from omegaconf import OmegaConf

from ConfigurationDiffuser.configuration_diffuser import SimpleTransformerDiffuser
from Data.basic_writerdatasets_st import DiffusionDataset

# from StructDiffusion.utils.rearrangement import show_pcs_color_order
# from StructDiffusion.data.dataset_v1_diffuser import SemanticArrangementDataset
# from StructDiffusion.data.tokenizer import Tokenizer
# from StructDiffusion.utils.rotation_continuity import compute_rotation_matrix_from_ortho6d
# from StructDiffusion.models.models import TransformerDiffuser


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

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

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

    return x


def get_struct_objs_poses(x):

    assert x.is_cuda, "compute_rotation_matrix_from_ortho6d requires input to be on gpu"
    device = x.device

    # important: the noisy x can go out of bounds
    x = torch.clamp(x, min=-1, max=1)

    # x: B, 1 + N, 9
    B = x.shape[0]
    N = x.shape[1] - 1

    # compute_rotation_matrix_from_ortho6d takes in [B, 6], outputs [B, 3, 3]
    x_6d = x[:, :, 3:].reshape(-1, 6)
    x_rot = compute_rotation_matrix_from_ortho6d(x_6d).reshape(B, N+1, 3, 3)  # B, 1 + N, 3, 3

    x_trans = x[:, :, :3] # B, 1 + N, 3

    x_full = torch.eye(4).repeat(B, 1 + N, 1, 1).to(device)
    x_full[:, :, :3, :3] = x_rot
    x_full[:, :, :3, 3] = x_trans

    struct_pose = x_full[:, 0].unsqueeze(1) # B, 1, 4, 4
    pc_poses_in_struct = x_full[:, 1:] # B, N, 4, 4

    return struct_pose, pc_poses_in_struct


def move_pc_and_create_scene(obj_xyzs, struct_pose, pc_poses_in_struct):

    device = obj_xyzs.device

    # obj_xyzs: B, N, P, 3
    # struct_pose: B, 1, 4, 4
    # pc_poses_in_struct: B, N, 4, 4

    B, N, _, _ = pc_poses_in_struct.shape
    _, _, P, _ = obj_xyzs.shape

    current_pc_poses = torch.eye(4).repeat(B, N, 1, 1).to(device)  # B, N, 4, 4
    # print(torch.mean(obj_xyzs, dim=2).shape)
    current_pc_poses[:, :, :3, 3] = torch.mean(obj_xyzs, dim=2)  # B, N, 4, 4
    current_pc_poses = current_pc_poses.reshape(B * N, 4, 4)  # B x N, 4, 4

    struct_pose = struct_pose.repeat(1, N, 1, 1) # B, N, 4, 4
    struct_pose = struct_pose.reshape(B * N, 4, 4)  # B x 1, 4, 4
    pc_poses_in_struct = pc_poses_in_struct.reshape(B * N, 4, 4)  # B x N, 4, 4

    goal_pc_pose = struct_pose @ pc_poses_in_struct  # B x N, 4, 4
    goal_pc_transform = goal_pc_pose @ torch.inverse(current_pc_poses)  # B x N, 4, 4

    # important: pytorch3d uses row-major ordering, need to transpose each transformation matrix
    transpose = tra3d.Transform3d(matrix=goal_pc_transform.transpose(1, 2))

    new_obj_xyzs = obj_xyzs.reshape(B * N, P, 3)  # B x N, P, 3
    new_obj_xyzs = transpose.transform_points(new_obj_xyzs)

    # put it back to B, N, P, 3
    new_obj_xyzs = new_obj_xyzs.reshape(B, N, P, 3)

    # visualize_batch_pcs(new_obj_xyzs, B, N, P)

    return new_obj_xyzs


def visualize_batch_pcs(obj_xyzs, B, N, P, verbose=True, limit_B=None, save_dir=None):
    if limit_B is None:
        limit_B = B

    vis_obj_xyzs = obj_xyzs.reshape(B, N, P, -1)
    vis_obj_xyzs = vis_obj_xyzs[:limit_B]

    if type(vis_obj_xyzs).__module__ == torch.__name__:
        if vis_obj_xyzs.is_cuda:
            vis_obj_xyzs = vis_obj_xyzs.detach().cpu()
        vis_obj_xyzs = vis_obj_xyzs.numpy()

    for bi, vis_obj_xyz in enumerate(vis_obj_xyzs):
        if verbose:
            print("example {}".format(bi))
            print(vis_obj_xyz.shape)

        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, "b{}.jpg".format(bi))
            show_pcs_color_order([xyz[:, :3] for xyz in vis_obj_xyz], None, visualize=False, add_coordinate_frame=True,
                                 side_view=True, save_path=save_path)
        else:
            show_pcs_color_order([xyz[:, :3] for xyz in vis_obj_xyz], None, visualize=True, add_coordinate_frame=True,
                                 side_view=True)


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


########################################################################################################################
def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


########################################################################################################################
def train_model(cfg, model, data_iter, noise_schedule, optimizer, warmup, num_epochs, device, save_best_model,
                grad_clipping=1.0):

    loss_type = cfg.diffusion.loss_type

    # if save_best_model:
    #     best_model_dir = os.path.join(cfg.experiment_dir, "best_model")
    #     print("best model will be saved to {}".format(best_model_dir))
    #     if not os.path.exists(best_model_dir):
    #         os.makedirs(best_model_dir)
    #     best_score = -np.inf

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()
        epoch_loss = 0

        with tqdm.tqdm(total=len(data_iter["train"])) as pbar:
            for step, batch in enumerate(data_iter["train"]):
                optimizer.zero_grad()
                (datapoint_pointclouds, transforms), images = batch
                # pointcloud is of shape (B, Num_Objects, Num_Points, 6 (xyzrgb))
                xyzs = datapoint_pointclouds[:,:,:, :3].to(device, non_blocking=True)
                B = xyzs.shape[0]
                # obj_pad_mask: we don't need it now since we are testing
                obj_xyztheta_inputs = transforms.to(device, non_blocking=True)

                t = torch.randint(0, noise_schedule.timesteps, (B,), dtype=torch.long).to(device, non_blocking=True)

                position_index = torch.tensor([[0, 1, 2, 3, 4, 5] for _ in range(B)]).to(device, non_blocking=True)

                #--------------
                x_start = get_diffusion_variables(obj_xyztheta_inputs)
                x_start = x_start + torch.randn_like(x_start, device=device) * 0.001
                noise = torch.randn_like(x_start, device=device)
                x_noisy = q_sample(x_start=x_start, t=t, noise_schedule=noise_schedule, noise=noise)

                obj_xyztheta_inputs = x_noisy[:, :, :]  # B, N, 3 + 6
                obj_xyztheta_outputs = model(t, xyzs, obj_xyztheta_inputs,
                position_index)

                predicted_noise =  obj_xyztheta_outputs
                if loss_type == 'l1':
                    loss = F.l1_loss(noise, predicted_noise)
                elif loss_type == 'l2':
                    loss = F.mse_loss(noise, predicted_noise)
                elif loss_type == "huber":
                    loss = F.smooth_l1_loss(noise, predicted_noise)
                else:
                    raise NotImplementedError()
                # --------------

                loss.backward()
                if grad_clipping != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)
                optimizer.step()
                epoch_loss += loss
                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss})

        if warmup is not None:
            warmup.step()

        print('[Epoch:{}]:  Training Loss:{:.4}'.format(epoch, epoch_loss))

        validate_model(cfg, model, data_iter, noise_schedule, epoch, device)

        # evaluate(gts, predictions, ["obj_x_outputs", "obj_y_outputs", "obj_z_outputs", "obj_theta_outputs",
        #                             "struct_x_inputs", "struct_y_inputs", "struct_z_inputs", "struct_theta_inputs"])
        
        # score = validate(cfg, model, data_iter["valid"], epoch, device)
        # if save_best_model and score > best_score:
        #     print("Saving best model so far...")
        #     best_score = score
        #     save_model(best_model_dir, cfg, epoch, model)

    return model


def validate_model(cfg, model, data_iter, noise_schedule, epoch, device):

    loss_type = cfg.diffusion.loss_type

    model.eval()

    epoch_loss = 0
    # gts = defaultdict(list)
    # predictions = defaultdict(list)
    with torch.no_grad():

        with tqdm.tqdm(total=len(data_iter["valid"])) as pbar:
            for step, batch in enumerate(data_iter["valid"]):

                (datapoint_pointclouds, transforms), images = batch
                # pointcloud is of shape (B, Num_Objects, Num_Points, 6 (xyzrgb))
                xyzs = datapoint_pointclouds[:,:,:, :3].to(device, non_blocking=True)
                B = xyzs.shape[0]
                # obj_pad_mask: we don't need it now since we are testing
                obj_xyztheta_inputs = transforms.to(device, non_blocking=True)

                t = torch.randint(0, noise_schedule.timesteps, (B,), dtype=torch.long).to(device, non_blocking=True)

                position_index = torch.tensor([[0, 1, 2, 3, 4, 5] for _ in range(B)]).to(device, non_blocking=True)

                #--------------
                x_start = get_diffusion_variables(obj_xyztheta_inputs)
                noise = torch.randn_like(x_start, device=device)
                x_noisy = q_sample(x_start=x_start, t=t, noise_schedule=noise_schedule, noise=noise)

                obj_xyztheta_inputs = x_noisy[:, :, :]  # B, N, 3 + 6
                obj_xyztheta_outputs = model(t, xyzs, obj_xyztheta_inputs,
                position_index)

                predicted_noise =  obj_xyztheta_outputs
                if loss_type == 'l1':
                    loss = F.l1_loss(noise, predicted_noise)
                elif loss_type == 'l2':
                    loss = F.mse_loss(noise, predicted_noise)
                elif loss_type == "huber":
                    loss = F.smooth_l1_loss(noise, predicted_noise)
                else:
                    raise NotImplementedError()

                epoch_loss += loss
                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss})

    print('[Epoch:{}]:  Val Loss:{:.4}'.format(epoch, epoch_loss))


def save_model(model_dir, cfg, epoch, model, optimizer=None, scheduler=None):
    state_dict = {'epoch': epoch,
                  'model_state_dict': model.state_dict()}
    if optimizer is not None:
        state_dict["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state_dict["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state_dict, os.path.join(model_dir, "model.tar"))
    OmegaConf.save(cfg, os.path.join(model_dir, "config.yaml"))


def load_model(model_dir):
    """
    Load transformer model
    Important: to use the model, call model.eval() or model.train()
    :param model_dir:
    :return:
    """
    # load dictionaries
    cfg = OmegaConf.load(os.path.join(model_dir, "config.yaml"))

    data_cfg = cfg.dataset

    # initialize model
    model_cfg = cfg.model
    model = SimpleTransformerDiffuser(num_attention_heads=model_cfg.num_attention_heads,
                                encoder_hidden_dim=model_cfg.encoder_hidden_dim,
                                encoder_dropout=model_cfg.encoder_dropout,
                                encoder_activation=model_cfg.encoder_activation,
                                encoder_num_layers=model_cfg.encoder_num_layers,
                                structure_dropout=model_cfg.structure_dropout,
                                object_dropout=model_cfg.object_dropout,
                                ignore_rgb=model_cfg.ignore_rgb)
    model.to(cfg.device)

    # load state dicts
    checkpoint = torch.load(os.path.join(model_dir, "model.tar"))
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = None
    if "optimizer_state_dict" in checkpoint:
        training_cfg = cfg.training
        optimizer = optim.Adam(model.parameters(), lr=training_cfg.learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = None
    if "scheduler_state_dict" in checkpoint:
        scheduler = None
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    noise_schedule = NoiseSchedule(cfg.diffusion.time_steps)

    epoch = checkpoint['epoch']
    return cfg, model, noise_schedule, optimizer, scheduler, epoch


def run_model(cfg):

    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.random_seed)
        torch.cuda.manual_seed_all(cfg.random_seed)
        torch.backends.cudnn.deterministic = True

    data_cfg = cfg.dataset

    train_dataset = DiffusionDataset(cfg.device, ds_roots=data_cfg.train_dirs) 
    valid_dataset = DiffusionDataset(cfg.device, ds_roots=data_cfg.valid_dirs)
    
    data_iter = {}
    data_iter["train"] = DataLoader(train_dataset, batch_size=data_cfg.batch_size, shuffle=True,
                                    # collate_fn=SemanticArrangementDataset.collate_fn,
                                    pin_memory=data_cfg.pin_memory, num_workers=data_cfg.num_workers)
    data_iter["valid"] = DataLoader(valid_dataset, batch_size=data_cfg.batch_size, shuffle=False,
                                    # collate_fn=SemanticArrangementDataset.collate_fn,
                                    pin_memory=data_cfg.pin_memory, num_workers=data_cfg.num_workers)

    # load model
    model_cfg = cfg.model
    model = SimpleTransformerDiffuser(num_attention_heads=model_cfg.num_attention_heads,
                                encoder_hidden_dim=model_cfg.encoder_hidden_dim,
                                encoder_dropout=model_cfg.encoder_dropout,
                                encoder_activation=model_cfg.encoder_activation,
                                encoder_num_layers=model_cfg.encoder_num_layers,
                                structure_dropout=model_cfg.structure_dropout,
                                object_dropout=model_cfg.object_dropout,
                                ignore_rgb=model_cfg.ignore_rgb)
    model.to(cfg.device)
    model.to(torch.double)
    
    training_cfg = cfg.training
    optimizer = optim.Adam(model.parameters(), lr=training_cfg.learning_rate, weight_decay=training_cfg.l2)
    scheduler = None
    warmup = None
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_cfg.lr_restart)
    # warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=training_cfg.warmup,
    #                                 after_scheduler=scheduler)

    noise_schedule = NoiseSchedule(cfg.diffusion.time_steps)

    train_model(cfg, model, data_iter, noise_schedule, optimizer, warmup, training_cfg.max_epochs, cfg.device, cfg.save_best_model)

    # save model
    if cfg.save_model:
        model_dir = os.path.join(cfg.experiment_dir, "model")
        print("Saving model to {}".format(model_dir))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        save_model(model_dir, cfg, cfg.training.max_epochs, model, optimizer, scheduler)


########################################################################################################################
# inference code
# @torch.no_grad()
# def p_sample(model, x, t, t_index):
#     betas_t = extract(betas, t, x.shape)
#     sqrt_one_minus_alphas_cumprod_t = extract(
#         sqrt_one_minus_alphas_cumprod, t, x.shape
#     )
#     sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
#
#     # Equation 11 in the paper
#     # Use our model (noise predictor) to predict the mean
#     model_mean = sqrt_recip_alphas_t * (
#             x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
#     )
#
#     if t_index == 0:
#         return model_mean
#     else:
#         posterior_variance_t = extract(posterior_variance, t, x.shape)
#         noise = torch.randn_like(x)
#         # Algorithm 2 line 4:
#         return model_mean + torch.sqrt(posterior_variance_t) * noise


def sampling(cfg, model, data_iter, noise_schedule, device):

    model.eval()

    with torch.no_grad():
        with tqdm.tqdm(total=len(data_iter["valid"])) as pbar:
            for step, batch in enumerate(data_iter["valid"]):

                # input
                (datapoint_pointclouds, transforms), images = batch
                # pointcloud is of shape (B, Num_Objects, Num_Points, 6 (xyzrgb))
                xyzs = datapoint_pointclouds[:,:,:, :3].to(device, non_blocking=True)
               
                B = xyzs.shape[0]
                # obj_pad_mask: we don't need it now since we are testing
                obj_xyztheta_inputs = transforms.to(device, non_blocking=True)
                position_index = torch.tensor([[0, 1, 2, 3, 4, 5] for _ in range(B)]).to(device, non_blocking=True)


                # --------------
                x_gt = get_diffusion_variables( obj_xyztheta_inputs)

                # start from random noise
                x = torch.randn_like(x_gt, device=device)
                xs = []
                for t_index in reversed(range(0, noise_schedule.timesteps)):

                    t = torch.full((B,), t_index, device=device, dtype=torch.long)

                    betas_t = extract(noise_schedule.betas, t, x.shape)
                    sqrt_one_minus_alphas_cumprod_t = extract(noise_schedule.sqrt_one_minus_alphas_cumprod, t, x.shape)
                    sqrt_recip_alphas_t = extract(noise_schedule.sqrt_recip_alphas, t, x.shape)

                    obj_xyztheta_inputs = x[:, :, :]  # B, N, 3 + 6
                    obj_xyztheta_outputs = model.forward(t, xyzs, obj_xyztheta_inputs,
                                                                                  position_index,
                                                                                  )
                    predicted_noise = obj_xyztheta_outputs

                    model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

                    if t_index == 0:
                        x = model_mean
                    else:
                        posterior_variance_t = extract(noise_schedule.posterior_variance, t, x.shape)
                        noise = torch.randn_like(x)
                        # Algorithm 2 line 4:
                        x = model_mean + torch.sqrt(posterior_variance_t) * noise

                    xs.append(x)
                # --------------
    return xs



def guidance_sampling(cfg, model, data_iter, noise_schedule, device, sampling_cfg, clip_model, embedder_model):

    model.eval()

    text = clip.tokenize(sampling_cfg.labels).to(model.device)
    text_features = model.encode_text(text).to(model.dtype)
    text_features.detach_()

    for step, batch in enumerate(data_iter["valid"]):

        # input
        (datapoint_pointclouds, transforms), images = batch
        # pointcloud is of shape (B, Num_Objects, Num_Points, 6 (xyzrgb))
        xyzs = datapoint_pointclouds[:,:,:, :3].to(device, non_blocking=True)
        
        B = xyzs.shape[0]
        # obj_pad_mask: we don't need it now since we are testing
        obj_xyztheta_inputs = transforms.to(device, non_blocking=True)
        position_index = torch.tensor([[0, 1, 2, 3, 4, 5] for _ in range(B)]).to(device, non_blocking=True)


        # --------------
        x_gt = get_diffusion_variables( obj_xyztheta_inputs)

        # start from random noise
        x = torch.randn_like(x_gt, device=device)
        xs = []
        for t_index in reversed(range(0, noise_schedule.timesteps)):

            t = torch.full((B,), t_index, device=device, dtype=torch.long)

            z_t = x[:, :, :]  # B, N, 3 + 6
            
            # Per step self reccurance should go here.
            num_recurrance_steps = 1 if not sampling_cfg.per_step_self_recurrance else sampling_cfg.per_step_k

            for _ in range(num_recurrance_steps):
                epsilon_noise = model.forward(t, xyzs, z_t,
                                                                                position_index,
                                                                                )
                predicted_noise = epsilon_noise

                hat_z_0 = UG_S(z_t, predicted_noise, t, noise_schedule)
                
                if sampling_cfg.forward_guidance:
                    z_t.requires_grad = True # Need grad for forward guidance
                    guidance_loss = compute_guidance_loss(hat_z_0, clip_model, embedder_model)
                    #Forward guidance
                    sampling_strength = sampling_cfg.guidance_strength_factor * extract(noise_schedule.sqrt_one_minus_alphas, t, z_t.shape) #TODO put in config file
                    noise_space_grad = z_t.grad
                    forward_guided_predicted_noise = predicted_noise + sampling_strength * noise_space_grad

                    predicted_noise = forward_guided_predicted_noise

                if sampling_cfg.backward_guidance:
                    delta_z = torch.zeros_like(hat_z_0)
                    delta_z.requires_grad = True
                    for idx in range(sampling_cfg.backwards_steps_m):
                        
                        loss = compute_guidance_loss(hat_z_0 + delta_z, text_features, embedder_model)
   
                        with torch.no_grad():
                            delta_z = delta_z - delta_z.grad 
                        delta_z.requires_grad = True
                    sqrt_alpha_over_one_minus_alphas = extract(noise_schedule.sqrt_alpha_over_one_minus_alphas, t, z_t.shape)
                    predicted_noise = predicted_noise - sqrt_alpha_over_one_minus_alphas * delta_z

                hat_z_0, hat_z_t_minus_one = S(z_t, predicted_noise, t, noise_schedule)

                #Resample z
                epsilon_noise = torch.randn_like(hat_z_t_minus_one)
                sqrt_alpha_over_alpha_prev  = extract(noise_schedule.sqrt_alpha_over_alpha_prev, t, z_t.shape)
                sqrt_one_minus_alpha_over_alpha_prev = extract(noise_schedule.sqrt_one_minus_alpha_over_alpha_prev, t, z_t.shape)
                z_t = sqrt_alpha_over_alpha_prev * hat_z_t_minus_one + sqrt_one_minus_alpha_over_alpha_prev * epsilon_noise
            
            if t_index == 0:
                xs.append(hat_z_0)
            else:
                xs.append(hat_z_t_minus_one)
        # --------------
    return xs

def compute_guidance_loss(z_0, text_features, clip_model, embedder_model):
    # We regularize the 9d form so it more closely matches its training set.
    z_0_regular = z_0#TODO do this
    image_features = embedder_model(z_0_regular)

    loss = F.mse_loss(image_features, text_features)
    loss.backward()
    return 


def S(x, predicted_noise, t, noise_schedule):
    betas_t = extract(noise_schedule.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(noise_schedule.sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(noise_schedule.sqrt_recip_alphas, t, x.shape)

    #\hat{z_0} https://arxiv.org/pdf/2006.11239.pdf
    model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

    hat_z_0 = model_mean

    posterior_variance_t = extract(noise_schedule.posterior_variance, t, x.shape)
    noise = torch.randn_like(x)
    # Algorithm 2 line 4: \hat{z_t-1} https://arxiv.org/pdf/2006.11239.pdf
    hat_z_t_minus_one = model_mean + torch.sqrt(posterior_variance_t) * noise

    return hat_z_0, hat_z_t_minus_one

# Equation 3 in https://arxiv.org/pdf/2302.07121.pdf, ref. in Algorithm 1
def UG_S(x, predicted_noise, t, noise_schedule):
    sqrt_recip_alphas_t = extract(noise_schedule.sqrt_recip_alphas, t, x.shape)
    sqrt_one_minus_alphas = extract(noise_schedule.sqrt_one_minus_alphas, t, x.shape)

    hat_z_0 = sqrt_recip_alphas_t * (x - sqrt_one_minus_alphas * predicted_noise )
    return hat_z_0


if __name__ == "__main__":
    torch.set_default_dtype(torch.double)
    
    parser = argparse.ArgumentParser(description="Run a simple model")
    parser.add_argument("--config_file", help='config yaml file',
                        default='ConfigurationDiffuser/Config/structfusion_example.yaml',
                        type=str)
    args = parser.parse_args()

    assert os.path.exists(args.config_file), "Cannot find config yaml file at {}".format(args.config_file)

    os.environ["DATETIME"] = time.strftime("%Y_%m_%d-%H:%M:%S")
    cfg = OmegaConf.load(args.config_file)

    if not os.path.exists(cfg.experiment_dir):
        os.makedirs(cfg.experiment_dir)

    OmegaConf.save(cfg, os.path.join(cfg.experiment_dir, "config.yaml"))

    run_model(cfg)