import math
from typing import Any, Optional
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pytorch_lightning as pl
import einops
import pickle
import random
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS

from Data.basic_writerdatasets_st import DiffusionDataset
from StructDiffusion.encoders import DropoutSampler, EncoderMLP
from StructDiffusion.point_transformer import PointTransformerEncoderSmall
from ConfigurationDiffuser.diffusion_utils import NoiseSchedule, extract, get_diffusion_variables, q_sample
from StructDiffusion.rotation_continuity import \
    compute_rotation_matrix_from_ortho6d


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimpleTransformerDiffuser(pl.LightningModule):
    """
    This model takes in point clouds of all objects
    """

    def __init__(self, cfg):

        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        print("Transformer Decoder with Point Transformer 6D All Objects")

        # object encode will have dim 256
        self.object_encoder = PointTransformerEncoderSmall(output_dim=256, input_dim=6, mean_center=True)

        # 256 = 80 (point cloud) + 80 (position) + 80 (time) + 16 (position idx)
        # Important: we set uses_pt to true because we want the model to consider the positions of objects that
        #  don't need to be rearranged.
        self.mlp = EncoderMLP(256, 80, uses_pt=True)
        self.position_encoder = nn.Sequential(nn.Linear(3 + 6, 80))
        #self.start_token_embeddings = torch.nn.Embedding(1, 256)

        # max number of objects or max length of sentence is 7
        self.position_embeddings = torch.nn.Embedding(7, 16)

        encoder_layers = TransformerEncoderLayer(
            256, cfg.model.num_attention_heads, cfg.model.encoder_hidden_dim, cfg.model.encoder_dropout, cfg.model.encoder_activation
            )
        self.encoder = TransformerEncoder(encoder_layers, cfg.model.encoder_num_layers)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(80),
            nn.Linear(80, 80),
            nn.GELU(),
            nn.Linear(80, 80),
        )

        self.obj_dist = DropoutSampler(256, 3 + 6, dropout_rate=cfg.model.object_dropout)

        if cfg.diffusion.loss_type == 'l1':
            self.loss_function = F.l1_loss
        elif cfg.diffusion.loss_type == 'l2':
            self.loss_function = F.mse_loss
        elif cfg.diffusion.loss_type == "huber":
            self.loss_function = F.smooth_l1_loss
        else:
            raise NotImplementedError()


        self.noise_schedule = NoiseSchedule(cfg.diffusion.time_steps)

    def encode_pc(self, xyzs, rgbs, batch_size, num_objects):
        center_xyz, x = self.object_encoder(xyzs, rgbs)
        pointclouds_embed = self.mlp(x, center_xyz)
        pointclouds_embed = pointclouds_embed.reshape(batch_size, num_objects, -1)
        return pointclouds_embed


    def _forward(self, t, xyzs, rgbs, transforms):
        """
            The forward pass of the model

        Args:
            t (int torch.tensor of shape (Batch size,)): contains the diffusion process timestep 
            xyzs (torch.tensor of shape (Batch Size, Number of Objects, Number of Points, 3 (xyz))): The segmented point cloud.
            transforms (torch.tensor of shape (Batch Size, Number of Objects, 9): Representing the transformations of objects
            position_index (torch.tensor of shape (Batch Size,)): An index from 0-max_objs representing position in sequence
            start_token (torch.zeors of shape (Batch Size)): 

        Returns:
            torch.tensor of shape (Batch Size, Number of Objects, 9): representing output positions.
        """

        batch_size, num_target_objects, num_pts, _ = xyzs.shape

        #########################
        xyzs = einops.rearrange( xyzs, " batch_size num_objects num_pts xyz -> (batch_size num_objects) num_pts xyz")
        rgbs = einops.rearrange( rgbs, " batch_size num_objects num_pts rgb -> (batch_size num_objects) num_pts rgb")
        
        
        pointclouds_embed = self.encode_pc(xyzs, rgbs, batch_size, num_target_objects) # gives batch_size num_objs embed_dim
        transforms_embed = self.position_encoder(transforms)
        position_embed = self.position_embeddings(torch.tensor([[0,1,2,3,4,5] for _ in range(batch_size)], device='cuda'))
        objects_embed = torch.cat([transforms_embed, pointclouds_embed], dim=-1)  # gives batch_size num_objs embed_dim

        #########################
        time_embed = self.time_mlp(t)  # B, dim update commented shape
        time_embed = time_embed.unsqueeze(1).repeat(1, num_target_objects, 1)  # B, N, dim
        
        sequence_embed = torch.cat([time_embed, objects_embed, position_embed], dim=-1)

        #########################
        # input to transformer needs to have dimenion [sequence_length, batch size, encoder input dimension]
        sequence_embed = einops.rearrange(sequence_embed,"batch seq_len enc_dim -> seq_len batch enc_dim")

        # encode: [sequence_length, batch_size, embedding_size]
        transformed_sequence_embed = self.encoder(sequence_embed)
        transformed_sequence_embed = einops.rearrange(sequence_embed,"seq_len batch enc_dim -> batch seq_len enc_dim")
        #########################
        output_transforms = self.obj_dist(transformed_sequence_embed)  # B, N, 3 + 6

        return output_transforms

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.cfg.training.learning_rate, weight_decay=self.cfg.training.l2)
        return optimizer
    
    def training_step(self, batch, batch_idx):

        datapoint_pointclouds, transforms = batch
        # pointcloud is of shape (B, Num_Objects, Num_Points, 6 (xyzrgb))
        xyzs = datapoint_pointclouds[:,:,:, :3].to(self.device, non_blocking=True)
        rgbs = datapoint_pointclouds[:,:,:, 3:].to(self.device, non_blocking=True)
        transforms = transforms.to(self.device, non_blocking=True)
        B = xyzs.shape[0]

        t = torch.randint(0, self.noise_schedule.timesteps, (B,), dtype=torch.long).to(self.device, non_blocking=True)

        #--------------
        transforms_0 = get_diffusion_variables(transforms)
        noise = torch.randn_like(transforms_0, device=self.device)
        transforms_t = q_sample(x_start=transforms_0, t=t, noise_schedule=self.noise_schedule, noise=noise)

        predicted_noise = self._forward(t, xyzs, rgbs, transforms_t)

        loss = self.loss_function(noise, predicted_noise)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        datapoint_pointclouds, transforms = batch
        # pointcloud is of shape (B, Num_Objects, Num_Points, 6 (xyzrgb))
        xyzs = datapoint_pointclouds[:,:,:, :3].to(self.device, non_blocking=True)
        rgbs = datapoint_pointclouds[:,:,:, 3:].to(self.device, non_blocking=True)
        transforms = transforms.to(self.device, non_blocking=True)
        B = xyzs.shape[0]

        t = torch.randint(0, self.noise_schedule.timesteps, (B,), dtype=torch.long).to(self.device, non_blocking=True)

        #--------------
        transforms_0 = get_diffusion_variables(transforms)
        noise = torch.randn_like(transforms_0, device=self.device)
        transforms_t = q_sample(x_start=transforms_0, t=t, noise_schedule=self.noise_schedule, noise=noise)

        predicted_noise = self._forward(t, xyzs, rgbs, transforms_t)

        betas_t = extract(self.noise_schedule.betas, t, transforms_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.noise_schedule.sqrt_one_minus_alphas_cumprod, t, transforms_t.shape)
        sqrt_recip_alphas_t = extract(self.noise_schedule.sqrt_recip_alphas, t, transforms_t.shape)
        hat_transforms_0 = sqrt_recip_alphas_t * (transforms_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        loss = self.loss_function(transforms_0, hat_transforms_0)
        self.log("val_pred_clean_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        loss = self.loss_function(noise, predicted_noise)
        self.log("val_noise_pred_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        if self.sampling_cfg.guidance_sampling:
            with torch.inference_mode(mode=False):
                self.guided_sample(batch, batch_idx)
        else:
            self.sample(batch, batch_idx)

    def sample(self, batch, batch_idx):
        # input
        datapoint_pointclouds, transforms = batch
        # pointcloud is of shape (B, Num_Objects, Num_Points, 6 (xyzrgb))
        xyzs = datapoint_pointclouds[:,:,:, :3].to(self.device, non_blocking=True)
        rgbs = datapoint_pointclouds[:,:,:, 3:].to(self.device, non_blocking=True)
        transforms = transforms.to(self.device, non_blocking=True)
        B = xyzs.shape[0]

        # --------------
        z_gt = get_diffusion_variables( transforms)

        # start from random noise
        z_t = torch.randn_like(z_gt, device=self.device)
        zs = []
        for t_index in reversed(range(0, self.noise_schedule.timesteps)):

            t = torch.full((B,), t_index, device=self.device, dtype=torch.long)

            betas_t = extract(self.noise_schedule.betas, t, z_t.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(self.noise_schedule.sqrt_one_minus_alphas_cumprod, t, z_t.shape)
            sqrt_recip_alphas_t = extract(self.noise_schedule.sqrt_recip_alphas, t, z_t.shape)

            predicted_noise = self._forward(t, xyzs, rgbs, z_t)

            hat_z_0 = sqrt_recip_alphas_t * (z_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

            if t_index == 0:
                z_t = hat_z_0
            else:
                posterior_variance_t = extract(self.noise_schedule.posterior_variance, t, z_t.shape)
                noise = torch.randn_like(z_t)
                # Algorithm 2 line 4:
                z_t = hat_z_0 + torch.sqrt(posterior_variance_t) * noise

            zs.append(z_t)
                # --------------
        z_0 =  zs[-1]
        # Now we'e sampled, save to self.poses_dir, set by testing script
        xyzs = z_0[:,:,:3]
        xyzs *= 0.1
        flattened_ortho6d = z_0[:,:,3:].reshape(-1, 6)
        flattened_rmats = compute_rotation_matrix_from_ortho6d(flattened_ortho6d)
        rmats = flattened_rmats.reshape(z_0.shape[0],z_0.shape[1], 3, 3)
        
        k = random.randint(0, 15)
        print(f"{k}th element in batch.")
        for i in range(z_0.shape[1]):
            print(f"XYZ: {xyzs[k][i]} ROTATION MATRIX:\n{rmats[k][i]}")

        print(f"Writing {xyzs.shape[0]} poses to {os.path.join(self.poses_dir, 'poses.pickle')}")
        with open(os.path.join(self.poses_dir, "poses.pickle"), 'wb') as file:
            pickle.dump((xyzs, rmats), file)
    
    def guided_sample(self, batch, batch_idx):
        
        datapoint_pointclouds, transforms = batch
        # pointcloud is of shape (B, Num_Objects, Num_Points, 6 (xyzrgb))
        xyzs = datapoint_pointclouds[:,:,:, :3].to(self.device, non_blocking=True)
        rgbs = datapoint_pointclouds[:,:,:, 3:].to(self.device, non_blocking=True)
        transforms = transforms.to(self.device, non_blocking=True)
        B = xyzs.shape[0]

        # --------------
        z_gt = get_diffusion_variables( transforms)

        # start from random noise
        z_t = torch.randn_like(z_gt, device=self.device, requires_grad=True)
        zs = []
        for t_index in reversed(range(0, self.noise_schedule.timesteps)):

            t = torch.full((B,), t_index, device=self.device, dtype=torch.long)

            num_recurrance_steps = 1 if not self.sampling_cfg.per_step_self_recurrance else self.sampling_cfg.per_step_k

            for _ in range(num_recurrance_steps):
                predicted_noise = self._forward(t, xyzs, rgbs, z_t)
                z_t.requires_grad = True

                hat_z_0 = self.UG_S(z_t, predicted_noise, t)

                if self.sampling_cfg.forward_universal_guidance:
                    guidance_loss = self.guidance_function(hat_z_0)
                    #Forward guidance
                    sampling_strength = self.sampling_cfg.guidance_strength_factor * extract(self.noise_schedule.sqrt_one_minus_alphas, t, z_t.shape) #TODO put in config file
                    noise_space_grad = z_t.grad
                    forward_guided_predicted_noise = predicted_noise + sampling_strength * noise_space_grad

                    predicted_noise = forward_guided_predicted_noise

                if self.sampling_cfg.backward_universal_guidance:
                    delta_z = torch.zeros_like(hat_z_0)
                    hat_z_0.detach_()
                    delta_z.requires_grad = True
                    for idx in range(self.sampling_cfg.backwards_steps_m):
                        loss = self.guidance_function(hat_z_0 + delta_z)
                        with torch.no_grad(): #IDK if this is needed, but I don't think it can hurt.
                            delta_z = delta_z - delta_z.grad 
                        delta_z.requires_grad = True
                    sqrt_alpha_over_one_minus_alphas = extract(self.noise_schedule.sqrt_alpha_over_one_minus_alphas, t, z_t.shape)
                    predicted_noise = predicted_noise - sqrt_alpha_over_one_minus_alphas * delta_z

                hat_z_0, hat_z_t_minus_one = self.S(z_t, predicted_noise, t)
                
                #Resample z
                epsilon_noise = torch.randn_like(hat_z_t_minus_one)
                sqrt_alpha_over_alpha_prev  = extract(self.noise_schedule.sqrt_alpha_over_alpha_prev, t, z_t.shape)
                sqrt_one_minus_alpha_over_alpha_prev = extract(self.noise_schedule.sqrt_one_minus_alpha_over_alpha_prev, t, z_t.shape)
                z_t = sqrt_alpha_over_alpha_prev * hat_z_t_minus_one + sqrt_one_minus_alpha_over_alpha_prev * epsilon_noise
                z_t.detach_()

            if t_index == 0:
                z_t = hat_z_0
            else:
                # Algorithm 2 line 4:
                z_t = hat_z_t_minus_one
            
            z_t.detach_()
            zs.append(z_t)
                # --------------
        z_0 =  zs[-1]
        # Now we'e sampled, save to self.poses_dir, set by testing script
        xyzs = z_0[:,:,:3]
        xyzs *= 0.1
        flattened_ortho6d = z_0[:,:,3:].reshape(-1, 6)
        flattened_rmats = compute_rotation_matrix_from_ortho6d(flattened_ortho6d)
        rmats = flattened_rmats.reshape(z_0.shape[0],z_0.shape[1], 3, 3)
        
        k = random.randint(0, 15)
        print(f"{k}th element in batch.")
        for i in range(z_0.shape[1]):
            print(f"XYZ: {xyzs[k][i]} ROTATION MATRIX:\n{rmats[k][i]}")

        print(f"Writing {xyzs.shape[0]} poses to {os.path.join(self.poses_dir, 'poses.pickle')}")
        with open(os.path.join(self.poses_dir, "poses.pickle"), 'wb') as file:
            pickle.dump((xyzs, rmats), file)

    def UG_S(self, x, predicted_noise, t):
        # Equation 3 in https://arxiv.org/pdf/2302.07121.pdf, ref. in Algorithm 1
        sqrt_recip_alphas_t = extract(self.noise_schedule.sqrt_recip_alphas, t, x.shape)
        sqrt_one_minus_alphas = extract(self.noise_schedule.sqrt_one_minus_alphas, t, x.shape)

        hat_z_0 = sqrt_recip_alphas_t * (x - sqrt_one_minus_alphas * predicted_noise )
        return hat_z_0
    
    def S(self, x, predicted_noise, t):
        betas_t = extract(self.noise_schedule.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.noise_schedule.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.noise_schedule.sqrt_recip_alphas, t, x.shape)

        #\hat{z_0} https://arxiv.org/pdf/2006.11239.pdf
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        hat_z_0 = model_mean

        posterior_variance_t = extract(self.noise_schedule.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4: \hat{z_t-1} https://arxiv.org/pdf/2006.11239.pdf
        hat_z_t_minus_one = model_mean + torch.sqrt(posterior_variance_t) * noise

        return hat_z_0, hat_z_t_minus_one

    def train_dataloader(self):
        train_dataset = DiffusionDataset(self.device, ds_roots=self.cfg.dataset.train_dirs) 
        
        return DataLoader(train_dataset, batch_size=self.cfg.dataset.batch_size, shuffle=True,
                                        pin_memory=self.cfg.dataset.pin_memory, num_workers=self.cfg.dataset.num_workers)
        
    def val_dataloader(self):
        valid_dataset = DiffusionDataset(self.device, ds_roots=self.cfg.dataset.valid_dirs)
        return DataLoader(valid_dataset, batch_size=self.cfg.dataset.batch_size, shuffle=False,
                                        pin_memory=self.cfg.dataset.pin_memory, num_workers=self.cfg.dataset.num_workers)
        