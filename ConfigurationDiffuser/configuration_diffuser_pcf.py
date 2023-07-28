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

from Data.pcf_dataset import DiffusionDataset
from StructDiffusion.encoders import DropoutSampler, EncoderMLP
from ConfigurationDiffuser.diffusion_utils import NoiseSchedule, extract, get_diffusion_variables, q_sample
from StructDiffusion.rotation_continuity import \
    compute_rotation_matrix_from_ortho6d
from pointconvformer_wrapper import PointConvFormerEncoder


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
        self.point_encoder = PointConvFormerEncoder(
            in_dim=3,
            out_dim=256,
            pool='max',
            hack=True
        )

        # 256 = 80 (point cloud) + 80 (position) + 80 (time) + 16 (position idx)
        # Important: we set uses_pt to true because we want the model to consider the positions of objects that
        #  don't need to be rearranged.
        self.mlp = EncoderMLP(256, 80, uses_pt=True)
        self.position_encoder = nn.Sequential(nn.Linear(3 + 6, 80))
        #self.start_token_embeddings = torch.nn.Embedding(1, 256)

        # max number of objects or max length of sentence is 7
        self.position_embeddings = torch.nn.Embedding(7, 16)

        encoder_layers = TransformerEncoderLayer(
            432, cfg.model.num_attention_heads, cfg.model.encoder_hidden_dim, cfg.model.encoder_dropout, cfg.model.encoder_activation
            )
        self.encoder = TransformerEncoder(encoder_layers, cfg.model.encoder_num_layers)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(80),
            nn.Linear(80, 80),
            nn.GELU(),
            nn.Linear(80, 80),
        )

        self.obj_dist = DropoutSampler(432, 3 + 6, dropout_rate=cfg.model.object_dropout)

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

    def create_batch(self, pointclouds_embed, transforms_embed, position_embed, time_embed, post_encoder_batch_idxs):
        max_seq_len = post_encoder_batch_idxs.bincount().max()
        batch_size = post_encoder_batch_idxs.bincount().shape[0]
        feature_length = pointclouds_embed.shape[1] #+ transforms_embed.shape[2] + position_embed.shape[2] + time_embed.shape[2]
        pose_encodings_feature_length = transforms_embed.shape[2]
        padded_sequences = torch.empty((max_seq_len, batch_size, feature_length), device="cuda", dtype=torch.double)
        src_key_padding_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool, device="cuda")
        
        for batch_idx in range(post_encoder_batch_idxs.max() + 1):
            seq_len = (post_encoder_batch_idxs == batch_idx).sum()

            # Object embeddings - Assuming pointclouds_embed contains object embeddings
            object_encodings = pointclouds_embed[post_encoder_batch_idxs == batch_idx]


            # Pad the sequences and create the mask
            padded_sequences[:seq_len, batch_idx, :] = object_encodings
            src_key_padding_mask[batch_idx, seq_len:] = True
        
        padded_sequences = padded_sequences.transpose(0, 1)
        # Concatenate all embeddings (object, pose, position, and time) #Fix this object_encodings.shape = torch.Size([4, 256])
        concatenated_embeddings = torch.cat((padded_sequences, transforms_embed, position_embed, time_embed), dim=2)
        
        return concatenated_embeddings, src_key_padding_mask
    
    def _forward(self,t, pc_rgbs, pc_xyzs, pc_norms, encoder_batch_idxs, post_encoder_batch_idxs, transforms_t):
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

        #########################
        batch_size = transforms_t.shape[0]

        pointclouds_embed = self.point_encoder(pc_xyzs, pc_rgbs, encoder_batch_idxs, norms=pc_norms)# gives batch_size num_objs embed_dim
        transforms_embed = self.position_encoder(transforms_t)
        position_embed = self.position_embeddings(torch.tensor([[0,1,2,3] for _ in range(batch_size)], device='cuda'))

        time_embed = self.time_mlp(t)  # B, dim update commented shape
        num_target_objects = 4
        time_embed = time_embed.unsqueeze(1).repeat(1, num_target_objects, 1)  # B, N, dim
        
        padded_sequences, mask = self.create_batch(pointclouds_embed, transforms_embed, position_embed, time_embed, post_encoder_batch_idxs)
        
        #objects_embed = torch.cat([transforms_embed, pointclouds_embed], dim=-1)  # gives batch_size num_objs embed_dim

        #########################
        
        #sequence_embed = torch.cat([time_embed, objects_embed, position_embed], dim=-1)

        #########################
        # input to transformer needs to have dimenion [sequence_length, batch size, encoder input dimension]
        sequence_embed = einops.rearrange(padded_sequences,"batch seq_len enc_dim -> seq_len batch enc_dim")
        mask = einops.rearrange(mask,"batch seq_len -> seq_len batch")

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
        pc_rgbs, pc_xyzs, pc_norms, encoder_batch_idxs, post_encoder_batch_idxs, transforms_batch = batch
        B = transforms_batch.shape[0]

        t = torch.randint(0, self.noise_schedule.timesteps, (B,), dtype=torch.long).to(self.device, non_blocking=True)

        #--------------
        transforms_0 = get_diffusion_variables(transforms_batch)
        noise = torch.randn_like(transforms_0, device=self.device)
        transforms_t = q_sample(x_start=transforms_0, t=t, noise_schedule=self.noise_schedule, noise=noise)

        predicted_noise = self._forward(t, pc_rgbs, pc_xyzs, pc_norms, encoder_batch_idxs, post_encoder_batch_idxs, transforms_t)

        loss = self.loss_function(noise, predicted_noise)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.cfg.dataset.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pc_rgbs, pc_xyzs, pc_norms, encoder_batch_idxs, post_encoder_batch_idxs, transforms_batch = batch
        B = transforms_batch.shape[0]

        t = torch.randint(0, self.noise_schedule.timesteps, (B,), dtype=torch.long).to(self.device, non_blocking=True)

        #--------------
        transforms_0 = get_diffusion_variables(transforms_batch)
        noise = torch.randn_like(transforms_0, device=self.device)
        transforms_t = q_sample(x_start=transforms_0, t=t, noise_schedule=self.noise_schedule, noise=noise)

        predicted_noise = self._forward(t, pc_rgbs, pc_xyzs, pc_norms, encoder_batch_idxs, post_encoder_batch_idxs, transforms_t)

        betas_t = extract(self.noise_schedule.betas, t, transforms_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.noise_schedule.sqrt_one_minus_alphas_cumprod, t, transforms_t.shape)
        sqrt_recip_alphas_t = extract(self.noise_schedule.sqrt_recip_alphas, t, transforms_t.shape)
        hat_transforms_0 = sqrt_recip_alphas_t * (transforms_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        ddim_hat_z_0 = self.UG_S(transforms_t, predicted_noise, t)
        loss = self.loss_function(transforms_0, ddim_hat_z_0)
        self.log("val_ddim_pred_clean_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,  batch_size=self.cfg.dataset.batch_size)

        loss = self.loss_function(noise, predicted_noise)
        self.log("val_noise_pred_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.cfg.dataset.batch_size)
        return loss
    
    def test_step(self, batch, batch_idx):
        if self.sampling_cfg.ddim:
            if self.sampling_cfg.guidance_sampling:
                with torch.inference_mode(mode=False): #Inference Mode != Eval 
                    self.ddim_guided_sample(batch, batch_idx, self.sampling_cfg.ddim_steps)
            else:
                self.ddim_sample(batch, batch_idx, self.sampling_cfg.ddim_steps)
        else:
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
    
    def ddim_sample(self, batch, batch_idx, num_steps):
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
        timesteps = range(0, self.noise_schedule.timesteps, self.noise_schedule.timesteps // num_steps)
        for step_idx in reversed(range(len(timesteps))):
            t_index = timesteps[step_idx]

            t = torch.full((B,), t_index, device=self.device, dtype=torch.long)

            predicted_noise = self._forward(t, xyzs, rgbs, z_t)

            hat_z_0 = self.UG_S(z_t, predicted_noise, t)
            if t_index == 0:
                break
            
            #Previous timesteps's cumulative noise
            t_tau_minus_one_index = timesteps[step_idx-1]
            t_tau_minus_one = torch.full((B,), t_tau_minus_one_index, device=self.device, dtype=torch.long)
            sqrt_one_minus_alpha_cumprod_posterior = extract(self.noise_schedule.sqrt_one_minus_alphas_cumprod, t_tau_minus_one, z_t.shape)
            sqrt_alpha_cumprod_posterior = extract(self.noise_schedule.sqrt_alphas_cumprod, t_tau_minus_one, z_t.shape)
            
            # Eq 12 in DDIM paper with noted special case sigma = 0
            hat_z_t_minus_one = hat_z_0 * sqrt_alpha_cumprod_posterior + sqrt_one_minus_alpha_cumprod_posterior * predicted_noise

            #Go to next step
            z_t = hat_z_t_minus_one
            zs.append(z_t)
                # --------------
        z_0 =  hat_z_0
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
                    guidance_loss = self.guidance_function(xyzs, rgbs ,hat_z_0)
                    sampling_strength = self.sampling_cfg.guidance_strength_factor * extract(self.noise_schedule.sqrt_one_minus_alphas_cumprod, t, z_t.shape)
                    noise_space_grad = z_t.grad
                    forward_guided_predicted_noise = predicted_noise + sampling_strength * noise_space_grad

                    predicted_noise = forward_guided_predicted_noise

                if self.sampling_cfg.backward_universal_guidance: 
                    hat_z_0.detach_()
                    delta_z = torch.zeros_like(hat_z_0, requires_grad=True)
                    optimizer = torch.optim.Adam([delta_z], lr=self.sampling_cfg.backward_guidance_lr)

                    for idx in range(self.sampling_cfg.backwards_steps_m):
                        optimizer.zero_grad()
                        loss = self.guidance_function(xyzs, rgbs, hat_z_0 + delta_z)
                        loss.mean().backward()
                        optimizer.step()

                    sqrt_alpha_cumprod_over_one_minus_alpha_cumprod = extract(self.noise_schedule.sqrt_alphas_cumprod_over_one_minus_alphas_cumprod, t, z_t.shape) # UG Paper (https://arxiv.org/pdf/2302.07121.pdf) Eq. 9, alpha's here were defined relative to the ddpm paper, and alphas in the ug paper refer to cumulative product of alphas in the ddpm paper.
                    predicted_noise = predicted_noise - sqrt_alpha_cumprod_over_one_minus_alpha_cumprod * delta_z

                hat_z_0, hat_z_t_minus_one = self.S(z_t, predicted_noise, t) # Literally just don't use this.
                
                #Resample z_t
                epsilon_noise = torch.randn_like(hat_z_t_minus_one)
                sqrt_alpha_cumprod  = extract(self.noise_schedule.sqrt_alphas_cumprod, t, z_t.shape)
                sqrt_alpha_cumprod_prev  = extract(self.noise_schedule.sqrt_alphas_cumprod_prev, t, z_t.shape)
                sqrt_alpha_over_alpha_prev = sqrt_alpha_cumprod / sqrt_alpha_cumprod_prev

                sqrt_one_minus_alpha_cumprod = extract(self.noise_schedule.sqrt_one_minus_alphas_cumprod, t, z_t.shape)
                sqrt_one_minus_alpha_over_alpha_prev = sqrt_one_minus_alpha_cumprod / sqrt_alpha_cumprod_prev

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

    #Refactoring a bit could be useful, or mabye just deleting non ddim sampling.
    def ddim_guided_sample(self, batch, batch_idx, num_steps):
        loss_printing = True
        update_z_zero_mid_guidance = False #This may decrease performance?
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

        timesteps = range(0, self.noise_schedule.timesteps, self.noise_schedule.timesteps // num_steps)
        for step_idx in reversed(range(len(timesteps))):
            t_index = timesteps[step_idx]

            t = torch.full((B,), t_index, device=self.device, dtype=torch.long)

            num_recurrance_steps = 1 if not self.sampling_cfg.per_step_self_recurrance else self.sampling_cfg.per_step_k

            for _ in range(num_recurrance_steps):
                predicted_noise = self._forward(t, xyzs, rgbs, z_t)
                z_t.requires_grad = True

                hat_z_0 = self.UG_S(z_t, predicted_noise, t)

                if self.sampling_cfg.forward_universal_guidance:
                    guidance_loss = self.guidance_function(xyzs, rgbs ,hat_z_0)
                    if loss_printing: print("\033[0mPre forward loss: \033[93m", guidance_loss.mean().item())
                    guidance_loss.mean().backward()
                    #Forward guidance
                    sampling_strength = self.sampling_cfg.guidance_strength_factor * extract(self.noise_schedule.sqrt_one_minus_alphas_cumprod, t, z_t.shape)
                    noise_space_grad = z_t.grad
                    forward_guided_predicted_noise = predicted_noise + sampling_strength * noise_space_grad

                    predicted_noise = forward_guided_predicted_noise

                # Updating z_0 before backward guidance?
                if update_z_zero_mid_guidance: hat_z_0 = self.UG_S(z_t, predicted_noise, t)
                if loss_printing:
                    hat_z_0_ = self.UG_S(z_t, predicted_noise, t)
                    guidance_loss = self.guidance_function(xyzs, rgbs ,hat_z_0_)
                    print("\033[0mPost forward loss: \033[91m", guidance_loss.mean().item())

                if self.sampling_cfg.backward_universal_guidance:
                    hat_z_0.detach_()
                    delta_z = torch.zeros_like(hat_z_0, requires_grad=True)
                    #optimizer = torch.optim.SGD([delta_z], lr=self.sampling_cfg.backward_guidance_lr)
                    optimizer = torch.optim.Adam([delta_z], lr=self.sampling_cfg.backward_guidance_lr)
                    for idx in range(self.sampling_cfg.backwards_steps_m):
                        optimizer.zero_grad()
                        loss = self.guidance_function(xyzs, rgbs, hat_z_0 + delta_z) #Why is there incosistent loss with no delta change?
                        if loss_printing: print("\033[0mMid backward loss: \033[96m", loss.mean().item())
                        loss.mean().backward()
                        optimizer.step()

                    sqrt_alpha_cumprod_over_one_minus_alpha_cumprod = extract(self.noise_schedule.sqrt_alphas_cumprod_over_one_minus_alphas_cumprod, t, z_t.shape)  # UG Paper (https://arxiv.org/pdf/2302.07121.pdf) Eq. 9, alpha's here were defined relative to the ddpm paper, and alphas in the ug paper refer to cumulative product of alphas in the ddpm paper.
                    predicted_noise = predicted_noise - sqrt_alpha_cumprod_over_one_minus_alpha_cumprod * delta_z

                #With backward guidance done, we need to move on to our next step.
                hat_z_0 = self.UG_S(z_t, predicted_noise, t)
                if loss_printing:
                    loss = self.guidance_function(xyzs, rgbs, hat_z_0)
                    print("\033[0mPost backward loss: \033[94m", loss.mean().item())

                if t_index == 0:
                    break

                #Previous timesteps's cumulative noise
                t_tau_minus_one_index = timesteps[step_idx-1]
                t_tau_minus_one = torch.full((B,), t_tau_minus_one_index, device=self.device, dtype=torch.long)
                sqrt_one_minus_alpha_cumprod_posterior = extract(self.noise_schedule.sqrt_one_minus_alphas_cumprod, t_tau_minus_one, z_t.shape)
                sqrt_alpha_cumprod_posterior = extract(self.noise_schedule.sqrt_alphas_cumprod, t_tau_minus_one, z_t.shape)
            
                # Eq 12 in DDIM paper with noted special case sigma = 0
                hat_z_t_minus_one = hat_z_0 * sqrt_alpha_cumprod_posterior + sqrt_one_minus_alpha_cumprod_posterior * predicted_noise

                # New noise (reintroducing variability lol)
                epsilon_prime = torch.randn_like(hat_z_t_minus_one)

                # Resample z_t
                sqrt_alpha_cumprod = extract(self.noise_schedule.sqrt_alphas_cumprod, t, z_t.shape)
                sqrt_one_minus_alpha_cumprod = extract(self.noise_schedule.sqrt_one_minus_alphas_cumprod, t, z_t.shape)
                z_t = (sqrt_alpha_cumprod / sqrt_alpha_cumprod_posterior) * hat_z_t_minus_one + (sqrt_one_minus_alpha_cumprod / sqrt_alpha_cumprod_posterior ) * epsilon_prime
                z_t.detach_()

            #Go to next step
            z_t = hat_z_t_minus_one
            zs.append(z_t)
            z_t.detach_()
                # --------------
        z_0 =  hat_z_0
        # Now we'e sampled, save to self.poses_dir, set by testing script
        translations = z_0[:,:,:3]
        translations *= 0.1
        flattened_ortho6d = z_0[:,:,3:].reshape(-1, 6)
        flattened_rmats = compute_rotation_matrix_from_ortho6d(flattened_ortho6d)
        rmats = flattened_rmats.reshape(z_0.shape[0],z_0.shape[1], 3, 3)
        
        # k = random.randint(0, 15)
        # print(f"{k}th element in batch.")
        # for i in range(z_0.shape[1]):
        #     print(f"XYZ: {translations[k][i]} ROTATION MATRIX:\n{rmats[k][i]}")

        # print(f"Writing {translations.shape[0]} poses to {os.path.join(self.poses_dir, 'poses.pickle')}")
        # with open(os.path.join(self.poses_dir, "poses.pickle"), 'wb') as file:
        #     pickle.dump((translations, rmats), file)

        if self.guidance_function:
            self.guidance_alignment = self.guidance_function(xyzs, rgbs, hat_z_0)

        return 

        

    def UG_S(self, x, predicted_noise, t):
        # Equation 3 in https://arxiv.org/pdf/2302.07121.pdf, ref. in Algorithm 1
        #sqrt_recip_alphas_t = extract(self.noise_schedule.sqrt_recip_alphas, t, x.shape)
        #sqrt_one_minus_alphas = extract(self.noise_schedule.sqrt_one_minus_alphas, t, x.shape)

        #hat_z_0 = sqrt_recip_alphas_t * (x - sqrt_one_minus_alphas * predicted_noise )
        
        sqrt_one_minus_alphas_cumprod = extract(self.noise_schedule.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_cumprod = extract(self.noise_schedule.sqrt_recip_alphas_cumprod, t, x.shape)

        hat_z_0 = sqrt_recip_alphas_cumprod * (x - sqrt_one_minus_alphas_cumprod * predicted_noise )
        return hat_z_0
    
    def S(self, x, predicted_noise, t):
        betas_t = extract(self.noise_schedule.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.noise_schedule.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.noise_schedule.sqrt_recip_alphas, t, x.shape)

        #\hat{z_0} https://arxiv.org/pdf/2006.11239.pdf
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        hat_z_0 = model_mean #This is not actually z_0, this was a mistake on my part.

        posterior_variance_t = extract(self.noise_schedule.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4: \hat{z_t-1} https://arxiv.org/pdf/2006.11239.pdf
        hat_z_t_minus_one = model_mean + torch.sqrt(posterior_variance_t) * noise

        return hat_z_0, hat_z_t_minus_one

    def train_dataloader(self):
        train_dataset = DiffusionDataset(self.device, ds_roots=self.cfg.dataset.train_dirs) 
        
        return DataLoader(train_dataset, batch_size=self.cfg.dataset.batch_size, shuffle=True,
                                        pin_memory=self.cfg.dataset.pin_memory, num_workers=self.cfg.dataset.num_workers,
                                        collate_fn=train_dataset.collate_fn)
        
    def val_dataloader(self):
        valid_dataset = DiffusionDataset(self.device, ds_roots=self.cfg.dataset.valid_dirs)
        return DataLoader(valid_dataset, batch_size=self.cfg.dataset.batch_size, shuffle=False,
                                        pin_memory=self.cfg.dataset.pin_memory, num_workers=self.cfg.dataset.num_workers,
                                        collate_fn=valid_dataset.collate_fn)
        

