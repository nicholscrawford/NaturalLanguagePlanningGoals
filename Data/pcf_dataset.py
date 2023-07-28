import torch
from torch.utils.data import Dataset
import os
from scipy.io import loadmat
from einops.einops import rearrange
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import random
from PIL import Image
from StructDiffusion.pointnet import farthest_point_sample, index_points
from torch_geometric.data import Data, Batch

# https://github.com/pytorch/pytorch/issues/13246#issuecomment-603823029

class AbstractDataset(Dataset):
    def __init__(self, device, ds_roots, clear_cache=True, max_size = None):

        self.device = device
        self.images = []
        self.pointcloud_rgbs = []
        self.pointcloud_xyzs = []
        self.pointcloud_segs = []
        self.pointcloud_norms = []
        self.bb_paths = []
        self.seg_nums = [2,3,4,5] # This should be saved in the data, TODO. Not sure where this is coming from though.

        for ds_root in ds_roots:
            for folder_idx, folder in tqdm(enumerate( os.listdir(ds_root)), desc=f"Lding {ds_root.split('/')[-2]}", total=len(os.listdir(ds_root))):
                cwd = os.path.join(ds_root, folder)
                files = os.listdir(cwd)
                # Get image locations
                imgs = [os.path.join(cwd, filename) for filename in files if "png" in filename]            
                for sim_idx_str in [name.split('/')[-1].split('_')[-1].split('.')[0] for name in imgs]:                
                    bb_path = os.path.join(cwd, f"bounding_box_3D_{sim_idx_str}.npy")
                    pointcloud_xyz_path = os.path.join(cwd, f"pointcloud_{sim_idx_str}.npy")
                    pointcloud_rgb_path = os.path.join(cwd, f"pointcloud_rgb_{sim_idx_str}.npy")
                    pointcloud_seg_path = os.path.join(cwd, f"pointcloud_semantic_{sim_idx_str}.npy")
                    pointcloud_normal_path = os.path.join(cwd, f"pointcloud_normals_{sim_idx_str}.npy")
                    image_path = os.path.join(cwd, f"rgb_{sim_idx_str}.png")
                    
                    # Save image path 
                    self.images.append(image_path)
                    self.pointcloud_rgbs.append(pointcloud_rgb_path)
                    self.pointcloud_xyzs.append(pointcloud_xyz_path)
                    self.pointcloud_segs.append(pointcloud_seg_path)
                    self.pointcloud_norms.append(pointcloud_normal_path)
                    self.bb_paths.append(bb_path)
        
        self.max_size = None
        if max_size:
            self.max_size = min(len(self), max_size)

        self.images = np.array(self.images).astype(np.string_)
        self.pointcloud_rgbs = np.array(self.pointcloud_rgbs).astype(np.string_)
        self.pointcloud_xyzs = np.array(self.pointcloud_xyzs).astype(np.string_)
        self.pointcloud_segs = np.array(self.pointcloud_segs).astype(np.string_)
        self.pointcloud_norms = np.array(self.pointcloud_norms).astype(np.string_)
        self.bb_paths = np.array(self.bb_paths).astype(np.string_)
        
    def __len__(self):
        if self.max_size:
            return self.max_size
        return len(self.images)
    
    def get_from_image_name(self, name):
        for idx, pth in enumerate(self.images):
            if name in pth.decode('UTF-8'):
                return self.__getitem__(idx)
    
    def get_pointcloud_rgb_xyz_norms(self, index):
        pointcloud_rgb = np.load(self.pointcloud_rgbs[index].decode('UTF-8'))
        pointcloud_xyz = np.load(self.pointcloud_xyzs[index].decode('UTF-8'))
        pointcloud_norms = np.load(self.pointcloud_norms[index].decode('UTF-8'))
        pointcloud_seg = np.load(self.pointcloud_segs[index].decode('UTF-8'))

        data_list = []

        for object_index, class_index in enumerate(self.seg_nums):
            class_mask = (pointcloud_seg == class_index)
            segmented_points_rgb = pointcloud_rgb[class_mask]
            segmented_points_xyz = pointcloud_xyz[class_mask]
            segmented_points_norms = pointcloud_norms[class_mask][:, :3] # Make norms regular cords, not homogenous.
            segmented_points_rgb = segmented_points_rgb[:, :3] / 255 # Remove alpha channel and norm to (0, 1)

            if len(segmented_points_rgb) == 0:
                segmented_points_rgb = np.array([[0.0, 0.0, 0.0]])
                segmented_points_xyz = np.array([[0.0, 0.0, 0.0]])
                segmented_points_norms = np.array([[0.0, 0.0, 0.0]])

            data = Data(pos=segmented_points_xyz, norm=segmented_points_norms, rgb=segmented_points_rgb)
            data_list.append(data)

        return data_list

    def get_transforms(self, index):
        tfs = np.load(self.bb_paths[index].decode('UTF-8'))['transform']
        tfs = torch.tensor(tfs).transpose(1, 2)
        return tfs

    def get_image(self, index):
        image = self.preprocess(Image.open(self.images[index].decode('UTF-8')))
        return image
    
class CLIPEmbedderDataset(AbstractDataset):
    def __init__(self, preprocess, device, ds_roots, max_size = None):
        super(CLIPEmbedderDataset, self).__init__(device, ds_roots=ds_roots, max_size=max_size)
        self.preprocess = preprocess
        
    def __getitem__(self, index):
        
        
        datapoint_pointclouds = self.get_pointcloud_rgb_xyz_norms(index)
        
        transforms = self.get_transforms(index)
        try:
            image = self.get_image(index)
        except:
            print(f"Warning: Image {self.images[index]} loaded as an object for some reason.")
            return self.__getitem__(random.randint(0, self.__len__() - 1))
        
        #if not ()

        x = (datapoint_pointclouds, transforms)
        y = image
        return (x, y)
    
    def collate_fn(self, data_list):
        transforms_list = []
        images = []

        npoints = sum(sum(element.pos.shape[0] for element in datum[0][0]) for datum in data_list)
        nobjs = sum(sum(1 for element in datum[0][0]) for datum in data_list)
        pc_rgbs = np.empty((npoints, 3), dtype=np.float32)
        pc_xyzs = np.empty((npoints, 3), dtype=np.float32)
        pc_norms = np.empty((npoints, 3), dtype=np.float32)
        encoder_batch_idxs = np.empty(npoints, dtype=np.int)
        post_encoder_batch_idxs = np.empty(nobjs, dtype=np.int)

        encoder_batch_idx = 0
        encoder_batch_idx_start = 0
        post_encoder_batch_idx = 0
        post_encoder_batch_idx_start = 0
        for element in data_list:
            (pointcloud_data, transforms), image = element
            transforms_list.append(transforms.unsqueeze(0))
            images.append(image.unsqueeze(0))

            end_index = post_encoder_batch_idx_start + len(pointcloud_data)
            post_encoder_batch_idxs[post_encoder_batch_idx_start: end_index] =post_encoder_batch_idx
            post_encoder_batch_idx += 1
            post_encoder_batch_idx_start = end_index
            for object_pointcloud in pointcloud_data:
                end_index = encoder_batch_idx_start + object_pointcloud.pos.shape[0]
                pc_rgbs[encoder_batch_idx_start:end_index] = object_pointcloud.rgb
                pc_xyzs[encoder_batch_idx_start:end_index] = object_pointcloud.pos
                pc_norms[encoder_batch_idx_start:end_index] = object_pointcloud.norm
                encoder_batch_idxs[encoder_batch_idx_start:end_index] = encoder_batch_idx
                encoder_batch_idx += 1
                encoder_batch_idx_start = end_index
            
        transforms_batch = torch.cat(transforms_list)
        images_batch = torch.cat(images)

        device = "cuda"
        dtype = torch.float64
        pc_rgbs = torch.tensor(pc_rgbs).to(device).to(dtype)
        pc_xyzs = torch.tensor(pc_xyzs).to(device).to(dtype)
        pc_norms = torch.tensor(pc_norms).to(device).to(dtype)
        encoder_batch_idxs = torch.tensor(encoder_batch_idxs).to(device)
        post_encoder_batch_idxs = torch.tensor(post_encoder_batch_idxs).to(device)
        transforms_batch = transforms_batch.to(device).to(dtype)

        return ((pc_rgbs, pc_xyzs, pc_norms, encoder_batch_idxs, post_encoder_batch_idxs, transforms_batch), images_batch)
    
class DiffusionDataset(AbstractDataset):
    def __getitem__(self, index):
        
        datapoint_pointclouds = self.get_pointcloud_rgb_xyz_norms(index)
        
        transforms = self.get_transforms(index)
        
        x = (datapoint_pointclouds, transforms)
        return x
    
    def collate_fn(self, data_list):
        transforms_list = []
        images = []

        npoints = sum(sum(element.pos.shape[0] for element in datum[0]) for datum in data_list)
        nobjs = sum([len(objs) for objs in [datum[0] for datum in data_list]])
        pc_rgbs = np.empty((npoints, 3), dtype=np.float32)
        pc_xyzs = np.empty((npoints, 3), dtype=np.float32)
        pc_norms = np.empty((npoints, 3), dtype=np.float32)
        encoder_batch_idxs = np.empty(npoints, dtype=np.int)
        post_encoder_batch_idxs = np.empty(nobjs, dtype=np.int)

        encoder_batch_idx = 0
        encoder_batch_idx_start = 0
        post_encoder_batch_idx = 0
        post_encoder_batch_idx_start = 0
        for element in data_list:
            pointcloud_data, transforms = element
            transforms_list.append(transforms.unsqueeze(0))

            end_index = post_encoder_batch_idx_start + len(pointcloud_data)
            post_encoder_batch_idxs[post_encoder_batch_idx_start: end_index] =post_encoder_batch_idx
            post_encoder_batch_idx += 1
            post_encoder_batch_idx_start = end_index
            for object_pointcloud in pointcloud_data:
                end_index = encoder_batch_idx_start + object_pointcloud.pos.shape[0]
                pc_rgbs[encoder_batch_idx_start:end_index] = object_pointcloud.rgb
                pc_xyzs[encoder_batch_idx_start:end_index] = object_pointcloud.pos
                pc_norms[encoder_batch_idx_start:end_index] = object_pointcloud.norm
                encoder_batch_idxs[encoder_batch_idx_start:end_index] = encoder_batch_idx
                encoder_batch_idx += 1
                encoder_batch_idx_start = end_index
            
        transforms_batch = torch.cat(transforms_list)

        device = "cuda"
        dtype = torch.float64
        pc_rgbs = torch.tensor(pc_rgbs).to(device).to(dtype)
        pc_xyzs = torch.tensor(pc_xyzs).to(device).to(dtype)
        pc_norms = torch.tensor(pc_norms).to(device).to(dtype)
        encoder_batch_idxs = torch.tensor(encoder_batch_idxs).to(device)
        post_encoder_batch_idxs = torch.tensor(post_encoder_batch_idxs).to(device)
        transforms_batch = transforms_batch.to(device).to(dtype)

        return (pc_rgbs, pc_xyzs, pc_norms, encoder_batch_idxs, post_encoder_batch_idxs, transforms_batch)


if __name__ == "__main__":        
    #AbstractDataset()
    import clip
    model, preprocess = clip.load("ViT-B/32", device="cuda")
    datapoint = CLIPEmbedderDataset(preprocess, "cuda", ds_roots=["/home/nicholscrawfordtaylor/data/NaturalLanguagePlanningGoals/jul4"]).__getitem__(0)    
    print(datapoint)
    # ((datapoint_pointclouds, transforms), image) = DiffusionDataset("cuda").__getitem__(0)
    # print(f"points shape: {datapoint_pointclouds.shape}")
    # print(f"transforms shape: {transforms.shape}")
    # print(f"image shape: {image.shape}")
