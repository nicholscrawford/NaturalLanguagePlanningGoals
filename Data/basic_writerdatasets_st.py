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

class AbstractDataset(Dataset):
    def __init__(self, device, ds_roots, clear_cache=False, max_size = None, resampling = False):

        self.resampling = resampling
        self.device = device
        self.images = []
        self.pointcloud_rgbs = []
        self.pointcloud_xyzs = []
        self.pointcloud_segs = []
        self.pointcloud_norms = []
        self.pointcloud_downsampled = []
        self.bb_paths = []
        self.seg_nums = [2,3,4,5] # This should be saved in the data, TODO. Not sure where this is coming from though.
        self.max_object_points = 256

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

                    # Cache pointclouds
                    pointcloud_downsampled_path = os.path.join(cwd, f"downsampled_pointcloud_{sim_idx_str}.pickle")
                    self.pointcloud_downsampled.append(pointcloud_downsampled_path)

                    if not os.path.exists(pointcloud_downsampled_path) or clear_cache:
                        
                        pointcloud_rgb = np.load(pointcloud_rgb_path)
                        pointcloud_xyz = np.load(pointcloud_xyz_path)
                        pointcloud_seg = np.load(pointcloud_seg_path)
                        
                        datapoint_pointclouds = []
                        unsampled_pointclouds = []
                        offset = [0] # Gives the offset an initial referance so we can add to it.
                        seg_nums = self.seg_nums # This should be saved in the data, TODO. Not sure where this is coming from though.

                        for class_index in seg_nums:
                            class_mask = (pointcloud_seg == class_index)
                            segmented_points_rgb = pointcloud_rgb[class_mask]
                            segmented_points_xyz = pointcloud_xyz[class_mask]

                            segmented_points_rgb = segmented_points_rgb[:, :3] / 255
                            num_obj_points = len(segmented_points_xyz)
                            
                            segmented_points_rgb = torch.tensor(segmented_points_rgb).unsqueeze(0)
                            segmented_points_xyz = torch.tensor(segmented_points_xyz).unsqueeze(0)

                            if  num_obj_points >= self.max_object_points:
                                point_idxs = farthest_point_sample(segmented_points_xyz, npoint=self.max_object_points)
                                ds_xyz = index_points(segmented_points_xyz, point_idxs).squeeze()
                                ds_rgb = index_points(segmented_points_rgb, point_idxs).squeeze()
                                
                                offset.append(offset[-1] + self.max_object_points)
                            elif num_obj_points == 0:
                                #print("Warning -- Object has no points")
                                ds_xyz = torch.zeros((1,3))
                                ds_rgb = torch.zeros((1,3))
                                offset.append(offset[-1] + 1)
                            elif num_obj_points == 1:
                                offset.append(offset[-1] + num_obj_points)    
                                ds_xyz = segmented_points_xyz.squeeze().unsqueeze(0)
                                ds_rgb = segmented_points_rgb.squeeze().unsqueeze(0)
                            else:
                                offset.append(offset[-1] + num_obj_points)    
                                ds_xyz = segmented_points_xyz.squeeze()
                                ds_rgb = segmented_points_rgb.squeeze()
                                
                            unsampled_pointclouds.append((segmented_points_xyz, segmented_points_rgb))
                            datapoint_pointclouds.append((ds_xyz, ds_rgb))
                            #o3d.visualization.draw_geometries([pcd])
                        #o3d.visualization.draw_geometries(datapoint_pointclouds)

                        offset = offset[1:]
                        with open(pointcloud_downsampled_path, "wb") as file:
                            # create one long list of all the points, with the offset to delineate them.
                            point_list = np.concatenate([
                                np.concatenate(xzys_rgbs, axis=1) for xzys_rgbs in datapoint_pointclouds 
                            ], axis = 0)
                            pickle.dump((point_list, offset), file)

                        # with open(pointcloud_downsampled_path, "wb") as file:
                        #         # cache unsampled segmented point list.
                        #         # point_list = [
                        #         #     (np.array(pc.points), np.array(pc.colors)) for pc in unsampled_pointclouds 
                        #         # ]
                        #         pickle.dump(unsampled_pointclouds, file)
        
        self.max_size = None
        if max_size:
            self.max_size = min(len(self), max_size)

    def __len__(self):
        if self.max_size:
            return self.max_size
        return len(self.images)
    
    def get_pointcloud(self, index):
        if not self.resampling:
            with open(self.pointcloud_downsampled[index], "rb") as file:
                datapoint_pointclouds, offset = pickle.load(file)
                datapoint_pointclouds = torch.tensor(datapoint_pointclouds, dtype=torch.double)
                offset = torch.tensor(offset, dtype = torch.double)
                
                if datapoint_pointclouds.shape[0] < 256 * 4:
                    return self.__getitem__(random.randint(0, self.__len__() - 1))
                
                datapoint_pointclouds = datapoint_pointclouds.reshape(4, 256 ,6)
            return datapoint_pointclouds
        else:
            with open(self.unsampled_pointclouds[index], "rb") as file:
                try:
                    datapoint_pointclouds = pickle.load(file)
                except:
                    #print(Unexpected EOF)
                    return False

                sampled_datapoint_pointclouds = []
                for object_points, object_colors in datapoint_pointclouds:
                    num_obj_points = len(object_points.squeeze())
                    if  num_obj_points >= self.max_object_points:
                        point_idxs = farthest_point_sample(object_points, npoint=self.max_object_points)
                        ds_xyz = index_points(object_points, point_idxs).squeeze()
                        ds_xyz = ds_xyz - ds_xyz.mean(dim=0)
                        ds_rgb = index_points(object_colors, point_idxs).squeeze()

                    elif num_obj_points == 0:
                        #print("Warning -- Object has no points")
                        return False
                    else:
                        return False
                    sampled_datapoint_pointclouds.append( np.concatenate((ds_xyz, ds_rgb), axis=1))
                datapoint_pointclouds = np.stack(sampled_datapoint_pointclouds)
                datapoint_pointclouds = torch.tensor(datapoint_pointclouds, dtype=torch.double)
            return datapoint_pointclouds
                # if datapoint_pointclouds.shape[0] < 256 * 6:
                #     return self.__getitem__(random.randint(0, self.__len__() - 1))
                
                # datapoint_pointclouds = datapoint_pointclouds.reshape(6, 256 ,6)

class CLIPEmbedderDataset(AbstractDataset):
    def __init__(self, preprocess, device, ds_roots):
        super(CLIPEmbedderDataset, self).__init__(device, ds_roots=ds_roots)
        self.preprocess = preprocess
    
    def get_from_image_name(self, name):
        for idx, pth in enumerate(self.images):
            if name in pth:
                return self.__getitem__(idx)
            
    def __getitem__(self, index):
        
        
        datapoint_pointclouds = self.get_pointcloud(index)
        if type(datapoint_pointclouds) == tuple:
            return datapoint_pointclouds #Hack to avoid low point objects, it's the return self.__getitem__ in the get_pointcloud func.
        if datapoint_pointclouds is False: #Give up if there's not enough points.
            #print(f"Warning: Not enough points for the datapoint with {self.unsampled_pointclouds[index]} as a component.")
            return self.__getitem__(random.randint(0, self.__len__() - 1))
        
        tfs = np.load(self.bb_paths[index])['transform']
        tfs = torch.tensor(tfs, dtype=torch.double).transpose(1, 2)

        try:
            image = self.preprocess(Image.open(self.images[index]))
        except:
            print(f"Warning: Image {self.images[index]} loaded as an object for some reason.")
            return self.__getitem__(random.randint(0, self.__len__() - 1))
        x = (datapoint_pointclouds, tfs)
        y = image
        return (x, y)
    
class DiffusionDataset(AbstractDataset):
    def __getitem__(self, index):
        
        datapoint_pointclouds = self.get_pointcloud(index)
        if type(datapoint_pointclouds) == tuple:
            return datapoint_pointclouds #Hack to avoid low point objects, it's the return self.__getitem__ in the get_pointcloud func.
        if datapoint_pointclouds is False: #Give up if there's not enough points.
            #print(f"Warning: Not enough points for the datapoint with {self.unsampled_pointclouds[index]} as a component.")
            return self.__getitem__(random.randint(0, self.__len__() - 1))
        
        tfs = np.load(self.bb_paths[index])['transform']
        tfs = torch.tensor(tfs).transpose(1, 2)
        
        x = (datapoint_pointclouds, tfs)
        return x

if __name__ == "__main__":        
    #AbstractDataset()
    #CLIPEmbedderDataset("cuda").__getitem__(0)    
    ((datapoint_pointclouds, transforms), image) = DiffusionDataset("cuda").__getitem__(0)
    print(f"points shape: {datapoint_pointclouds.shape}")
    print(f"transforms shape: {transforms.shape}")
    print(f"image shape: {image.shape}")
