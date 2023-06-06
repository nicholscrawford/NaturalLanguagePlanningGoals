import torch
from torch.utils.data import Dataset
import os
from scipy.io import loadmat
from einops.einops import rearrange
from PIL import Image
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from open3d.geometry import PointCloud
import random
from PIL import Image

class _PointCloudTransmissionFormat:
    def __init__(self, pointcloud: PointCloud):
        self.points = np.array(pointcloud.points)
        self.colors = np.array(pointcloud.colors)

    def create_pointcloud(self) -> PointCloud:
        pointcloud = PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(self.points)
        pointcloud.colors = o3d.utility.Vector3dVector(self.colors)
        return pointcloud

class AbstractDataset(Dataset):
    def __init__(self, device, ds_roots, clear_cache=False, max_size = None):

        # This strategy relies on consistent ordering in each array, and in each arrays construction. It may be better to make the relationship between each portion of the datapoint explicit.
        self.device = device
        self.images = []
        self.pointclouds = []
        self.object_transforms = []
        self.cls_indexes = [] # Segmentation labels
        # camera_homo = np.array([
        #     [1.0000000,  0.0000000,  0.0000000, 0],
        #     [0.0000000, -0.7660444,  0.6427876, 1.2],
        #     [0.0000000, -0.6427876, -0.7660444, -1.2],
        #     [0.0000000,  0.0000000,  0.0000000, 1.000000]
        #     ])
        self.max_object_points = 256
        
        # Check for cached folders?

        for ds_root in ds_roots:
            for folder_idx, folder in tqdm(enumerate( os.listdir(ds_root)), desc=f"Loading dataset root folder {ds_root}", total=len(os.listdir(ds_root))):
                cwd = os.path.join(ds_root, folder)
                files = os.listdir(cwd)
                self.sim_idx = []
                # Get image locations
                imgs = [os.path.join(cwd, filename) for filename in files if "png" in filename]            
                
                for sim_idx_str in [name[-8:-4] for name in imgs]:
                    
                    bb_path = os.path.join(cwd, f"bounding_box_3d_{sim_idx_str}.npy")
                    pointcloud_xyz_path = os.path.join(cwd, f"pointcloud_{sim_idx_str}.npy")
                    pointcloud_rgb_path = os.path.join(cwd, f"pointcloud_rgb_{sim_idx_str}.npy")
                    pointcloud_seg_path = os.path.join(cwd, f"pointcloud_semantic_{sim_idx_str}.npy")
                    image_path = os.path.join(cwd, f"rgb_{sim_idx_str}.png")
                    
                    # Save image path 
                    self.images.append(image_path)
                    
                    # Save poses
                    #poses = rearrange(poses, "height width object_idx -> object_idx height width")
                    poses = torch.tensor(np.load(bb_path)['transform']).transpose(1,2)
                    self.object_transforms.append(poses)

                    # Cache pointclouds
                    pointcloud_cache_path = os.path.join(cwd, f"downsampled_pointcloud_{sim_idx_str}.pickle")
                    self.pointclouds.append(pointcloud_cache_path)

                    if not os.path.exists(pointcloud_cache_path) or clear_cache:
                        
                        pointcloud_rgb = np.load(pointcloud_rgb_path)
                        pointcloud_xyz = np.load(pointcloud_xyz_path)
                        pointcloud_seg = np.load(pointcloud_seg_path)
                        
                        datapoint_pointclouds = []
                        offset = [0] # Gives the offset an initial referance so we can add to it.
                        #num_objs = len(cls_idxs[0])
                        seg_nums = [2,3,4,5,6,7] # This should be saved in the data, TODO. Not sure where this is coming from though.
                        #self.cls_indexes.append(cls_idxs)
                        self.cls_indexes.append(seg_nums)
                        for class_index in seg_nums:
                            class_mask = (pointcloud_seg == class_index)
                            segmented_points_rgb = pointcloud_rgb[class_mask]
                            segmented_points_xyz = pointcloud_xyz[class_mask]
                            
                            #segmented_points = rearrange(points, "h w c -> (h w) c")
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(segmented_points_xyz)
                            pcd.colors = o3d.utility.Vector3dVector(segmented_points_rgb[:, :3] / 255)
                            num_obj_points = len(pcd.points)
                            
                            #o3d.visualization.draw_geometries([pcd])
                            #print(num_obj_points)
                            if  num_obj_points >= self.max_object_points:
                                pcd = pcd.farthest_point_down_sample(self.max_object_points)
                                offset.append(offset[-1] + self.max_object_points)
                            elif num_obj_points == 0:
                                #print("Warning -- Object has no points")
                                pcd.points = o3d.utility.Vector3dVector(np.zeros((1,3)))
                                pcd.colors = o3d.utility.Vector3dVector(np.zeros((1,3)))
                                offset.append(offset[-1] + 1)
                            else:
                                offset.append(offset[-1] + num_obj_points)    
                            
                            datapoint_pointclouds.append(pcd)    
                            #o3d.visualization.draw_geometries([pcd])
                        #o3d.visualization.draw_geometries(datapoint_pointclouds)

                        offset = offset[1:]
                        with open(pointcloud_cache_path, "wb") as file:
                            # create one long list of all the points, with the offset to delineate them.
                            point_list = np.concatenate([
                                np.concatenate((np.array(pc.points), np.array(pc.colors)), axis=1) for pc in datapoint_pointclouds 
                            ], axis = 0)
                            pickle.dump((point_list, offset), file)
        
        self.max_size = None
        if max_size:
            self.max_size = min(len(self), max_size)

    def __len__(self):
        if self.max_size:
            return self.max_size
        return len(self.images)

class CLIPEmbedderDataset(AbstractDataset):
    def __init__(self, preprocess, device, ds_roots):
        super(CLIPEmbedderDataset, self).__init__(device, ds_roots=ds_roots)
        self.preprocess = preprocess
        
    def __getitem__(self, index):
        
        with open(self.pointclouds[index], "rb") as file:
            datapoint_pointclouds, offset = pickle.load(file)
            datapoint_pointclouds = torch.tensor(datapoint_pointclouds, dtype=torch.float)
            offset = torch.tensor(offset, dtype = torch.float)
            
            if datapoint_pointclouds.shape[0] < 256 * 6:
                return self.__getitem__(random.randint(0, self.__len__() - 1))
            
            datapoint_pointclouds = datapoint_pointclouds.reshape(6, 256 ,6)
        
        transforms = self.object_transforms[index].to(dtype=torch.float)

        image = self.preprocess(Image.open(self.images[index]))

        x = (datapoint_pointclouds, transforms)
        y = image
        return (x, y)
    
class DiffusionDataset(AbstractDataset):
    def __getitem__(self, index):
        
        with open(self.pointclouds[index], "rb") as file:
            datapoint_pointclouds, offset = pickle.load(file)
            datapoint_pointclouds = torch.tensor(datapoint_pointclouds, dtype=torch.double)
            offset = torch.tensor(offset, dtype = torch.double)
            
            if datapoint_pointclouds.shape[0] < 256 * 6:
                return self.__getitem__(random.randint(0, self.__len__() - 1))
            
            datapoint_pointclouds = datapoint_pointclouds.reshape(6, 256 ,6)
        
        transforms = self.object_transforms[index].to(dtype=torch.double)

        image = torch.tensor(np.asarray(Image.open(self.images[index])), dtype=torch.double)

        x = (datapoint_pointclouds, transforms)
        y = image
        return (x, y)

if __name__ == "__main__":        
    #AbstractDataset()
    #CLIPEmbedderDataset("cuda").__getitem__(0)    
    ((datapoint_pointclouds, transforms), image) = DiffusionDataset("cuda").__getitem__(0)
    print(f"points shape: {datapoint_pointclouds.shape}")
    print(f"transforms shape: {transforms.shape}")
    print(f"image shape: {image.shape}")
