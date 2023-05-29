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
    def __init__(self, ds_root = "/home/nichols/Data/may26/", clear_cache=False):

        # This strategy relies on consistent ordering in each array, and in each arrays construction. It may be better to make the relationship between each portion of the datapoint explicit.
        self.images = []
        self.pointclouds = []
        self.object_transforms = []
        self.cls_indexes = [] # Segmentation labels
        camera_homo = np.array([
            [1.0000000,  0.0000000,  0.0000000, 0],
            [0.0000000, -0.7660444,  0.6427876, 1.2],
            [0.0000000, -0.6427876, -0.7660444, -1.2],
            [0.0000000,  0.0000000,  0.0000000, 1.000000]
            ])
        self.max_object_points = 256
        
        # Check for cached folders?

        for folder_idx, folder in tqdm(enumerate( os.listdir(ds_root))):
            cwd = os.path.join(ds_root, folder)
            files = os.listdir(cwd)
            self.sim_idx = []
            # Get metadata locations
            metamat = [os.path.join(cwd, filename) for filename in files if "meta_" in filename]            
            
            for sim_idx_str in [name[-8:-4] for name in metamat]:
                metamat_path = os.path.join(cwd, f"meta_{sim_idx_str}.mat")
                # Not actually used?
                # instance_segmentation_path = os.path.join(cwd, f"instance_segmentation_{sim_idx_str}.npy") 
                # depth_path = os.path.join(cwd, f"distance_to_image_plane_{sim_idx_str}.npy")
                pointcloud_xyz_path = os.path.join(cwd, f"pointcloud_{sim_idx_str}.npy")
                pointcloud_rgb_path = os.path.join(cwd, f"pointcloud_rgb_{sim_idx_str}.npy")
                pointcloud_seg_path = os.path.join(cwd, f"pointcloud_semantic_{sim_idx_str}.npy")
                image_path = os.path.join(cwd, f"rgb_{sim_idx_str}.png")
                
                # Save image path 
                self.images.append(image_path)
                
                # Load from metadata file
                annots = loadmat(metamat_path)
                cls_idxs = annots["cls_indexes"]
                poses = annots["poses"]
                
                # Save poses
                poses = rearrange(poses, "height width object_idx -> object_idx height width")
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

    def __len__(self):
        return len(self.images)

class CLIPEmbedderDataset(AbstractDataset):
    def __getitem__(self, index):
        
        with open(self.pointclouds[index], "rb") as file:
            datapoint_pointclouds, offset = pickle.load(file)
            datapoint_pointclouds = torch.tensor(datapoint_pointclouds, dtype=torch.float).to('cuda')
            offset = torch.tensor(offset, dtype = torch.double).to('cuda')
        
        transforms = torch.tensor(self.object_transforms[index], dtype=torch.float).to('cuda')

        image = torch.tensor(np.asarray(Image.open(self.images[index])), dtype=torch.float).to('cuda')

        x = ((datapoint_pointclouds, offset), transforms)
        y = image
        return (x, y)
    
class DiffusionDataset(AbstractDataset):
    def __getitem__(self, index):
        
        with open(self.pointclouds[index], "rb") as file:
            datapoint_pointclouds, offset = pickle.load(file)
            datapoint_pointclouds = torch.tensor(datapoint_pointclouds, dtype=torch.float).to('cuda')
            offset = torch.tensor(offset, dtype = torch.double).to('cuda')
        
        transforms = torch.tensor(self.object_transforms[index], dtype=torch.float).to('cuda')

        image = torch.tensor(np.asarray(Image.open(self.images[index])), dtype=torch.float).to('cuda')

        x = ((datapoint_pointclouds, offset), transforms)
        y = image
        return (x, y)

if __name__ == "__main__":        
    #AbstractDataset()
    CLIPEmbedderDataset().__getitem__(0)    
    (((datapoint_pointclouds, offset), transforms), image) = DiffusionDataset().__getitem__(0)
    print(f"points shape: {datapoint_pointclouds.shape}")
    print(f"points offset: {offset.shape}")
    print(f"transforms shape: {transforms.shape}")
    print(f"image shape: {image.shape}")