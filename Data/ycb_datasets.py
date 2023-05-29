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
    def __init__(self, ds_root = "/home/nicholscrawfordtaylor/data/may22/", clear_cache=False):

        # This strategy relies on consistent ordering in each array, and in each arrays construction. It may be better to make the relationship between each portion of the datapoint explicit.
        self.images = []
        self.is_realistic = []
        self.sim_idx = []
        self.pointclouds = []
        self.object_transforms = []
        self.cls_indexes = [] # Segmentation labels
        camera_homo = np.array([
            [1.0000000,  0.0000000,  0.0000000, 0],
            [0.0000000, -0.7660444,  0.6427876, 1.2],
            [0.0000000, -0.6427876, -0.7660444, -1.2],
            [0.0000000,  0.0000000,  0.0000000, 1.000000]
            ])
        self.max_object_points = 128
        
        # Check for cached folders?

        for folder_idx, folder in tqdm(enumerate( os.listdir(ds_root))):
            cwd = os.path.join(ds_root, folder, "data", "YCB_Video", "data", "0000")
            files = os.listdir(cwd)
            
            # Get metadata locations
            metamat = [os.path.join(cwd, filename) for filename in files if "meta" in filename]
            self.sim_idx += [int(name[-15:-9]) for name in metamat]
            
            # Get image locations
            image = [os.path.join(cwd, filename) for filename in files if "color" in filename]
            self.images += image

            # Get depth and seg locations
            depth = [os.path.join(cwd, filename) for filename in files if "depth" in filename]
            seg = [os.path.join(cwd, filename) for filename in files if "label" in filename]
            
            for sim_idx_str in [name[-15:-9] for name in metamat]:
                metamat_path = [pth for pth in metamat if str(sim_idx_str) in pth][0]
                depth_path = [pth for pth in depth if str(sim_idx_str) in pth][0]
                seg_path = [pth for pth in seg if str(sim_idx_str) in pth][0]
                image_path = [pth for pth in image if str(sim_idx_str) in pth][0]

                annots = loadmat(metamat_path)
                cls_idxs = annots["cls_indexes"]
                self.cls_indexes.append(cls_idxs)
                poses = annots["poses"]
                poses = rearrange(poses, "height width object_idx -> object_idx height width")
                self.object_transforms.append(poses)

                # Cache pointclouds
                pointcloud_cache_path = os.path.join(cwd, f"{sim_idx_str}-downsampled_pc.pickle")
                self.pointclouds.append(pointcloud_cache_path)

                if not os.path.exists(pointcloud_cache_path) or clear_cache:
                    depth_img = np.asarray(o3d.io.read_image(depth_path)) #/ annots["factor_depth"].item()
                    seg_img = np.asarray(o3d.io.read_image(seg_path) )
                    img = np.asarray(o3d.io.read_image(image_path))
                    fx = annots["intrinsic_matrix"][0, 0]
                    fy = annots["intrinsic_matrix"][1, 1]
                    cx = annots["intrinsic_matrix"][0, 2]
                    cy = annots["intrinsic_matrix"][1, 2]

                    #camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(1280, 720, fx, fy, cx, cy)
                    #rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img, depth_img, depth_scale=annots["factor_depth"].item(), depth_trunc=10000000, convert_rgb_to_intensity=False)
                    depth_height, depth_width = depth_img.shape
                    v, u = np.meshgrid(range(depth_height), range(depth_width), indexing='ij')
                    points_x = (u - cx) * depth_img / fx
                    points_y = (v - cy) * depth_img / fy
                    points_z = depth_img
                    points = np.stack((points_x, points_y, points_z), axis=-1)
                    
                    datapoint_pointclouds = []
                    offset = [0]
                    num_objs = len(cls_idxs[0])
                    for class_index in range(num_objs):
                        class_mask = (seg_img == cls_idxs[0][class_index])
                        segmented_points = points[class_mask]
                        
                        #segmented_points = rearrange(points, "h w c -> (h w) c")
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(segmented_points)
                        pcd.colors = o3d.utility.Vector3dVector(img[class_mask][:, :3] / 255.0)
                        num_obj_points = class_mask.sum()
                        if  class_mask.sum() >= self.max_object_points:
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

if __name__ == "__main__":        
    #AbstractDataset()
    CLIPEmbedderDataset().__getitem__(0)    