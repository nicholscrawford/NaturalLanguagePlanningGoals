import os
import random
import sys
import urllib.request
import zipfile

import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torch

from PointCloudRenderer.sampling_methods import uniform_d10


def get_dataset_and_cache(dataset = "sun"):
    '''
    Returns a dataset.
    If the dataset hasn't been downloaded, save it to $HOME/data/NaturalLanguagePlanningGoals/name_of_dataset/
    If it has, load the dataset from there.
    '''
    if dataset == "sun":
        data_dir = os.path.join(os.path.expanduser("~"), "data", "NaturalLanguagePlanningGoals", "sun_rgbd")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        url = "https://rgbd.cs.princeton.edu/data/SUNRGBD.zip"
        zip_file_path = os.path.join(data_dir, "SUNRGBD.zip")
        if not os.path.exists(zip_file_path):
            print(f"Downloading {url}...")
            urllib.request.urlretrieve(url, zip_file_path)
            print("Extracting files...")
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print("Done!")
    
    else:
        print(f"Unknown dataset: {dataset}")
        sys.exit(1)

    return data_dir

class SUNRGBDDataset(Dataset):
    def __init__(self, data_dir, k_points, sampling_method):
        self.data_dir = data_dir
        self.k_points = k_points
        self.sampling_method = sampling_method
        self.image_paths = []
        self.depth_paths = []
        self.extrinsics_paths = []
        self.intrinsics_paths = []
        self.load_data_paths()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        depth_path = self.depth_paths[index]
        extrinsics_path = self.extrinsics_paths[index]
        intrinsics_path = self.intrinsics_paths[index]
        debug = False
        

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        

        extrinsics = np.loadtxt(extrinsics_path)
        if extrinsics.ndim == 1:
            extrinsics = extrinsics.reshape((3, 4))
        elif extrinsics.shape != (3, 4):
            raise ValueError("Extrinsics matrix should be a 3x4 matrix or a 1D array with 12 values.")
                
        intrinsics = np.loadtxt(intrinsics_path)
        if intrinsics.ndim == 1:
            intrinsics = intrinsics.reshape((3, 3))
        elif intrinsics.shape != (3, 3):
            raise ValueError("Intrinsics matrix should be a 3x3 matrix or a 1D array with 9 values.")
                
        depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

        # Convert the raw depth values to meters using the https://github.com/ankurhanda/sunrgbd-meta-data
        depth_img = depth_img / 10000

        # Get the pixel image location, our x val
        image_pixel = self.select_img_pixel(img)

        # From that image pixel, choose k_points pixels in the depth image
        depth_pixels = self.sampling_method(depth_img, image_pixel, self.k_points)

        # Then translate those pixels into a pointcloud in 3d space using the camera extrinsic and intrinsic matrix
        pointcloud = self.get_pointcloud(depth_pixels, depth_img, img, extrinsics, intrinsics)

        if debug:
            print(f"Image path {image_path}")
            print(f"Image point {image_pixel}")

        return (pointcloud, img[image_pixel].astype(np.float32), torch.tensor(image_pixel), intrinsics, extrinsics)

    def load_data_paths(self):
        for top_level_dir in ["realsense", "kv1", "kv2"]:
            for lower_level_dir in [folder for folder in os.listdir(os.path.join(self.data_dir, "SUNRGBD", top_level_dir)) if not folder[0] == '.']:
                for item_dir in [folder for folder in os.listdir(os.path.join(self.data_dir, "SUNRGBD", top_level_dir, lower_level_dir)) if not folder[0] == '.']:
                    item_path = os.path.join(self.data_dir, "SUNRGBD", top_level_dir, lower_level_dir, item_dir)
        

                    image_folder = os.path.join(item_path, 'image')
                    image_path = os.path.join(image_folder, os.listdir(image_folder)[0])

                    depth_folder = os.path.join(item_path, 'depth_bfx')
                    depth_path = os.path.join(depth_folder, os.listdir(depth_folder)[0])
                    
                    extrinsics_folder = os.path.join(item_path, 'extrinsics')
                    extrinsics_path = os.path.join(extrinsics_folder, os.listdir(extrinsics_folder)[0])

                    intrinsics_path = os.path.join(item_path, 'intrinsics.txt')

                    self.image_paths.append(image_path)
                    self.depth_paths.append(depth_path)
                    self.extrinsics_paths.append(extrinsics_path)
                    self.intrinsics_paths.append(intrinsics_path)

    def select_img_pixel(self, img: np.ndarray):
        # Choose a random point on the image
        height, width, _ = img.shape
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        return y, x
    
    def get_pointcloud(self, depth_pixels, depth_img, img, extrinsics, intrinsics):
        # Translate the depth pixels into a colored pointcloud in 3D space
        points = []
        for sy, sx in depth_pixels:
            depth = depth_img[sy, sx]
            if depth == 0:
                continue
            x = (sx - intrinsics[0, 2]) / intrinsics[0, 0] * depth
            y = (sy - intrinsics[1, 2]) / intrinsics[1, 1] * depth
            point = np.dot(extrinsics, [x, y, depth, 1])
            color = img[sy, sx]
            points.append([point[0], point[1], point[2], color[2], color[1], color[0]])
        
        # Convert the list of points into a numpy array
        pointcloud = np.array(points, dtype=np.float32)
        
        return pointcloud

    


def get_dataloader(data_dir, k_points=5, sampling_method=uniform_d10, batch_size=32):
    dataset = SUNRGBDDataset(data_dir, k_points, sampling_method)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def get_train_val_dl(data_dir, k_points=5, sampling_method=uniform_d10, batch_size=32):
        
        dataset = SUNRGBDDataset(data_dir, k_points, sampling_method)

        train_size = int(0.8 * len(dataset))

        # Define the indices for the training and test sets
        indices = list(range(len(dataset)))
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        # Define the samplers for the training and test sets
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        # Create data loaders for the training and test sets
        train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

        return train_dataloader, test_dataloader
