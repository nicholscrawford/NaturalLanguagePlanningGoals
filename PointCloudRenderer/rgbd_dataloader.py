import os
import random
import sys
import urllib.request
import zipfile

import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset

from PointCloudRenderer.sampling_methods import uniform


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
        self.rgb_paths = []
        self.load_data_paths()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        depth_path = self.depth_paths[index]
        rgb_path = self.rgb_paths[index]

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)

        points = self.sampling_method(depth, self.k_points)
        rgb_vals = self.get_rgb_values(rgb, points)

        return points, rgb_vals

    def load_data_paths(self):
        with open(os.path.join(self.data_dir, "trainvalsplit", "train.txt"), 'r') as f:
            for line in f.readlines():
                line = line.strip()
                image_path = os.path.join(self.data_dir, "image", line + ".jpg")
                depth_path = os.path.join(self.data_dir, "depth", line + ".png")
                rgb_path = os.path.join(self.data_dir, "seg_img", line + ".png")
                self.image_paths.append(image_path)
                self.depth_paths.append(depth_path)
                self.rgb_paths.append(rgb_path)

    def get_rgb_values(self, rgb, points):
        h, w, _ = rgb.shape
        x = np.clip(np.int32(np.round(points[:, 0])), 0, w-1)
        y = np.clip(np.int32(np.round(points[:, 1])), 0, h-1)
        return rgb[y, x]

def get_dataloader(data_dir, k_points=5, sampling_method=uniform, batch_size=32):
    dataset = SUNRGBDDataset(data_dir, k_points, sampling_method)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
