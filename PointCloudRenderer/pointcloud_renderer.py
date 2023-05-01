import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from PointCloudRenderer.point_to_rgb_mlp import MLPPointToRGBModule
from PointCloudRenderer.point_to_rgb_transformer import \
    TransformerPointsToRGBModule

import torch

class PointCloudRenderer():
    def __init__(self, pointcloud, model="mlp", camera_K = [], k_points=5) -> None:
        self.pointcloud = pointcloud
        self.camera_K = camera_K
        self.k_points = k_points
        self.model = model


    def render_images(self, num_images=1, img_resolution_x = 1280, img_resolution_y = 720):
        images = []
        for img_idx in range(num_images):
            image = np.zeros((img_resolution_y, img_resolution_x, 3))
            for y_idx in range(img_resolution_y):
                for x_idx in range(img_resolution_x):
                    R , t = self.get_random_camera_pose()
                    points = self.get_k_nearest_points(R, t, x_idx, y_idx)
                    rgb = self.model(points)
                    image[y_idx, x_idx] = rgb
            images.append(image)

        return images
    
    def get_k_nearest_points(self, camera_pose_R, camera_pose_t, pixel_location_x, pixel_location_y):
        # Extract camera intrinsic and extrinsic parameters
        K = self.camera_K
        R = camera_pose_R
        t = camera_pose_t
        
        # Compute the ray direction in world coordinates
        pixel_homogeneous = np.array([pixel_location_y, pixel_location_x, 1])
        pixel_homogeneous = pixel_homogeneous.reshape((3,1))
        ray_direction = np.linalg.inv(K @ R) @ pixel_homogeneous
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        
        # Compute the ray origin in world coordinates
        ray_origin = -np.linalg.inv(R) @ t
        
        # Compute the distances to all points in the point cloud along the ray
        distances_along_ray = np.dot(self.pointcloud[:, :3] - ray_origin, ray_direction)
        
        # Sort the distances and get the indices of the k nearest points
        indices = np.argsort(distances_along_ray)[:self.k_points]
        
        # Return the k nearest points with the same type as the input pointcloud array
        if isinstance(self.pointcloud, np.ndarray):
            return self.pointcloud[indices]
        elif isinstance(self.pointcloud, torch.Tensor):
            self.pointcloud.index_select()
            return self.pointcloud.index_select(0, torch.tensor(indices))
        else:
            raise ValueError("Input point cloud array type not supported.")
    
    def get_center(self, outlier_threshold=None):
        """
        Computes the "center" of a point cloud by taking the average of all
        points within a certain distance threshold from the mean.
        
        Returns:
        numpy array: a 1D array of shape (3,) representing the computed center point.
        """

        if isinstance(self.pointcloud, torch.Tensor):
            pointcloud = self.pointcloud.detach().cpu().numpy()
        else:
            pointcloud = self.pointcloud
        
        if outlier_threshold == None:
            outlier_threshold = self.compute_outlier_threshold()
        # Compute the mean of all points in the cloud
        cloud_mean = np.mean(pointcloud[:, :3], axis=0)
        
        # Compute the distances of all points to the mean
        distances = np.linalg.norm(pointcloud[:, :3] - cloud_mean, axis=1)
        
        # Identify points within the outlier threshold
        inliers = pointcloud[distances <= outlier_threshold]
        
        # Compute the mean of the inliers
        center = np.mean(inliers[:, :3], axis=0)
        
        return center
    
    def compute_outlier_threshold(self, k=1.5):
        """
        Computes a reasonable outlier threshold for a point cloud using the interquartile
        range (IQR) method.
        
        Parameters:
        point_cloud (numpy array): a 2D array of shape (N, 3) where N is the number
            of points in the cloud and each row represents a 3D point as [x, y, z].
        k (float): a scaling factor to adjust the sensitivity of the outlier threshold.
            A larger value of k will result in a higher threshold.
        
        Returns:
        float: the computed outlier threshold.
        """
        if isinstance(self.pointcloud, torch.Tensor):
            pointcloud = self.pointcloud.detach().cpu().numpy()
        else:
            pointcloud = self.pointcloud

        # Compute the distances of all points to the mean
        distances = np.linalg.norm(pointcloud - np.mean(pointcloud, axis=0), axis=1)
        
        # Compute the first and third quartiles of the distances
        q1, q3 = np.percentile(distances, [25, 75])
        
        # Compute the interquartile range (IQR)
        iqr = q3 - q1
        
        # Compute the outlier threshold as k times the IQR
        outlier_threshold = q3 + k * iqr
        
        return outlier_threshold
    


    def get_random_camera_pose(self, outlier_threshold = None, distance_from_center= None):
        """
        Computes a random camera pose for taking a picture of a point cloud. The camera
        position is generated by translating away from the center of the cloud, and the
        camera orientation is computed to point towards the center.
        
        Parameters:
        point_cloud (numpy array): a 2D array of shape (N, 3) where N is the number
            of points in the cloud and each row represents a 3D point as [x, y, z].
        outlier_threshold (float): a threshold value for identifying outliers. Points
            more than this distance away from the mean will be excluded from the center calculation.
        distance_from_center (float): the desired distance between the camera position and
            the center of the cloud.
        
        Returns:
        tuple: a tuple containing the camera rotation matrix as a NumPy array of shape (3, 3)
            and the camera translation parameters as a NumPy array of shape (3,).
        """
        if outlier_threshold == None:
            outlier_threshold = self.compute_outlier_threshold()

        if distance_from_center == None:
            distance_from_center = 10

        # Compute the center of the point cloud
        center = self.get_center(outlier_threshold)
        
        # Generate a random unit vector for camera orientation
        orientation = np.random.randn(3)
        orientation /= np.linalg.norm(orientation)
        
        # Compute the camera position by translating away from the center
        position = center - orientation * distance_from_center
        
        # Compute the rotation matrix to point the camera towards the center
        z_axis = center - position
        z_axis /= np.linalg.norm(z_axis)
        x_axis = np.cross([0, 0, 1], z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        rotation_matrix = np.stack([x_axis, y_axis, z_axis], axis=1)
        
        return rotation_matrix, position



