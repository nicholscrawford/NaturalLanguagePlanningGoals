import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from PointCloudRenderer.point_to_rgb_mlp import MLPPointToRGBModule
from PointCloudRenderer.point_to_rgb_transformer import \
    TransformerPointsToRGBModule

import torch

def convert_numpy_to_torch(array):
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array).to(device='cuda').to(torch.float)
    else:
        return array.to(device='cuda').to(torch.float)
    
class PointCloudRenderer():
    def __init__(self, pointcloud, model, camera_K, initial_camera_pose, k_points=5) -> None:
        self.pointcloud = convert_numpy_to_torch(pointcloud)
        self.camera_K = convert_numpy_to_torch(camera_K)
        self.initial_camera_pose = convert_numpy_to_torch(initial_camera_pose)
        self.k_points = k_points
        self.model = model

    '''
    Moved to GPU since it takes way too long otherwise.
    '''
    def render_images(self, num_images=1, img_resolution_x = 480, img_resolution_y = 320, batch_size = 480):
        images = []
        for img_idx in range(num_images):
            image = torch.zeros((img_resolution_y, img_resolution_x, 3)).to('cuda')
            R, t = self.get_random_camera_pose()
            for y_idx in range(img_resolution_y):
                points = []
                for x_idx in range(img_resolution_x):
                    point = self.get_k_nearest_points(R, t, x_idx, y_idx)
                    points.append(point)
                points = torch.stack(points)
                with torch.no_grad():
                    batch_size = min(batch_size, img_resolution_x) # Make sure batch_size is not larger than image width
                    for i in range(0, img_resolution_x, batch_size):
                        batch_points = points[:, i:i+batch_size, :]
                        batch_rgb = self.model(batch_points)
                        image[y_idx, i:i+batch_size] = batch_rgb
            images.append(image)
        return images
    
    def get_ray(self, camera_pose_R, camera_pose_t, pixel_location_x, pixel_location_y):
            # Extract camera intrinsic and extrinsic parameters
            K = self.camera_K
            R = camera_pose_R
            t = camera_pose_t.squeeze() #Handle t extracted from homogenous or combined matrix by squeezin

            # Compute the ray direction in world coordinates
            pixel_homogeneous = torch.tensor([pixel_location_y, pixel_location_x, 1.0]).to('cuda')
            pixel_homogeneous = pixel_homogeneous.reshape((3,1))
            ray_direction = torch.inverse(K @ R) @ pixel_homogeneous
            ray_direction = ray_direction / torch.norm(ray_direction)
            
            # Compute the ray origin in world coordinates
            ray_origin = -torch.inverse(R) @ t

            return ray_direction, ray_origin

    def get_k_nearest_points(self, camera_pose_R, camera_pose_t, pixel_location_x, pixel_location_y):
        ray_direction, ray_origin = self.get_ray(camera_pose_R, camera_pose_t, pixel_location_x, pixel_location_y)
        
        # Compute the distances to all points in the point cloud along the ray
        distances_along_ray = torch.tensordot(self.pointcloud[:, :3] - ray_origin, ray_direction.T)

        # Compute the perpendicular distances from the ray, by finding the norm of the distance from each projecetion along the ray to the each point.
        distances_from_ray = torch.norm((ray_direction * distances_along_ray.squeeze()).T - self.pointcloud[:, :3], dim = 1)
        
        # Sort the distances and get the indices of the k nearest points
        indices = torch.argsort(distances_from_ray.squeeze())[:self.k_points]
        
        return self.pointcloud.index_select(0, indices)
    
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
    


    def get_random_camera_pose(self):
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
        # Compute the center of the point cloud
        pointcloud_center = torch.from_numpy(self.get_center()).to('cuda')

        # Get the initial camera pose
        R, t = self.initial_camera_pose[:, :3], self.initial_camera_pose[:, 3]

        # Compute a random translation away from the initial pose
        delta_t = torch.normal(mean=0.0, std=0.05, size=(3,)).to('cuda')
        t_new = t + delta_t

        # Compute the direction vector pointing towards the center
        direction = pointcloud_center - t_new
        direction /= torch.norm(direction)

        # Compute angle between that vector and the positive z vector
        z = torch.tensor([0.0, 0.0, 1.0]).to('cuda')
        angle = torch.acos(torch.dot(direction, z))

        # Compute a random rotation around the direction vector
        rotation_axis = torch.cross(z,direction)
        rotation_axis /= torch.norm(rotation_axis)
        rotation_matrix = torch.tensor(Rotation.from_rotvec((angle * rotation_axis).cpu()).as_matrix()).to('cuda').to(torch.float)

        # Return the new camera pose
        return rotation_matrix, t_new


