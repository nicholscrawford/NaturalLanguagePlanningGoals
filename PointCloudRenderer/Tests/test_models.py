import random
import struct

import geometry_msgs.msg
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import visualization_msgs.msg
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from tf import transformations
import cv2

from PointCloudRenderer.rgbd_dataloader import (SUNRGBDDataset, get_dataloader,
                                                get_dataset_and_cache)
from PointCloudRenderer.sampling_methods import uniform_d10
from PointCloudRenderer.point_to_rgb_transformer import TransformerPointsToRGBModule



if __name__ == "__main__":
    data_dir = get_dataset_and_cache()
    print("Loaded default dataset SUNRGBD")
    dataloader = get_dataloader(data_dir)

    transformer = TransformerPointsToRGBModule(5, nhead=2, num_layers=3)
    for batch in dataloader:
        pred_color = transformer(batch[0])
        print(f"Batch of predicted colors shape: {pred_color.shape}")
        exit(0)