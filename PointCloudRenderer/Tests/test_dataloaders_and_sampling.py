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

def publish_camera_frame(points, T_camera_world, camera_name='camera'):
    """
    Publish a camera frame in RViz using a given camera extrinsic matrix.

    T_camera_world: a 3x4 numpy array representing the camera extrinsic matrix in the world frame.
    camera_name: a string representing the name of the camera frame.
    """
    # Initialize ROS node and transform broadcaster
    rospy.init_node('camera_frame_visualizer')
    tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

    # Create a transform message
    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = 'map'
    t.child_frame_id = camera_name

    # Extract the translation and rotation from the extrinsic matrix
    translation = T_camera_world[:3, 3]
    rotation = T_camera_world[:3, :3]

    camera_direction_global = -rotation @ np.array([0, 0, 1])
    
    # Convert the rotation matrix to a quaternion
    q = transformations.quaternion_from_matrix(np.vstack((T_camera_world, [0, 0, 0, 1])))
    
    print(f"Translation {translation}")

    # Fill in the transform message
    t.transform.translation.x = translation[0]
    t.transform.translation.y = translation[1]
    t.transform.translation.z = translation[2]
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]

    # Publish the transform message
    tf_broadcaster.sendTransform(t)
    
    # Publish a visualization marker for the camera frame
    marker = visualization_msgs.msg.Marker()
    marker.header.frame_id = camera_name
    marker.header.stamp = rospy.Time.now()
    #marker.ns = 'camera_frame'
    marker.id = 0
    marker.type = visualization_msgs.msg.Marker.ARROW
    marker.action = visualization_msgs.msg.Marker.ADD
    marker.pose.position.x = 0
    marker.pose.position.y = 0
    marker.pose.position.z = 0

    # Set the orientation of the marker to point in the direction of the camera's optical axis
    marker_orientation = transformations.quaternion_from_euler(0, np.arctan2(camera_direction_global[1], camera_direction_global[0]), -np.arctan2(camera_direction_global[2], np.linalg.norm(camera_direction_global)))
    marker.pose.orientation.x = marker_orientation[0]
    marker.pose.orientation.y = marker_orientation[1]
    marker.pose.orientation.z = marker_orientation[2]
    marker.pose.orientation.w = marker_orientation[3]


    marker.scale.x = 0.2
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 1.0

    marker_pub = rospy.Publisher('visualization_marker', visualization_msgs.msg.Marker, queue_size=1)


    # Create a ROS publisher for the point cloud topic
    pub = rospy.Publisher('/image_pointcloud', PointCloud2, queue_size=10)

    # Create a header for the point cloud
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'map'

    # Define the point cloud fields
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgba', 12, PointField.UINT32, 1)
    ]


    msg_points = []
    for i in range(points.shape[0]):
        point = (points[i][0].item(), points[i][1].item(), points[i][2].item())
        color = (int(points[i][3].item()), int(points[i][4].item()), int(points[i][5].item()), 255)
        r, g, b, a = color
        rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
        msg_points.append([point[0], point[1], point[2], rgb])
    
    cloud = pc2.create_cloud(header, fields, msg_points)


    while not rospy.is_shutdown():
        marker.header.stamp = rospy.Time.now()
        cloud.header.stamp = rospy.Time.now()
        marker_pub.publish(marker)
        pub.publish(cloud)
        rospy.sleep(0.1)


if __name__ == "__main__":
    data_dir = get_dataset_and_cache()
    print("Loaded default dataset SUNRGBD")
    dataset = SUNRGBDDataset(data_dir, 5, uniform_d10)
    index = random.randint(0, len(dataset)-1)

    image_path = dataset.image_paths[index]
    depth_path = dataset.depth_paths[index]
    extrinsics_path = dataset.extrinsics_paths[index]
    intrinsics_path = dataset.intrinsics_paths[index]
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
    image_pixel = dataset.select_img_pixel(img)

    # For the test, we'll just grab all the pixels in the image. Alternatively, uncomment the other depth pixels to display those. 
    #depth_pixels = dataset.sampling_method(depth_img, image_pixel, dataset.k_points)
    depth_pixels = [(y, x) for y in range(depth_img.shape[0]-1) for x in range(depth_img.shape[1]-1)]

    # Then translate those pixels into a pointcloud in 3d space using the camera extrinsic and intrinsic matrix
    pointcloud = dataset.get_pointcloud(depth_pixels, depth_img, img, extrinsics, intrinsics)


    publish_camera_frame(pointcloud, extrinsics)

    exit(0)