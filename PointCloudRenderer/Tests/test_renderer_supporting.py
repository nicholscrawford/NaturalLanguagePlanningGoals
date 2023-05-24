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
import torch

from PointCloudRenderer.rgbd_dataloader import (SUNRGBDDataset, get_dataloader,
                                                get_dataset_and_cache)
from PointCloudRenderer.sampling_methods import uniform_d10

from PointCloudRenderer.pointcloud_renderer import PointCloudRenderer

def publish_camera_frame(points,selected_points,raycast_points, T_camera_world, pointcloud_center, ray, camera_name='camera'):
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

    # Create a transform message
    ct = geometry_msgs.msg.TransformStamped()
    ct.header.stamp = rospy.Time.now()
    ct.header.frame_id = 'map'
    ct.child_frame_id = 'pointcloud_center'

    # Extract the translation and rotation from the extrinsic matrix
    translation = T_camera_world[:3, 3]
    rotation = T_camera_world[:3, :3]

    camera_direction_global = -rotation @ np.array([0, 0, 1])
    
    # Convert the rotation matrix to a quaternion
    q = transformations.quaternion_from_matrix(np.vstack((T_camera_world, [0, 0, 0, 1])))
    
    # Fill in the transform message
    t.transform.translation.x = translation[0]
    t.transform.translation.y = translation[1]
    t.transform.translation.z = translation[2]
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]

    ct.transform.translation.x = pointcloud_center[0]
    ct.transform.translation.y = pointcloud_center[1]
    ct.transform.translation.z = pointcloud_center[2]
    ct.transform.rotation.x = 0
    ct.transform.rotation.y = 0
    ct.transform.rotation.z = 0
    ct.transform.rotation.w = 1

    # Publish the transform message
    tf_broadcaster.sendTransform([t, ct])

    
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
    #marker_orientation = transformations.quaternion_from_euler(0, np.arctan2(camera_direction_global[1], camera_direction_global[0]), -np.arctan2(camera_direction_global[2], np.linalg.norm(camera_direction_global)))
    marker.pose.orientation.x = -0.5#marker_orientation[0]
    marker.pose.orientation.y = -0.5#marker_orientation[1]
    marker.pose.orientation.z = -0.5#marker_orientation[2]
    marker.pose.orientation.w = 0.5#marker_orientation[3]


    marker.scale.x = 0.2
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 1.0

    marker_pub = rospy.Publisher('visualization_marker', visualization_msgs.msg.Marker, queue_size=1)


    # Publish a visualization marker for the ray
    ray_direction, ray_origin = ray
    ray_direction = ray_direction.cpu().numpy()

    ray_origin = ray_origin.cpu().numpy()

    ray_marker = visualization_msgs.msg.Marker()
    ray_marker.header.frame_id = "map"
    ray_marker.header.stamp = rospy.Time.now()
    #ray_marker.ns = 'camera_frame'
    ray_marker.id = 0
    ray_marker.type = visualization_msgs.msg.Marker.ARROW
    marker.action = visualization_msgs.msg.Marker.ADD
    ray_marker.pose.position.x = ray_origin[0]
    ray_marker.pose.position.y = ray_origin[1]
    ray_marker.pose.position.z = ray_origin[2]

    # Set the orientation of the marker to point in the direction of the ray
    direction_norm = np.linalg.norm(ray_direction)
    if direction_norm != 0:
        direction_unit = ray_direction / direction_norm
    else:
        direction_unit = np.array([1, 0, 0])  # Default unit direction if the provided direction is zero vector


    marker_orientation = transformations.quaternion_from_euler(0, 0, 0)
    marker_orientation[1] = direction_unit[1] * np.sin(0.5 * np.arccos(direction_unit[0]))
    marker_orientation[2] = direction_unit[2] * np.sin(0.5 * np.arccos(direction_unit[0]))
    marker_orientation[3] = direction_unit[0] * np.sin(0.5 * np.arccos(direction_unit[0]))
    marker_orientation[0] = np.cos(0.5 * np.arccos(direction_unit[0]))

    ray_marker.pose.orientation.x = marker_orientation[0]
    ray_marker.pose.orientation.y = marker_orientation[1]
    ray_marker.pose.orientation.z = marker_orientation[2]
    ray_marker.pose.orientation.w = marker_orientation[3]


    ray_marker.scale.x = 2
    ray_marker.scale.y = 0.01
    ray_marker.scale.z = 0.01
    ray_marker.color.a = 1.0
    ray_marker.color.r = 0.0
    ray_marker.color.g = 0.0
    ray_marker.color.b = 1.0

    ray_marker_pub = rospy.Publisher('ray_marker', visualization_msgs.msg.Marker, queue_size=1)



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

    # Create a ROS publisher for the point cloud topic
    sel_pub = rospy.Publisher('/selected_pointcloud', PointCloud2, queue_size=10)

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
    for i in range(selected_points.shape[0]):
        point = (selected_points[i][0].item(), selected_points[i][1].item(), selected_points[i][2].item())
        color = (int(selected_points[i][3].item()), int(selected_points[i][4].item()), int(selected_points[i][5].item()), 255)
        r, g, b, a = color
        rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
        msg_points.append([point[0], point[1], point[2], rgb])
    
    selected_cloud = pc2.create_cloud(header, fields, msg_points)

    # Create a ROS publisher for the point cloud topic
    cast_pub = rospy.Publisher('/raycast_pointcloud', PointCloud2, queue_size=10)

    msg_points = []
    for i in range(raycast_points.shape[0]):
        point = (raycast_points[i][0].item(), raycast_points[i][1].item(), raycast_points[i][2].item())
        color = (int(raycast_points[i][3].item()), int(raycast_points[i][4].item()), int(raycast_points[i][5].item()), 255)
        r, g, b, a = color
        rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
        msg_points.append([point[0], point[1], point[2], rgb])
    
    cast_cloud = pc2.create_cloud(header, fields, msg_points)

    while not rospy.is_shutdown():
        marker.header.stamp = rospy.Time.now()
        cloud.header.stamp = rospy.Time.now()
        marker_pub.publish(marker)
        ray_marker_pub.publish(ray_marker)
        pub.publish(cloud)
        sel_pub.publish(selected_cloud)
        cast_pub.publish(cast_cloud)
        rospy.sleep(0.1)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float)
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

    # For the test, we'll just grab all the pixels in the image. And display some selected ones based on depth sampling.
    selected_depth_pixels = dataset.sampling_method(depth_img, image_pixel, dataset.k_points)
    depth_pixels = [(y, x) for y in range(depth_img.shape[0]-1) for x in range(depth_img.shape[1]-1)]

    # Then translate those pixels into a pointcloud in 3d space using the camera extrinsic and intrinsic matrix
    pointcloud = dataset.get_pointcloud(depth_pixels, depth_img, img, extrinsics, intrinsics)
    selected_pointcloud = dataset.get_pointcloud(selected_depth_pixels, depth_img, img, extrinsics, intrinsics)
    
    renderer = PointCloudRenderer(pointcloud, None, intrinsics, extrinsics)

    # And let's get some points using the raycasting approach.
    ray_direction, ray_origin = renderer.get_ray(torch.tensor(extrinsics[:, :3]).to('cuda').float(), torch.tensor(extrinsics[:, 3:]).to('cuda').float(), image_pixel[0], image_pixel[1])
    print(f"ray dir: {ray_direction}, ray origin: {ray_origin}")
    ray = (ray_direction, ray_origin)
    raycast_pointcloud = renderer.get_k_nearest_points(torch.tensor(extrinsics[:, :3]).to('cuda').float(), torch.tensor(extrinsics[:, 3:]).to('cuda').float(), image_pixel[0], image_pixel[1])

    R, t = renderer.get_random_camera_pose()
    new_extrinsics = torch.cat((R, t[:, np.newaxis]), dim=1)

    print(new_extrinsics)

    pointcloud_center = renderer.get_center()

    new_extrinsics = new_extrinsics.cpu().numpy()
    publish_camera_frame(pointcloud, selected_pointcloud, raycast_pointcloud, new_extrinsics, pointcloud_center, ray)

    exit(0)