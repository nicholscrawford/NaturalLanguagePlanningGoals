import roslaunch
import rospy
import random
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from geometry_msgs.msg import Point
import struct

def test_get_center():

    # Initialize ROS node
    rospy.init_node('random_pointcloud_publisher')

    # Create a ROS publisher for the point cloud topic
    pub = rospy.Publisher('/random_pointcloud', PointCloud2, queue_size=10)

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


    points = []
    for i in range(100):
        point = (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)
        r, g, b, a = color
        rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
        points.append([point[0], point[1], point[2], rgb])
    
    cloud = pc2.create_cloud(header, fields, points)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        pub.publish(cloud)
        rate.sleep()


if __name__ == "__main__":
    test_get_center()