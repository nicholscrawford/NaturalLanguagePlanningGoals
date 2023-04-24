import roslaunch

def start_rviz():
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)

    # Create the launch configuration for RViz
    launch_config = roslaunch.config.ROSLaunchConfig()
    rviz_node = launch_config.add_node(roslaunch.core.Node("rviz", "rviz"))

    # Launch RViz
    launch = roslaunch.scriptapi.ROSLaunch()
    launch.start()
    process = launch.launch(rviz_node)

    # Wait for RViz to start
    while not process.is_alive():
        pass

def test_get_center():
    start_rviz()

    print("Is the marker in the center of the point cloud? (y/n)")
    assert(input() == "y")

if __name__ == "__main__":
    test_get_center()