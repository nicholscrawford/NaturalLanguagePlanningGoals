#!/bin/bash

# Start roscore
gnome-terminal --tab --title="roscore" -- bash -c "roscore; $SHELL"

# Wait for roscore to start up
sleep 2

# Start rviz in a new terminal window
gnome-terminal --tab --title="rviz" -- bash -c "rviz; $SHELL"

# Wait for rviz to start up
sleep 2

# Start your Python script in a new terminal window
gnome-terminal --tab --title="python script" -- bash -c "python test_renderer_supporting.py; $SHELL"


# Print instructions to the terminal
echo "Make sure roscore, rviz, and test Python script are all running. Then add the visualizations by topic for the pointcloud, camera marker, and if desired, tfs."
