FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-humble-pcl-ros \
    ros-humble-pcl-conversions \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-tf2-tools \
    zstd \
    && rm -rf /var/lib/apt/lists/*

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

WORKDIR /workspace/ros2_ws

COPY ["ROS2 folder/src", "/workspace/ros2_ws/src"]
RUN . /opt/ros/humble/setup.sh && colcon build --symlink-install

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
