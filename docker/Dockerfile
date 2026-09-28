FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-colcon-common-extensions \
    python3-numpy \
    python3-scipy \
    python3-yaml \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-tf2-tools \
    && rm -rf /var/lib/apt/lists/*

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

WORKDIR /workspace/ros2_ws
COPY ["ROS2 folder/src", "src"]
RUN . /opt/ros/humble/setup.sh && colcon build --symlink-install

COPY docker/cyclonedds.xml /etc/cyclonedds.xml
ENV CYCLONEDDS_URI=file:///etc/cyclonedds.xml

COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/ros2 /usr/local/bin/ros2
RUN chmod +x /entrypoint.sh /usr/local/bin/ros2

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
