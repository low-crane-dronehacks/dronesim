# Local PX4 SITL rig (Phase 1): ROS 2 Jazzy + Gazebo Harmonic (from the base image), PX4 v1.17.0 SITL, Micro-XRCE-DDS-Agent,
# px4_msgs (release/1.17, matches PX4). Build with deploy/rig/run_rig.sh build (JOBS caps the compile parallelism).
FROM osrf/ros:jazzy-desktop-full
ARG JOBS=8
ARG PX4_TAG=v1.17.0
ARG MSGS_BRANCH=release/1.17
ARG XRCE_TAG=v3.0.1
# Upstream git URLs of PX4-Autopilot, Micro-XRCE-DDS-Agent and px4_msgs. Override with --build-arg to use a mirror
# (run_rig.sh forwards PX4_REPO / XRCE_REPO / MSGS_REPO from the environment when they are set).
ARG PX4_REPO=https://github.com/PX4/PX4-Autopilot.git
ARG XRCE_REPO=https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
ARG MSGS_REPO=https://github.com/PX4/px4_msgs.git
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake ninja-build build-essential ccache python3-pip python3-venv python3-dev xsltproc libxml2-utils \
      libeigen3-dev gdb procps iproute2 \
    && rm -rf /var/lib/apt/lists/*
# PX4 SITL
RUN git clone --depth 1 --branch ${PX4_TAG} --recurse-submodules --shallow-submodules -j${JOBS} \
      ${PX4_REPO} /opt/PX4-Autopilot
# px_update_git_header.py wants a nuttx-X.Y.Z tag in the (shallow) NuttX submodule; SITL does not use NuttX, so a placeholder tag
# only affects the version string.
RUN git -C /opt/PX4-Autopilot/platforms/nuttx/NuttX/nuttx tag nuttx-12.4.0 \
 && pip install --break-system-packages --ignore-installed -r /opt/PX4-Autopilot/Tools/setup/requirements.txt
# Gazebo Harmonic comes from the ROS vendor packages: expose their cmake configs to the PX4 gz_bridge build
RUN . /opt/ros/jazzy/setup.sh && cd /opt/PX4-Autopilot \
 && CMAKE_PREFIX_PATH=$(ls -d /opt/ros/jazzy/opt/*_vendor | tr "\n" ":")$CMAKE_PREFIX_PATH HEADLESS=1 make -j${JOBS} px4_sitl_default
# Micro XRCE-DDS agent
RUN git clone --depth 1 --branch ${XRCE_TAG} ${XRCE_REPO} /opt/xrce \
 && cmake -S /opt/xrce -B /opt/xrce/build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build /opt/xrce/build -j${JOBS} && cmake --install /opt/xrce/build && ldconfig && rm -rf /opt/xrce/build
# px4_msgs
RUN mkdir -p /ws/src && git clone --depth 1 --branch ${MSGS_BRANCH} ${MSGS_REPO} /ws/src/px4_msgs \
 && . /opt/ros/jazzy/setup.sh && cd /ws && MAKEFLAGS=-j${JOBS} colcon build --packages-select px4_msgs --parallel-workers 2
ENV NVIDIA_DRIVER_CAPABILITIES=all NVIDIA_VISIBLE_DEVICES=all
WORKDIR /rig
