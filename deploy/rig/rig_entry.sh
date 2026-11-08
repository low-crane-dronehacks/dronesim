#!/bin/bash
# Runs INSIDE the container: gz server (headless render, our world) + PX4 SITL x500_depth + uXRCE agent + ROS bridges.
# Logs in /tmp/rig/*.log. The policy node is started separately (run_rig.sh policy).
set -u
RIG=/rig/deploy/rig; PX4=/opt/PX4-Autopilot; WORLD=dronehacks; MODEL=x500_depth_0
mkdir -p /tmp/rig; cd /tmp/rig
set +u; . /opt/ros/jazzy/setup.bash; . /ws/install/setup.bash; set -u
export HEADLESS=1 PX4_GZ_WORLDS=$RIG/world PX4_GZ_WORLD=$WORLD PX4_SYS_AUTOSTART=4002 PX4_SIM_MODEL=gz_x500_depth
export PX4_GZ_MODEL_POSE="${SPAWN_POSE:-$(grep -v '^#' $RIG/world/spawn.txt | head -1)}"   # SPAWN_POSE=x,y,z,r,p,yaw (ENU) overrides spawn.txt
B=$PX4/build/px4_sitl_default
export GZ_SIM_RESOURCE_PATH=${GZ_SIM_RESOURCE_PATH:-} GZ_SIM_SYSTEM_PLUGIN_PATH=${GZ_SIM_SYSTEM_PLUGIN_PATH:-}
. $B/rootfs/gz_env.sh                 # PX4's model/plugin/server-config paths
python3 $RIG/add_normals.py $RIG/world /tmp/rig/world   # gz-physics crashes on normal-less OBJ meshes
export PX4_GZ_WORLDS=/tmp/rig/world   # our world (gz_env.sh points it at PX4's own worlds)
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:/tmp/rig/world

# The nvidia container runtime (snap docker) ships the EGL libs but not the glvnd vendor file: without it ogre2 falls back to mesa and
# segfaults in headless rendering.
mkdir -p /usr/share/glvnd/egl_vendor.d
echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# 1. gazebo server alone, EGL headless rendering (the depth camera needs the GPU renderer)
gz sim -r -s --headless-rendering --verbose=1 /tmp/rig/world/$WORLD.sdf > gz.log 2>&1 &
until gz topic -l 2>/dev/null | grep -q "^/world/$WORLD/clock"; do sleep 1; done

# 1b. spawn the x500_depth ourselves with the full launch pose: PX4 v1.16 only applies the x y z of PX4_GZ_MODEL_POSE, not the yaw.
# PX4 then attaches to the existing model by name (PX4_GZ_MODEL_NAME).
IFS=, read -r PX PY PZ PR PP PYAW <<< "$PX4_GZ_MODEL_POSE"
QZ=$(python3 -c "import math;print(math.sin(${PYAW:-0}/2))"); QW=$(python3 -c "import math;print(math.cos(${PYAW:-0}/2))")
# Battery: BATTERY_SOC (0..1, default 1.0) sets the pack's state of charge at takeoff. The 4S 5000 mAh LiPo model
# (dronehacks/battery.py + assets/battery_4s5000.json) gives the loaded voltage at hover and the thrust it leaves
# (~(V/V_ref)^2); the x500 motor constants are scaled by that factor in a private copy of the model. Static per
# flight (a 3-min run drains ~5% more thrust that is not modelled here); PX4 is told about the pack below.
SOC=${BATTERY_SOC:-1.0}
TF=$(cd /rig && PYTHONPATH=/rig python3 -c "from dronehacks.battery import LiPoPack as P; p=P.from_json('assets/battery_4s5000.json'); v,i=p.solve($SOC,p.hover_thrust_n); print(round(float(p.thrust_factor(v)),4))")
mkdir -p /tmp/rig/models && cp -r $PX4_GZ_MODELS/x500 $PX4_GZ_MODELS/x500_base $PX4_GZ_MODELS/x500_depth /tmp/rig/models/ 2>/dev/null || cp -r $PX4_GZ_MODELS/x500 $PX4_GZ_MODELS/x500_depth /tmp/rig/models/
python3 - "$TF" <<'PYEOF'
import re, sys, glob
f = float(sys.argv[1])
for path in glob.glob('/tmp/rig/models/*/model.sdf'):
    s = open(path).read()
    s2 = re.sub(r'<motorConstant>([0-9.eE+-]+)</motorConstant>', lambda m: f'<motorConstant>{float(m.group(1)) * f:.6e}</motorConstant>', s)
    if s2 != s:
        open(path, 'w').write(s2)
        print(f'battery: scaled motorConstant x{f} in {path}')
PYEOF
echo "battery: SoC $SOC -> hover thrust factor $TF" | tee battery.log
export GZ_SIM_RESOURCE_PATH=/tmp/rig/models:$GZ_SIM_RESOURCE_PATH
gz service -s /world/$WORLD/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 10000 \
  --req "sdf_filename: \"/tmp/rig/models/x500_depth/model.sdf\", name: \"$MODEL\", allow_renaming: false, pose: {position: {x: $PX, y: $PY, z: $PZ}, orientation: {z: $QZ, w: $QW}}" > spawn.log 2>&1
export PX4_GZ_MODEL_NAME=$MODEL

# 2. PX4 params for the indoor / external-vision estimate (appended to the airframe of this container's build)
AF=$B/etc/init.d-posix/airframes/4001_gz_x500
grep -q "EKF2_EV_CTRL" $AF || cat $RIG/px4_params.txt >> $AF
rm -f $B/rootfs/parameters*.bson $B/rootfs/dataman     # fresh params each start

# 3. uXRCE agent + PX4 (attaches to the running world, spawns the x500_depth at the launch mark)
MicroXRCEAgent udp4 -p 8888 > agent.log 2>&1 &
(cd $B/rootfs && exec $B/bin/px4 -d $B/etc > /tmp/rig/px4.log 2>&1) &
until gz topic -l 2>/dev/null | grep -q "depth_camera"; do sleep 1; done

# 4. ROS side: /clock, depth + camera_info, ground-truth pose, static TF base_link -> camera optical frame
DEPTH=$(gz topic -l | grep -m1 -E "depth_camera$"); INFO=$(gz topic -l | grep -m1 -E "depth_camera/camera_info$|camera_info$")
ros2 run ros_gz_bridge parameter_bridge \
  "/world/$WORLD/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock" \
  "$DEPTH@sensor_msgs/msg/Image[gz.msgs.Image" "$INFO@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo" \
  --ros-args -r /world/$WORLD/clock:=/clock -r "$DEPTH:=/camera/depth/image_raw_gz" -r "$INFO:=/camera/depth/camera_info" > bridge.log 2>&1 &
# optical frame of the OAK-D-Lite stereo camera on the x500_depth: x500 pose (.12,.03,.242) + sensor (.01233,-.03,.01878), optical = x right, y down, z forward
ros2 run tf2_ros static_transform_publisher --x 0.13233 --y 0.0 --z 0.26078 --qx -0.5 --qy 0.5 --qz -0.5 --qw 0.5 \
  --frame-id base_link --child-frame-id camera_optical_frame > tf.log 2>&1 &
python3 $RIG/depth_relay.py --ros-args -p use_sim_time:=true > relay.log 2>&1 &
python3 $RIG/ev_bridge.py --ros-args -p model:=$MODEL -p use_sim_time:=false > ev.log 2>&1 &
echo "rig up; logs in /tmp/rig"; wait
