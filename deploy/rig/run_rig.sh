#!/bin/bash
# Host-side wrapper. Shared desktop: the container is capped at 8 CPUs, low CPU shares, started only for tests.
#   run_rig.sh build   build the image (JOBS=8 compile parallelism)
#   run_rig.sh up      (SPAWN_POSE=x,y,z,roll,pitch,yaw_enu overrides world/spawn.txt; BATTERY_SOC=0..1 pack charge) start the container (detached), then gz + PX4 + agent + bridges
#   run_rig.sh sh      shell in the running container (ROS sourced)    run_rig.sh policy [ros args]   dry-run policy node
#   run_rig.sh logs    run_rig.sh down
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"; NAME=dronehacks-rig
SRC='. /opt/ros/jazzy/setup.bash; . /ws/install/setup.bash'
case "${1:-}" in
  build) cd "$HERE" && exec docker build --build-arg JOBS=8 --build-arg PX4_REPO --build-arg XRCE_REPO --build-arg MSGS_REPO -t dronehacks-rig:phase1 . ;;
  up) docker rm -f $NAME >/dev/null 2>&1 || true
      docker run -d --name $NAME --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all --cpus 8 ${SPAWN_POSE:+-e SPAWN_POSE=$SPAWN_POSE} ${BATTERY_SOC:+-e BATTERY_SOC=$BATTERY_SOC} --cpu-shares 256 --memory 24g -v "$REPO:/rig:ro" dronehacks-rig:phase1 /rig/deploy/rig/rig_entry.sh ;;
  sh) exec docker exec -it $NAME bash -c "$SRC; exec bash" ;;
  policy) shift; exec docker exec $([ -t 0 ] && echo -it) $NAME bash -c "$SRC; cd /rig/deploy && exec python3 main_policy.py --ros-args -p use_sim_time:=true -p range_source:=altitude $*" ;;
  logs) exec docker exec $NAME bash -c 'tail -n 20 /tmp/rig/*.log' ;;
  down) docker rm -f $NAME ;;
  *) sed -n 2,6p "$0"; exit 1 ;;
esac
