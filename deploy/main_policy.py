"""Entry point: run from the directory that contains policynav/ (a ROS 2 / PX4 companion environment), then

    python3 main_policy.py --ros-args -p use_sim_time:=true -p range_source:=sim            # dry run: never arms
    python3 main_policy.py --ros-args -p use_sim_time:=true -p range_source:=sim -p arm:=true   # flies (team approval)
"""
import sys

from policynav import node

if __name__ == '__main__':
    node.main(sys.argv)
