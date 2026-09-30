# cyberwave_handeye_test_ws

ROS 2 Jazzy workspace for eye-in-hand calibration (ChArUco + AX=XB), dome
sampling with MoveIt, and 3D reconstruction on the SO-101 / OpenArm.

## Packages

| Package | Purpose |
| --- | --- |
| `handeye_calibration` | Python nodes: intrinsics, capture, calibration, auto-calibration, replay |
| `handeye_calibration_cpp` | C++ port, drop-in compatible (see its README; needs OpenCV >= 4.7) |
| `handeye_calibration_interfaces` | Shared services |
| `dome_sampler_pkg`, `dome_moveit_pkg` | Generate and execute dome-shaped camera poses |
| `record_positions` | Record / snapshot joint positions into a manifest |
| `reconstruct_3d` | Scene capture and reconstruction |
| `so100_follower_description` | SO-100 URDF and meshes |

`camera_tuner.py` and `teach_waypoints_to_manifest.py` are standalone helpers.

## Setup

Ubuntu 24.04 + ROS 2 Jazzy.

```bash
# 1. Third-party sources (pinned in thirdparty.repos, our changes in patches/)
sudo apt install python3-vcstool python3-colcon-common-extensions
vcs import src < thirdparty.repos
./patches/apply.sh

# 2. System / ROS dependencies. openarm_can comes from the OpenArm PPA;
#    ament_python and warehouse_ros_mongo have no rosdep entry on Jazzy.
sudo add-apt-repository ppa:openarm/main
sudo apt install libopenarm-can-dev
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -y -r \
  --skip-keys "ament_python warehouse_ros_mongo openarm_can"

# 3. Python venv for pip-only packages
python3 -m venv --system-site-packages ros_venv
ros_venv/bin/pip install -r requirements.txt

# 4. Build. Running colcon with the venv interpreter makes it the shebang of
#    every installed node, so no activation is needed at runtime.
ros_venv/bin/python -m colcon build --symlink-install \
  --packages-skip handeye_calibration_cpp
source install/setup.bash
```

`handeye_calibration_cpp` needs OpenCV >= 4.7 (Ubuntu ships 4.6); see its
README to build it against a separate OpenCV.

The venv and `build/`/`install/` hardcode absolute paths. After moving or
renaming the workspace, delete `ros_venv/`, `build/` and `install/` and
repeat steps 3 and 4.

## Third-party packages

Third-party repos are not committed. They are pinned in `thirdparty.repos`, and
our local changes to them live in `patches/<name>.patch`.

After changing a patched package, run `./patches/refresh.sh` and commit the
updated patch. To start patching a package that has no patch yet, create an
empty `patches/<name>.patch` first. To move to a newer upstream, bump
`version` in `thirdparty.repos` and re-apply (fix conflicts, then refresh).

Large data (`archive/`, `3d_reconstruction_dataset/`, capture images, `.ply`
files) is gitignored and kept locally only.
