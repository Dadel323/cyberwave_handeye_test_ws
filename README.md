# cyberwave_handeye_test_ws

ROS 2 Jazzy workspace for eye-in-hand calibration of a wrist webcam
(ChArUco + AX=XB) on the SO-101 / OpenArm, plus dome sampling with MoveIt and
3D scene capture.

## Packages

| Package | Purpose |
| --- | --- |
| `handeye_calibration` | Intrinsics, sample capture, hand-eye solve, replay and analysis scripts |
| `handeye_calibration_interfaces` | Services shared by the nodes |
| `record_positions` | Teach joint positions into a manifest for replay |
| `dome_sampler_pkg`, `dome_moveit_pkg` | Generate and execute dome-shaped camera poses with MoveIt |
| `reconstruct_3d` | Capture images + poses for 3D reconstruction |
| `so100_follower_description` | SO-100 URDF and meshes |

Helpers at the root: `camera_tuner.py` (live V4L2 tuning with the board
overlay) and `teach_waypoints_to_manifest.py` (so101_teach waypoints to a
`record_positions` manifest).

## Usage

All launch files document their arguments in their header; the usual order is:

```bash
ros2 launch handeye_calibration viewer.launch.py         # check framing/focus
ros2 launch handeye_calibration intrinsics.launch.py     # once per camera
ros2 launch handeye_calibration so101_handeye.launch.py  # hand-eye, SO-101 placed by hand
ros2 launch handeye_calibration pipeline.launch.py       # hand-eye, replaying a manifest
ros2 launch handeye_calibration publish_handeye_tf.launch.py
```

Config files use absolute paths under
`/home/cyb/projects/cyberwave_handeye_test_ws`; adjust them if the workspace
lives elsewhere.

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

# 3. Python venv for pip-only packages (COLCON_IGNORE keeps colcon out of it)
python3 -m venv --system-site-packages ros_venv
ros_venv/bin/pip install -r requirements.txt
touch ros_venv/COLCON_IGNORE

# 4. Build inside the venv, so CMake and every installed node's shebang use
#    its interpreter (no activation needed at runtime)
source ros_venv/bin/activate
python -m colcon build --symlink-install
source install/setup.bash
```

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
