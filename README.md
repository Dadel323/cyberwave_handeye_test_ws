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

## Third-party packages

Third-party repos are not committed. They are pinned in `thirdparty.repos`, and
our local changes to them live in `patches/<name>.patch`.

```bash
sudo apt install python3-vcstool
vcs import src < thirdparty.repos
./patches/apply.sh
```

After changing a patched package, run `./patches/refresh.sh` and commit the
updated patch. To start patching a package that has no patch yet, create an
empty `patches/<name>.patch` first. To move to a newer upstream, bump
`version` in `thirdparty.repos` and re-apply (fix conflicts, then refresh).

## Build

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Large data (`archive/`, `3d_reconstruction_dataset/`, capture images, `.ply`
files) is gitignored and kept locally only.
