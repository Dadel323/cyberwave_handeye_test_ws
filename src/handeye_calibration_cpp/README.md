# handeye_calibration_cpp

C++ port of the `handeye_calibration` package: ChArUco detection, sample
collection, and the AX=XB eye-in-hand solve.

Both packages share `handeye_calibration_interfaces`, so the nodes are drop-in
interchangeable — you can run the C++ `capture_node` against the Python
`auto_calibration_node`, or the other way round.

## Requirements

**OpenCV >= 4.7.** `cv::aruco::CharucoDetector` and
`CharucoBoard::matchImagePoints` are 4.7+ APIs, and `setLegacyPattern()` — which
the `legacy_pattern: true` board configuration depends on — does not exist
before then. CMake fails at configure time with an explanatory message if an
older OpenCV is found, rather than building against a board model that silently
differs from the Python one.

Ubuntu 24.04's `libopencv-dev` is 4.6, so a newer OpenCV has to be built or
installed separately and pointed at:

```bash
colcon build --packages-select handeye_calibration_cpp \
  --cmake-args -DOpenCV_DIR=/path/to/opencv/build
```

Note that the Python package sidesteps this entirely: it uses the
`opencv-python` wheel in `ros_venv` (4.14), which ships no C++ headers or
libraries and so cannot be linked against.

## Nodes

| Executable | Services | Notes |
|---|---|---|
| `capture_node` | `~/sample_position` | Image + TF gripper pose -> `dataset.yaml` |
| `calibration_node` | `calibrate_dataset` | Re-detects boards, solves, writes `handeye_result.yaml` |
| `auto_calibration_node` | — | Orchestrates move → capture → solve |
| `intrinsics_node` | `~/capture`, `~/calibrate`, `~/reset` | Monocular intrinsics from the same board |
| `run_points_node` | `~/move_to_next_point` | Replays a recorded joint manifest |
| `viewer_node` | — | Live view of `<image_topic>/compressed` |

Launch files (`pipeline`, `intrinsics`, `viewer`, `webcam`) and the config files
mirror the Python package's, repointed at this package.

## Not ported

The offline analysis and visualization scripts stay in the Python package —
they are matplotlib/Open3D tools with no sensible C++ equivalent, and nothing
in the runtime pipeline depends on them:

`visualize_scene.py`, `accuracy_vs_samples.py`, `deviation_vs_samples.py`,
`process_dataset.py`, `visualize_board_poses.py`.

They read `dataset.yaml` and `handeye_result.yaml`, which this package writes in
the same format, so they work unchanged against C++-produced datasets.

## Verification

The solver was checked against the Python on the `data_so101` dataset. Given
the *same* board poses, the two agree to 15 decimal places on translation,
quaternion and leave-one-out standard deviation.

Detection is where they can diverge, and the cause is the OpenCV version rather
than the port. On images where the board is marginal, OpenCV 4.10 recovers more
ChArUco corners than 4.14 does — on `data_so101`, 4.10 accepted `sample_000.png`
(13 corners) where 4.14 rejects it (4 corners), which changed the solve. Images
both versions accept give board poses agreeing to ~0.1 mm.

The practical consequence: **the OpenCV version is part of your calibration
setup.** A dataset re-solved under a different OpenCV can admit or reject
borderline captures and move the result. If you compare a C++ result against a
Python one, match the versions first, and treat `num_samples`/`num_skipped` in
`handeye_result.yaml` as the first thing to check when two runs disagree.

## Differences from the Python

These are deliberate, and each removes a hazard the Python version documents in
its own comments:

- **`cv_bridge` instead of the hand-rolled `imgmsg_to_cv2`.** That helper exists
  only because `cv_bridge`'s Python extension is built against NumPy 1.x's ABI
  and crashes under NumPy 2.x. That is a Python-specific problem; the C++
  `cv_bridge` is the right dependency here.

- **Quaternions are produced in `xyzw` order by `matrix_to_quat_xyzw`.** The
  Python went through `transforms3d`, whose `mat2quat` returns `wxyz`; reading
  that positionally as `xyzw` once rotated the camera frame by ~180° in RViz.
  A single fixed-order function means that class of bug cannot recur.

- **`calibration_node` logs *which* entries were skipped**, not just how many.
  The Python counted them and discarded which — the information you need to go
  and look at the bad capture.

- **`run_points_node` uses a multi-threaded executor.** Its service handler
  sleeps while the arm travels; on a single-threaded executor that would also
  stall the `joint_states` subscription.

- **`auto_calibration_node` spins the executor on a separate thread**, since
  `run()` blocks on service futures and C++ has no equivalent of re-entering
  the loop via `spin_until_future_complete`.
