#!/usr/bin/env python3
"""
teach_waypoints_to_manifest.py

Convert so101_teach's waypoints file into the manifest run_points_node replays.

so101_teach writes

    waypoints:
    - {shoulder_pan: 0.1, shoulder_lift: -0.2, ...}
    generated: '...'

while run_points_node reads manifest.yaml, the format snapshot_node produces:

    entries:
    - joint_positions:
        shoulder_pan: 0.1
        ...

Both hold the same numbers in the same units -- radians relative to the
calibration's homing offset, by the same (raw - home) * 2*pi / 4096 that
motor_bridge uses to command them. Only the nesting differs.

    ./teach_waypoints_to_manifest.py \
        ~/.so101_teach_waypoints.yaml \
        src/record_positions/data/manifest_so100.yaml
"""

import pathlib
import sys

import yaml

EXPECTED_JOINTS = {'shoulder_pan', 'shoulder_lift', 'elbow_flex',
                   'wrist_flex', 'wrist_roll', 'gripper'}


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip())
        return 2

    src = pathlib.Path(argv[1]).expanduser()
    dst = pathlib.Path(argv[2]).expanduser()

    loaded = yaml.safe_load(src.read_text(encoding='utf-8')) or {}
    waypoints = loaded.get('waypoints', [])
    if not waypoints:
        print(f'No waypoints in {src}.', file=sys.stderr)
        return 1

    entries = []
    for i, wp in enumerate(waypoints, start=1):
        missing = EXPECTED_JOINTS - set(wp)
        if missing:
            # A short read on the bus can drop a joint. Replaying a pose with a
            # joint missing silently leaves that joint wherever it was, so the
            # recorded pose and the pose actually reached differ -- drop it.
            print(f'  waypoint {i}: missing {sorted(missing)}, skipped',
                  file=sys.stderr)
            continue
        entries.append({'joint_positions': {k: float(v) for k, v in wp.items()}})

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, 'w', encoding='utf-8') as fp:
        yaml.safe_dump({'entries': entries}, fp, sort_keys=False)

    print(f'{len(entries)} of {len(waypoints)} waypoints -> {dst}')
    if len(entries) < len(waypoints):
        print('Set num_samples in so100_params.yaml to the number written.')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
