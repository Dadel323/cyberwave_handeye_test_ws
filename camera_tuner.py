#!/usr/bin/env python3
"""
camera_tuner.py -- live view + V4L2 control panel for the calibration webcam.

Purpose: find the camera settings that make the ChArUco board detect reliably,
then copy them into webcam_params.yaml / the launch `focus:=` argument.

The controls are read from `v4l2-ctl -L` at startup, so whatever the device
exposes is what you get sliders for -- nothing is hardcoded per camera.

The board overlay uses handeye_calibration.utils.build_board with the board
block from pipeline_params.yaml, so what you see here is exactly what
intrinsics_node/capture_node will see.

Run it with the workspace sourced, and with NOTHING else holding the camera
(usb_cam refuses to share, and V4L2 control writes get EACCES while it
streams):

    source install/setup.bash
    ros_venv/bin/python camera_tuner.py           # --device /dev/video0
"""

import argparse
import re
import subprocess
import sys

import cv2
import yaml
from ament_index_python.packages import get_package_share_directory
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QGridLayout,
                             QGroupBox, QHBoxLayout, QLabel, QMainWindow,
                             QPushButton, QScrollArea, QSlider, QSpinBox,
                             QVBoxLayout, QWidget)

from handeye_calibration.utils import build_board

CONFIG = (f'{get_package_share_directory("handeye_calibration")}'
          '/config/webcam_params.yaml')


# ---------------------------------------------------------------- v4l2 controls
def list_controls(device):
    out = subprocess.run(['v4l2-ctl', '-d', device, '-L'],
                         capture_output=True, text=True).stdout
    ctrls, cur = [], None
    for line in out.splitlines():
        m = re.match(r'\s+(\w+)\s+0x[0-9a-f]+\s+\((\w+)\)\s*:\s*(.*)$', line)
        if m:
            name, ctype, rest = m.groups()
            nums = dict(re.findall(r'\b(min|max|step|default|value)=(-?\d+)', rest))
            cur = {'name': name, 'type': ctype, 'menu': [],
                   'min': int(nums.get('min', 0)), 'max': int(nums.get('max', 1)),
                   'step': int(nums.get('step', 1)), 'value': int(nums['value']),
                   'default': int(nums.get('default', 0)),
                   'inactive': 'inactive' in rest}
            ctrls.append(cur)
            continue
        m = re.match(r'\s+(\d+):\s*(.+)$', line)
        if m and cur and cur['type'] == 'menu':
            cur['menu'].append((int(m.group(1)), m.group(2).strip()))
    return ctrls


def set_control(device, name, value):
    r = subprocess.run(['v4l2-ctl', '-d', device, '--set-ctrl', f'{name}={value}'],
                       capture_output=True, text=True)
    return (r.stderr or r.stdout).strip()


# ---------------------------------------------------------------- board overlay
def config_controls():
    """The control names webcam.launch.py applies, in the order it applies
    them -- read from the config so this list cannot drift from it."""
    cfg = yaml.safe_load(open(CONFIG)) or {}
    return list(cfg.get('v4l2_controls', {}).get('ros__parameters', {}))


def load_board():
    cfg = yaml.safe_load(open(
        f'{get_package_share_directory("handeye_calibration")}'
        '/config/pipeline_params.yaml'))
    p = cfg['intrinsics_node']['ros__parameters']
    board, detector = build_board(p['squares_x'], p['squares_y'],
                                  p['square_length_m'], p['marker_length_m'],
                                  p['dictionary'], p['legacy_pattern'])
    label = (f"{p['squares_x']}x{p['squares_y']} {p['dictionary']} "
             f"legacy={p['legacy_pattern']}")
    return detector, label, p['min_charuco_corners']


class Tuner(QMainWindow):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.detector, self.board_label, self.min_corners = load_board()

        # This OpenCV build's V4L2 backend cannot open a device by name
        # ("can't be used to capture by name"), so address it by index.
        index = int(re.sub(r'\D', '', device) or 0)
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            sys.exit(f'Cannot open {device} -- is usb_cam still running?')
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.setWindowTitle(f'camera_tuner -- {device}')
        self.video = QLabel(alignment=Qt.AlignCenter)
        self.video.setMinimumSize(960, 540)
        self.stats = QLabel('')
        self.stats.setStyleSheet('font-family: monospace;')
        self.overlay = QCheckBox('ChArUco overlay')
        self.overlay.setChecked(True)

        left = QVBoxLayout()
        left.addWidget(self.video, 1)
        left.addWidget(self.stats)
        left.addWidget(self.overlay)

        self.widgets = {}
        panel = self.build_panel()
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(430)

        buttons = QHBoxLayout()
        for text, slot in [('Refresh', self.refresh_panel),
                           ('Defaults', self.restore_defaults),
                           ('Print settings', self.print_settings)]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            buttons.addWidget(b)

        right = QVBoxLayout()
        right.addWidget(scroll, 1)
        right.addLayout(buttons)

        root = QHBoxLayout()
        root.addLayout(left, 1)
        root.addLayout(right)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

    # ------------------------------------------------------------ control panel
    def build_panel(self):
        group = QGroupBox('V4L2 controls')
        gl = QGridLayout(group)
        for row, c in enumerate(list_controls(self.device)):
            gl.addWidget(QLabel(c['name']), row, 0)
            if c['type'] == 'bool':
                w = QCheckBox()
                w.setChecked(bool(c['value']))
                w.stateChanged.connect(
                    lambda s, n=c['name']: self.apply(n, 1 if s else 0))
                gl.addWidget(w, row, 1, 1, 2)
            elif c['type'] == 'menu':
                w = QComboBox()
                for val, text in c['menu']:
                    w.addItem(f'{val}: {text}', val)
                w.setCurrentIndex(max(0, [v for v, _ in c['menu']].index(c['value'])
                                       if c['value'] in [v for v, _ in c['menu']] else 0))
                w.currentIndexChanged.connect(
                    lambda i, n=c['name'], b=w: self.apply(n, b.itemData(i)))
                gl.addWidget(w, row, 1, 1, 2)
            else:
                w = QSlider(Qt.Horizontal)
                w.setRange(c['min'], c['max'])
                w.setSingleStep(c['step'])
                w.setValue(c['value'])
                box = QSpinBox()
                box.setRange(c['min'], c['max'])
                box.setSingleStep(c['step'])
                box.setValue(c['value'])
                w.valueChanged.connect(box.setValue)
                box.valueChanged.connect(w.setValue)
                w.valueChanged.connect(
                    lambda v, n=c['name'], s=c['step']: self.apply(n, v - v % s))
                gl.addWidget(w, row, 1)
                gl.addWidget(box, row, 2)
            w.setEnabled(not c['inactive'])
            self.widgets[c['name']] = w
        return group

    def apply(self, name, value):
        err = set_control(self.device, name, value)
        self.statusBar().showMessage(err or f'{name} = {value}', 4000)
        self.refresh_panel()            # toggling an "auto" flips others active

    def refresh_panel(self):
        for c in list_controls(self.device):
            w = self.widgets.get(c['name'])
            if w is None:
                continue
            w.setEnabled(not c['inactive'])
            w.blockSignals(True)
            if isinstance(w, QCheckBox):
                w.setChecked(bool(c['value']))
            elif isinstance(w, QComboBox):
                idx = w.findData(c['value'])
                if idx >= 0:
                    w.setCurrentIndex(idx)
            else:
                w.setValue(c['value'])
            w.blockSignals(False)

    def restore_defaults(self):
        for c in list_controls(self.device):
            set_control(self.device, c['name'], c['default'])
        self.refresh_panel()

    def print_settings(self):
        """Dump every control, then the subset webcam_params.yaml drives."""
        ctrls = list_controls(self.device)
        vals = {c['name']: c['value'] for c in ctrls}
        print('\n--- all camera settings ---')
        for c in ctrls:
            print(f'  {c["name"]:32} {c["value"]}')

        print('\n--- paste into config/webcam_params.yaml ---')
        print('v4l2_controls:')
        print('  ros__parameters:')
        for name in config_controls():
            if name in vals:
                print(f'    {name}: {vals[name]}')
        if vals.get('focus_automatic_continuous') != 0:
            print('\n  !! autofocus is ON -- turn it off before calibrating')
        if vals.get('auto_exposure') != 1:
            print('  !! auto_exposure is not Manual Mode (1)')
        print()

    # ------------------------------------------------------------------- video
    def tick(self):
        ok, bgr = self.cap.read()
        if not ok:
            return
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        n_corners = 0
        if self.overlay.isChecked():
            cc, ci, mc, mi = self.detector.detectBoard(gray)
            if mi is not None:
                cv2.aruco.drawDetectedMarkers(bgr, mc, mi)
            if ci is not None:
                n_corners = len(ci)
                cv2.aruco.drawDetectedCornersCharuco(bgr, cc, ci, (0, 0, 255))

        ok_mark = '  OK' if n_corners >= self.min_corners else '  too few'
        self.stats.setText(
            f'board {self.board_label}   corners {n_corners}/{self.min_corners}'
            f'{ok_mark}   sharpness {sharp:7.1f}   frame {bgr.shape[1]}x{bgr.shape[0]}')

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.video.setPixmap(QPixmap.fromImage(img).scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        super().closeEvent(event)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='/dev/video0')
    args = ap.parse_args()

    app = QApplication(sys.argv)
    win = Tuner(args.device)
    win.show()
    sys.exit(app.exec_())
