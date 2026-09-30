from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'handeye_calibration'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='you@example.com',
    description='Eye-in-hand calibration for SO-101 + Orbbec Gemini 335Lg.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'auto_calibration_node = handeye_calibration.auto_calibration_node:main',
            'calibration_node = handeye_calibration.calibration_node:main',
            'capture_node = handeye_calibration.capture_node:main',
            'intrinsics_node = handeye_calibration.intrinsics_node:main',
            'run_points_node = handeye_calibration.run_points_node:main',
            'viewer_node = handeye_calibration.viewer_node:main',
            'visualize_scene = handeye_calibration.visualize_scene:main',
        ],
    },
)
