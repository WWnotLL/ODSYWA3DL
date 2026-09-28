import os
from glob import glob

from setuptools import setup

package_name = 'bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz') + glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Egor',
    maintainer_email='egor.zhuravlev0407@gmail.com',
    description='Запуск детекции препятствий: положение лидара в TF, нода детекции, RViz',
    license='Apache-2.0',
)
