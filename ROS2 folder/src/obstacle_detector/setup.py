from glob import glob

from setuptools import find_packages, setup

package_name = 'obstacle_detector'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(include=[package_name, package_name + '.*', 'core', 'core.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Egor',
    maintainer_email='egor.zhuravlev0407@gmail.com',
    description='Детекция посторонних объектов в габарите поезда по облаку точек лидара',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'obstacle_detector_node = obstacle_detector.detector_node:main',
        ],
    },
)
