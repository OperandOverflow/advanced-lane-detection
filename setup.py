from setuptools import setup

package_name = 'lane_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/lane_detection_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Advanced lane detection: lane lines, road curvature and vehicle offset from the camera image',
    license='MIT',
    entry_points={
        'console_scripts': [
            'lane_detection_node = lane_detection.lane_detection_node:main',
        ],
    },
)
