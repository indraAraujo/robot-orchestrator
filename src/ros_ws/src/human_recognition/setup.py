import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'human_recognition'
database_dir = os.path.join('human_recognition', 'database')

database_data_files = [
    (os.path.join('share', package_name, 'database', person),
     sorted(glob(os.path.join(database_dir, person, '*'))))
    for person in sorted(os.listdir(database_dir))
    if os.path.isdir(os.path.join(database_dir, person))
]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
    ] + database_data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='indra',
    maintainer_email='indra.araujo.santos@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_publisher = human_recognition.camera_publisher:main',
            'face_recognizer = human_recognition.face_recognizer:main',
            'recognition_visualizer = human_recognition.recognition_visualizer:main',
        ],
    },
)
