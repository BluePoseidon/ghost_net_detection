from setuptools import setup

package_name = 'ghost_net_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dag',
    maintainer_email='dag@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detection_node=ghost_net_detection.detection_node:main',
            'train_svm=ghost_net_detection.svm_model:train_and_save_model',
            'create_synthetic_training_data=ghost_net_detection.synthetic_histogram_data:create_synthetic_histogram_data',
        ],
    },
)
