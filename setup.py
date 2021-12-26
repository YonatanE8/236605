from distutils.core import setup
from setuptools import find_packages

import os

cwd = os.getcwd()

setup(
    name='AdvancedDL',
    version='1.0',
    description=(
        'This package implements models for the course 236605 - Advanced deep learning.'),
    author='Yonatan Elul, Eyal Rozenberg',
    author_email='renedal@gmail.com',
    url='https://github.com/YonatanE8/236605.git',
    license='',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Private',
        'Topic :: Software Development :: Deep Learning Course',
        'Programming Language :: Python :: 3.8',
    ],
    package_dir={'AdvancedDL': os.path.join(cwd, 'AdvancedDL')},
    packages=find_packages(
        exclude=['data', 'logs']
    ),
    install_requires=[
        'torch',
        'torchvision',
        'torchmetrics',
        'numpy',
        'scipy',
        'matplotlib',
        'tqdm',
        'pandas',
    ],
)
