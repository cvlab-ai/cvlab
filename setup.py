#!/usr/bin/env python3

import sys
import os
from distutils.core import setup

from setuptools import find_packages
from glob import glob


data_files = glob("cvlab/images/*.*") + \
             glob("cvlab/styles/*/*.*") + \
             glob("cvlab/styles/*/*/*.*")
data_files = list(map(lambda x:x[6:], data_files))

sample_files = glob("cvlab_samples/*.cvlab")
sample_files = list(map(lambda x:x[14:], sample_files))

if sys.version_info.major <= 2:
    raise Exception("Only python 3+ is supported!")

is_windows = os.name == 'nt'
is_linux = os.name == "posix"


requirements = [
    "numpy",
    "scipy",
    "pygments>=2",
    "matplotlib",
    "tinycss2",
    "sip",
]

try:
    import cv2
    if str(cv2.__version__) < "3":
        print("WARNING! OpenCV version 2.x detected. It is *strongly advised* to install OpenCV 3.x")
        print("Please visit: https://opencv.org/releases.html")
except ImportError:
    print("ERROR! OpenCV is required. Trying to use opencv-python package...")
    print("If it doesn't work, please visit: https://opencv.org/releases.html")
    requirements.append("opencv-python")


try:
    import PyQt5
except ImportError:
    requirements.append("pyqt5")


__version__ = None
package_name = None
exec(compile(open('cvlab/version.py').read(), 'cvlab/version.py', 'exec'))


this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

description = long_description.splitlines()[0].strip()

setup(
    name=package_name,
    version=__version__,
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Adam Brzeski, Jan Cychnerski',
    author_email='adam.m.brzeski@gmail.com, jan.cychnerski@gmail.com',
    url='https://github.com/cvlab-ai/cvlab',
    packages=find_packages(exclude=["diagrams","tools","samples","temp"]),
    package_data={"cvlab":data_files, "cvlab_samples":sample_files},
    entry_points={'gui_scripts': ['cvlab=cvlab.__main__:main']},
    license="AGPL-3.0+",
    python_requires='>=3.3',
    install_requires=requirements,
)
