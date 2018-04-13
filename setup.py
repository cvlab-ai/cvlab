#!/usr/bin/env python

from __future__ import print_function

import sys
import os
from distutils.core import setup

from setuptools import find_packages
from glob import glob


data_files = glob("cvlab/images/*.*")+glob("cvlab/styles/*/*.*")+glob("cvlab/styles/*/*/*.*")
data_files = list(map(lambda x:x[6:], data_files))

is_python2 = sys.version_info.major == 2
is_windows = os.name == 'nt'
is_linux = os.name == "posix"


requirements = [
    "numpy",
    "scipy",
    "pygments>=2",
    "matplotlib",
    "tinycss2"
]


if is_python2:
    requirements += [
        "future>=0.16",
        "configparser",
    ]


try:
    import cv2
    if str(cv2.__version__) < "3":
        print("WARNING! OpenCV version 2.x detected. It is *strongly advised* to install OpenCV 3.x")
        print("Please visit: https://opencv.org/releases.html")
except ImportError:
    print("ERROR! OpenCV is required. Trying to use python-opencv package...")
    print("If it doesn't work, please visit: https://opencv.org/releases.html")
    requirements.append("python-opencv")


try:
    import PyQt4
except ImportError:
    print("ERROR! You must install PyQt4 to use CV-Lab!")
    if is_linux:
        print("Under ubuntu, you can run: sudo apt-get install python-qt4")
    if is_windows:
        print("Under Windows, it is advised to use Anaconda or Python(x,y)")
        print("You can also try: pip install python-qt5")
    print("With conda, you can run: conda install pyqt=4")
    print("Please read: https://www.riverbankcomputing.com/software/pyqt/download")
    exit(1)


__version__ = None
package_name = None
exec(compile(open('cvlab/version.py').read(), 'cvlab/version.py', 'exec'))


setup(
    name=package_name,
    version=__version__,
    description='CV Lab - Computer Vision Laboratory - GUI for computer vision algorithm design, prototyping and implementation',
    author='Adam Brzeski, Jan Cychnerski',
    author_email='adam.m.brzeski@gmail.com, jan.cychnerski@gmail.com',
    url='https://github.com/cvlab-ai/cvlab',
    packages=find_packages(exclude=["diagrams","tools","samples","temp"]),
    package_data={"cvlab":data_files},
    entry_points={'gui_scripts': ['cvlab=cvlab.__main__:main']},
    license="AGPL-3.0+",
    python_requires='>=2.7',
    install_requires=requirements,
)
