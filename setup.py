#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  setup.py
#
#  Copyright 2017 Matthew D. Cutone <cutonem (at) yorku.ca>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#
import os
import platform
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize, build_ext
import Cython

# this is a mess right now but it works okay, needs to be improved soon.
if platform.system() == 'Windows':
    # set enviornment variables for build
    os.environ["MSSdk"] = "1"
    os.environ["DISTUTILS_USE_SDK"] = "1"
    # get paths
    OCULUS_SDK_PATH = os.getenv('OCULUS_SDK_DIR', r'C:\OculusSDK')
    OCULUS_SDK_INCLUDE = os.path.join(OCULUS_SDK_PATH, 'LibOVR', 'Include')
    OCULUS_SDK_INCLUDE_EXTRAS = os.path.join(OCULUS_SDK_INCLUDE, 'Extras')
    OCULUS_SDK_LIB = os.path.join(
        OCULUS_SDK_PATH, 'LibOVR', 'Lib', 'Windows', 'x64', 'Release', 'VS2015')
    # set enviornment variables for build
    LIBRARIES = ['opengl32', 'User32', "LibOVR"]
    LIB_DIRS = [OCULUS_SDK_LIB]
else:
    raise Exception('Trying to install PsychHMD on an unsupported '
        'operating system. Exiting.')


# extensions to build
ext_modules = [
    Extension("psychxr.ovr.rift", ["psychxr/ovr/rift.pyx"],
              include_dirs=[OCULUS_SDK_INCLUDE,
                            OCULUS_SDK_INCLUDE_EXTRAS,
                            "psychxr/ovr/",
                            "include/"],
              libraries=LIBRARIES,
              library_dirs=LIB_DIRS,
              language="c++",
              extra_compile_args=[''])
]

setup_pars = {
    "name" : "psychxr",
    "author" : "Matthew D. Cutone",
    "author_email" : "cutonem(at)yorku.ca",
    "packages" : ['psychxr',
                  'psychxr.ovr'],
    #"package_data": {"psychxr.vrheadset.rift.ovrsdk": ["*.pyd"],
    #                 "": ["*.md", "*.txt"]},
    "license" : "GPLv3",
    "description":
        "API access from Python for eXended reality displays, used for "
        "developing psychology experiments.",
    "long_description": "",
    "classifiers" : [
        'Development Status :: 3 - Alpha',
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'Operating System :: Microsoft :: Windows :: Windows 8.1',
        'Operating System :: Microsoft :: Windows :: Windows 7',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Cython',
        'Intended Audience :: Science/Research'],
    "ext_modules" : cythonize(ext_modules),
    "requires" : ["Cython"],
    'py_modules' : [],
    "cmdclass" : {"build_ext": build_ext}}

setup(**setup_pars)
