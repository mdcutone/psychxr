#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  setup.py
#
#  Copyright 2017 Matthew D. Cutone <cutonem (at) yorku.ca>
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#
"""Installation script for PsychXR.

Installation is straightforward as long as your build environment is properly
configured. Significant portions of this library are written in Cython which is
converted to C++ code. Therefore, you must have a C++ compiler and SDKs
installed prior to building PsychXR from source. See
https://github.com/mdcutone/psychxr/blob/master/README.md for instructions on
how to build from source.

"""
import os
import platform
from pathlib import Path, PureWindowsPath
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize, build_ext
import numpy

# compiler related data
_include_dir_ = [numpy.get_include()]
_lib_dirs_ = []
_libraries_ = []
_build_ext_ = []

# additional package data
PACKAGES = ['psychxr']
DATA_FILES = []

# list of built extensions
ext_modules = []

# setup build environments on supported platforms
if platform.system() == 'Windows':
    os.environ["MSSdk"] = '1'  # this makes sure the correct compiler is used
    os.environ["DISTUTILS_USE_SDK"] = '1'
    _libraries_.extend(['opengl32', 'User32'])  # required Windows libraries
else:
    raise Exception(
        "Trying to install `PsychXR` on an unsupported operating system. "
        "Exiting.")

# check if which HMD were building libraries for and build the extensions
if os.environ.get('PSYCHXR_BUILD_LIBOVR', '1') == '1':
    print("building `LibOVR` extension modules ...")

    # build parameters for LibOVR passed to the compiler and linker
    libovr_package_data = {
        'psychxr.drivers.libovr': ['*.pxi', '*.pxd', '*.pyx', '*.cpp']}
    libovr_data_files = {'psychxr/drivers/libovr': ['*.pyd', '*.pxi']}
    libovr_build_params = {
        'libs': ['LibOVR'],
        'lib_dir': [],
        'include': [],
        'packages': ['psychxr.drivers.libovr'],
        'package_data': libovr_package_data,
        'data_files': libovr_data_files}

    # get the path to the SDK, uses the `cohorts` folder if not defined
    env_libovr_sdk_path = os.environ.get(
        'PSYCHXR_LIBOVR_SDK_PATH',
        r'cohorts\LibOVR')

    # convert to a path object, make absolute
    libovr_sdk_path = Path(PureWindowsPath(env_libovr_sdk_path)).absolute()

    # check if the path is a directory
    if not libovr_sdk_path.is_dir():
        raise NotADirectoryError(
            "ERROR: Cannot find the Oculus PC SDK at the specified location "
            "'{}'. Make sure `PSYCHXR_LIBOVR_SDK_PATH` is set.".format(
                str(libovr_sdk_path)))

    # tell the user where the setup is looking for the SDK
    print(r"Using `PSYCHXR_LIBOVR_SDK_PATH={}`".format(str(libovr_sdk_path)))

    # base paths within the SDK
    libovr_base_path = libovr_sdk_path.joinpath('LibOVR')
    libovr_include_path = libovr_base_path.joinpath('Include')
    libovr_lib_path = libovr_base_path.joinpath('Lib')

    # add paths
    libovr_build_params['include'] = [
        str(libovr_include_path),
        str(libovr_include_path.joinpath('Extras'))]
    libovr_build_params['lib_dir'] = [
        str(libovr_lib_path.joinpath(
            'Windows', 'x64', 'Release', 'VS2017'))]

    # compile the `libovr` extension
    cythonize(
        "psychxr/drivers/libovr/_libovr.pyx",
        include_path=libovr_build_params['include'],
        compiler_directives={
            'embedsignature': True,
            'language_level': 3})
    ext_modules.extend([
        Extension(
            "psychxr.drivers.libovr._libovr",
            ["psychxr/drivers/libovr/_libovr" + ".cpp"],
            include_dirs=_include_dir_ + libovr_build_params['include'],
            libraries=_libraries_ + libovr_build_params['libs'],
            library_dirs=_lib_dirs_ + libovr_build_params['lib_dir'],
            language="c++",
            extra_compile_args=[''])
    ])
    PACKAGES.extend(libovr_build_params['packages'])

# build the OpenHMD driver
if os.environ.get('PSYCHXR_BUILD_OPENHMD', '1') == '1':
    print("building `OpenHMD` extension modules ...")

    ohmd_package_data = {
        'psychxr.drivers.libovr': ['*.pxi', '*.pxd', '*.pyx', '*.cpp']}
    ohmd_data_files = {'psychxr/drivers/openhmd': ['*.pyd', '*.pxi']}
    ohmd_build_params = {
        'libs': ['LibOVR'],
        'lib_dir': [],
        'include': [],
        'packages': ['psychxr.drivers.openhmd'],
        'package_data': ohmd_package_data,
        'data_files': ohmd_data_files}


setup_pars = {
    "name": "psychxr",
    "author": "Matthew D. Cutone, Laurie M. Wilcox",
    "author_email": "mcutone@opensciencetools.com",
    "maintainer": "Matthew D. Cutone",
    "maintainer_email": "mcutone@opensciencetools.com",
    "packages": PACKAGES,
    "url": "http://psychxr.org",
    "include_package_data": True,
    "version": "0.2.4",
    "license": "MIT",
    "description":
        "Python extension library for interacting with eXtended Reality "
        "displays (HMDs), intended for research in neuroscience and "
        "psychology.",
    "long_description": "",
    "classifiers": [
        'Development Status :: 5 - Production/Stable',
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'Operating System :: Microsoft :: Windows :: Windows 8.1',
        'Operating System :: Microsoft :: Windows :: Windows 7',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Human Machine Interfaces',
        'Intended Audience :: Science/Research'],
    "ext_modules": ext_modules,
    "install_requires": ["Cython>=0.29.3", "numpy>=1.13.3"],
    "requires": [],
    "cmdclass": {"build_ext": build_ext},
    "zip_safe": False
}

setup(**setup_pars)
