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
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize, build_ext
import numpy

# Cython specific include directories
CYTHON_INCLUDE_DIRS = [numpy.get_include(), '.']

# C++ include directories
CPP_INCLUDE_DIRS = [numpy.get_include(), '.', 'include/']

# library directories and library names
LIB_DIRS = ['lib/']
LIBRARIES = ['opengl32']  # required

# extension modules to add to the package
EXT_MODULES = []

# platform and build information
THIS_PLATFORM = platform.system()
BUILD_LIBOVR = False
BUILD_OPENHMD = True

# setup build environment, needs to use MSVC on windows
if platform.system() == 'Windows':
    # This makes sure the correct compiler is used, even if not explicitly
    # specified.
    os.environ["MSSdk"] = '1'
    os.environ["DISTUTILS_USE_SDK"] = '1'

    # only can be set on windows
    BUILD_LIBOVR = os.environ.get('PSYCHXR_BUILD_LIBOVR', '1') == '1'
    LIBRARIES.extend(['User32'])  # required

# print out the build configuration
if BUILD_LIBOVR:
    print("Configured to build `LibOVR` extension modules.")

if BUILD_OPENHMD:
    print("Configured to build `OpenHMD` extension modules.")

# additional package data
PACKAGES = ['psychxr']
DATA_FILES = []

# Build LibOVR extensions which uses the official Oculus driver, this is
# optional and only supported on Windows.
if BUILD_LIBOVR:
    # get SDK base path
    libovr_sdk_path = os.environ.get('PSYCHXR_LIBOVR_SDK_PATH', None)
    if libovr_sdk_path is None:
        libovr_sdk_path = r"C:\OculusSDK"  # default
    libovr_sdk_path = os.path.join(libovr_sdk_path, 'LibOVR')

    # get libraries and include paths
    libovr_libs = LIBRARIES + ['LibOVR']
    libovr_include = CPP_INCLUDE_DIRS + [
        os.path.join(libovr_sdk_path, 'Include'),
        os.path.join(libovr_sdk_path, 'Include', 'Extras')]
    libovr_libdir = LIB_DIRS + [
        os.path.join(
            libovr_sdk_path, 'Lib', 'Windows', 'x64', 'Release', 'VS2017')]

    # compile from Cython to C++
    cythonize("psychxr/libovr/_libovr.pyx",
              include_path=CYTHON_INCLUDE_DIRS,
              compiler_directives={'embedsignature': True,
                                   'language_level': 3})

    # build the module and add it to the package directory
    EXT_MODULES.extend([
        Extension(
            "psychxr.libovr._libovr",
            ["psychxr/libovr/_libovr" + ".cpp"],  # cythonized file
            include_dirs=libovr_include,
            libraries=libovr_libs,
            library_dirs=libovr_libdir,
            language="c++",
            extra_compile_args=['']
        )]
    )

    PACKAGES.extend(['psychxr.libovr'])


# Build the open source OpenHMD VR drivers, these are always built since we
# ship with the libraries
if BUILD_OPENHMD:
    # get libraries and include paths
    ohmd_libs = LIBRARIES + ['hidapi', 'openhmd']
    ohmd_include = CPP_INCLUDE_DIRS
    if THIS_PLATFORM == 'Windows':
        # use the DLLs we shipped with
        ohmd_libdir = LIB_DIRS + [os.path.join('lib', 'win', 'x64')]
    else:
        # other platforms will use the system libs
        ohmd_libdir = LIB_DIRS

    # compile from Cython to C++
    cythonize("psychxr/openhmd/_openhmd.pyx",
              include_path=CYTHON_INCLUDE_DIRS,
              compiler_directives={'embedsignature': True,
                                   'language_level': 3})

    # build the module and add it to the package directory
    EXT_MODULES.extend([
        Extension(
            "psychxr.openhmd._openhmd",
            ["psychxr/openhmd/_openhmd" + ".cpp"],  # cythonized file
            include_dirs=ohmd_include,
            libraries=ohmd_libs,
            library_dirs=ohmd_libdir,
            language="c++",
            extra_compile_args=['']
        )]
    )

    PACKAGES.extend(['psychxr.openhmd'])


# Setup parameters
setup_pars = {
    "name": "psychxr",
    "author": "Matthew D. Cutone, Laurie M. Wilcox",
    "author_email": "cutonem@yorku.ca",
    "maintainer": "Matthew D. Cutone",
    "maintainer_email": "cutonem@yorku.ca",
    "packages": PACKAGES,
    "url": "http://psychxr.org",
    #"package_data": PACKAGE_DATA,
    "include_package_data": True,
    "version": "0.2.3",
    "license" : "MIT",
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
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Cython',
        'Intended Audience :: Science/Research'],
    "ext_modules": EXT_MODULES,
    #"data_files": DATA_FILES,
    "include_dirs": ["."],
    "install_requires" : ["Cython>=0.29.3", "numpy>=1.13.3"],
    "requires" : [],
    "cmdclass" : {"build_ext": build_ext}}

setup(**setup_pars)


