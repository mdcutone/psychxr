#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  setup.py
#
#  Copyright 2021 Matthew D. Cutone <mcutone(at)opensciencetools.com>
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

# Version string for the project. Make sure this is updated to reflect the
# project branch version.
PSYCHXR_VERSION_STRING = '0.2.4rc2'

# environment variable values
ENV_TRUE = '1'
ENV_FALSE = '0'

# Cython and C++ include directories
CYTHON_INCLUDE_DIRS = [numpy.get_include(), '.']
CPP_INCLUDE_DIRS = CYTHON_INCLUDE_DIRS + ['include/', 'psychxr/tools/']

# library directories and library names
LIBRARY_DIRS = LIB_DIRS = ['lib/']
LIBRARIES = []  # required

# Extension modules to add to the package. This grows as extensions are built.
EXT_MODULES = []

# additional package data
PACKAGES = ['psychxr']
PACKAGE_DATA = ['*.pxi', '*.pxd', '*.pyx', '*.cpp', '*.h', '*.c']
DATA_FILES = ['*.pyd', '*.pxi', '*.dll', '*.lib']

# platform and build information
THIS_PLATFORM = platform.system()
BUILD_OPENXR = os.environ.get('PSYCHXR_BUILD_OPENXR', ENV_TRUE) == ENV_TRUE
BUILD_LIBOVR = os.environ.get('PSYCHXR_BUILD_LIBOVR', ENV_TRUE) == ENV_TRUE
BUILD_OPENHMD = os.environ.get('PSYCHXR_BUILD_OPENHMD', ENV_TRUE) == ENV_TRUE


# ------------------------------------------------------------------------------
# Setup build environment
#

# print out the build configuration
if BUILD_OPENXR:
    print("Configured to build `OpenXR` extension modules "
          "(`PSYCHXR_BUILD_OPENXR=1`).")

if BUILD_LIBOVR:
    print("Configured to build `LibOVR` extension modules "
          "(`PSYCHXR_BUILD_LIBOVR=1`).")

if BUILD_OPENHMD:
    print("Configured to build `OpenHMD` extension modules "
          "(`PSYCHXR_BUILD_OPENHMD=1`).")


# setup build environments on supported platforms
if THIS_PLATFORM == 'Windows':
    os.environ["MSSdk"] = ENV_TRUE  # ensure correct compiler is used
    os.environ["DISTUTILS_USE_SDK"] = ENV_TRUE
    LIBRARIES.extend(['opengl32', 'User32'])  # required Windows libraries

    if BUILD_OPENHMD:
        LIBRARY_DIRS.extend(
            [os.path.join('psychxr/drivers/openhmd/lib', 'win', 'x64')])

    if BUILD_OPENXR:
        LIBRARY_DIRS.extend(
            [os.path.join('psychxr/drivers/openxr/lib', 'win', 'x64')])

else:
    if BUILD_LIBOVR:  # windows only
        print("WARNING: Cannot build `LibOVR`, platform is not supported.")
        BUILD_LIBOVR = False


# ------------------------------------------------------------------------------
# Helper functions for the installer
#

def fix_path(path):
    """Fix a path and convert to an absolute location.

    Parameters
    ----------
    path : str
        Path to fix.

    Returns
    -------
    str
        Absolute path.

    """
    if THIS_PLATFORM == 'Windows':
        path = PureWindowsPath(path)

    return str(Path(path).absolute())


def build_extension(name, **kwargs):
    """Build an extension.

    Parameters
    ----------
    name : str
        FQN of the extension module.

    Returns
    -------
    Extension
        Object referencing the build extension module.

    """
    # build the module and add it to the package directory
    pyx_file_path = [r"/".join(name.split('.')) + ".pyx"]
    pxd_include_path = kwargs.get('include_path', []) + CYTHON_INCLUDE_DIRS
    cpp_file_path = [r"/".join(name.split('.')) + ".cpp"]
    cpp_include_dirs = kwargs.get('include_dirs', []) + CPP_INCLUDE_DIRS
    cpp_libraries = kwargs.get('libraries', []) + LIBRARIES
    cpp_library_dirs = kwargs.get('library_dirs', []) + LIBRARY_DIRS

    cythonize(
        pyx_file_path,
        include_path=pxd_include_path,
        compiler_directives={
            'embedsignature': True,
            'language_level': 3})

    ext = Extension(
        name,
        cpp_file_path,
        include_dirs=cpp_include_dirs,
        libraries=cpp_libraries,
        library_dirs=cpp_library_dirs,
        language="c++",
        extra_compile_args=[''])

    return ext


# ------------------------------------------------------------------------------
# Build common libraries
#
print("Building `psychxr.tools.vrmath` extension module ...")

vrmath_package_data = {
    'psychxr.tools': PACKAGE_DATA}
vrmath_data_files = {
    'psychxr/tools': DATA_FILES}
vrmath_build_params = {
    'libraries': LIBRARIES + [],
    'library_dirs': LIBRARY_DIRS,
    'include_dirs': [fix_path('include/linmath')],
    'packages': ['psychxr.tools'],
    'package_data': vrmath_package_data,
    'data_files': vrmath_data_files}

# compile the `tools` extension
EXT_MODULES.extend(
    [build_extension(
        "psychxr.tools.vrmath",
        **vrmath_build_params)])
PACKAGES.extend(vrmath_build_params['packages'])


# ------------------------------------------------------------------------------
# Build driver extension libraries (e.g., libovr, openhmd, etc.)
#
if BUILD_OPENXR:
    print("Building `psychxr.drivers.openxr` extension module ...")

    openxr_package_data = {
        'psychxr.drivers.openxr': PACKAGE_DATA}
    openxr_data_files = {
        'psychxr/drivers/openxr': DATA_FILES}
    openxr_build_params = {
        'libraries': LIBRARIES + ['OpenXR'],
        'library_dirs': LIBRARY_DIRS,
        'include_dirs': [  # header files for needed libraries
            fix_path('include/openxr')],
        'packages': ['psychxr.drivers.openxr'],
        'package_data': openxr_package_data,
        'data_files': openxr_data_files}

    # compile the `openhmd` extension
    EXT_MODULES.extend(
        [build_extension(
            "psychxr.drivers.openxr._openxr",
            **openxr_build_params)])
    PACKAGES.extend(openxr_build_params['packages'])


if BUILD_LIBOVR:
    print("Building `psychxr.drivers.libovr` extension module ...")

    # build parameters for LibOVR passed to the compiler and linker
    libovr_package_data = {
        'psychxr.drivers.libovr': PACKAGE_DATA}
    libovr_data_files = {'psychxr/drivers/libovr': DATA_FILES}
    libovr_build_params = {
        'libraries': ['LibOVR'],
        'library_dirs': [],
        'include_dirs': [],
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
    libovr_build_params['include_dirs'] = [
        fix_path(libovr_include_path),
        fix_path(libovr_include_path.joinpath('Extras'))]
    libovr_build_params['library_dirs'] = [
        fix_path(libovr_lib_path.joinpath(
            'Windows', 'x64', 'Release', 'VS2017'))]

    # compile the `libovr` extension
    EXT_MODULES.extend(
        [build_extension(
            "psychxr.drivers.libovr._libovr",
            **libovr_build_params)])
    PACKAGES.extend(libovr_build_params['packages'])

# build the OpenHMD driver
if BUILD_OPENHMD:
    print("building `psychxr.drivers.openhmd` extension module ...")

    ohmd_package_data = {
        'psychxr.drivers.openhmd': PACKAGE_DATA}
    ohmd_data_files = {
        'psychxr/drivers/openhmd': DATA_FILES}
    ohmd_build_params = {
        'libraries': LIBRARIES + ['hidapi', 'openhmd'],
        'library_dirs': LIBRARY_DIRS,
        'include_dirs': [  # header files for needed libraries
            fix_path('include/linmath'),
            fix_path('include/openhmd')],
        'packages': ['psychxr.drivers.openhmd'],
        'package_data': ohmd_package_data,
        'data_files': ohmd_data_files}

    # compile the `openhmd` extension
    EXT_MODULES.extend(
        [build_extension(
            "psychxr.drivers.openhmd._openhmd",
            **ohmd_build_params)])
    PACKAGES.extend(ohmd_build_params['packages'])


# ------------------------------------------------------------------------------
# Packaging and setup
#

setup_pars = {
    "name": "psychxr",
    "author": "Matthew D. Cutone, Laurie M. Wilcox",
    "author_email": "mcutone@opensciencetools.com",
    "maintainer": "Matthew D. Cutone",
    "maintainer_email": "mcutone@opensciencetools.com",
    "packages": PACKAGES,
    "url": "http://psychxr.org",
    "include_package_data": True,
    "version": PSYCHXR_VERSION_STRING,
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
    "ext_modules": EXT_MODULES,
    "install_requires": ["Cython>=0.29.3", "numpy>=1.13.3"],
    "requires": [],
    "cmdclass": {"build_ext": build_ext},
    "zip_safe": False
}

setup(**setup_pars)
