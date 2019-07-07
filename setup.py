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

# compiler related data
_include_dir_ = [numpy.get_include()]
_lib_dirs_ = []
_libraries_ = []
_build_ext_ = []

# build flags
_build_libovr_ = '0'  # Oculus PC SDK extensions

# SDK related data
_sdk_data_ = {}

# additional package data
PACKAGES = ['psychxr']
DATA_FILES = []

if platform.system() == 'Windows':
    # This makes sure the correct compiler is used, even if not explicitly
    # specified.
    os.environ["MSSdk"] = '1'
    os.environ["DISTUTILS_USE_SDK"] = '1'
    _libraries_.extend(['opengl32', 'User32'])  # required Windows libraries

    # check if which HMD were building libraries for
    _build_libovr_ = os.environ.get('PSYCHXR_BUILD_LIBOVR', '1')

    if _build_libovr_ == '1':  # build libovr extensions
        print("building libovr extension modules ...")
        _sdk_data_['libovr'] = {}
        _sdk_data_['libovr']['libs'] = ['LibOVR']
        env_sdk_path = os.environ.get('PSYCHXR_LIBOVR_SDK_PATH', None)
        if env_sdk_path is not None:
            _sdk_data_['libovr']['include'] = [
                os.path.join(env_sdk_path, 'LibOVR', 'Include'),
                os.path.join(env_sdk_path, 'LibOVR', 'Include', 'Extras')]
            _sdk_data_['libovr']['lib_dir'] = [
                os.path.join(
                    env_sdk_path, 'LibOVR', 'Lib', 'Windows', 'x64', 'Release',
                    'VS2017')]
            _sdk_data_['libovr']['libs'] = []
        else:
            # use a default
            _sdk_data_['libovr']['include'] = \
                [r"C:\OculusSDK\LibOVR\Include",
                 r"C:\OculusSDK\LibOVR\Include\Extras"]
            _sdk_data_['libovr']['lib_dir'] = \
                [r"C:\OculusSDK\LibOVR\Lib\Windows\x64\Release\VS2017"]

        # package data
        _sdk_data_['libovr']['packages'] = ['psychxr.libovr']
        _sdk_data_['libovr']['package_data'] = \
            {'psychxr.libovr': ['*.pxd', '*.pyx', '*.cpp']}
        _sdk_data_['libovr']['data_files'] = {'psychxr/libovr': ['*.pyd']}

else:
    raise Exception("Trying to install PsychXR on an unsupported operating "
                    "system. Exiting.")

# add configured extensions
ext_modules = []
if _build_libovr_ == '1':
    cythonize("psychxr/libovr/_libovr.pyx",
              include_path=_sdk_data_['libovr']['include'],
              compiler_directives = {'embedsignature': True,
                                     'language_level': 3})
    ext_modules.extend([
        Extension(
            "psychxr.libovr._libovr",
            ["psychxr/libovr/_libovr"+".cpp"],
            include_dirs=_include_dir_ + _sdk_data_['libovr']['include'],
            libraries=_libraries_ + _sdk_data_['libovr']['libs'],
            library_dirs=_lib_dirs_ + _sdk_data_['libovr']['lib_dir'],
            language="c++",
            extra_compile_args=[''])
    ])
    PACKAGES.extend(_sdk_data_['libovr']['packages'])

setup_pars = {
    "name" : "psychxr",
    "author" : "Matthew D. Cutone, Laurie M. Wilcox",
    "author_email" : "cutonem@yorku.ca",
    "maintainer": "Matthew D. Cutone",
    "maintainer_email": "cutonem@yorku.ca",
    "packages" : PACKAGES,
    "url": "https://github.com/mdcutone/psychxr",
    #"package_data": PACKAGE_DATA,
    "include_package_data": True,
    "version": "0.2.0",
    "license" : "MIT",
    "description":
        "Python extension library for interacting with eXtended Reality "
        "displays (HMDs), intended for research in neuroscience and "
        "psychology.",
    "long_description": "",
    "classifiers" : [
        'Development Status :: 4 - Beta',
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'Operating System :: Microsoft :: Windows :: Windows 8.1',
        'Operating System :: Microsoft :: Windows :: Windows 7',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Cython',
        'Intended Audience :: Science/Research'],
    "ext_modules": ext_modules,
    #"data_files": DATA_FILES,
    "install_requires" : ["Cython>=0.29.3"],
    "requires" : [],
    "cmdclass" : {"build_ext": build_ext}}

setup(**setup_pars)


