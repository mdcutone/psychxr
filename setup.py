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
