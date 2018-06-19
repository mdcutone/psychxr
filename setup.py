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
installed prior to building PsychXR from source.

First, you must configure the installer to build extensions for a target HMD
API. As of now, only the Oculus PC SDK is supported, therefore only one
configuration command is available. Command arguments are used to specify API
specific build options. You must indicate where the compiler can find header
and library files. See the example command below:

    python setup.py libovr --include-dir=C:\OculusSDK\LibOVR\Include
        --lib-dir=C:\OculusSDK\...\VS2015

After running the above command, build the library by calling:

    python setup.py build

NOTE: On Windows, you need to use the "Visual C++ 2015 (or 2017) Native Build
Tools Command Prompt" when executing the above commands. Make sure your
LibOVR.lib file matches the version of Visual C++ you are using!

"""
import os
import json
import platform
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize, build_ext

# compiler related data
INCLUDE_DIRS = []
LIB_DIRS = []
LIBRARIES = []
BUILD_EXT = []

# packaging data
PACKAGES = ['psychxr']
PACKAGE_DATA = {'psychxr': ["*.pxd"]}

if platform.system() == 'Windows':
    # This makes sure the correct compiler is used, even if not explicitly
    # specified.
    os.environ["MSSdk"] = "1"
    os.environ["DISTUTILS_USE_SDK"] = "1"
    LIBRARIES.extend(['opengl32', 'User32'])  # required Windows libraries
else:
    raise Exception("Trying to install PsychXR on an unsupported operating "
                    "system. Exiting.")

class SetupLibOvrSdkCommand(build_ext):
    """Command class for setting the Oculus PC SDK directory."""

    description = 'specify the Oculus SDK directory'
    user_options = [
        ('include-dir=', None, 'path to LibOVR include directory'),
        ('lib-dir=', None, 'path to LibOVR library directory')]

    def initialize_options(self):
        # Default paths
        self.include_dir = r'C:\\OculusSDK\\LibOVR\\Include'  # default
        self.include_extras_dir = os.path.join(self.include_dir, 'Extras')
        self.lib_dir = \
            r'C:\\OculusSDK\\LibOVR\\Lib\\Windows\\x64\\Release\\VS2015'

    def finalize_options(self):
        if not os.path.exists(self.include_dir):
            raise FileNotFoundError(
                r"Include directory '{}' does not exist or has moved.".format(
                    self.include_dir))

        if not os.path.exists(self.include_extras_dir):
            raise FileNotFoundError(
                r"Extras directory '{}' does not exist or has moved.".format(
                    self.include_extras_dir))

        if not os.path.exists(self.lib_dir):
            raise FileNotFoundError(
                r"Library directory '{}' does not exist or has moved.".format(
                    self.lib_dir))

    def run(self):
        """Write an install configuration file.

        """
        # open or create a configuration file
        if os.path.exists("build_config.json"):
            with open("build_config.json", 'r') as fp:
                install_cfg = json.loads(fp.read())
        else:
            install_cfg = {}

        install_cfg['libovr'] = {}
        install_cfg['libovr']['include'] = \
            [self.include_dir, self.include_extras_dir, "psychxr/ovr/"]
        install_cfg['libovr']['lib_dir'] = [self.lib_dir]
        install_cfg['libovr']['libs'] = ['LibOVR']
        install_cfg['libovr']['packages'] = ['psychxr.ovr']
        install_cfg['libovr']['package_data'] = \
            {'psychxr.ovr': ['*.pxd', '*.pyd']}

        # write the installation configuration file
        with open("build_config.json", 'w') as write_file:
            json.dump(install_cfg,
                      write_file,
                      sort_keys=True,
                      indent=4,
                      separators=(',', ': '))

        # extensions to cythonize
        cythonize("psychxr/ovr/capi.pyx",
                  include_path=install_cfg['libovr']['include'])

        cythonize("psychxr/ovr/math.pyx",
                  include_path=install_cfg['libovr']['include'])


# load the build configuration file
with open("build_config.json", 'r') as fp:
    install_cfg = json.loads(fp.read())
    if not install_cfg.keys():
        print("No extension module build configurations found!")

# add configured extensions
ext_modules = []
if 'libovr' in install_cfg.keys():
    ext_modules.extend([
        Extension(
            "psychxr.ovr.capi",
            ["psychxr/ovr/capi.cpp"],
            include_dirs=INCLUDE_DIRS + install_cfg['libovr']['include'],
            libraries=LIBRARIES + install_cfg['libovr']['libs'],
            library_dirs=LIB_DIRS + install_cfg['libovr']['lib_dir'],
            language="c++",
            extra_compile_args=['']),
        Extension(
            "psychxr.ovr.math",
            ["psychxr/ovr/math.cpp"],
            include_dirs=INCLUDE_DIRS + install_cfg['libovr']['include'],
            libraries=LIBRARIES + install_cfg['libovr']['libs'],
            library_dirs=LIB_DIRS + install_cfg['libovr']['lib_dir'],
            language="c++",
            extra_compile_args=[''])
    ])
    PACKAGES.extend(install_cfg['libovr']['packages'])
    PACKAGE_DATA.update(install_cfg['libovr']['package_data'])

setup_pars = {
    "name" : "psychxr",
    "author" : "Matthew D. Cutone",
    "author_email" : "cutonem(at)yorku.ca",
    "maintainer": "Matthew D. Cutone",
    "maintainer_email": "cutonem(at)yorku.ca",
    "packages" : PACKAGES,
    "package_data": PACKAGE_DATA,
    "include_package_data": True,
    "version": "0.1.2",
    "license" : "MIT",
    "description":
        "Python extension library for interacting with eXtended Reality "
        "displays (HMDs), intended for research in neuroscience and "
        "psychology.",
    "long_description": "",
    "classifiers" : [
        'Development Status :: 3 - Alpha',
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'Operating System :: Microsoft :: Windows :: Windows 8.1',
        'Operating System :: Microsoft :: Windows :: Windows 7',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Cython',
        'Intended Audience :: Science/Research'],
    "ext_modules": ext_modules,
    "requires" : ["Cython", "PyOpenGL"],
    'py_modules' : [],
    "cmdclass" : {"libovr": SetupLibOvrSdkCommand,
                  "build_ext": build_ext}}

setup(**setup_pars)


