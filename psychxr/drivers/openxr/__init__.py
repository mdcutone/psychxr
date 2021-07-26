"""Interface and tools for HMDs using OpenXR.

"""
# ------------------
# Module information
# ------------------
#
__author__ = "Matthew D. Cutone"
__credits__ = ["Matthew D. Cutone"]
__copyright__ = "Copyright 2021 Matthew D. Cutone"
__license__ = "MIT"
__version__ = '0.2.4rc2'
__status__ = "Stable"
__maintainer__ = "Matthew D. Cutone"
__email__ = "mcutone@opensciencetools.org"

# -------
# Imports
# -------
#

# add library path
import os
import sys
import platform

# Add library directory to PATH before loading the pyd, points to the OpenXR
# loader.
if platform.system() == "Windows":
    here, _ = os.path.split(sys.modules[__name__].__file__)
    os.environ["PATH"] += r";" + os.path.join(here, 'lib', 'win', 'x64')

# load exported data from the _openxr extension module into the namespace
from ._openxr import *