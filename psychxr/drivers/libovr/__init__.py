"""Interface and tools for the Oculus Rift head-mounted display (HMD) using the
official Oculus PC SDK.

The Oculus PC SDK is Copyright (c) Facebook Technologies, LLC and its
affiliates. All rights reserved.

"""
# ------------------
# Module information
# ------------------
#
__author__ = "Matthew D. Cutone"
__credits__ = ["Laurie M. Wilcox"]
__copyright__ = "Copyright 2021 Matthew D. Cutone"
__license__ = "MIT"
__version__ = '0.2.4rc3'
__status__ = "Stable"
__maintainer__ = "Matthew D. Cutone"
__email__ = "mcutone@opensciencetools.org"

# -------
# Imports
# -------
#
# load exported data from the _libovr extension module into the namespace
from ._libovr import *
