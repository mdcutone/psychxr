# distutils: language=c++
#  =============================================================================
#  _openhmd.pyx - Python Interface Module for OpenHMD
#  =============================================================================
#
#  Copyright 2021 Matthew Cutone <mcutone@opensciencetools.com> and Laurie M. Wilcox
#  <lmwilcox(a)yorku.ca>; The Centre For Vision Research, York University,
#  Toronto, Canada
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
"""This extension module exposes the LibOVR API to Python using the FOSS OpenHMD
driver interface.

OpenHMD is a project aimed at providing free and open source drivers for many
commercial HMDs. The driver interface is portable and cross-platform, however
feature support varies depending on the HMD used.

"""
# ------------------------------------------------------------------------------
# Module information
#
__author__ = "Matthew D. Cutone"
__credits__ = ["Laurie M. Wilcox"]
__copyright__ = "Copyright 2019 Matthew D. Cutone"
__license__ = "MIT"
__version__ = "0.2.4"
__status__ = "Stable"
__maintainer__ = "Matthew D. Cutone"
__email__ = "mcutone@opensciencetools.com"

# ------------------------------------------------------------------------------
# Module information
#

__all__ = [
    "OHMD_STR_SIZE",
    "OHMD_S_OK",
    "OHMD_S_UNKNOWN_ERROR", 
    "OHMD_S_INVALID_PARAMETER",
    "OHMD_S_UNSUPPORTED",
    "OHMD_S_INVALID_OPERATION",
    "OHMD_S_USER_RESERVED",
    "OHMD_VENDOR",
    "OHMD_PRODUCT",
    "OHMD_PATH",
    "OHMD_GLSL_DISTORTION_VERT_SRC",
    "OHMD_GLSL_DISTORTION_FRAG_SRC",
    "OHMD_GLSL_330_DISTORTION_VERT_SRC",
    "OHMD_GLSL_330_DISTORTION_FRAG_SRC",
    "OHMD_GLSL_ES_DISTORTION_VERT_SRC",
    "OHMD_GLSL_ES_DISTORTION_FRAG_SRC",
    "OHMD_GENERIC",
    "OHMD_TRIGGER",
    "OHMD_TRIGGER_CLICK",
    "OHMD_SQUEEZE",
    "OHMD_MENU",
    "OHMD_HOME",
    "OHMD_ANALOG_X",
    "OHMD_ANALOG_Y",
    "OHMD_ANALOG_PRESS",
    "OHMD_BUTTON_A",
    "OHMD_BUTTON_B",
    "OHMD_BUTTON_X",
    "OHMD_BUTTON_Y",
    "OHMD_VOLUME_PLUS",
    "OHMD_VOLUME_MINUS",
    "OHMD_MIC_MUTE",
    "OHMD_ROTATION_QUAT",
    "OHMD_LEFT_EYE_GL_MODELVIEW_MATRIX",
    "OHMD_RIGHT_EYE_GL_MODELVIEW_MATRIX",
    "OHMD_LEFT_EYE_GL_PROJECTION_MATRIX",
    "OHMD_RIGHT_EYE_GL_PROJECTION_MATRIX",
    "OHMD_POSITION_VECTOR",
    "OHMD_SCREEN_HORIZONTAL_SIZE",
    "OHMD_SCREEN_VERTICAL_SIZE",
    "OHMD_LENS_HORIZONTAL_SEPARATION",
    "OHMD_LENS_VERTICAL_POSITION",
    "OHMD_LEFT_EYE_FOV",
    "OHMD_LEFT_EYE_ASPECT_RATIO",
    "OHMD_RIGHT_EYE_FOV",
    "OHMD_RIGHT_EYE_ASPECT_RATIO",
    "OHMD_EYE_IPD",
    "OHMD_PROJECTION_ZFAR",
    "OHMD_PROJECTION_ZNEAR",
    "OHMD_DISTORTION_K",
    "OHMD_EXTERNAL_SENSOR_FUSION",
    "OHMD_UNIVERSAL_DISTORTION_K",
    "OHMD_UNIVERSAL_ABERRATION_K",
    "OHMD_CONTROLS_STATE",
    "OHMD_SCREEN_HORIZONTAL_RESOLUTION",
    "OHMD_SCREEN_VERTICAL_RESOLUTION",
    "OHMD_DEVICE_CLASS",
    "OHMD_DEVICE_FLAGS",
    "OHMD_CONTROL_COUNT",
    "OHMD_CONTROLS_HINTS",
    "OHMD_CONTROLS_TYPES",
    "OHMD_DRIVER_DATA",
    "OHMD_DRIVER_PROPERTIES",
    "OHMD_IDS_AUTOMATIC_UPDATE",
    "OHMD_DEVICE_CLASS_HMD",
    "OHMD_DEVICE_CLASS_CONTROLLER",
    "OHMD_DEVICE_CLASS_GENERIC_TRACKER",
    "OHMD_DEVICE_FLAGS_POSITIONAL_TRACKING",
    "OHMD_DEVICE_FLAGS_ROTATIONAL_TRACKING",
    "OHMD_DEVICE_FLAGS_LEFT_CONTROLLER",
    "OHMD_DEVICE_FLAGS_RIGHT_CONTROLLER",
    "OpenHMDPose",
    "OpenHMDDeviceInfo",
    "create",
    "destroy"
]

cimport numpy as np
import numpy as np
from . cimport openhmd_capi as ohmd

# ------------------------------------------------------------------------------
# Constants

# max string length including termination
OHMD_STR_SIZE = ohmd.OHMD_STR_SIZE

# status codes, used for all functions that return an error
OHMD_S_OK = ohmd.OHMD_S_OK
OHMD_S_UNKNOWN_ERROR = ohmd.OHMD_S_UNKNOWN_ERROR
OHMD_S_INVALID_PARAMETER = ohmd.OHMD_S_INVALID_PARAMETER
OHMD_S_UNSUPPORTED = ohmd.OHMD_S_UNSUPPORTED
OHMD_S_INVALID_OPERATION = ohmd.OHMD_S_INVALID_OPERATION
OHMD_S_USER_RESERVED = ohmd.OHMD_S_USER_RESERVED

# string information types, max length is `OHMD_STR_SIZE`
OHMD_VENDOR = ohmd.OHMD_VENDOR
OHMD_PRODUCT = ohmd.OHMD_PRODUCT
OHMD_PATH = ohmd.OHMD_PATH

# long string descriptions
OHMD_GLSL_DISTORTION_VERT_SRC = ohmd.OHMD_GLSL_DISTORTION_VERT_SRC
OHMD_GLSL_DISTORTION_FRAG_SRC = ohmd.OHMD_GLSL_DISTORTION_FRAG_SRC
OHMD_GLSL_330_DISTORTION_VERT_SRC = ohmd.OHMD_GLSL_330_DISTORTION_VERT_SRC
OHMD_GLSL_330_DISTORTION_FRAG_SRC = ohmd.OHMD_GLSL_330_DISTORTION_FRAG_SRC
OHMD_GLSL_ES_DISTORTION_VERT_SRC = ohmd.OHMD_GLSL_ES_DISTORTION_VERT_SRC
OHMD_GLSL_ES_DISTORTION_FRAG_SRC = ohmd.OHMD_GLSL_ES_DISTORTION_FRAG_SRC

# standard controls
OHMD_GENERIC = ohmd.OHMD_GENERIC
OHMD_TRIGGER = ohmd.OHMD_TRIGGER
OHMD_TRIGGER_CLICK = ohmd.OHMD_TRIGGER_CLICK
OHMD_SQUEEZE = ohmd.OHMD_SQUEEZE
OHMD_MENU = ohmd.OHMD_MENU
OHMD_HOME = ohmd.OHMD_HOME
OHMD_ANALOG_X = ohmd.OHMD_ANALOG_X
OHMD_ANALOG_Y = ohmd.OHMD_ANALOG_Y
OHMD_ANALOG_PRESS = ohmd.OHMD_ANALOG_PRESS
OHMD_BUTTON_A = ohmd.OHMD_BUTTON_A
OHMD_BUTTON_B = ohmd.OHMD_BUTTON_B
OHMD_BUTTON_X = ohmd.OHMD_BUTTON_X
OHMD_BUTTON_Y = ohmd.OHMD_BUTTON_Y
OHMD_VOLUME_PLUS = ohmd.OHMD_VOLUME_PLUS
OHMD_VOLUME_MINUS = ohmd.OHMD_VOLUME_MINUS
OHMD_MIC_MUTE = ohmd.OHMD_MIC_MUTE

# float values
OHMD_ROTATION_QUAT = ohmd.OHMD_ROTATION_QUAT
OHMD_LEFT_EYE_GL_MODELVIEW_MATRIX = ohmd.OHMD_LEFT_EYE_GL_MODELVIEW_MATRIX
OHMD_RIGHT_EYE_GL_MODELVIEW_MATRIX = ohmd.OHMD_RIGHT_EYE_GL_MODELVIEW_MATRIX
OHMD_LEFT_EYE_GL_PROJECTION_MATRIX = ohmd.OHMD_LEFT_EYE_GL_PROJECTION_MATRIX
OHMD_RIGHT_EYE_GL_PROJECTION_MATRIX = ohmd.OHMD_RIGHT_EYE_GL_PROJECTION_MATRIX
OHMD_POSITION_VECTOR = ohmd.OHMD_POSITION_VECTOR
OHMD_SCREEN_HORIZONTAL_SIZE = ohmd.OHMD_SCREEN_HORIZONTAL_SIZE
OHMD_SCREEN_VERTICAL_SIZE = ohmd.OHMD_SCREEN_VERTICAL_SIZE
OHMD_LENS_HORIZONTAL_SEPARATION = ohmd.OHMD_LENS_HORIZONTAL_SEPARATION
OHMD_LENS_VERTICAL_POSITION = ohmd.OHMD_LENS_VERTICAL_POSITION
OHMD_LEFT_EYE_FOV = ohmd.OHMD_LEFT_EYE_FOV
OHMD_LEFT_EYE_ASPECT_RATIO = ohmd.OHMD_LEFT_EYE_ASPECT_RATIO
OHMD_RIGHT_EYE_FOV = ohmd.OHMD_RIGHT_EYE_FOV
OHMD_RIGHT_EYE_ASPECT_RATIO = ohmd.OHMD_RIGHT_EYE_ASPECT_RATIO
OHMD_EYE_IPD = ohmd.OHMD_EYE_IPD
OHMD_PROJECTION_ZFAR = ohmd.OHMD_PROJECTION_ZFAR
OHMD_PROJECTION_ZNEAR = ohmd.OHMD_PROJECTION_ZNEAR
OHMD_DISTORTION_K = ohmd.OHMD_DISTORTION_K
OHMD_EXTERNAL_SENSOR_FUSION = ohmd.OHMD_EXTERNAL_SENSOR_FUSION
OHMD_UNIVERSAL_DISTORTION_K = ohmd.OHMD_UNIVERSAL_DISTORTION_K
OHMD_UNIVERSAL_ABERRATION_K = ohmd.OHMD_UNIVERSAL_ABERRATION_K
OHMD_CONTROLS_STATE = ohmd.OHMD_CONTROLS_STATE

# int values
OHMD_SCREEN_HORIZONTAL_RESOLUTION = ohmd.OHMD_SCREEN_HORIZONTAL_RESOLUTION
OHMD_SCREEN_VERTICAL_RESOLUTION = ohmd.OHMD_SCREEN_VERTICAL_RESOLUTION
OHMD_DEVICE_CLASS = ohmd.OHMD_DEVICE_CLASS
OHMD_DEVICE_FLAGS = ohmd.OHMD_DEVICE_FLAGS
OHMD_CONTROL_COUNT = ohmd.OHMD_CONTROL_COUNT
OHMD_CONTROLS_HINTS = ohmd.OHMD_CONTROLS_HINTS
OHMD_CONTROLS_TYPES = ohmd.OHMD_CONTROLS_TYPES

OHMD_DRIVER_DATA = ohmd.OHMD_DRIVER_DATA
OHMD_DRIVER_PROPERTIES = ohmd.OHMD_DRIVER_PROPERTIES

OHMD_IDS_AUTOMATIC_UPDATE = ohmd.OHMD_IDS_AUTOMATIC_UPDATE

OHMD_DEVICE_CLASS_HMD = ohmd.OHMD_DEVICE_CLASS_HMD
OHMD_DEVICE_CLASS_CONTROLLER = ohmd.OHMD_DEVICE_CLASS_CONTROLLER
OHMD_DEVICE_CLASS_GENERIC_TRACKER = ohmd.OHMD_DEVICE_CLASS_GENERIC_TRACKER

OHMD_DEVICE_FLAGS_POSITIONAL_TRACKING = ohmd.OHMD_DEVICE_FLAGS_POSITIONAL_TRACKING
OHMD_DEVICE_FLAGS_ROTATIONAL_TRACKING = ohmd.OHMD_DEVICE_FLAGS_ROTATIONAL_TRACKING
OHMD_DEVICE_FLAGS_LEFT_CONTROLLER = ohmd.OHMD_DEVICE_FLAGS_LEFT_CONTROLLER
OHMD_DEVICE_FLAGS_RIGHT_CONTROLLER = ohmd.OHMD_DEVICE_FLAGS_RIGHT_CONTROLLER


# --------------------------------------
# Initialize module
#
cdef ohmd.ohmd_context* _ctx = NULL


# ------------------------------------------------------------------------------
# C-API for OpenHMD
#

def getError():
    """Get the error message."""
    cdef const char* err_msg = ohmd.ohmd_ctx_get_error(_ctx)

    # return as a python string


cdef class OpenHMDPose(object):
    """Class representing a 3D pose in space.

    Parameters
    ----------
    pos : ArrayLike
        Position vector (x, y, z).
    ori : ArrayLike
        Orientation quaternion (x, y, z, w), where x, y, z are real and w is
        imaginary.

    See Also
    --------
    psychxr.drivers.libovr.LibOVRPose

    """
    cdef np.ndarray _pos
    cdef np.ndarray _ori

    def __init__(self, pos=(0., 0., 0.), ori=(0., 0., 0., 1.)):
        self._pos[:] = pos
        self._ori[:] = ori

    def __cinit__(self, *args, **kwargs):
        # define the storage arrays here
        self._pos = np.empty((3,), dtype=np.float32)
        self._ori = np.empty((4,), dtype=np.float32)

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, object value):
        self._pos[:] = value

    @property
    def ori(self):
        return self._ori

    @ori.setter
    def ori(self, object value):
        self._ori[:] = value


cdef class OpenHMDDeviceInfo(object):
    """Device information class."""
    cdef np.ndarray _fov
    cdef np.ndarray _aspect


def getDeviceInfo():
    """Get device information."""
    pass


def create():
    """Create a new OpenHMD context/session.

    At this time only a single context can be created per session. You must call
    this function prior to using any other API calls.

    """
    global _ctx
    _ctx = ohmd.ohmd_ctx_create()

    if _ctx is not NULL:
        return 0

    return 1


def destroy():
    """Destroy the current context/session."""
    global _ctx
    ohmd.ohmd_ctx_destroy(_ctx)
    _ctx = NULL