# distutils: language=c++
#  =============================================================================
#  _openhmd.pyx - Python Interface Module for OpenHMD
#  =============================================================================
#
#  Copyright 2021 Matthew Cutone <mcutone@opensciencetools.com> and Laurie M.
#  Wilcox <lmwilcox(a)yorku.ca>; The Centre For Vision Research, York
#  University, Toronto, Canada
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
commercial HMDs and other VR related devices. The driver interface is portable
and cross-platform, however feature support varies depending on the HMD used.

Unlike `LibOVR`, OpenHMD does not come with a compositor. Therefore, the user
must implement their own system to present scenes to the display. You can also
use OpenHMD in conjunction with other drivers. For instance, you can use OpenHMD
to add controllers and motion trackers to your project not supported by other
drivers.

"""
# ------------------------------------------------------------------------------
# Module information
#
__author__ = "Matthew D. Cutone"
__credits__ = []
__copyright__ = "Copyright 2021 Matthew D. Cutone"
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
    "OHMDPose",
    "OHMDDeviceInfo",
    "OHMDDisplayInfo",
    "OHMDControllerInfo",
    "success",
    "failure",
    "getVersion",
    "create",
    "destroy",
    "probe",
    "getDevices",
    "getDisplayInfo",
    "openDevice",
    "getError",
    "update",
    "getString",
    "getListString",
    "getListInt"
]

cimport numpy as np
import numpy as np
from . cimport openhmd as ohmd
from libc.time cimport clock, clock_t
from cpython.mem cimport PyMem_Malloc, PyMem_Free


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

# custom enums, used for indexing displays and controllers
OHMD_EYE_LEFT = ohmd.OHMD_EYE_LEFT
OHMD_EYE_RIGHT = ohmd.OHMD_EYE_RIGHT
OHMD_EYE_COUNT = ohmd.OHMD_EYE_COUNT
OHMD_HAND_LEFT = ohmd.OHMD_HAND_LEFT
OHMD_HAND_RIGHT = ohmd.OHMD_HAND_RIGHT
OHMD_HAND_COUNT = ohmd.OHMD_HAND_COUNT


# ------------------------------------------------------------------------------
# OpenHMD specific exceptions
#
class OHMDNoContextError(RuntimeError):
    """Raised if trying to perform an action without having a valid context."""
    pass


class OHMDContextNotProbedError(RuntimeError):
    """Raised if trying to perform an action that requires the context be probed
    first."""
    pass


class OHMDWrongDeviceClassError(RuntimeError):
    """Raised if performing an action on a device not belonging to the required
    class."""
    pass


class OHMDDeviceNotOpenError(RuntimeError):
    """Raised if performing an action on a device that is closed but needs to
    be opened."""
    pass

# ------------------------------------------------------------------------------
# Initialize module
#
cdef ohmd.ohmd_context* _ctx = NULL  # handle for the context

# found devices end up here
cdef Py_ssize_t _deviceCount = 0
cdef tuple _deviceInfoList = ()  # stores device info instances
cdef int _contextProbed = 0

# is auto-update enabled?
cdef int _automatic_update = 0

# ------------------------------------------------------------------------------
# Constants and helper functions
#
cdef double NS_TO_SECS = 1e-9


cdef double cpu_time():
    """Get the current CPU time (ns). Used for timestamping of events.
    
    Returns
    -------
    double
        CPU time in seconds.
    
    """
    cdef clock_t time_ns = clock()
    return <double>time_ns * NS_TO_SECS


cdef void clear_device_info():
    """Clear internal store of device information descriptors.
    """
    global _deviceCount
    global _deviceInfoList

    _deviceCount = 0
    _deviceInfoList = ()


# ------------------------------------------------------------------------------
# Python API for OpenHMD
#

cdef class OHMDPose(object):
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


cdef class OHMDDeviceInfo(object):
    """Information class for OpenHMD devices.

    OpenHMD supports a wide range of devices (to varying degrees), including
    HMDs, controller, and trackers. This class stores data about a given device,
    providing the user details (e.g., type of device, vendor/product name,
    capabilities like tracking, etc.) about it.

    Instances of this class are returned by calling :func:`getDevices`. You can
    then pass instances to :func:`openDevice` to open them.

    """
    cdef ohmd.ohmd_device* c_device
    cdef ohmd.ohmdDeviceInfo c_device_info

    def __cinit__(self):
        self.c_device = NULL

    def __del__(self):
        # try to close the device if we lost all references to this object
        if self.c_device is not NULL:
            ohmd.ohmd_close_device(self.c_device)
            self.c_device = NULL

    @property
    def isOpen(self):
        """Is the device open? (`bool`)."""
        return self.c_device is not NULL

    @property
    def vendorName(self):
        """Device vendor name (`str`)."""
        return self.c_device_info.vendorName.decode('utf-8')

    @property
    def manufacturer(self):
        """Device manufacturer name, alias of `vendorName` (`str`)"""
        return self.c_device_info.vendorName.decode('utf-8')

    @property
    def productName(self):
        """Device product name (`str`)."""
        return self.c_device_info.productName.decode('utf-8')

    @property
    def deviceIdx(self):
        """Enumerated index of the device (`int`)."""
        return self.c_device_info.deviceIdx

    @property
    def deviceClass(self):
        """Device class identifier (`int`)."""
        return <int>self.c_device_info.deviceClass

    @property
    def isHMD(self):
        """``True`` if this device is an HMD (`bool`)."""
        return self.c_device_info.deviceClass == ohmd.OHMD_DEVICE_CLASS_HMD

    @property
    def isController(self):
        """``True`` if this device is a controller (`bool`)."""
        return self.c_device_info.deviceClass == ohmd.OHMD_DEVICE_CLASS_CONTROLLER

    @property
    def isTracker(self):
        """``True`` if this device is a generic tracker (`bool`)."""
        return self.c_device_info.deviceClass == \
               ohmd.OHMD_DEVICE_CLASS_GENERIC_TRACKER

    @property
    def deviceFlags(self):
        """Device flags (`int`).

        Examples
        --------
        Check if a device has positional and orientation tracking support::

            hmdInfo = getHmdInfo()
            flags = OHMD_DEVICE_FLAGS_POSITIONAL_TRACKING |
                OHMD_DEVICE_FLAGS_ROTATIONAL_TRACKING

            hasFullTracking = (self.c_data.deviceFlags & flags) == flags

        """
        return <int>self.c_device_info.deviceFlags

    @property
    def hasOrientationTracking(self):
        """``True`` if capable of tracking orientation (`bool`)."""
        return (self.c_device_info.deviceFlags &
                ohmd.OHMD_DEVICE_FLAGS_ROTATIONAL_TRACKING) == \
               ohmd.OHMD_DEVICE_FLAGS_ROTATIONAL_TRACKING

    @property
    def hasPositionTracking(self):
        """``True`` if capable of tracking position (`bool`)."""
        return (self.c_device_info.deviceFlags &
                ohmd.OHMD_DEVICE_FLAGS_POSITIONAL_TRACKING) == \
               ohmd.OHMD_DEVICE_FLAGS_POSITIONAL_TRACKING

    @property
    def isDebugDevice(self):
        """``True`` if a virtual debug (null or dummy) device (`bool`)."""
        return (self.c_device_info.deviceFlags &
                ohmd.OHMD_DEVICE_FLAGS_NULL_DEVICE) == \
               ohmd.OHMD_DEVICE_FLAGS_NULL_DEVICE


cdef class OHMDDisplayInfo(object):
    """Class for information about a display device (HMD).

    This class provides additional information about hardware belonging to the
    HMD device class (i.e. ``OHMD_DEVICE_CLASS_HMD``).

    """
    cdef ohmd.ohmdDisplayInfo c_data

    @property
    def resolution(self):
        """Horizontal and vertical resolution of the display (`ndarray`)."""
        return np.asarray(
            (self.c_data.resolution[0],
             self.c_data.resolution[1]), dtype=int)

    @property
    def screenSize(self):
        """Horizontal and vertical size of the display in meters (`ndarray`)."""
        return np.asarray(
            (self.c_data.screenSize[0],
             self.c_data.screenSize[1]), dtype=np.float32)

    @property
    def aspect(self):
        """Physical display aspect ratios for the left and right eye (`ndarray`).
        """
        return np.asarray(
            (self.c_data.aspect[0],
             self.c_data.aspect[1]), dtype=np.float32)

    @property
    def eyeFov(self):
        """Physical field of view for each eye in degrees (`ndarray`).
        """
        return np.asarray(
            (self.c_data.fov[0],
             self.c_data.fov[1]), dtype=np.float32)

    @property
    def ipd(self):
        """Interpupilary distance in meters reported by the device (`float`)."""
        return self.c_data.ipd


cdef class OHMDControllerInfo(object):
    """Class for information about a controller.

    """
    cdef ohmd.ohmdControllerInfo c_data

    @property
    def controlCount(self):
        """Number of analog and digital controls on the device (`int`)."""
        return self.c_data.controlCount


def success(int result):
    """Check if a function returned successfully.

    Parameters
    ----------
    result : int
        Return value of a function.

    Returns
    -------
    bool
        `True` if the return code indicates success.

    """
    return ohmd.OHMD_S_OK == result


def failure(int result):
    """Check if a function returned an error.

    Parameters
    ----------
    result : int
        Return value of a function.

    Returns
    -------
    bool
        `True` if the return code indicates failure.

    """
    return ohmd.OHMD_S_OK > result


def getVersion():
    """Get the version of the `OpenHMD` library presently loaded.

    Returns
    -------
    tuple
        Major (`int`), minor (`int`), and patch (`int`) version.

    Examples
    --------
    Get the version for the OpenHMD library::

        major_version, minor_version, patch_version = getVersion()

    """
    cdef:
        int major_version = 0
        int minor_version = 0
        int patch_version = 0

    ohmd.ohmd_get_version(&major_version, &minor_version, &patch_version)

    return major_version, minor_version, patch_version


def create():
    """Create a new OpenHMD context/session.

    Calling this will create a context, enumerate devices and open them. Only
    the first connected HMD will be opened, all other device types that are
    controllers or external trackers will be opened.

    At this time only a single context can be created per session. You must call
    this function prior to using any other API calls other than
    :func:`destroy()`.

    Returns
    -------
    int
        Returns value of ``OHMD_S_OK`` if a context was created successfully,
        else ``OHMD_S_USER_RESERVED``. You can check the result using the
        :func:`success()` and :func:`failure()` functions.

    Examples
    --------
    Create a new OpenHMD context, starting the session::

        import sys
        import psychxr.drivers.openhmd as openhmd

        result = openhmd.create()  # create a context
        if failure(result):  # if we failed to create a context, exit with error
            sys.exit(1)

    """
    global _ctx

    if _ctx is not NULL:  # check if a context is already opened
        return ohmd.OHMD_S_USER_RESERVED

    _ctx = ohmd.ohmd_ctx_create()  # create the context

    if _ctx is NULL:  # check if context failed to open
        return ohmd.OHMD_S_USER_RESERVED

    return ohmd.OHMD_S_OK


def probe():
    """Probe for devices.

    Probes for and enumerates supported devices attached to the system. After
    calling this function you may use :func:`getDevices()` which will return
    a list of descriptors representing found devices.

    Returns
    -------
    int
        Number of devices found on the system.

    """
    global _ctx
    global _deviceCount
    global _deviceInfoList
    global _contextProbed

    if _ctx is NULL:
        raise OHMDNoContextError()

    # probe for devices
    cdef int probe_device_count = ohmd.ohmd_ctx_probe(_ctx)
    if not probe_device_count:  # no devices found, just return
        return probe_device_count

    _contextProbed = 1

    # inter over devices, open them and get information
    cdef int device_idx = 0
    cdef list devices_list = []
    cdef OHMDDeviceInfo desc  # Python descriptor class for the device
    cdef ohmd.ohmdDeviceInfo* device_info
    for device_idx in range(probe_device_count):
        # properties common to all device types
        desc = OHMDDeviceInfo()
        device_info = &(<OHMDDeviceInfo>desc).c_device_info
        device_info.deviceIdx = device_idx
        device_info.productName = ohmd.ohmd_list_gets(  # product name
            _ctx,
            device_idx,
            ohmd.OHMD_PRODUCT)
        device_info.vendorName = ohmd.ohmd_list_gets(  # vendor name
            _ctx,
            device_idx,
            ohmd.OHMD_VENDOR)
        ohmd.ohmd_list_geti(  # device class
            _ctx,
            device_idx,
            ohmd.OHMD_DEVICE_CLASS,
            &device_info.deviceClass)
        ohmd.ohmd_list_geti(  # device flags
            _ctx,
            device_idx,
            ohmd.OHMD_DEVICE_FLAGS,
            &device_info.deviceFlags)

        # create a python wrapper around the descriptor
        devices_list.append(desc)  # add to list

    _deviceInfoList = tuple(devices_list)

    return probe_device_count


def destroy():
    """Destroy the current context/session."""
    global _ctx
    global _contextProbed

    if _ctx is NULL:  # nop if no context created
        return

    clear_device_info()

    ohmd.ohmd_ctx_destroy(_ctx)
    _ctx = NULL
    _contextProbed = 0


def getError():
    """Get the last error as a human readable string.

    Call this after a function returns a code indicating an error to get a
    string describing what went wrong.

    Returns
    -------
    str
        Human-readable string describing the cause of the last error.

    """
    cdef const char* err_msg

    if _ctx is NULL:
        raise OHMDNoContextError()

    err_msg = ohmd.ohmd_ctx_get_error(_ctx)

    return err_msg.decode('utf-8')


def getDevices(int deviceClass=0):
    """Get devices found during the last call to :func:`probe`.

    Parameters
    ----------
    deviceClass : int
        Only get devices belonging to the specified device class. Values can be
        one of ``OHMD_DEVICE_CLASS_CONTROLLER``, ``OHMD_DEVICE_CLASS_HMD``, or
        ``OHMD_OHMD_DEVICE_CLASS_GENERIC_TRACKER``. Set to zero to get all
        devices (the default).

    Returns
    -------
    list
        List of :class:`OpenHMDDeviceInfo` descriptors.

    Examples
    --------
    Get all HMDs found on the system::

        if probe():  # >0 if devices have been found
            all_devices = getDevices()
            only_hmds = [dev for dev in all_devices if dev.isHMD]

    The same as above but using the `deviceClass` argument::

        # assume we probed already
        only_hmds = getDevices(deviceClass=OHMD_DEVICE_CLASS_HMD)

    """
    global _deviceCount
    global _deviceInfoList
    cdef tuple to_return

    if deviceClass == 0:
        to_return = _deviceInfoList
        return to_return

    cdef list device_list = []
    cdef OHMDDeviceInfo device_info
    for device_info in _deviceInfoList:
        if deviceClass == device_info.deviceClass:
            device_list.append(device_info)

    to_return = tuple(device_list)  # must be tuple

    return to_return


def openDevice(object device):
    """Open a device.

    Parameters
    ----------
    device : OHMDDeviceInfo or int
        Device descriptor or enumerated index to retrieve information from. Must
        be a device belonging to the ``OHMD_DEVICE_CLASS_HMD`` display class.
        Best practice is to pass a descriptor instead of an `int`.

    """
    # must have context and be probed
    if _ctx is NULL:
        raise OHMDNoContextError()

    if not _contextProbed:
        raise OHMDContextNotProbedError()

    cdef int result = ohmd.OHMD_S_OK
    cdef OHMDDeviceInfo desc

    if isinstance(device, int):  # enum index provided
        desc = _deviceInfoList[device]
    elif isinstance(device, OHMDDeviceInfo):  # device object provided
        desc = OHMDDeviceInfo
    else:
        raise ValueError(
            'Parameter `device` must be type `int` or `OHMDDeviceInfo`.')

    # check if the target object has the correct display class
    if desc.c_device_info.deviceClass != ohmd.OHMD_DEVICE_CLASS_HMD:
        raise OHMDWrongDeviceClassError("Device is not a display.")

    # pointer to device handle
    desc.c_device = ohmd.ohmd_list_open_device(_ctx, desc.deviceIdx)


def closeDevice(object device):
    """Close a device."""
    pass


def getDisplayInfo(object device):
    """Get information about a display device (head-mounted display usually)
    from OpenHMD.

    Parameters
    ----------
    device : OHMDDeviceInfo or int
        Device descriptor or enumerated index to retrieve information from. Must
        be a device belonging to the ``OHMD_DEVICE_CLASS_HMD`` display class.
        Best practice is to pass a descriptor instead of an `int`.

    Returns
    -------
    OHMDDisplayInfo
        Descriptor containing information about the display.

    Examples
    --------
    Get information about a display, here we attempt to get the physical
    screen width and height::

        if probe():  # must be called at least once after `create`
            hmd_devices = getDevices(displayClass=OHMD_DEVICE_CLASS_HMD)
            if hmd_devices:
                hmd = hmd_devices[0]  # use the first found
                openDevice(hmd)   # open the device
                display_info = getDisplayInfo(hmd)
                screenWidth, screenHeight = display_info.screenSize

    """
    # must have context and be probed
    if _ctx is NULL:
        raise OHMDNoContextError()

    if not _contextProbed:
        raise OHMDContextNotProbedError()

    cdef int result = ohmd.OHMD_S_OK
    cdef OHMDDeviceInfo desc

    if isinstance(device, int):  # enum index provided
        desc = _deviceInfoList[device]
    elif isinstance(device, OHMDDeviceInfo):  # device object provided
        desc = OHMDDeviceInfo
    else:
        raise ValueError(
            'Parameter `device` must be type `int` or `OHMDDeviceInfo`.')

    # check if the target object has the correct display class
    if desc.c_device_info.deviceClass != ohmd.OHMD_DEVICE_CLASS_HMD:
        raise OHMDWrongDeviceClassError("Device is not a display.")

    # check if the device has been opened
    if desc.c_device is NULL:
        raise OHMDDeviceNotOpenError(
            "Device is not opened, cannot retrieve information.")

    # pointer to device handle
    cdef ohmd.ohmd_device* this_device = desc.c_device

    # create a descriptor and populate its fields
    cdef OHMDDisplayInfo to_return = OHMDDisplayInfo()
    cdef ohmd.ohmdDisplayInfo* display_info = &to_return.c_data

    display_info.deviceIdx = desc.c_device_info.deviceIdx

    ohmd.ohmd_device_getf(
        this_device,
        ohmd.OHMD_SCREEN_HORIZONTAL_SIZE,
        &display_info.screenSize[0])
    ohmd.ohmd_device_getf(
        this_device,
        ohmd.OHMD_SCREEN_VERTICAL_SIZE,
        &display_info.screenSize[1])
    ohmd.ohmd_device_getf(
        this_device,
        ohmd.OHMD_EYE_IPD,
        &display_info.ipd)

    # binocular properties
    ohmd.ohmd_device_getf(
        this_device,
        ohmd.OHMD_LEFT_EYE_FOV,
        &display_info.eyeFov[0])
    ohmd.ohmd_device_getf(
        this_device,
        ohmd.OHMD_RIGHT_EYE_FOV,
        &display_info.eyeFov[1])
    ohmd.ohmd_device_getf(
        this_device,
        ohmd.OHMD_LEFT_EYE_ASPECT_RATIO,
        &display_info.eyeFov[0])
    ohmd.ohmd_device_getf(
        this_device,
        ohmd.OHMD_RIGHT_EYE_ASPECT_RATIO,
        &display_info.eyeFov[1])

    return desc


def update():
    """Update the values for the devices handled by a context.

    Call this once per frame. If PsychXR is running in a background thread, it
    is recommended that you call this every 10-20 milliseconds.

    """
    global _ctx

    if _ctx is NULL:
        raise OHMDNoContextError()

    ohmd.ohmd_ctx_update(_ctx)


def getString(int stype):
    """Get a string from OpenHMD.

    Parameters
    ----------
    stype : int
        Type of string data to fetch, either one of
        ``OHMD_GLSL_DISTORTION_FRAG_SRC`` or ``OHMD_GLSL_DISTORTION_FRAG_SRC``.

    Returns
    -------
    tuple
        Result of the ``ohmd_gets`` C-API call (`int`) and the description text
        (`str`).

    """
    cdef int result
    cdef const char* out
    cdef str str_return

    result = ohmd.ohmd_gets(<ohmd.ohmd_string_description>stype, &out)
    str_return = out.decode('utf-8')

    return result, str_return


def getListString(int index, int type_):
    """Get a device description string from an enumeration list index.

    Can only be called after :func:`probe`.

    Parameters
    ----------
    index : int
        An index between 0 and the value returned by calling :func:`probe`.
    type_ : int
        Type of string description data to fetch, either one of ``OHMD_VENDOR``,
        ``OHMD_PRODUCT`` and ``OHMD_PATH``.

    Returns
    -------
    tuple
        The string description text (`str`).

    """
    cdef const char* result
    cdef str str_return

    if _ctx is NULL:
        raise OHMDNoContextError()

    if not _contextProbed:
        raise OHMDContextNotProbedError()

    result = ohmd.ohmd_list_gets(_ctx, index, <ohmd.ohmd_string_value>type_)
    str_return = result.decode('utf-8')

    return str_return


def getListInt(int index, int type_):
    """Get an integer value from an enumeration list index.

    Can only be called after :func:`probe`.

    Parameters
    ----------
    index : int
        An index between 0 and the value returned by calling :func:`probe`.
    type_ : int
        Type of string description data to fetch.

    Returns
    -------
    tuple
        Result of the ``ohmd_list_geti`` C-API call (`int`) and the value
        (`int`).

    """
    cdef int result
    cdef int int_return

    if _ctx is NULL:
        raise OHMDNoContextError()

    if not _contextProbed:
        raise OHMDContextNotProbedError()

    result = ohmd.ohmd_list_geti(_ctx, index, <ohmd.ohmd_int_value>type_, &int_return)
    str_return = result.decode('utf-8')

    return result, int_return