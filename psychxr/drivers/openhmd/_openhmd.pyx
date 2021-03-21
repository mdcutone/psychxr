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
    "success",
    "failure",
    "create",
    "destroy",
    "probe",
    "getDevices",
    "getError",
    "update",
    "getString"
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

# ------------------------------------------------------------------------------
# OpenHMD specific exceptions
#
class OpenHMDNoContextError(RuntimeError):
    """Raised if trying to perform an action without having a valid context."""
    pass


# ------------------------------------------------------------------------------
# Initialize module
#
cdef ohmd.ohmd_context* _ctx = NULL  # handle for the context

# Keep track of devices found using an array of descriptors. This is populated
# when `probe` is called and freed when `destroy` is called. Calling `probe`
# will repopulate this array. The user can call `getDevices` to get a list of
# device descriptors. Calling `openDevice` using the descriptor will open it.
cdef Py_ssize_t _deviceCount = 0
# cdef ohmd.ohmdDeviceInfo** _deviceInfo = NULL
cdef tuple _deviceInfoList = ()  # stores device info instances

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
    """Device information class for OpenHMD devices.

    This class is used to identify devices enumerated by OpenHMD, such as HMDs,
    controllers and trackers. OpenHMD manages devices differently from LibOVR,
    where this class is used for all types of devices, not just HMDs. Therefore,
    data about an HMD, such as display resolution, are not provided by instances
    of this class.

    """
    cdef ohmd.ohmdDeviceInfo* c_data
    cdef bint ptr_owner

    def __init__(self):
        self.newStruct()

    def __cinit__(self):
        self.ptr_owner = False

    @staticmethod
    cdef OHMDDeviceInfo fromPtr(ohmd.ohmdDeviceInfo* ptr, bint owner=False):
        # bypass __init__ if wrapping a pointer
        cdef OHMDDeviceInfo wrapper = OHMDDeviceInfo.__new__(
            OHMDDeviceInfo)
        wrapper.c_data = ptr
        wrapper.ptr_owner = owner

        return wrapper

    cdef void newStruct(self):
        if self.c_data is not NULL:  # already allocated, __init__ called twice?
            return

        cdef ohmd.ohmdDeviceInfo* _ptr = \
            <ohmd.ohmdDeviceInfo*>PyMem_Malloc(
                sizeof(ohmd.ohmdDeviceInfo))

        if _ptr is NULL:
            raise MemoryError

        self.c_data = _ptr
        self.ptr_owner = True

    def __dealloc__(self):
        if self.c_data is not NULL and self.ptr_owner is True:
            PyMem_Free(self.c_data)
            self.c_data = NULL

    @property
    def vendorName(self):
        """Device vendor name (`str`)."""
        return self.c_data[0].vendorName.decode('utf-8')

    @property
    def manufacturer(self):
        """Device manufacturer name, alias of `vendorName` (`str`)"""
        return self.c_data[0].vendorName.decode('utf-8')

    @property
    def productName(self):
        """Device product name (`str`)."""
        return self.c_data[0].productName.decode('utf-8')

    # @property
    # def resolution(self):
    #     """Horizontal and vertical resolution of the display (`ndarray`)."""
    #     return np.asarray(
    #         (self.c_data[0].resolution[0],
    #          self.c_data[0].resolution[1]), dtype=int)
    #
    # @property
    # def screenSize(self):
    #     """Horizontal and vertical size of the display in meters (`ndarray`)."""
    #     return np.asarray(
    #         (self.c_data[0].screenSize[0],
    #          self.c_data[0].screenSize[1]), dtype=np.float32)
    #
    # @property
    # def aspect(self):
    #     """Physical display aspect ratios for the left and right eye (`ndarray`).
    #     """
    #     return np.asarray(
    #         (self.c_data[0].aspect[0],
    #          self.c_data[0].aspect[1]), dtype=np.float32)
    #
    # @property
    # def eyeFov(self):
    #     """Physical field of view for each eye in degrees (`ndarray`).
    #     """
    #     return np.asarray(
    #         (self.c_data[0].fov[0],
    #          self.c_data[0].fov[1]), dtype=np.float32)
    #
    # @property
    # def ipd(self):
    #     """Interpupilary distance in meters reported by the device (`float`)."""
    #     return self.c_data[0].ipd

    @property
    def deviceIdx(self):
        """Enumerated index of the device (`int`)."""
        return self.c_data[0].deviceIdx

    @property
    def deviceClass(self):
        """Device class identifier (`int`)."""
        return <int>self.c_data[0].deviceClass

    @property
    def isHMD(self):
        """``True`` if this device is an HMD (`bool`)."""
        return self.c_data[0].deviceClass == ohmd.OHMD_DEVICE_CLASS_HMD

    @property
    def isController(self):
        """``True`` if this device is a controller (`bool`)."""
        return self.c_data[0].deviceClass == ohmd.OHMD_DEVICE_CLASS_CONTROLLER

    @property
    def isTracker(self):
        """``True`` if this device is a generic tracker (`bool`)."""
        return self.c_data[0].deviceClass == \
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
        return <int>self.c_data[0].deviceFlags

    @property
    def hasOrientationTracking(self):
        """``True`` if the HMD is capable of tracking orientation."""
        return (self.c_data.deviceFlags &
                ohmd.OHMD_DEVICE_FLAGS_ROTATIONAL_TRACKING) == \
               ohmd.OHMD_DEVICE_FLAGS_ROTATIONAL_TRACKING

    @property
    def hasPositionTracking(self):
        """``True`` if the HMD is capable of tracking position."""
        return (self.c_data.deviceFlags &
                ohmd.OHMD_DEVICE_FLAGS_POSITIONAL_TRACKING) == \
               ohmd.OHMD_DEVICE_FLAGS_POSITIONAL_TRACKING

    @property
    def isDebugDevice(self):
        """``True`` if the HMD is a virtual debug (null or dummy) device."""
        return (self.c_data.deviceFlags &
                ohmd.OHMD_DEVICE_FLAGS_NULL_DEVICE) == \
               ohmd.OHMD_DEVICE_FLAGS_NULL_DEVICE


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
    global _deviceInfo

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

    if _ctx is NULL:
        raise OpenHMDNoContextError()

    # probe for devices
    cdef int probe_result = ohmd.ohmd_ctx_probe(_ctx)
    if not probe_result:  # no devices found, just return
        return probe_result

    print(f"device count: {probe_result}")

    print('before open')

    # inter over devices, open them and get information
    cdef int device_idx = 0
    cdef ohmd.ohmd_device* this_device
    cdef ohmd.ohmdDeviceInfo* device_info
    cdef OHMDDeviceInfo desc
    cdef list devices_list = []
    for device_idx in range(probe_result):
        # open device
        this_device = ohmd.ohmd_list_open_device(_ctx, device_idx)
        if this_device is NULL:
            continue

        desc = OHMDDeviceInfo()

        print('opened device')
        # populate device info
        device_info = desc.c_data
        print(f"set device info for {device_idx}")
        device_info.productName = ohmd.ohmd_list_gets(  # product name
            _ctx,
            device_idx,
            ohmd.OHMD_PRODUCT)
        device_info.vendorName = ohmd.ohmd_list_gets(  # vendor name
            _ctx,
            device_idx,
            ohmd.OHMD_VENDOR)
        print("device getter API")
        device_info[0].deviceIdx = device_idx  # device index
        ohmd.ohmd_device_geti(  # device class
            this_device,
            ohmd.OHMD_DEVICE_CLASS,
            <int*>(&device_info.deviceClass))
        ohmd.ohmd_device_geti(  # device flags
            this_device,
            ohmd.OHMD_DEVICE_FLAGS,
            <int*>(&device_info.deviceFlags))

        # close the device
        if ohmd.ohmd_close_device(this_device) < ohmd.OHMD_S_OK:
            clear_device_info()

            return probe_result

        # create a python wrapper around the descriptor
        print(f'create wrapper for {desc.productName}')
        devices_list.append(desc)  # add to list

    _deviceInfoList = tuple(_deviceInfoList)

    return probe_result


def destroy():
    """Destroy the current context/session."""
    global _ctx

    if _ctx is NULL:  # nop if no context created
        return

    clear_device_info()

    ohmd.ohmd_ctx_destroy(_ctx)
    _ctx = NULL


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

    """
    global _deviceCount
    cdef tuple to_return

    print('getting device list')

    if deviceClass == 0:
        to_return = _deviceInfoList
        return to_return

    cdef list device_list = []
    cdef OHMDDeviceInfo device_info
    for device_info in _deviceInfoList:
        if deviceClass == device_info.deviceClass:
            device_list.append(device_info)

    to_return = tuple(device_list)

    return to_return


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
        raise OpenHMDNoContextError()

    err_msg = ohmd.ohmd_ctx_get_error(_ctx)

    return err_msg.decode('utf-8')


def update():
    """Update the values for the devices handled by a context.

    Call this once per frame. If PsychXR is running in a background thread, it
    is recommended that you call this every 10-20 milliseconds.

    """
    global _ctx

    if _ctx is NULL:
        raise OpenHMDNoContextError()

    ohmd.ohmd_ctx_update(_ctx)


def getString(int stype):
    """Get a string from the API."""
    cdef int result
    cdef const char* out
    cdef str str_return

    if _ctx is NULL:
        raise OpenHMDNoContextError()

    result = ohmd.ohmd_gets(<ohmd.ohmd_string_description>stype, &out)
    str_return = out.decode('utf-8')

    return result, str_return
