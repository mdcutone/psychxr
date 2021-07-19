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

"""
# ------------------------------------------------------------------------------
# Module information
#
__author__ = "Matthew D. Cutone"
__credits__ = []
__copyright__ = "Copyright 2021 Matthew D. Cutone"
__license__ = "MIT"
__version__ = '0.2.4rc2'
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
    "OHMDDeviceInfo",
    "success",
    "failure",
    "getVersion",
    "create",
    "destroy",
    "probe",
    "isContextProbed",
    "getDeviceCount",
    "getError",
    "getDevices",
    "openDevice",
    "closeDevice",
    "getDevicePose",
    "lastUpdateTimeElapsed",
    "externalSensorFusion",
    "getEyeViewMatrix",
    "getEyeProjectionMatrix",
    "getEyeAspectRatio",
    "getDeviceParamf",
    "setDeviceParamf",
    "getDeviceParami",
    "update",
    "getString",
    "getListString",
    "getListInt"
]

cimport numpy as np
import numpy as np
np.import_array()
from . cimport openhmd as ohmd
from libc.time cimport clock, clock_t
from cpython.mem cimport PyMem_Malloc, PyMem_Free
# from libc.math cimport fabs
cimport psychxr.tools.vrmath as vrmath
import psychxr.tools.vrmath as vrmath


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
# Initialize module
#
cdef ohmd.ohmd_context* _ctx = NULL  # handle for the context

# found devices end up here
cdef Py_ssize_t _deviceCount = 0
cdef tuple _deviceInfoList = ()  # stores device info instances
cdef int _contextProbed = 0

# is auto-update enabled?
cdef int _automatic_update = 0
cdef double _last_update_time = 0.0


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


cdef ohmd.ohmd_device* get_device_from_param(object dev):
    """Get a device handle given a descriptor or enumerated index. This is an 
    internal function.

    Parameters
    ----------
    dev : OHMDDeviceInfo or int
        Descriptor or enumerated index of a device.

    Returns
    -------
    ohmd_device*
        Opaque pointer as a device handle. Value is `NULL` if there is no valid
        device.

    """
    cdef ohmd.ohmd_device*to_return = NULL
    if isinstance(dev, int):  # enum index provided
        try:
            to_return = (<OHMDDeviceInfo> _deviceInfoList[dev]).c_device
        except IndexError:
            return to_return  # catch this error, just return NULL
    elif isinstance(dev, OHMDDeviceInfo):  # device object provided
        to_return = (<OHMDDeviceInfo> dev).c_device

    # anything else will give NULL

    return to_return


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


class OHMDUnknownError(RuntimeError):
    """Raised if an API call returns `OHMD_S_UNKNOWN_ERROR`."""
    pass


class OHMDInvalidParameterError(RuntimeError):
    """Raised if an API call returns `OHMD_S_INVALID_PARAMETER`."""
    pass


class OHMDUnsupportedError(RuntimeError):
    """Raised if an API call returns `OHMD_S_UNSUPPORTED`."""
    pass


class OHMDInvalidOperationError(RuntimeError):
    """Raised if an API call returns `OHMD_S_INVALID_OPERATION`."""
    pass


# ------------------------------------------------------------------------------
# Python API for OpenHMD
#

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
    _deviceCount = probe_device_count

    return probe_device_count


def isContextProbed():
    """Check if :func:`probe` was called on the current context.

    Returns
    -------
    bool
        ``True`` if the context was probed since :func:``create`` was called.

    """
    return <bint>_contextProbed


def getDeviceCount():
    """Number of devices found during the last call to :func:`probe`.

    This function returns the same number that was returned by the last
    ``probe`` call. If referencing devices by their enumerated index, values
    from ``0`` to ``getDeviceCount() - 1`` are valid.

    Returns
    -------
    int
        Number of devices found on this system that OpenHMD can use.

    """
    return _deviceCount


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


def getDevices(int deviceClass=-1, bint openOnly=False, bint nullDevices=False):
    """Get devices found during the last call to :func:`probe`.

    Parameters
    ----------
    deviceClass : int
        Only get devices belonging to the specified device class. Values can be
        one of ``OHMD_DEVICE_CLASS_CONTROLLER``, ``OHMD_DEVICE_CLASS_HMD``, or
        ``OHMD_OHMD_DEVICE_CLASS_GENERIC_TRACKER``. Set to ``-1`` to get all
        devices (the default).
    openOnly : bool
        Only get devices that are currently opened. Default is `False` which
        will return all devices found during the last `probe` call whether they
        are opened or not.
    nullDevices : bool
        Include null/debug devices.

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

    if deviceClass < 0:
        to_return = _deviceInfoList
        return to_return

    cdef list device_list = []
    cdef OHMDDeviceInfo desc
    for desc in _deviceInfoList:
        if openOnly and not desc.isOpen:
            continue

        if desc.isDebugDevice and not nullDevices:
            continue

        if deviceClass == desc.deviceClass:
            device_list.append(desc)

    to_return = tuple(device_list)  # must be tuple

    return to_return


def openDevice(object device):
    """Open a device.

    Parameters
    ----------
    device : OHMDDeviceInfo or int
        Descriptor or enumerated index of a device to open. Best practice is to
        pass a descriptor instead of an `int`.

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
        desc = device
    else:
        raise ValueError(
            'Parameter `device` must be type `int` or `OHMDDeviceInfo`.')

    # check if the target object has the correct display class
    if desc.c_device_info.deviceClass != ohmd.OHMD_DEVICE_CLASS_HMD:
        raise OHMDWrongDeviceClassError("Device is not a display.")

    # get pointer to device handle
    desc.c_device = ohmd.ohmd_list_open_device(_ctx, desc.deviceIdx)
    desc.c_device_info.isOpened = 1  # flag as opened


def closeDevice(object device):
    """Close a device.

    Parameters
    ----------
    device : OHMDDeviceInfo or int
        Descriptor or enumerated index of a device to close. Best practice is to
        pass a descriptor instead of an `int`.

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
        desc = device
    else:
        raise ValueError(
            'Parameter `device` must be type `int` or `OHMDDeviceInfo`.')

    # actually close the device
    result = ohmd.ohmd_close_device(desc.c_device)
    desc.c_device_info.isOpened = 0  # flag as closed

    return result


def externalSensorFusion(object device,
                         float deltaT,
                         object linearAcceleration,
                         object angularAcceleration,
                         object absRef):
    """Perform external sensor fusion on the specified device.

    This function allows for external sensor data to be used in pose estimation
    for a given device. Data can be obtained from an IMU or similar device.

    Parameters
    ----------
    device : OHMDDeviceInfo or int
        Descriptor or enumerated index of a device. Best practice is to pass a
        descriptor instead of an `int`.
    deltaT : float
        Elapsed time in seconds sensor values were sampled since the last
        :func:`update` call.
    linearAcceleration : ArrayLike
        Linear acceleration reading (X, Y, Z) from an accelerometer.
    angularAcceleration : ArrayLike
        Angular acceleration reading (X, Y, Z) from a gyro.
    absRef : ArrayLike
        Absolute direction reference (X, Y, Z) from a magnetometer or similar.

    """
    # must have context and be probed
    if _ctx is NULL:
        raise OHMDNoContextError()

    if not _contextProbed:
        raise OHMDContextNotProbedError()

    cdef int result = ohmd.OHMD_S_OK
    cdef ohmd.ohmd_device* c_device = get_device_from_param(device)

    if c_device is NULL:
        raise ValueError(
            'Invalid device identifier specified to parameter `device`.')

    # populate the array to pass to the sensor fusion API
    cdef float[10] sensor_fusion_values

    # time elapsed
    sensor_fusion_values[0] = deltaT

    # gyro
    sensor_fusion_values[1] = <float>angularAcceleration[0]
    sensor_fusion_values[2] = <float>angularAcceleration[1]
    sensor_fusion_values[3] = <float>angularAcceleration[2]

    # accelerometer
    sensor_fusion_values[4] = <float>linearAcceleration[0]
    sensor_fusion_values[5] = <float>linearAcceleration[1]
    sensor_fusion_values[6] = <float>linearAcceleration[2]

    # magnetometer
    sensor_fusion_values[7] = <float>absRef[0]
    sensor_fusion_values[8] = <float>absRef[1]
    sensor_fusion_values[9] = <float>absRef[2]

    # call the API
    result = ohmd.ohmd_device_setf(
        c_device,
        ohmd.OHMD_EXTERNAL_SENSOR_FUSION,
        &sensor_fusion_values[0])

    return result


def getEyeViewMatrix(object device, int eye, np.ndarray[np.float32_t, ndim=2] out=None):
    """Get the eye view matrix for a given tracked device and eye. Value
    returned can be used directly with OpenGL.

    Parameters
    ----------
    device : OHMDDeviceInfo or int
        Descriptor or enumerated index of a device. Best practice is to pass a
        descriptor instead of an `int`.
    eye : int
        Eye index. Either 0 for the left and 1 for the right. You can also use
        constants `OHMD_LEFT_EYE` or `OHMD_RIGHT_EYE`.
    out : ndarray
        Optional output array for the matrix values.

    Returns
    -------
    ndarray
        4x4 view matrix.

    """
    # must have context and be probed
    if _ctx is NULL:
        raise OHMDNoContextError()

    if not _contextProbed:
        raise OHMDContextNotProbedError()

    cdef int result = ohmd.OHMD_S_OK
    cdef ohmd.ohmd_device* c_device = get_device_from_param(device)

    if c_device is NULL:
        raise ValueError(
            'Invalid device identifier specified to parameter `device`.')

    cdef ohmd.ohmd_float_value matrix_type
    if eye == ohmd.OHMD_EYE_LEFT:
        matrix_type = ohmd.OHMD_LEFT_EYE_GL_MODELVIEW_MATRIX
    elif eye == ohmd.OHMD_EYE_RIGHT:
        matrix_type = ohmd.OHMD_RIGHT_EYE_GL_MODELVIEW_MATRIX
    else:
        raise ValueError(
            "Value specified to parameter `eyeIndex` myst be one of "
            "`OHMD_EYE_LEFT` or `OHMD_EYE_RIGHT`.")

    cdef np.ndarray[np.float32_t, ndim=2] to_return

    if out is None:
        to_return = np.empty((4, 4), dtype=np.float32)
    else:
        to_return = out

    result = ohmd.ohmd_device_getf(
        c_device, matrix_type, <float*>to_return.data)

    return to_return


def getEyeProjectionMatrix(object device, int eye, np.ndarray[np.float32_t, ndim=2] out=None):
    """Get the projection matrix for the specified device and eye. Value
    returned can be used directly with OpenGL.

    Parameters
    ----------
    device : OHMDDeviceInfo or int
        Descriptor or enumerated index of a device. Best practice is to pass a
        descriptor instead of an `int`.
    eye : int
        Eye index. Either 0 for the left and 1 for the right. You can also use
        constants `OHMD_LEFT_EYE` or `OHMD_RIGHT_EYE`.
    out : ndarray
        Optional output array for the matrix values. Needs to have a `dtype` of
        `float32` and be C contiguous.

    Returns
    -------
    ndarray
        4x4 projection matrix.

    """
    # must have context and be probed
    if _ctx is NULL:
        raise OHMDNoContextError()

    if not _contextProbed:
        raise OHMDContextNotProbedError()

    cdef int result = ohmd.OHMD_S_OK
    cdef ohmd.ohmd_device* c_device = get_device_from_param(device)

    if c_device is NULL:
        raise ValueError(
            'Invalid device identifier specified to parameter `device`.')

    cdef ohmd.ohmd_float_value matrix_type
    if eye == ohmd.OHMD_EYE_LEFT:
        matrix_type = ohmd.OHMD_LEFT_EYE_GL_PROJECTION_MATRIX
    elif eye == ohmd.OHMD_EYE_RIGHT:
        matrix_type = ohmd.OHMD_RIGHT_EYE_GL_PROJECTION_MATRIX
    else:
        raise ValueError(
            "Value specified to parameter `eye` must be one of `OHMD_EYE_LEFT` "
            "or `OHMD_EYE_RIGHT`.")

    cdef np.ndarray[np.float32_t, ndim=2] to_return

    if out is None:
        to_return = np.empty((4, 4), dtype=np.float32)
    else:
        to_return = out

    result = ohmd.ohmd_device_getf(
        c_device, matrix_type, <float*>to_return.data)

    return to_return


def getEyeAspectRatio(object device, int eye):
    """Get the aspect ratio of an eye.

    Parameters
    ----------
    device : OHMDDeviceInfo or int
        Descriptor or enumerated index of a device. Best practice is to pass a
        descriptor instead of an `int`.
    eye : int
        Eye index. Either 0 for the left and 1 for the right. You can also use
        constants `OHMD_LEFT_EYE` or `OHMD_RIGHT_EYE`.

    Returns
    -------
    float
        Aspect ratio of the eye's physical display.

    """
    # must have context and be probed
    if _ctx is NULL:
        raise OHMDNoContextError()

    if not _contextProbed:
        raise OHMDContextNotProbedError()

    cdef int result = ohmd.OHMD_S_OK
    cdef ohmd.ohmd_device* c_device = get_device_from_param(device)

    if c_device is NULL:
        raise ValueError(
            'Invalid device identifier specified to parameter `device`.')

    cdef ohmd.ohmd_float_value aspect_type
    if eye == ohmd.OHMD_EYE_LEFT:
        aspect_type = ohmd.OHMD_LEFT_EYE_ASPECT_RATIO
    elif eye == ohmd.OHMD_EYE_RIGHT:
        aspect_type = ohmd.OHMD_RIGHT_EYE_ASPECT_RATIO
    else:
        raise ValueError(
            "Value specified to parameter `eye` must be one of `OHMD_EYE_LEFT` "
            "or `OHMD_EYE_RIGHT`.")

    cdef float to_return = 0.0
    result = ohmd.ohmd_device_getf(
        c_device, aspect_type, &to_return)

    return to_return


# These control how we can specify and retrieve each parameter. We subtract 1
# from the value of the symbolic constant and look up the required value in the
# following arrays.

cdef Py_ssize_t _ohmd_float_value_sz[22]  # for float parameters
_ohmd_float_value_sz[:] = [
    4,   # OHMD_ROTATION_QUAT
    16,  # OHMD_LEFT_EYE_GL_MODELVIEW_MATRIX
    16,  # OHMD_RIGHT_EYE_GL_MODELVIEW_MATRIX
    16,  # OHMD_LEFT_EYE_GL_PROJECTION_MATRIX
    16,  # OHMD_RIGHT_EYE_GL_PROJECTION_MATRIX
    3,   # OHMD_POSITION_VECTOR
    1,   # OHMD_SCREEN_HORIZONTAL_SIZE
    1,   # OHMD_SCREEN_VERTICAL_SIZE
    1,   # OHMD_LENS_HORIZONTAL_SEPARATION
    1,   # OHMD_LENS_VERTICAL_POSITION
    1,   # OHMD_LEFT_EYE_FOV
    1,   # OHMD_LEFT_EYE_ASPECT_RATIO
    1,   # OHMD_RIGHT_EYE_FOV
    1,   # OHMD_RIGHT_EYE_ASPECT_RATIO
    1,   # OHMD_EYE_IPD
    1,   # OHMD_PROJECTION_ZFAR
    1,   # OHMD_PROJECTION_ZNEAR
    6,   # OHMD_DISTORTION_K
    10,  # OHMD_EXTERNAL_SENSOR_FUSION
    4,   # OHMD_UNIVERSAL_DISTORTION_K
    3,   # OHMD_UNIVERSAL_ABERRATION_K
    0    # OHMD_CONTROLS_STATE (unused)
]

cdef Py_ssize_t _ohmd_int_value_sz[7]  # for integer parameters, start at 0
_ohmd_int_value_sz[:] = [
    1,   # OHMD_SCREEN_HORIZONTAL_RESOLUTION
    1,   # OHMD_SCREEN_VERTICAL_RESOLUTION
    1,   # OHMD_DEVICE_CLASS
    1,   # OHMD_DEVICE_FLAGS
    1,   # OHMD_CONTROL_COUNT
    # special handling for these so they are unfilled ...
    0,   # OHMD_CONTROLS_HINTS
    0    # OHMD_CONTROLS_TYPES
]

# Are the parameters settable? We check using the access flags below.
# 0 = read-only (get)
# 1 = write (set)
# 2 = read/write (get/set)
cdef Py_ssize_t _ohmd_float_value_access_flags[22]  # floats
_ohmd_float_value_access_flags[:] = [
    0,   # OHMD_ROTATION_QUAT
    0,   # OHMD_LEFT_EYE_GL_MODELVIEW_MATRIX
    0,   # OHMD_RIGHT_EYE_GL_MODELVIEW_MATRIX
    0,   # OHMD_LEFT_EYE_GL_PROJECTION_MATRIX
    0,   # OHMD_RIGHT_EYE_GL_PROJECTION_MATRIX
    0,   # OHMD_POSITION_VECTOR
    0,   # OHMD_SCREEN_HORIZONTAL_SIZE
    0,   # OHMD_SCREEN_VERTICAL_SIZE
    0,   # OHMD_LENS_HORIZONTAL_SEPARATION
    0,   # OHMD_LENS_VERTICAL_POSITION
    0,   # OHMD_LEFT_EYE_FOV
    0,   # OHMD_LEFT_EYE_ASPECT_RATIO
    0,   # OHMD_RIGHT_EYE_FOV
    0,   # OHMD_RIGHT_EYE_ASPECT_RATIO
    2,   # OHMD_EYE_IPD
    2,   # OHMD_PROJECTION_ZFAR
    2,   # OHMD_PROJECTION_ZNEAR
    0,   # OHMD_DISTORTION_K
    1,   # OHMD_EXTERNAL_SENSOR_FUSION
    0,   # OHMD_UNIVERSAL_DISTORTION_K
    0,   # OHMD_UNIVERSAL_ABERRATION_K
    0    # OHMD_CONTROLS_STATE
]

cdef Py_ssize_t _ohmd_int_value_access_flags[7]  # for integer parameters
_ohmd_int_value_access_flags[:] = [
    0,   # OHMD_SCREEN_HORIZONTAL_RESOLUTION
    0,   # OHMD_SCREEN_VERTICAL_RESOLUTION
    0,   # OHMD_DEVICE_CLASS
    0,   # OHMD_DEVICE_FLAGS
    0,   # OHMD_CONTROL_COUNT
    # special handling for these so they are unfilled ...
    0,   # OHMD_CONTROLS_HINTS
    0    # OHMD_CONTROLS_TYPES
]


def setDeviceParamf(object device, int param, object value):
    """Set a floating point parameter for a given device.

    This calls to `ohmd_device_setf` on the specified `device`, passing the
    value specified by `param`.

    Parameters
    ----------
    device : OHMDDeviceInfo or int
        Descriptor or enumerated index of a device. Best practice is to pass a
        descriptor instead of an `int`.
    param : int
        Symbolic constant representing the parameter to set. Parameters can
        be one of the following constants (parameter type and length if
        applicable in parentheses):

            - ``OHMD_EYE_IPD`` (float)
            - ``OHMD_PROJECTION_ZFAR`` (float)
            - ``OHMD_PROJECTION_ZNEAR`` (float)
            - ``OHMD_EXTERNAL_SENSOR_FUSION`` (ndarray, length 10)

    Examples
    --------
    Set the eye IPD for the device used to compute view matrices::

        ohmd.setDeviceParamf(
            hmd_device,
            ohmd.OHMD_EYE_IPD,
            0.062)  # in meters

    """
    # must have context and be probed
    if _ctx is NULL:
        raise OHMDNoContextError()

    if not _contextProbed:
        raise OHMDContextNotProbedError()

    cdef OHMDDeviceInfo desc

    if isinstance(device, int):  # enum index provided
        desc = _deviceInfoList[device]
    elif isinstance(device, OHMDDeviceInfo):  # device object provided
        desc = device
    else:
        raise ValueError(
            "Parameter `device` must be type `int` or `OHMDDeviceInfo`.")

    # check if the device is presently opened
    if not desc.c_device_info.isOpened:
        raise OHMDDeviceNotOpenError(
            "Trying to obtain the pose of a device that is presently closed.")

    if 1 > param > 22:  # equal to the length of `_ohmd_float_value_sz`
        raise ValueError("Value for `param` is not valid.")

    # check if the parameter is write only
    if _ohmd_float_value_access_flags[param-1] < 1:
        raise OHMDInvalidParameterError(
            "Specified parameter is not writeable (read-only).")

    # this is pretty messy for now, it works... but a better solution may be
    # required to avoid hard-coded indices
    cdef Py_ssize_t param_sz
    cdef Py_ssize_t i
    cdef int result  # API call result
    cdef object to_return  # returned to the user in Python land

    # array allocated to hold the returned data from the API call
    cdef float* api_param_val
    param_sz = _ohmd_float_value_sz[param-1]

    # all params are handled here
    api_param_val = <float*>PyMem_Malloc(param_sz * sizeof(float))
    if not api_param_val:
        raise MemoryError()

    if param_sz > 1:
        # make sure that the value being set has the right length for the
        # parameter
        if not hasattr(value, '__len__') and hasattr(value, '__getitem__'):
            raise TypeError(
                "`value` must be iterable for to set the specified parameter."
            )
        if len(value) != param_sz:  # check size
            raise ValueError(
                "`value` does not have required length for specified parameter."
            )

        # write values to temp array
        for i in range(param_sz):
            api_param_val[i] = value[i]
    else:
        # single value parameters
        api_param_val[0] = <float>value

    # make the APi call to get the required data
    result = ohmd.ohmd_device_setf(  # get the data from the API
        desc.c_device,
        <ohmd.ohmd_float_value>param,
        &api_param_val[0])

    PyMem_Free(api_param_val)  # clean up

    # caught an error
    if result < ohmd.OHMD_S_OK:
        if result == ohmd.OHMD_S_INVALID_PARAMETER:
            raise OHMDInvalidParameterError()
        elif result == ohmd.OHMD_S_UNSUPPORTED:
            raise OHMDUnsupportedError()
        elif result == ohmd.OHMD_S_INVALID_OPERATION:
            raise OHMDInvalidOperationError()
        else:  # catchall
            raise OHMDUnknownError()


def getDeviceParamf(object device, int param):
    """Get a floating point parameter from a device.

    This calls to `ohmd_device_getf` on the specified `device`, retrieving the
    value requested by `param`.

    Parameters
    ----------
    device : OHMDDeviceInfo or int
        Descriptor or enumerated index of a device. Best practice is to pass a
        descriptor instead of an `int`.
    param : int
        Symbolic constant representing the parameter to retrieve. Parameters can
        be one of the following constants (return type and length if applicable
        in parentheses):

            - ``OHMD_ROTATION_QUAT`` (ndarray, length 4)
            - ``OHMD_LEFT_EYE_GL_MODELVIEW_MATRIX`` (ndarray, length 16)
            - ``OHMD_RIGHT_EYE_GL_MODELVIEW_MATRIX`` (ndarray, length 16)
            - ``OHMD_LEFT_EYE_GL_PROJECTION_MATRIX`` (ndarray, length 16)
            - ``OHMD_RIGHT_EYE_GL_PROJECTION_MATRIX`` (ndarray, length 16)
            - ``OHMD_POSITION_VECTOR`` (ndarray, length 3)
            - ``OHMD_SCREEN_HORIZONTAL_SIZE`` (float)
            - ``OHMD_SCREEN_VERTICAL_SIZE`` (float)
            - ``OHMD_LENS_HORIZONTAL_SEPARATION`` (float)
            - ``OHMD_LENS_VERTICAL_POSITION`` (float)
            - ``OHMD_LEFT_EYE_FOV`` (float)
            - ``OHMD_LEFT_EYE_ASPECT_RATIO`` (float)
            - ``OHMD_RIGHT_EYE_FOV`` (float)
            - ``OHMD_RIGHT_EYE_ASPECT_RATIO`` (float)
            - ``OHMD_EYE_IPD`` (float)
            - ``OHMD_PROJECTION_ZFAR`` (float)
            - ``OHMD_PROJECTION_ZNEAR`` (float)
            - ``OHMD_DISTORTION_K`` (ndarray, length 6)
            - ``OHMD_UNIVERSAL_DISTORTION_K`` (ndarray, length 4)
            - ``OHMD_UNIVERSAL_ABERRATION_K`` (ndarray, length 3)
            - ``OHMD_CONTROLS_STATE`` (ndarray, length is the value of
              ``OHMD_CONTROL_COUNT`` returned by the API)
    value : object
        Value to set, must have the same length as the what is required by the
        parameter.

    Returns
    -------
    float or ndarray
        Single floating-point value or array. The return value data type and
        size is automatically determined by the type of parameter.

    Examples
    --------
    Get the horizontal and vertical screen size for a device::

        physical_horiz_size = ohmd.getDeviceParamf(
            hmd_device,
            ohmd.OHMD_SCREEN_HORIZONTAL_SIZE)
        physical_vert_size = ohmd.getDeviceParamf(
            hmd_device,
            ohmd.OHMD_SCREEN_VERTICAL_SIZE)

    Get the left eye model/view matrix::

        left_modelview_matrix = ohmd.getDeviceParamf(
            hmd_device,
            ohmd.OHMD_LEFT_EYE_GL_MODELVIEW_MATRIX)

        # you may want to reshape it
        left_modelview_matrix = left_modelview_matrix.reshape((4, 4))

    """
    # must have context and be probed
    if _ctx is NULL:
        raise OHMDNoContextError()

    if not _contextProbed:
        raise OHMDContextNotProbedError()

    cdef OHMDDeviceInfo desc

    if isinstance(device, int):  # enum index provided
        desc = _deviceInfoList[device]
    elif isinstance(device, OHMDDeviceInfo):  # device object provided
        desc = device
    else:
        raise ValueError(
            "Parameter `device` must be type `int` or `OHMDDeviceInfo`.")

    # check if the device is presently opened
    if not desc.c_device_info.isOpened:
        raise OHMDDeviceNotOpenError(
            "Trying to obtain the pose of a device that is presently closed.")

    if 1 > param > 22:  # equal to the length of `_ohmd_float_value_sz`
        raise ValueError("Value for `param` is not valid.")

    # check if the parameter is write only
    if _ohmd_float_value_access_flags[param-1] == 1:
        raise OHMDInvalidParameterError(
            "Specified parameter is not readable (write-only).")

    # this is pretty messy for now, it works... but a better solution may be
    # required to avoid hard-coded indices
    cdef Py_ssize_t ret_sz
    cdef Py_ssize_t i
    cdef int ctrl_count  # used if `param` == `OHMD_CONTROLS_STATE`
    cdef int result  # API call result
    cdef object to_return  # returned to the user in Python land

    # array allocated to hold the returned data from the API call
    cdef float* api_ret_val
    if param == ohmd.OHMD_CONTROLS_STATE:
        # getting a controller button state requires getting
        # `OHMD_CONTROL_COUNT` first
        result = ohmd.ohmd_list_geti(
            _ctx,
            desc.c_device_info.deviceIdx,
            <ohmd.ohmd_int_value>ohmd.OHMD_CONTROL_COUNT,
            &ctrl_count)

        # caught an error getting this parameter
        if result < ohmd.OHMD_S_OK:
            if result == ohmd.OHMD_S_INVALID_PARAMETER:
                raise OHMDInvalidParameterError()
            elif result == ohmd.OHMD_S_UNSUPPORTED:
                raise OHMDUnsupportedError()
            elif result == ohmd.OHMD_S_INVALID_OPERATION:
                raise OHMDInvalidOperationError()
            else:  # catchall
                raise OHMDUnknownError()

        ret_sz = ctrl_count
    else:
        ret_sz = _ohmd_float_value_sz[param-1]

    # all params are handled here
    api_ret_val = <float*>PyMem_Malloc(ret_sz * sizeof(float))
    if not api_ret_val:
        raise MemoryError()

    # make the APi call to get the required data
    result = ohmd.ohmd_device_getf(  # get the data from the API
        desc.c_device,
        <ohmd.ohmd_float_value>param,
        &api_ret_val[0])

    # caught an error
    if result < ohmd.OHMD_S_OK:
        PyMem_Free(api_ret_val)
        if result == ohmd.OHMD_S_INVALID_PARAMETER:
            raise OHMDInvalidParameterError()
        elif result == ohmd.OHMD_S_UNSUPPORTED:
            raise OHMDUnsupportedError()
        elif result == ohmd.OHMD_S_INVALID_OPERATION:
            raise OHMDInvalidOperationError()
        else:  # catchall
            raise OHMDUnknownError()

    # handle the return value
    if ret_sz > 1:  # requires an array to be returned
        to_return = np.empty((ret_sz,), dtype=np.float32)
        for i in range(ret_sz):
            to_return[i] = api_ret_val[i]
    else:
        to_return = <float>api_ret_val[0]  # just a Python float

    PyMem_Free(api_ret_val)  # clean up
    return to_return


def getDeviceParami(object device, int param):
    """Get an integer parameter from a device.

    This calls to `ohmd_device_geti` on the specified `device`, retrieving the
    value requested by `param`.

    Parameters
    ----------
    device : OHMDDeviceInfo or int
        Descriptor or enumerated index of a device. Best practice is to pass a
        descriptor instead of an `int`.
    param : int
        Symbolic constant representing the parameter to retrieve. Parameters can
        be one of the following constants (return type and length if applicable
        in parentheses):

            - ``OHMD_SCREEN_HORIZONTAL_RESOLUTION`` (int)
            - ``OHMD_SCREEN_VERTICAL_RESOLUTION`` (int)
            - ``OHMD_DEVICE_CLASS`` (int)
            - ``OHMD_DEVICE_FLAGS`` (int)
            - ``OHMD_CONTROL_COUNT`` (int)
            - ``OHMD_CONTROLS_HINTS`` (ndarray, length is the value of
              ``OHMD_CONTROL_COUNT`` returned by the API)
            - ``OHMD_CONTROLS_TYPES`` (ndarray, length is the value of
              ``OHMD_CONTROL_COUNT`` returned by the API)

    Returns
    -------
    float or ndarray
        Single integer value or array. The return value data type and
        size is automatically determined by the type of parameter.

    Examples
    --------
    Get the horizontal and vertical screen resolution for a device::

        hres = ohmd.getDeviceParamf(
            hmd_device,
            ohmd.OHMD_SCREEN_HORIZONTAL_RESOLUTION)
        vres = ohmd.getDeviceParamf(
            hmd_device,
            ohmd.OHMD_SCREEN_VERTICAL_RESOLUTION)

    """
    # must have context and be probed
    if _ctx is NULL:
        raise OHMDNoContextError()

    if not _contextProbed:
        raise OHMDContextNotProbedError()

    cdef OHMDDeviceInfo desc

    if isinstance(device, int):  # enum index provided
        desc = _deviceInfoList[device]
    elif isinstance(device, OHMDDeviceInfo):  # device object provided
        desc = device
    else:
        raise ValueError(
            "Parameter `device` must be type `int` or `OHMDDeviceInfo`.")

    # check if the device is presently opened
    if not desc.c_device_info.isOpened:
        raise OHMDDeviceNotOpenError(
            "Trying to obtain the pose of a device that is presently closed.")

    if 0 > param > 6:  # equal to the length of `_ohmd_int_value_sz`
        raise ValueError("Value for `param` is not valid.")

    # check if the parameter is write only
    if _ohmd_int_value_access_flags[param] == 1:
        raise OHMDInvalidParameterError(
            "Specified parameter is not readable (write-only).")

    cdef Py_ssize_t ret_sz
    cdef Py_ssize_t i
    cdef int ctrl_count  # used if `param` == `OHMD_CONTROLS_STATE`
    cdef int result  # API call result
    cdef object to_return  # returned to the user in Python land

    # array allocated to hold the returned data from the API call
    cdef int* api_ret_val
    if param == ohmd.OHMD_CONTROLS_HINTS or param == ohmd.OHMD_CONTROLS_TYPES:
        result = ohmd.ohmd_list_geti(
            _ctx,
            desc.c_device_info.deviceIdx,
            <ohmd.ohmd_int_value>ohmd.OHMD_CONTROL_COUNT,
            &ctrl_count)

        # caught an error getting this parameter
        if result < ohmd.OHMD_S_OK:
            if result == ohmd.OHMD_S_INVALID_PARAMETER:
                raise OHMDInvalidParameterError()
            elif result == ohmd.OHMD_S_UNSUPPORTED:
                raise OHMDUnsupportedError()
            elif result == ohmd.OHMD_S_INVALID_OPERATION:
                raise OHMDInvalidOperationError()
            else:  # catchall
                raise OHMDUnknownError()

        ret_sz = ctrl_count
    else:
        ret_sz = _ohmd_int_value_sz[param]

    # all params are handled here
    api_ret_val = <int*>PyMem_Malloc(ret_sz * sizeof(int))
    if not api_ret_val:
        raise MemoryError()

    # make the APi call to get the required data
    result = ohmd.ohmd_device_geti(  # get the data from the API
        desc.c_device,
        <ohmd.ohmd_int_value>param,
        &api_ret_val[0])

    # caught an error
    if result < ohmd.OHMD_S_OK:
        PyMem_Free(api_ret_val)
        if result == ohmd.OHMD_S_INVALID_PARAMETER:
            raise OHMDInvalidParameterError()
        elif result == ohmd.OHMD_S_UNSUPPORTED:
            raise OHMDUnsupportedError()
        elif result == ohmd.OHMD_S_INVALID_OPERATION:
            raise OHMDInvalidOperationError()
        else:  # catchall
            raise OHMDUnknownError()

    # handle the return value
    if ret_sz > 1:  # requires an array to be returned
        to_return = np.empty((ret_sz,), dtype=int)
        for i in range(ret_sz):
            to_return[i] = api_ret_val[i]
    else:
        to_return = <int>api_ret_val[0]  # just a Python int

    PyMem_Free(api_ret_val)  # clean up
    return to_return


def getDevicePose(object device):
    """Get the pose of a device.

    Parameters
    ----------
    device : OHMDDeviceInfo or int
        Descriptor or enumerated index of a device to open. Best practice is to
        pass a descriptor instead of an `int`.

    Returns
    -------
    RigidBodyPose
        Object representing the pose of the device.

    Examples
    --------
    Get the position (vector) and orientation (quaternion) of a device::

        myDevicePose = getDevicePose(myDevice)  # HMD, controller, etc.
        pos, ori = myDevicePose.posOri

    You can get the eye poses from the pose (assuming its an HMD)::

        import psychxr.tools.vrmath as vrmath
        leftEyePose, rightEyePose = vrmath.calcEyePoses(device, ipd=0.062)

    These can be converted to eye view matrices::

        leftViewMatrix = leftEyePose.viewMatrix
        rightViewMatrix = rightViewMatrix.viewMatrix

    """
    # must have context and be probed
    if _ctx is NULL:
        raise OHMDNoContextError()

    if not _contextProbed:
        raise OHMDContextNotProbedError()

    cdef OHMDDeviceInfo desc

    if isinstance(device, int):  # enum index provided
        desc = _deviceInfoList[device]
    elif isinstance(device, OHMDDeviceInfo):  # device object provided
        desc = device
    else:
        raise ValueError(
            'Parameter `device` must be type `int` or `OHMDDeviceInfo`.')

    # check if the device is presently opened
    # NB - maybe we should just return an identity pose?
    if not desc.c_device_info.isOpened:
        raise OHMDDeviceNotOpenError(
            "Trying to obtain the pose of a device that is presently closed.")

    # get the position from the device
    cdef float[3] device_pos  # position vector [x, y, z]
    cdef float[4] device_ori  # orientation quaternion [x, y, z, w]

    ohmd.ohmd_device_getf(  # get the position from the device
        desc.c_device,
        ohmd.OHMD_POSITION_VECTOR,
        &device_pos[0])
    ohmd.ohmd_device_getf(  # get the orientation from the device
        desc.c_device,
        ohmd.OHMD_ROTATION_QUAT,
        &device_ori[0])

    # actual pose object to return
    cdef vrmath.RigidBodyPose the_pose = vrmath.RigidBodyPose()

    # set values - find a better way to do this
    the_pose.pos = (
        device_pos[0], device_pos[1], device_pos[2])
    the_pose.ori = (
        device_ori[0], device_ori[1], device_ori[2], device_ori[3])

    return the_pose


def getDistortionCoefs(object device, int coefType, np.ndarray[np.float32_t, ndim=1] out=None):
    """Get the distortion and aberration coefficients for the device.

    Distortion coefficients are used by the shaders to apply barrel distortion
    to for lens correction. You may specify one of the following values to the
    parameter `coefType`:

    * ``OHMD_DISTORTION_K`` - Device specific distortion values, returns an
      array of 6 values.
    * ``OHMD_UNIVERSAL_DISTORTION_K`` - Universal shader distortion coefficients
      following the PanoTools lens correction model. Returns an array of 4
      values (`a`, `b`, `c`, `d`).
    * ``OHMD_UNIVERSAL_ABERRATION_K`` - Universal shader aberration coefficients
      for post-warp scaling of RGB channels. Returns 3 values specifying the
      `r`, `g`, and `b` scaling coefficients.

    Parameters
    ----------
    device : OHMDDeviceInfo or int
        Descriptor or enumerated index of a device. Best practice is to pass a
        descriptor instead of an `int`.
    coefType : int
        Symbolic constant specifiying the coefficient type to retrieve, can be
        either one of ``OHMD_DISTORTION_K``, ``OHMD_UNIVERSAL_DISTORTION_K`` or
        ``OHMD_UNIVERSAL_ABERRATION_K``.
    out : ndarray or None
        Optional output array for the matrix values. Must be the same length as
        what would be expected to be returned by this function given `coefType`.

    Returns
    -------
    ndarray
        Array of coefficients. Same object as `out` if that was specified.

    """
    # must have context and be probed
    if _ctx is NULL:
        raise OHMDNoContextError()

    if not _contextProbed:
        raise OHMDContextNotProbedError()

    cdef int result = ohmd.OHMD_S_OK
    cdef ohmd.ohmd_device* c_device = get_device_from_param(device)

    if c_device is NULL:
        raise ValueError(
            'Invalid device identifier specified to parameter `device`.')

    cdef np.ndarray[np.float32_t, ndim=1] to_return
    cdef Py_ssize_t ret_len
    if coefType == ohmd.OHMD_DISTORTION_K:
        ret_len = 6
    elif coefType == ohmd.OHMD_UNIVERSAL_DISTORTION_K:
        ret_len = 4
    elif coefType == ohmd.OHMD_UNIVERSAL_ABERRATION_K:
        ret_len = 3
    else:
        raise ValueError("Invalid value specified tp parameter `coefType`.")

    if out is None:
        to_return = np.empty((ret_len,), dtype=np.float32)
    else:
        to_return = out
        # make sure we are using the correct size
        if len(to_return) != ret_len:
            raise ValueError(
                "Array specified to parameter `out` must have length {} for the"
                " given `coefType`.".format(ret_len))

    result = ohmd.ohmd_device_getf(
        c_device, <ohmd.ohmd_float_value>coefType, <float*>to_return.data)

    if result < 0:
        raise RuntimeError("Error getting value from `ohmd_device_getf`.")

    return to_return


def lastUpdateTimeElapsed():
    """Get the time elapsed in seconds since the last :func:`update` call.

    Returns
    -------
    float
        Elapsed time in seconds.

    """
    return cpu_time() - _last_update_time


def update():
    """Update the values for the devices handled by a context.

    Call this once per frame. If PsychXR is running in a background thread, it
    is recommended that you call this every 10-20 milliseconds.

    """
    global _ctx
    global _last_update_time

    if _ctx is NULL:
        raise OHMDNoContextError()

    ohmd.ohmd_ctx_update(_ctx)
    _last_update_time = cpu_time()


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

    return result, int_return