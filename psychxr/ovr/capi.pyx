#  =============================================================================
#  Python Interface Module for LibOVR
#  =============================================================================
#
#  capi.pxy
#
#  Copyright 2018 Matthew Cutone <cutonem(a)yorku.ca> and Laurie M. Wilcox
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
"""This file exposes LibOVR functions to Python.

"""
__author__ = "Matthew D. Cutone"
__credits__ = ["Laurie M. Wilcox"]
__copyright__ = "Copyright 2019 Matthew D. Cutone"
__license__ = "MIT"
__version__ = "0.2.0"
__status__ = "Production"
__maintainer__ = "Matthew D. Cutone"
__email__ = "cutonem@yorku.ca"

# exports
__all__ = [
    'LibOVRSession', 'LibOVRPose', 'LibOVRPoseState', 'LibOVRInputState',
    'LibOVRTrackerInfo', 'LibOVRSessionStatus', 'isOculusServiceRunning',
    'isHmdConnected', 'LIBOVR_SUCCESS', 'LIBOVR_SUCCESS_NOT_VISIBLE',
    'LIBOVR_SUCCESS_BOUNDARY_INVALID', 'LIBOVR_SUCCESS_DEVICE_UNAVAILABLE',
    'LIBOVR_ERROR_TEXTURE_SWAP_CHAIN_FULL'
    ]

from .cimport ovr_capi
from .cimport ovr_math
from .math cimport *

from libc.stdint cimport int32_t, uint32_t

cimport numpy as np
import numpy as np

# -----------------
# Initialize module
# -----------------
#
# Function to check for errors returned by OVRLib functions
#
cdef ovr_capi.ovrErrorInfo _last_error_info_  # store our last error here
def check_result(result):
    if ovr_capi.OVR_FAILURE(result):
        ovr_capi.ovr_GetLastErrorInfo(&_last_error_info_)
        raise RuntimeError(
            str(result) + ": " + _last_error_info_.ErrorString.decode("utf-8"))

# helper functions
cdef float maxf(float a, float b):
    return a if a >= b else b

# Enable error checking on OVRLib functions by setting 'debug_mode=True'. All
# LibOVR functions that return a 'ovrResult' type will be checked. A
# RuntimeError will be raised if the returned value indicates failure with the
# associated message passed from LibOVR.
#
debug_mode = False

# Store controller states.
#
cdef ovr_capi.ovrInputState _ctrl_states_[5]
cdef ovr_capi.ovrInputState _ctrl_states_prev_[5]  # previous controller states

# Look-up table of button values to test which are pressed.
#
cdef dict ctrl_button_lut = {
    "A": ovr_capi.ovrButton_A,
    "B": ovr_capi.ovrButton_B,
    "RThumb": ovr_capi.ovrButton_RThumb,
    "RShoulder": ovr_capi.ovrButton_RShoulder,
    "X": ovr_capi.ovrButton_X,
    "Y": ovr_capi.ovrButton_Y,
    "LThumb": ovr_capi.ovrButton_LThumb,
    "LShoulder": ovr_capi.ovrButton_LThumb,
    "Up": ovr_capi.ovrButton_Up,
    "Down": ovr_capi.ovrButton_Down,
    "Left": ovr_capi.ovrButton_Left,
    "Right": ovr_capi.ovrButton_Right,
    "Enter": ovr_capi.ovrButton_Enter,
    "Back": ovr_capi.ovrButton_Back,
    "VolUp": ovr_capi.ovrButton_VolUp,
    "VolDown": ovr_capi.ovrButton_VolDown,
    "Home": ovr_capi.ovrButton_Home,
    "Private": ovr_capi.ovrButton_Private,
    "RMask": ovr_capi.ovrButton_RMask,
    "LMask": ovr_capi.ovrButton_LMask}

# Python accessible list of valid button names.
button_names = [*ctrl_button_lut.keys()]

# Look-up table of controller touches.
#
cdef dict ctrl_touch_lut = {
    "A": ovr_capi.ovrTouch_A,
    "B": ovr_capi.ovrTouch_B,
    "RThumb": ovr_capi.ovrTouch_RThumb,
    "RThumbRest": ovr_capi.ovrTouch_RThumbRest,
    "RIndexTrigger": ovr_capi.ovrTouch_RThumb,
    "X": ovr_capi.ovrTouch_X,
    "Y": ovr_capi.ovrTouch_Y,
    "LThumb": ovr_capi.ovrTouch_LThumb,
    "LThumbRest": ovr_capi.ovrTouch_LThumbRest,
    "LIndexTrigger": ovr_capi.ovrTouch_LIndexTrigger,
    "RIndexPointing": ovr_capi.ovrTouch_RIndexPointing,
    "RThumbUp": ovr_capi.ovrTouch_RThumbUp,
    "LIndexPointing": ovr_capi.ovrTouch_LIndexPointing,
    "LThumbUp": ovr_capi.ovrTouch_LThumbUp}

# Python accessible list of valid touch names.
touch_names = [*ctrl_touch_lut.keys()]

# Performance information for profiling.
#
cdef ovr_capi.ovrPerfStats _perf_stats_

# Color texture formats supported by OpenGL, can be used for creating swap
# chains.
#
cdef dict _supported_texture_formats = {
    "R8G8B8A8_UNORM": ovr_capi.OVR_FORMAT_R8G8B8A8_UNORM,
    "R8G8B8A8_UNORM_SRGB": ovr_capi.OVR_FORMAT_R8G8B8A8_UNORM_SRGB,
    "R16G16B16A16_FLOAT": ovr_capi.OVR_FORMAT_R16G16B16A16_FLOAT,
    "R11G11B10_FLOAT": ovr_capi.OVR_FORMAT_R11G11B10_FLOAT
}

# Performance HUD modes
#
cdef dict _performance_hud_modes = {
    "Off" : ovr_capi.ovrPerfHud_Off,
    "PerfSummary": ovr_capi.ovrPerfHud_PerfSummary,
    "AppRenderTiming" : ovr_capi.ovrPerfHud_AppRenderTiming,
    "LatencyTiming" : ovr_capi.ovrPerfHud_LatencyTiming,
    "CompRenderTiming" : ovr_capi.ovrPerfHud_CompRenderTiming,
    "AswStats" : ovr_capi.ovrPerfHud_AswStats,
    "VersionInfo" : ovr_capi.ovrPerfHud_VersionInfo
}

# mirror texture options
#
cdef dict _mirror_texture_options = {
    "Default" : ovr_capi.ovrMirrorOption_Default,
    "PostDistortion" : ovr_capi.ovrMirrorOption_PostDistortion,
    "LeftEyeOnly" : ovr_capi.ovrMirrorOption_LeftEyeOnly,
    "RightEyeOnly" : ovr_capi.ovrMirrorOption_RightEyeOnly,
    "IncludeGuardian" : ovr_capi.ovrMirrorOption_IncludeGuardian,
    "IncludeNotifications" : ovr_capi.ovrMirrorOption_IncludeNotifications,
    "IncludeSystemGui" : ovr_capi.ovrMirrorOption_IncludeSystemGui
}

# Button values
#
cdef dict _controller_buttons = {
    "A": ovr_capi.ovrButton_A,
    "B": ovr_capi.ovrButton_B,
    "RThumb": ovr_capi.ovrButton_RThumb,
    "RShoulder": ovr_capi.ovrButton_RShoulder,
    "X": ovr_capi.ovrButton_X,
    "Y": ovr_capi.ovrButton_Y,
    "LThumb": ovr_capi.ovrButton_LThumb,
    "LShoulder": ovr_capi.ovrButton_LThumb,
    "Up": ovr_capi.ovrButton_Up,
    "Down": ovr_capi.ovrButton_Down,
    "Left": ovr_capi.ovrButton_Left,
    "Right": ovr_capi.ovrButton_Right,
    "Enter": ovr_capi.ovrButton_Enter,
    "Back": ovr_capi.ovrButton_Back,
    "VolUp": ovr_capi.ovrButton_VolUp,
    "VolDown": ovr_capi.ovrButton_VolDown,
    "Home": ovr_capi.ovrButton_Home,
    "Private": ovr_capi.ovrButton_Private,
    "RMask": ovr_capi.ovrButton_RMask,
    "LMask": ovr_capi.ovrButton_LMask}

# Touch states
#
cdef dict _touch_states = {
    "A": ovr_capi.ovrTouch_A,
    "B": ovr_capi.ovrTouch_B,
    "RThumb": ovr_capi.ovrTouch_RThumb,
    "RThumbRest": ovr_capi.ovrTouch_RThumbRest,
    "RIndexTrigger": ovr_capi.ovrTouch_RThumb,
    "X": ovr_capi.ovrTouch_X,
    "Y": ovr_capi.ovrTouch_Y,
    "LThumb": ovr_capi.ovrTouch_LThumb,
    "LThumbRest": ovr_capi.ovrTouch_LThumbRest,
    "LIndexTrigger": ovr_capi.ovrTouch_LIndexTrigger,
    "RIndexPointing": ovr_capi.ovrTouch_RIndexPointing,
    "RThumbUp": ovr_capi.ovrTouch_RThumbUp,
    "LIndexPointing": ovr_capi.ovrTouch_LIndexPointing,
    "LThumbUp": ovr_capi.ovrTouch_LThumbUp}

# Controller types
#
cdef dict _controller_types = {
    'Xbox' : ovr_capi.ovrControllerType_XBox,
    'Remote' : ovr_capi.ovrControllerType_Remote,
    'Touch' : ovr_capi.ovrControllerType_Touch,
    'LeftTouch' : ovr_capi.ovrControllerType_LTouch,
    'RightTouch' : ovr_capi.ovrControllerType_RTouch}

# return success codes, values other than 'LIBOVR_SUCCESS' are conditional
LIBOVR_SUCCESS = ovr_capi.ovrSuccess
LIBOVR_SUCCESS_NOT_VISIBLE = ovr_capi.ovrSuccess_NotVisible
LIBOVR_SUCCESS_DEVICE_UNAVAILABLE = ovr_capi.ovrSuccess_DeviceUnavailable
LIBOVR_SUCCESS_BOUNDARY_INVALID = ovr_capi.ovrSuccess_BoundaryInvalid

# return error code
LIBOVR_ERROR_MEMORY_ALLOCATION_FAILURE = ovr_capi.ovrError_MemoryAllocationFailure
LIBOVR_ERROR_INVALID_SESSION = ovr_capi.ovrError_InvalidSession
LIBOVR_ERROR_TIMEOUT = ovr_capi.ovrError_Timeout
LIBOVR_ERROR_NOT_INITIALIZED = ovr_capi.ovrError_NotInitialized
LIBOVR_ERROR_INVALID_PARAMETER = ovr_capi.ovrError_InvalidParameter
LIBOVR_ERROR_SERVICE_ERROR = ovr_capi.ovrError_ServiceError
LIBOVR_ERROR_NO_HMD = ovr_capi.ovrError_NoHmd
LIBOVR_ERROR_UNSUPPORTED = ovr_capi.ovrError_Unsupported
LIBOVR_ERROR_TEXTURE_SWAP_CHAIN_FULL = ovr_capi.ovrError_TextureSwapChainFull


def isOculusServiceRunning(int timeout_ms=100):
    """Check if the Oculus Runtime is loaded and running.

    Parameters
    ----------
    timeout_ms : int
        Timeout in milliseconds.

    Returns
    -------
    bool

    """
    cdef ovr_capi.ovrDetectResult result = ovr_capi.ovr_Detect(
        timeout_ms)

    return <bint>result.IsOculusServiceRunning

def isHmdConnected(int timeout_ms=100):
    """Check if an HMD is connected.

    Parameters
    ----------
    timeout_ms : int
        Timeout in milliseconds.

    Returns
    -------
    bool

    """
    cdef ovr_capi.ovrDetectResult result = ovr_capi.ovr_Detect(
        timeout_ms)

    return <bint>result.IsOculusHMDConnected


def libovr_success(result):
    pass

def libovr_failure(result):
    pass

def libovr_unqualified_success(result):
    pass


cdef class LibOVRSession(object):
    """Session object for LibOVR.

    This class provides an interface for LibOVR sessions. Once initialized,
    the LibOVRSession instance provides configuration, data acquisition (e.g.
    sensors, inputs, etc.), and control of the HMD via attributes and methods.

    LibOVR API functions which return matrix and vector data types are converted
    to Numpy arrays.

    """
    # Session related pointers and information
    cdef ovr_capi.ovrInitParams initParams  # initialization parameters
    cdef ovr_capi.ovrSession ptrSession  # session pointer
    cdef ovr_capi.ovrGraphicsLuid ptrLuid  # LUID
    cdef ovr_capi.ovrHmdDesc hmdDesc  # HMD information descriptor
    cdef ovr_capi.ovrBoundaryLookAndFeel boundryStyle

    # VR related data persistent across frames
    cdef ovr_capi.ovrLayerEyeFov eyeLayer
    cdef ovr_capi.ovrEyeRenderDesc[2] eyeRenderDesc

    # texture swap chains, for eye views and mirror
    cdef ovr_capi.ovrTextureSwapChain[2] swapChains  # fixme!
    cdef ovr_capi.ovrMirrorTexture mirrorTexture

    # status and performance information
    cdef ovr_capi.ovrSessionStatus sessionStatus
    cdef ovr_capi.ovrPerfStats perfStats

    # tracking states

    # controller states

    # error information
    cdef ovr_capi.ovrErrorInfo errorInfo  # store our last error here

    # debug mode
    cdef bint debugMode

    # view objects
    cdef np.ndarray _viewport_left
    cdef np.ndarray _viewport_right

    def __init__(self, raiseErrors=False, timeout=100, *args, **kwargs):
        """Constructor for LibOVRSession.

        Parameters
        ----------
        raiseErrors : bool
            Raise exceptions when LibOVR functions return errors. If False,
            returned values of some methods will need to be checked for error
            conditions.
        timeout : int
            Connection timeout in milliseconds.

        """
        pass

    def __cinit__(self, bint raiseErrors=False, int timeout=100, *args, **kwargs):
        self.debugMode = raiseErrors
        self.ptrSession = NULL

        # view objects
        self._viewport_left = np.empty((4,), dtype=np.int)
        self._viewport_right = np.empty((4,), dtype=np.int)

        # check if the driver and service are available
        cdef ovr_capi.ovrDetectResult result = \
            ovr_capi.ovr_Detect(<int>timeout)

        if not result.IsOculusServiceRunning:
            raise RuntimeError("Oculus service is not running, it may be "
                               "disabled or not installed.")

        if not result.IsOculusHMDConnected:
            raise RuntimeError("No Oculus HMD connected! Check connections "
                               "and try again.")

    def __dealloc__(self):
        pass

    @property
    def userHeight(self):
        """User's calibrated height in meters.

        Getter
        ------
        float
            Distance from floor to the top of the user's head in meters reported
            by LibOVR. If not set, the default value is 1.778 meters.

        """
        cdef float to_return = ovr_capi.ovr_GetFloat(
            self.ptrSession,
            b"PlayerHeight",
            <float> 1.778)

        return to_return

    @property
    def eyeHeight(self):
        """Calibrated eye height from floor in meters.

        Getter
        ------
        float
            Distance from floor to the user's eye level in meters.

        """
        cdef float to_return = ovr_capi.ovr_GetFloat(
            self.ptrSession,
            b"EyeHeight",
            <float> 1.675)

        return to_return

    @property
    def neckEyeDist(self):
        """Distance from the neck to eyes in meters.

        Getter
        ------
        float
            Distance in meters.

        """
        cdef float vals[2]

        cdef unsigned int ret = ovr_capi.ovr_GetFloatArray(
            self.ptrSession,
            b"NeckEyeDistance",
            vals,
            <unsigned int>2)

        return <float> vals[0], <float> vals[1]

    @property
    def eyeToNoseDist(self):
        """Distance between the nose and eyes in meters.

        Getter
        ------
        float
            Distance in meters.

        """
        cdef float vals[2]

        cdef unsigned int ret = ovr_capi.ovr_GetFloatArray(
            self.ptrSession,
            b"EyeToNoseDist",
            vals,
            <unsigned int> 2)

        return <float>vals[0], <float> vals[1]

    @property
    def productName(self):
        """Get the product name for this device.

        Getter
        ------
        str
            Product name string (utf-8).

        """
        return self.hmdDesc.ProductName.decode('utf-8')

    @property
    def manufacturerName(self):
        """Get the device manufacturer name.

        Getter
        ------
        str
            Manufacturer name string (utf-8).

        """
        return self.hmdDesc.Manufacturer.decode('utf-8')

    @property
    def screenSize(self):
        """Horizontal and vertical resolution of the display in pixels.

        Getter
        ------
        ndarray of int
            Resolution of the display [w, h].

        """
        return np.asarray(
            (self.hmdDesc.Resolution.w, self.hmdDesc.Resolution.h),
            dtype=int)

    @property
    def refreshRate(self):
        """Nominal refresh rate in Hertz of the display.

        Getter
        ------
        float
            Refresh rate in Hz.

        """
        return <float>self.hmdDesc.DisplayRefreshRate

    @property
    def hid(self):
        """USB human interface device class identifiers.

        Getter
        ------
        tuple
            USB HIDs (vendor, product).

        """
        return <int>self.hmdDesc.VendorId, <int>self.hmdDesc.ProductId

    @property
    def firmwareVersion(self):
        """Firmware version for this device.

        Getter
        ------
        tuple
            Firmware version (major, minor).

        """
        return <int>self.hmdDesc.FirmwareMajor, <int>self.hmdDesc.FirmwareMinor

    @property
    def versionString(self):
        """LibOVRRT version as a string.

        Getter
        ------
        str
            Runtime version information as a UTF-8 encoded string.

        """
        cdef const char* version = ovr_capi.ovr_GetVersionString()
        return version.decode('utf-8')  # already UTF-8?

    def initialize(self, bint focusAware=False, int connectionTimeout=0):
        """Initialize the session.

        Parameters
        ----------
        focusAware : bool
            Client is focus aware.
        connectionTimeout : bool
            Timeout in milliseconds for connecting to the server.

        Returns
        -------
        int
            Return code of the LibOVR API call 'ovr_Initialize'. Returns
            LIBOVR_SUCCESS if completed without errors. In the event of an
            error, possible return values are:

            - :data:`LIBOVR_ERROR_INITIALIZE`: Initialization error.
            - :data:`LIBOVR_ERROR_LIB_LOAD`:  Failed to load LibOVRRT.
            - :data:`LIBOVR_ERROR_LIB_VERSION`:  LibOVRRT version incompatible.
            - :data:`LIBOVR_ERROR_SERVICE_CONNECTION`:  Cannot connect to OVR service.
            - :data:`LIBOVR_ERROR_SERVICE_VERSION`: OVR service version is incompatible.
            - :data:`LIBOVR_ERROR_INCOMPATIBLE_OS`: Operating system version is incompatible.
            - :data:`LIBOVR_ERROR_DISPLAY_INIT`: Unable to initialize the HMD.
            - :data:`LIBOVR_ERROR_SERVER_START`:  Cannot start a server.
            - :data:`LIBOVR_ERROR_REINITIALIZATION`: Reinitialized with a different version.

        Raises
        ------
        RuntimeError
            Raised if 'debugMode' is True and the API call to
            'ovr_Initialize' returns an error.

        """
        cdef int32_t flags = ovr_capi.ovrInit_RequestVersion
        if focusAware is True:
            flags |= ovr_capi.ovrInit_FocusAware

        #if debug is True:
        #    flags |= ovr_capi.ovrInit_Debug

        self.initParams.Flags = flags
        self.initParams.RequestedMinorVersion = 32
        self.initParams.LogCallback = NULL  # not used yet
        self.initParams.ConnectionTimeoutMS = <uint32_t>connectionTimeout
        cdef ovr_capi.ovrResult result = ovr_capi.ovr_Initialize(
            &self.initParams)

        if self.debugMode:
            check_result(result)

        return result  # failed to initalize, return error code

    def create(self):
        """Create a new session. Control is handed over to the application from
        Oculus Home.

        Starting a session will initialize and create a new session. Afterwards
        API functions will return valid values.

        """
        result = ovr_capi.ovr_Create(&self.ptrSession, &self.ptrLuid)
        check_result(result)
        if ovr_capi.OVR_FAILURE(result):
            return result  # failed to create session, return error code

        # if we got to this point, everything should be fine
        # get HMD descriptor
        self.hmdDesc = ovr_capi.ovr_GetHmdDesc(self.ptrSession)

        # configure the eye render descriptor to use the recommended FOV, this
        # can be changed later
        cdef Py_ssize_t i = 0
        for i in range(ovr_capi.ovrEye_Count):
            self.eyeRenderDesc[i] = ovr_capi.ovr_GetRenderDesc(
                self.ptrSession,
                <ovr_capi.ovrEyeType>i,
                self.hmdDesc.DefaultEyeFov[i])

            self.eyeLayer.Fov[i] = self.eyeRenderDesc[i].Fov

        # prepare the render layer
        self.eyeLayer.Header.Type = ovr_capi.ovrLayerType_EyeFov
        self.eyeLayer.Header.Flags = \
            ovr_capi.ovrLayerFlag_TextureOriginAtBottomLeft | \
            ovr_capi.ovrLayerFlag_HighQuality
        self.eyeLayer.ColorTexture[0] = self.eyeLayer.ColorTexture[1] = NULL

        return result

    def destroy(self):
        """Destroy all resources associated with this session.

        """
        # free all swap chains
        cdef int i = 0
        for i in range(32):
            ovr_capi.ovr_DestroyTextureSwapChain(
                self.ptrSession, self.swapChains[i])
            self.swapChains[i] = NULL

        # null eye textures in eye layer
        self.eyeLayer.ColorTexture[0] = self.eyeLayer.ColorTexture[1] = NULL

        # destroy the mirror texture
        if self.mirrorTexture is NULL:
            ovr_capi.ovr_DestroyMirrorTexture(self.ptrSession, self.mirrorTexture)

        # destroy the current session and shutdown
        ovr_capi.ovr_Destroy(self.ptrSession)

    def shutdown(self):
        """End the current session.

        Clean-up routines are executed that destroy all swap chains and mirror
        texture buffers, afterwards control is returned to Oculus Home. This
        must be called after every successful 'initialize' call.

        """
        ovr_capi.ovr_Shutdown()

    @property
    def debugMode(self):
        """Enable session debugging. Python exceptions are raised if the LibOVR
        API returns an error. If 'debugMode=False', API errors will be silently
        ignored.

        """
        return self.debugMode

    @debugMode.setter
    def debugMode(self, value):
        self.debugMode = value

    @property
    def highQuality(self):
        """True when high quality mode is enabled.

        The distortion compositor applies 4x anisotropic texture filtering which
        reduces the visibility of artifacts, particularly in the periphery.

        This is enabled by default when a session is started.

        """
        return (self.eyeLayer.Header.Flags &
                ovr_capi.ovrLayerFlag_HighQuality) == \
               ovr_capi.ovrLayerFlag_HighQuality

    @highQuality.setter
    def highQuality(self, value):
        if value:
            self.eyeLayer.Header.Flags |= ovr_capi.ovrLayerFlag_HighQuality
        else:
            self.eyeLayer.Header.Flags &= ~ovr_capi.ovrLayerFlag_HighQuality

    @property
    def headLocked(self):
        """True when head-locked mode is enabled.

        This is disabled by default when a session is started. Head locking
        places the rendered image as a 'billboard' in front of the viewer.

        """
        return (self.eyeLayer.Header.Flags &
                ovr_capi.ovrLayerFlag_HeadLocked) == \
               ovr_capi.ovrLayerFlag_HeadLocked

    @headLocked.setter
    def headLocked(self, value):
        if value:
            self.eyeLayer.Header.Flags |= ovr_capi.ovrLayerFlag_HeadLocked
        else:
            self.eyeLayer.Header.Flags &= ~ovr_capi.ovrLayerFlag_HeadLocked

    @property
    def trackerCount(self):
        """Number of connected trackers."""
        cdef unsigned int trackerCount = ovr_capi.ovr_GetTrackerCount(
            self.ptrSession)

        return <int>trackerCount

    @property
    def defaultEyeFOVs(self):
        """Default or recommended eye field-of-views (FOVs) provided by the API.

        Returns
        -------
        tuple of ndarray
            Pair of left and right eye FOVs specified as tangent angles [Up,
            Down, Left, Right].

        """
        cdef np.ndarray fovLeft = np.asarray([
            self.hmdDesc.DefaultEyeFov[0].UpTan,
            self.hmdDesc.DefaultEyeFov[0].DownTan,
            self.hmdDesc.DefaultEyeFov[0].LeftTan,
            self.hmdDesc.DefaultEyeFov[0].RightTan],
            dtype=np.float32)

        cdef np.ndarray fovRight = np.asarray([
            self.hmdDesc.DefaultEyeFov[1].UpTan,
            self.hmdDesc.DefaultEyeFov[1].DownTan,
            self.hmdDesc.DefaultEyeFov[1].LeftTan,
            self.hmdDesc.DefaultEyeFov[1].RightTan],
            dtype=np.float32)

        return fovLeft, fovRight

    @property
    def maxEyeFOVs(self):
        """Maximum eye field-of-views (FOVs) provided by the API.

        Returns
        -------
        tuple of ndarray
            Pair of left and right eye FOVs specified as tangent angles in
            radians [Up, Down, Left, Right].

        """
        cdef np.ndarray[float, ndim=1] fov_left = np.asarray([
            self.hmdDesc.MaxEyeFov[0].UpTan,
            self.hmdDesc.MaxEyeFov[0].DownTan,
            self.hmdDesc.MaxEyeFov[0].LeftTan,
            self.hmdDesc.MaxEyeFov[0].RightTan],
            dtype=np.float32)

        cdef np.ndarray[float, ndim=1] fov_right = np.asarray([
            self.hmdDesc.MaxEyeFov[1].UpTan,
            self.hmdDesc.MaxEyeFov[1].DownTan,
            self.hmdDesc.MaxEyeFov[1].LeftTan,
            self.hmdDesc.MaxEyeFov[1].RightTan],
            dtype=np.float32)

        return fov_left, fov_right

    @property
    def symmetricEyeFOVs(self):
        """Symmetric field-of-views (FOVs) for mono rendering.

        By default, the Rift uses off-axis FOVs. These frustum parameters make
        it difficult to converge monoscopic stimuli.

        Returns
        -------
        tuple of ndarray of float
            Pair of left and right eye FOVs specified as tangent angles in
            radians [Up, Down, Left, Right]. Both FOV objects will have the same
            values.

        """
        cdef ovr_capi.ovrFovPort fov_left = self.hmdDesc.DefaultEyeFov[0]
        cdef ovr_capi.ovrFovPort fov_right = self.hmdDesc.DefaultEyeFov[1]

        cdef ovr_capi.ovrFovPort fov_max
        fov_max.UpTan = maxf(fov_left.UpTan, fov_right.Uptan)
        fov_max.DownTan = maxf(fov_left.DownTan, fov_right.DownTan)
        fov_max.LeftTan = maxf(fov_left.LeftTan, fov_right.LeftTan)
        fov_max.RightTan = maxf(fov_left.RightTan, fov_right.RightTan)

        cdef float tan_half_fov_horz = maxf(fov_max.LeftTan, fov_max.RightTan)
        cdef float tan_half_fov_vert = maxf(fov_max.DownTan, fov_max.UpTan)

        cdef ovr_capi.ovrFovPort fov_both
        fov_both.LeftTan = fov_both.RightTan = tan_half_fov_horz
        fov_both.UpTan = fov_both.DownTan = tan_half_fov_horz

        cdef np.ndarray[float, ndim=1] fov_left_out = np.asarray([
            fov_both.UpTan,
            fov_both.DownTan,
            fov_both.LeftTan,
            fov_both.RightTan],
            dtype=np.float32)

        cdef np.ndarray[float, ndim=1] fov_right_out = np.asarray([
            fov_both.UpTan,
            fov_both.DownTan,
            fov_both.LeftTan,
            fov_both.RightTan],
            dtype=np.float32)

        return fov_left_out, fov_right_out

    @property
    def eyeRenderFOVs(self):
        """Field-of-view to use for rendering.

        The FOV for a given eye are defined as a tuple of tangent angles (Up,
        Down, Left, Right). By default, this function will return the default
        FOVs after 'start' is called (see 'defaultEyeFOVs'). You can override
        these values using 'maxEyeFOVs' and 'symmetricEyeFOVs', or with
        custom values (see Examples below).

        Examples
        --------
        Setting eye render FOVs to symmetric (needed for mono rendering)::

            hmd.eye_render_fovs = hmd.symmetric_eye_fovs

        Getting the tangent angles::

            left_fov, right_fov = hmd.eye_render_fovs
            # left FOV tangent angles, do the same for the right
            up_tan, down_tan, left_tan, right_tan =  left_fov

        Using custom values::

            # Up, Down, Left, Right tan angles
            left_fov = [1.0, -1.0, -1.0, 1.0]
            right_fov = [1.0, -1.0, -1.0, 1.0]
            hmd.eye_render_fovs = left_fov, right_fov

        """
        cdef np.ndarray left_fov = np.asarray([
            self.eyeRenderDesc[0].Fov.UpTan,
            self.eyeRenderDesc[0].Fov.DownTan,
            self.eyeRenderDesc[0].Fov.LeftTan,
            self.eyeRenderDesc[0].Fov.RightTan],
            dtype=np.float32)

        cdef np.ndarray right_fov = np.asarray([
            self.eyeRenderDesc[1].Fov.UpTan,
            self.eyeRenderDesc[1].Fov.DownTan,
            self.eyeRenderDesc[1].Fov.LeftTan,
            self.eyeRenderDesc[1].Fov.RightTan],
            dtype=np.float32)

        return left_fov, right_fov

    @eyeRenderFOVs.setter
    def eyeRenderFOVs(self, object value):
        cdef ovr_capi.ovrFovPort fov_in
        cdef int i = 0
        for i in range(ovr_capi.ovrEye_Count):
            fov_in.UpTan = <float>value[i][0]
            fov_in.DownTan = <float>value[i][1]
            fov_in.LeftTan = <float>value[i][2]
            fov_in.RightTan = <float>value[i][3]

            self.eyeRenderDesc[i] = ovr_capi.ovr_GetRenderDesc(
                self.ptrSession,
                <ovr_capi.ovrEyeType>i,
                fov_in)

            self.eyeLayer.Fov[i] = self.eyeRenderDesc[i].Fov

    def getEyeRenderFOV(self, int eye):
        """Get the field-of-view of a given eye used to compute the projection
        matrix.

        Returns
        -------
        tuple of ndarray
            Eye FOVs specified as tangent angles [Up, Down, Left, Right].

        """
        cdef np.ndarray to_return = np.asarray([
            self.eyeRenderDesc[eye].Fov.UpTan,
            self.eyeRenderDesc[eye].Fov.DownTan,
            self.eyeRenderDesc[eye].Fov.LeftTan,
            self.eyeRenderDesc[eye].Fov.RightTan],
            dtype=np.float32)

        return to_return

    def setEyeRenderFOV(self, int eye, object fov):
        """Set the field-of-view of a given eye. This is used to compute the
        projection matrix.

        Parameters
        ----------
        eye : int
            Eye index.
        fov : tuple, list or ndarray of floats
        texelPerPixel : float

        """
        cdef ovr_capi.ovrFovPort fov_in
        fov_in.UpTan = <float>fov[0]
        fov_in.DownTan = <float>fov[1]
        fov_in.LeftTan = <float>fov[2]
        fov_in.RightTan = <float>fov[3]

        self.eyeRenderDesc[<int>eye] = ovr_capi.ovr_GetRenderDesc(
            self.ptrSession,
            <ovr_capi.ovrEyeType>eye,
            fov_in)

        # set in eye layer too
        self.eyeLayer.Fov[eye] = self.eyeRenderDesc[eye].Fov

    def calcEyeBufferSizes(self, texelsPerPixel=1.0):
        """Get the recommended buffer (texture) sizes for eye buffers.

        Should be called after 'eye_render_fovs' is set. Returns left and
        right buffer resolutions (w, h). The values can be used when configuring
        a framebuffer for rendering to the HMD eye buffers.

        Parameters
        ----------
        texelsPerPixel : float
            Display pixels per texture pixels at the center of the display.
            Use a value less than 1.0 to improve performance at the cost of
            resolution. Specifying a larger texture is possible, but not
            recommended by the manufacturer.

        Returns
        -------
        tuple of tuples
            Buffer widths and heights (w, h) for each eye.

        Examples
        --------
        Getting the buffer sizes::

            hmd.eyeRenderFOVs = hmd.defaultEyeFOVs  # set the FOV
            leftBufferSize, rightBufferSize = hmd.calcEyeBufferSizes()
            left_w, left_h = leftBufferSize
            right_w, right_h = rightBufferSize
            # combined size if using a single texture buffer for both eyes
            w, h = left_w + right_w, max(left_h, right_h)
            # make the texture ...

        Notes
        -----
        This function returns the recommended texture resolution for each eye.
        If you are using a single buffer for both eyes, that buffer should be
        as wide as the combined width of both returned size.

        """
        cdef ovr_capi.ovrSizei sizeLeft = ovr_capi.ovr_GetFovTextureSize(
            self.ptrSession,
            <ovr_capi.ovrEyeType>0,
            self.eyeRenderDesc[0].Fov,
            <float>texelsPerPixel)

        cdef ovr_capi.ovrSizei sizeRight = ovr_capi.ovr_GetFovTextureSize(
            self.ptrSession,
            <ovr_capi.ovrEyeType>1,
            self.eyeRenderDesc[1].Fov,
            <float>texelsPerPixel)

        return (sizeLeft.w, sizeLeft.h), (sizeRight.w, sizeRight.h)

    def getSwapChainLengthGL(self, eye):
        """Get the swap chain length for a given eye."""
        cdef int out_length
        cdef ovr_capi.ovrResult result = 0

        # check if there is a swap chain in the slot
        if self.eyeLayer.ColorTexture[eye] == NULL:
            raise RuntimeError(
                "Cannot get swap chain length, NULL eye buffer texture.")

        # get the current texture index within the swap chain
        result = ovr_capi.ovr_GetTextureSwapChainLength(
            self.ptrSession, self.swapChains[eye], &out_length)

        return out_length

    def getSwapChainCurrentIndex(self, eye):
        """Get the current index of the swap chain for a given eye."""
        cdef int current_idx = 0
        cdef ovr_capi.ovrResult result = 0

        # check if there is a swap chain in the slot
        if self.eyeLayer.ColorTexture[eye] == NULL:
            raise RuntimeError(
                "Cannot get buffer ID, NULL eye buffer texture.")

        # get the current texture index within the swap chain
        result = ovr_capi.ovr_GetTextureSwapChainCurrentIndex(
            self.ptrSession, self.swapChains[eye], &current_idx)

        return current_idx

    def getTextureBufferGL(self, eye, index):
        """Get the texture buffer as an OpenGL name at a specific index in the
        swap chain for a given eye.

        """
        cdef unsigned int tex_id = 0
        cdef ovr_capi.ovrResult result = 0

        # get the next available texture ID from the swap chain
        result = ovr_capi.ovr_GetTextureSwapChainBufferGL(
            self.ptrSession, self.swapChains[eye], index, &tex_id)

        return tex_id

    def getNextTextureBufferGL(self, eye):
        """Get the next available texture buffer as an OpenGL name in the swap
        chain for a given eye.

        Calling this automatically handles getting the next available swap chain
        index. The index is incremented when 'commit_swap_chain' is called.

        Parameters
        ----------
        eye : int
            Swap chain belonging to a given eye to get the texture ID.

        Returns
        -------
        int
            OpenGL texture handle.

        """
        cdef int current_idx = 0
        cdef unsigned int tex_id = 0
        cdef ovr_capi.ovrResult result = 0

        # check if there is a swap chain in the slot
        if self.eyeLayer.ColorTexture[eye] == NULL:
            raise RuntimeError(
                "Cannot get buffer ID, NULL eye buffer texture.")

        # get the current texture index within the swap chain
        result = ovr_capi.ovr_GetTextureSwapChainCurrentIndex(
            self.ptrSession, self.swapChains[eye], &current_idx)

        if self.debugMode:
            check_result(result)

        # get the next available texture ID from the swap chain
        result = ovr_capi.ovr_GetTextureSwapChainBufferGL(
            self.ptrSession, self.swapChains[eye], current_idx, &tex_id)

        if self.debugMode:
            check_result(result)

        return tex_id

    def createTextureSwapChainGL(self, eye, width, height, textureFormat='R8G8B8A8_UNORM_SRGB', levels=1):
        """Initialize an texture swap chain for eye images.

        Parameters
        ----------
        eye : int
            Eye index to initialize.
        textureFormat : str
            Texture format, valid texture formats are 'R8G8B8A8_UNORM',
            'R8G8B8A8_UNORM_SRGB', 'R16G16B16A16_FLOAT', and 'R11G11B10_FLOAT'.
        width : int
            Width of texture in pixels.
        height : int
            Height of texture in pixels.
        levels : int
            Mip levels to use, default is 1.

        """
        # configure the texture
        cdef ovr_capi.ovrTextureSwapChainDesc swapConfig
        swapConfig.Type = ovr_capi.ovrTexture_2D
        swapConfig.Format = _supported_texture_formats[textureFormat]
        swapConfig.ArraySize = 1
        swapConfig.Width = <int>width
        swapConfig.Height = <int>height
        swapConfig.MipLevels = <int>levels
        swapConfig.SampleCount = 1
        swapConfig.StaticImage = ovr_capi.ovrFalse
        swapConfig.MiscFlags = ovr_capi.ovrTextureMisc_None
        swapConfig.BindFlags = ovr_capi.ovrTextureBind_None

        # create the swap chain
        cdef ovr_capi.ovrResult result = \
            ovr_capi.ovr_CreateTextureSwapChainGL(
                self.ptrSession,
                &swapConfig,
                &self.swapChains[eye])

        self.eyeLayer.ColorTexture[eye] = self.swapChains[eye]

    def createMirrorTexture(
            self,
            width,
            height,
            texture_format='R8G8B8A8_UNORM_SRGB'):
        """Create a mirror texture displaying the contents of the rendered
        images being presented on the HMD. The image is automatically refreshed
        to reflect the current content on the display. This displays the
        post-distortion texture.

        Parameters
        ----------
        width : int
            Width of texture in pixels.
        height : int
            Height of texture in pixels.
        texture_format : str
            Texture format. Valid texture formats are: 'R8G8B8A8_UNORM',
            'R8G8B8A8_UNORM_SRGB', 'R16G16B16A16_FLOAT', and 'R11G11B10_FLOAT'.

        """
        # additional options
        #cdef unsigned int mirror_options = ovr_capi.ovrMirrorOption_Default
        # set the mirror texture mode
        #if mirrorMode == 'Default':
        #    mirror_options = <ovr_capi.ovrMirrorOptions>ovr_capi.ovrMirrorOption_Default
        #elif mirrorMode == 'PostDistortion':
        #    mirror_options = <ovr_capi.ovrMirrorOptions>ovr_capi.ovrMirrorOption_PostDistortion
        #elif mirrorMode == 'LeftEyeOnly':
        #    mirror_options = <ovr_capi.ovrMirrorOptions>ovr_capi.ovrMirrorOption_LeftEyeOnly
        #elif mirrorMode == 'RightEyeOnly':
        #    mirror_options = <ovr_capi.ovrMirrorOptions>ovr_capi.ovrMirrorOption_RightEyeOnly
        #else:
        #    raise RuntimeError("Invalid 'mirrorMode' mode specified.")

        #if include_guardian:
        #    mirror_options |= ovr_capi.ovrMirrorOption_IncludeGuardian
        #if include_notifications:
        #    mirror_options |= ovr_capi.ovrMirrorOption_IncludeNotifications
        #if include_system_gui:
        #    mirror_options |= ovr_capi.ovrMirrorOption_IncludeSystemGui

        # create the descriptor
        cdef ovr_capi.ovrMirrorTextureDesc mirrorDesc

        mirrorDesc.Format = ovr_capi.OVR_FORMAT_R8G8B8A8_UNORM_SRGB
        mirrorDesc.Width = <int>width
        mirrorDesc.Height = <int>height
        mirrorDesc.MiscFlags = ovr_capi.ovrTextureMisc_None
        mirrorDesc.MirrorOptions = ovr_capi.ovrMirrorOption_Default

        cdef ovr_capi.ovrResult result = ovr_capi.ovr_CreateMirrorTextureGL(
            self.ptrSession, &mirrorDesc, &self.mirrorTexture)

        if self.debugMode:
            check_result(result)

    @property
    def mirrorTexture(self):
        """Mirror texture ID."""
        cdef unsigned int mirror_id

        if self.mirrorTexture is NULL:  # no texture created
            return None

        cdef ovr_capi.ovrResult result = \
            ovr_capi.ovr_GetMirrorTextureBufferGL(
                self.ptrSession,
                self.mirrorTexture,
                &mirror_id)

        return <unsigned int>mirror_id

    def getPoses(self, double abs_time, bint latency_marker=True):
        """Get the current poses of the head and hands.

        Parameters
        ----------
        abs_time : float
            Absolute time in seconds which the tracking state refers to.
        latency_marker : bool
            Insert a latency marker for motion-to-photon calculation.

        Returns
        -------
        tuple of LibOVRPoseState
            Pose state for the head, left and right hands.

        Examples
        --------
        Getting the head pose and calculating eye render poses::

            t = hmd.get_predicted_display_time()
            head, left_hand, right_hand = hmd.get_poses(t)

            # check if tracking
            if head.orientation_tracked and head.position_tracked:
                hmd.calc_eye_poses(head)  # calculate eye poses

        """
        cdef ovr_capi.ovrBool use_marker = \
            ovr_capi.ovrTrue if latency_marker else ovr_capi.ovrFalse

        cdef ovr_capi.ovrTrackingState tracking_state = \
            ovr_capi.ovr_GetTrackingState(self.ptrSession, abs_time, use_marker)

        cdef LibOVRPoseState head_pose = LibOVRPoseState()
        head_pose.c_data[0] = tracking_state.HeadPose
        head_pose.status_flags = tracking_state.StatusFlags

        # for computing app photon-to-motion latency
        self.eyeLayer.SensorSampleTime = tracking_state.HeadPose.TimeInSeconds

        cdef LibOVRPoseState left_hand_pose = LibOVRPoseState()
        left_hand_pose.c_data[0] = tracking_state.HandPoses[0]
        left_hand_pose.status_flags = tracking_state.HandStatusFlags[0]

        cdef LibOVRPoseState right_hand_pose = LibOVRPoseState()
        right_hand_pose.c_data[0] = tracking_state.HandPoses[1]
        right_hand_pose.status_flags = tracking_state.HandStatusFlags[1]

        return head_pose, left_hand_pose, right_hand_pose

    def calcEyePoses(self, LibOVRPoseState head_pose):
        """Calcuate eye poses using a given pose state.

        Eye poses are derived from the head pose stored in the pose state and
        the HMD to eye poses reported by LibOVR. Calculated eye poses are stored
        and passed to the compositor when 'end_frame' is called for additional
        rendering.

        You can access the computed poses via the 'render_poses' attribute.

        """
        cdef ovr_capi.ovrPosef[2] hmd_to_eye_poses
        hmd_to_eye_poses[0] = self.eyeRenderDesc[0].HmdToEyePose
        hmd_to_eye_poses[1] = self.eyeRenderDesc[1].HmdToEyePose

        ovr_capi.ovr_CalcEyePoses2(
            head_pose.c_data[0].ThePose,
            hmd_to_eye_poses,
            self.eyeLayer.RenderPose)

    @property
    def hmdToEyePoses(self):
        """HMD to eye poses (`tuple` of `LibOVRPose`).

        These are the prototype eye poses specified by LibOVR, defined only
        after 'start' is called. These poses are transformed by the head pose
        by 'calcEyePoses' to get 'eyeRenderPoses'.

        Notes
        -----
            The horizontal (x-axis) separation of the eye poses are determined
            by the configured lens spacing (slider adjustment). This spacing is
            supposed to correspond to the actual inter-ocular distance (IOD) of
            the user. You can get the IOD used for rendering by adding up the
            absolute values of the x-components of the eye poses, or by
            multiplying the value of 'eyeToNoseDist' by two. Furthermore, the
            IOD values can be altered, prior to calling 'calcEyePoses', to
            override the values specified by LibOVR.

        Returns
        -------
        tuple of LibOVRPose
            Copies of the HMD to eye poses for the left and right eye.

        """
        cdef LibOVRPose leftHmdToEyePose = LibOVRPose()
        cdef LibOVRPose rightHmdToEyePose = LibOVRPose()

        leftHmdToEyePose.c_data[0] = self.eyeRenderDesc[0].HmdToEyePose
        leftHmdToEyePose.c_data[1] = self.eyeRenderDesc[1].HmdToEyePose

        return leftHmdToEyePose, rightHmdToEyePose

    @hmdToEyePoses.setter
    def hmdToEyePoses(self, value):
        self.eyeRenderDesc[0].HmdToEyePose = (<LibOVRPose>value[0]).c_data[0]
        self.eyeRenderDesc[1].HmdToEyePose = (<LibOVRPose>value[1]).c_data[1]

    @property
    def renderPoses(self):
        """Eye render poses.

        Pose are those computed by the last 'calcEyePoses' call. Returned
        objects are copies of the data stored internally by the session
        instance. These poses are used to define the view matrix when rendering
        for each eye.

        Notes
        -----
            The returned LibOVRPose objects reference data stored in the session
            instance. Changing their values will immediately update render
            poses.

        """
        cdef LibOVRPose left_eye_pose = LibOVRPose()
        cdef LibOVRPose right_eye_pose = LibOVRPose()

        left_eye_pose.c_data = &self.eyeLayer.RenderPose[0]
        right_eye_pose.c_data = &self.eyeLayer.RenderPose[1]

        return left_eye_pose, right_eye_pose

    @renderPoses.setter
    def renderPoses(self, object value):
        self.eyeLayer.RenderPose[0] = (<LibOVRPose>value[0]).c_data[0]
        self.eyeLayer.RenderPose[1] = (<LibOVRPose>value[1]).c_data[1]

    def getMirrorTexture(self):
        """Get the mirror texture handle.

        Returns
        -------
        int
            OpenGL texture handle.

        """
        cdef unsigned int mirror_id
        cdef ovr_capi.ovrResult result = \
            ovr_capi.ovr_GetMirrorTextureBufferGL(
                self.ptrSession,
                self.mirrorTexture,
                &mirror_id)

        return <unsigned int> mirror_id

    def getTextureSwapChainBufferGL(self, int eye):
        """Get the next available swap chain buffer for a specified eye.

        Parameters
        ----------
        eye : int
            Swap chain belonging to a given eye to get the texture ID.

        Returns
        -------
        int
            OpenGL texture handle.

        """
        cdef int current_idx = 0
        cdef unsigned int tex_id = 0
        cdef ovr_capi.ovrResult result = 0

        # check if there is a swap chain in the slot
        if self.eyeLayer.ColorTexture[eye] == NULL:
            raise RuntimeError(
                "Cannot get buffer ID, NULL eye buffer texture.")

        # get the current texture index within the swap chain
        result = ovr_capi.ovr_GetTextureSwapChainCurrentIndex(
            self.ptrSession, self.swapChains[eye], &current_idx)

        if self.debugMode:
            check_result(result)

        # get the next available texture ID from the swap chain
        result = ovr_capi.ovr_GetTextureSwapChainBufferGL(
            self.ptrSession, self.swapChains[eye], current_idx, &tex_id)

        if self.debugMode:
            check_result(result)

        return tex_id

    def getEyeProjectionMatrix(self, int eye, float nearClip=0.1, float farClip=1000.0):
        """Compute the projection matrix.

        The projection matrix is computed by the runtime using the eye FOV
        parameters set with '~ovr.LibOVRSession.setEyeRenderFov' calls.

        Parameters
        ----------
        eye : int
            Eye index.
        nearClip : float
            Near clipping plane in meters.
        farClip : float
            Far clipping plane in meters.

        Returns
        -------
        ndarray of floats
            4x4 projection matrix.

        """
        cdef ovr_capi.ovrMatrix4f projMat = \
            ovr_capi.ovrMatrix4f_Projection(
                self.eyeRenderDesc[eye].Fov,
                nearClip,
                farClip,
                ovr_capi.ovrProjection_ClipRangeOpenGL)

        cdef np.ndarray to_return = np.zeros((4, 4), dtype=np.float32)

        # fast copy matrix to numpy array
        cdef float [:, :] mv = to_return
        cdef Py_ssize_t i, j
        i = j = 0
        for i in range(4):
            for j in range(4):
                mv[i, j] = projMat.M[i][j]

        return to_return

    @property
    def eyeRenderViewports(self):
        """Eye viewports."""
        self._viewport_left.data = <char*>&self.eyeLayer.Viewport[0]
        self._viewport_right.data = <char*>&self.eyeLayer.Viewport[1]

        return self._viewport_left, self._viewport_right

    @eyeRenderViewports.setter
    def eyeRenderViewports(self, object values):
        cdef int i = 0
        for i in range(ovr_capi.ovrEye_Count):
            self.eyeLayer.Viewport[i].Pos.x = <int>values[i][0]
            self.eyeLayer.Viewport[i].Pos.y = <int>values[i][1]
            self.eyeLayer.Viewport[i].Size.w = <int>values[i][2]
            self.eyeLayer.Viewport[i].Size.h = <int>values[i][3]

    def getEyeViewMatrix(self, int eye, bint flatten=False):
        """Compute a view matrix for a specified eye.

        View matrices are derived from the eye render poses calculated by the
        last 'calcEyePoses' call.

        Parameters
        ----------
        eye : int
            Eye index.
        flatten : bool
            Flatten the matrix into a 1D vector. This will create an array
            suitable for use with OpenGL functions accepting column-major, 4x4
            matrices as a length 16 vector of floats.

        Returns
        -------
        ndarray
            4x4 view matrix (16x1 if flatten=True).

        """
        cdef ovr_math.Vector3f pos = \
            <ovr_math.Vector3f>self.eyeLayer.RenderPose[eye].Position
        cdef ovr_math.Quatf ori = \
            <ovr_math.Quatf>self.eyeLayer.RenderPose[eye].Orientation

        if not ori.IsNormalized():  # make sure orientation is normalized
            ori.Normalize()

        cdef ovr_math.Matrix4f rm = ovr_math.Matrix4f(ori)
        cdef ovr_math.Vector3f up = \
            rm.Transform(ovr_math.Vector3f(0., 1., 0.))
        cdef ovr_math.Vector3f forward = \
            rm.Transform(ovr_math.Vector3f(0., 0., -1.))
        cdef ovr_math.Matrix4f view_mat = ovr_math.Matrix4f.LookAtRH(
            pos, pos + forward, up)

        # output array
        cdef np.ndarray to_return
        cdef Py_ssize_t i, j, k, N
        i = j = k = 0
        N = 4
        if flatten:
            to_return = np.zeros((16,), dtype=np.float32)
            for i in range(N):
                for j in range(N):
                    to_return[k] = view_mat.M[j][i]
                    k += 1
        else:
            to_return = np.zeros((4, 4), dtype=np.float32)
            for i in range(N):
                for j in range(N):
                    to_return[i, j] = view_mat.M[i][j]

        return to_return

    def getPredictedDisplayTime(self, unsigned int frame_index=0):
        """Get the predicted time a frame will be displayed.

        Parameters
        ----------
        frame_index : int
            Frame index.

        Returns
        -------
        float
            Absolute frame mid-point time for the given frame index in seconds.

        """
        cdef double t_sec = ovr_capi.ovr_GetPredictedDisplayTime(
            self.ptrSession,
            frame_index)

        return t_sec

    @property
    def timeInSeconds(self):
        """Absolute time in seconds.

        Returns
        -------
        float
            Time in seconds.

        """
        cdef double t_sec = ovr_capi.ovr_GetTimeInSeconds()

        return t_sec

    def perfHudMode(self, str mode):
        """Display a performance information HUD.
        
        Parameters
        ----------
        mode : str
            Performance HUD mode to present. Valid mode strings are:
            'PerfSummary', 'LatencyTiming', 'AppRenderTiming', 
            'CompRenderTiming', 'AswStats', 'VersionInfo' and 'Off'. Specifying 
            'Off' hides the performance HUD.
            
        Warning
        -------
        The performance HUD remains visible until 'Off' is specified, even after
        the application quits.
        
        """
        cdef int perfHudMode = 0

        try:
            perfHudMode = <int>_performance_hud_modes[mode]
        except KeyError:
            raise KeyError("Invalid performance HUD mode specified.")

        cdef ovr_capi.ovrBool ret = ovr_capi.ovr_SetInt(
            self.ptrSession, b"PerfHudMode", perfHudMode)

    def hidePerfHud(self):
        """Hide the performance HUD.

        This is a convenience function that is equivalent to calling
        'perf_hud_mode('Off').

        """
        cdef ovr_capi.ovrBool ret = ovr_capi.ovr_SetInt(
            self.ptrSession, b"PerfHudMode", ovr_capi.ovrPerfHud_Off)

    @property
    def perfHudModes(self):
        """List of valid performance HUD modes."""
        return [*_performance_hud_modes]

    def setEyeViewport(self, eye, rect):
        """Set the viewport for a given eye.

        Parameters
        ----------
        eye : int
            Which eye to set the viewport, where left=0 and right=1.
        rect : ndarray, list or tuple of float
            Rectangle specifying the viewport's position and dimensions on the
            eye buffer.

        """
        cdef ovr_capi.ovrRecti viewportRect
        viewportRect.Pos.x = <int>rect[0]
        viewportRect.Pos.y = <int>rect[1]
        viewportRect.Size.w = <int>rect[2]
        viewportRect.Size.h = <int>rect[3]

        self.eyeLayer.Viewport[eye] = viewportRect

    def getEyeViewport(self, eye):
        """Get the viewport for a given eye.

        Parameters
        ----------
        eye : int
            Which eye to set the viewport, where left=0 and right=1.

        """
        cdef ovr_capi.ovrRecti viewportRect = \
            self.eyeLayer.Viewport[eye]
        cdef np.ndarray to_return = np.asarray(
            [viewportRect.Pos.x,
             viewportRect.Pos.y,
             viewportRect.Size.w,
             viewportRect.Size.h],
            dtype=np.float32)

        return to_return

    def waitToBeginFrame(self, unsigned int frameIndex=0):
        """Wait until a buffer is available and frame rendering can begin. Must
        be called before 'beginFrame'.

        Parameters
        ----------
        frameIndex : int
            The target frame index.

        Returns
        -------
        int
            Return code of the LibOVR API call 'ovr_WaitToBeginFrame'. Returns
            LIBOVR_SUCCESS if completed without errors. May return
            LIBOVR_ERROR_DISPLAY_LOST if the device was removed, rendering the
            current session invalid.

        Raises
        ------
        RuntimeError
            Raised if 'debugMode' is True and the API call to
            'ovr_WaitToBeginFrame' returns an error.

        """
        cdef ovr_capi.ovrResult result = \
            ovr_capi.ovr_WaitToBeginFrame(self.ptrSession, frameIndex)

        return <int>result

    def beginFrame(self, unsigned int frameIndex=0):
        """Begin rendering the frame. Must be called prior to drawing and
        'endFrame'.

        Parameters
        ----------
        frameIndex : int
            The target frame index.

        Returns
        -------
        int
            Error code returned by 'ovr_BeginFrame'.

        """
        cdef ovr_capi.ovrResult result = \
            ovr_capi.ovr_BeginFrame(self.ptrSession, frameIndex)

        return <int> result

    def commitSwapChain(self, int eye):
        """Commit changes to a given eye's texture swap chain. When called, the
        runtime is notified that the texture is ready for use, and the swap
        chain index is incremented.

        Parameters
        ----------
        eye : int
            Eye buffer index.

        Returns
        -------
        int
            Error code returned by API call 'ovr_CommitTextureSwapChain'. Will
            return :data:`LIBOVR_SUCCESS` if successful. Returns error code
            :data:`LIBOVR_ERROR_TEXTURE_SWAP_CHAIN_FULL` if called too many
            times without calling 'endFrame'.

        Raises
        ------
        RuntimeError
            Raised if 'debugMode' is True and the API call to
            'ovr_CommitTextureSwapChain' returns an error.

        Warning
        -------
            No additional drawing operations are permitted once the texture is
            committed until the SDK dereferences it, making it available again.

        """
        cdef ovr_capi.ovrResult result = ovr_capi.ovr_CommitTextureSwapChain(
            self.ptrSession,
            self.swapChains[eye])

        if self.debugMode:
            check_result(result)

            return result

    def endFrame(self, unsigned int frameIndex=0):
        """Call when rendering a frame has completed. Buffers which have been
        committed are passed to the compositor for distortion.

        Parameters
        ----------
        frameIndex : int
            The target frame index.

        Returns
        -------
        int
            Error code returned by API call 'ovr_EndFrame'. Check against
            LIBOVR_SUCCESS, LIBOVR_SUCCESS_NOT_VISIBLE,
            LIBOVR_SUCCESS_BOUNDARY_INVALID, LIBOVR_SUCCESS_DEVICE_UNAVAILABLE.

        Raises
        ------
        RuntimeError
            Raised if 'debugMode' is True and the API call to 'ovr_EndFrame'
            returns an error.

        """
        cdef ovr_capi.ovrLayerHeader* layers = &self.eyeLayer.Header
        cdef ovr_capi.ovrResult result = ovr_capi.ovr_EndFrame(
            self.ptrSession,
            frameIndex,
            NULL,
            &layers,
            <unsigned int>1)

        if self.debugMode:
            check_result(result)

        return result

    @property
    def isVisible(self):
        """Application has focus and is visible in the HMD."""
        return <bint>self.sessionStatus.IsVisible

    @property
    def hmdPresent(self):
        """HMD is present."""
        return <bint>self.sessionStatus.HmdPresent

    @property
    def hmdMounted(self):
        """HMD is being worn by the user."""
        return <bint>self.sessionStatus.HmdMounted

    @property
    def displayLost(self):
        """Display has been lost."""
        return <bint>self.sessionStatus.DisplayLost

    @property
    def shouldQuit(self):
        """The application should quit."""
        return <bint>self.sessionStatus.ShouldQuit

    @property
    def shouldRecenter(self):
        """The application should recenter."""
        return <bint>self.sessionStatus.ShouldRecenter

    @property
    def hasInputFocus(self):
        """The application has input focus."""
        return <bint>self.sessionStatus.HasInputFocus

    @property
    def overlayPresent(self):
        """The system overlay is present."""
        return <bint>self.sessionStatus.OverlayPresent

    @property
    def depthRequested(self):
        """Depth buffers are requested by the runtime."""
        return <bint>self.sessionStatus.DepthRequested

    def resetFrameStats(self):
        """Reset frame statistics.

        Returns
        -------
        int
            Error code returned by 'ovr_ResetPerfStats'.

        """
        cdef ovr_capi.ovrResult result = ovr_capi.ovr_ResetPerfStats(
            self.ptrSession)

        return result

    @property
    def trackingOriginType(self):
        """Tracking origin type.

        The tracking origin type specifies where the origin is placed when
        computing the pose of tracked objects (i.e. the head and touch
        controllers.) Valid values are 'floor' and 'eye'.

        """
        cdef ovr_capi.ovrTrackingOrigin origin = \
            ovr_capi.ovr_GetTrackingOriginType(self.ptrSession)

        if origin == ovr_capi.ovrTrackingOrigin_FloorLevel:
            return 'floor'
        elif origin == ovr_capi.ovrTrackingOrigin_EyeLevel:
            return 'eye'


    @trackingOriginType.setter
    def trackingOriginType(self, str value):
        cdef ovr_capi.ovrResult result
        if value == 'floor':
            result = ovr_capi.ovr_SetTrackingOriginType(
                self.ptrSession, ovr_capi.ovrTrackingOrigin_FloorLevel)
        elif value == 'eye':
            result = ovr_capi.ovr_SetTrackingOriginType(
                self.ptrSession, ovr_capi.ovrTrackingOrigin_EyeLevel)

        if self.debugMode:
            check_result(result)

    def recenterTrackingOrigin(self):
        """Recenter the tracking origin.

        Returns
        -------
        None

        """
        cdef ovr_capi.ovrResult result = ovr_capi.ovr_RecenterTrackingOrigin(
            self.ptrSession)

        if self.debugMode:
            check_result(result)

    def getTrackerFrustum(self, int trackerIndex):
        """Get the frustum parameters of a specified position tracker/sensor.

        Parameters
        ----------
        trackerIndex : int
            The index of the sensor to query. Valid values are between 0 and
            '~LibOVRSession.trackerCount'.

        Returns
        -------
        ndarray of float
            Frustum parameters of the tracker's camera. The returned array
            contains the horizontal and vertical FOV's in radians and the near
            and far clipping planes in meters.

        """
        cdef ovr_capi.ovrTrackerDesc tracker_desc = ovr_capi.ovr_GetTrackerDesc(
            self.ptrSession, <unsigned int>trackerIndex)

        cdef np.ndarray to_return = np.asarray([
            tracker_desc.FrustumHFovInRadians,
            tracker_desc.FrustumVFovInRadians,
            tracker_desc.FrustumNearZInMeters,
            tracker_desc.FrustumFarZInMeters],
            dtype=np.float32)

        return to_return

    def getTrackerInfo(self, int trackerIndex):
        """Get information about a given tracker.

        Parameters
        ----------
        trackerIndex : int
            The index of the sensor to query. Valid values are between 0 and
            '~LibOVRSession.trackerCount'.

        """
        cdef LibOVRTrackerInfo to_return = LibOVRTrackerInfo()

        # set the descriptor data
        to_return.c_ovrTrackerDesc = ovr_capi.ovr_GetTrackerDesc(
            self.ptrSession, <unsigned int>trackerIndex)
        # get the tracker pose
        to_return.c_ovrTrackerPose = ovr_capi.ovr_GetTrackerPose(
            self.ptrSession, <unsigned int>trackerIndex)

        return to_return

    @property
    def maxProvidedFrameStats(self):
        """Maximum number of frame stats provided."""
        return 5

    @property
    def frameStatsCount(self):
        """Number of frame stats available."""
        pass

    @property
    def anyFrameStatsDropped(self):
        """Have any frame stats been dropped?"""
        pass

    @property
    def adaptiveGpuPerformanceScale(self):
        """Adaptive GPU performance scaling factor."""
        pass

    @property
    def isAswAvailable(self):
        """Is ASW available?"""
        pass

    def getLastErrorInfo(self):
        """Get the last error code and information string reported by the API.
        """
        pass

    def setBoundaryColor(self, red, green, blue):
        """Set the boundary color.

        The boundary is drawn by the compositor which overlays the extents of
        the physical space where the user can safely move.

        Parameters
        ----------
        red : float
            Red component of the color from 0.0 to 1.0.
        green : float
            Green component of the color from 0.0 to 1.0.
        blue : float
            Blue component of the color from 0.0 to 1.0.

        """
        cdef ovr_capi.ovrColorf color
        color.r = <float>red
        color.g = <float>green
        color.b = <float>blue

        self.boundryStyle.Color = color

        cdef ovr_capi.ovrResult result = ovr_capi.ovr_SetBoundaryLookAndFeel(
            self.ptrSession,
            &self.boundryStyle)

        if self.debugMode:
            check_result(result)

    def resetBoundaryColor(self):
        """Reset the boundary color to system default.

        """
        cdef ovr_capi.ovrResult result = ovr_capi.ovr_ResetBoundaryLookAndFeel(
            self.ptrSession)

        if self.debugMode:
            check_result(result)

    @property
    def isBoundryVisible(self):
        """Check if the Guardian boundary is visible.

        The boundary is drawn by the compositor which overlays the extents of
        the physical space where the user can safely move.

        """
        cdef ovr_capi.ovrBool is_visible
        cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetBoundaryVisible(
            self.ptrSession, &is_visible)

        if self.debugMode:
            check_result(result)

        return <bint> is_visible

    def showBoundary(self):
        """Show the boundary.

        The boundary is drawn by the compositor which overlays the extents of
        the physical space where the user can safely move.

        """
        cdef ovr_capi.ovrResult result = ovr_capi.ovr_RequestBoundaryVisible(
            self.ptrSession, ovr_capi.ovrTrue)

        return result

    def hideBoundary(self):
        """Hide the boundry."""
        cdef ovr_capi.ovrResult result = ovr_capi.ovr_RequestBoundaryVisible(
            self.ptrSession, ovr_capi.ovrFalse)

        return result

    def getBoundaryDimensions(self, str boundaryType='PlayArea'):
        """Get the dimensions of the boundary.

        Parameters
        ----------
        boundaryType : str
            Boundary type, can be 'PlayArea' or 'Outer'.

        Returns
        -------
        ndarray
            Dimensions of the boundary meters [x, y, z].

        """
        cdef ovr_capi.ovrBoundaryType btype
        if boundaryType == 'PlayArea':
            btype = ovr_capi.ovrBoundary_PlayArea
        elif boundaryType == 'Outer':
            btype = ovr_capi.ovrBoundary_Outer
        else:
            raise ValueError("Invalid boundary type specified.")

        cdef ovr_capi.ovrVector3f vec_out
        cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetBoundaryDimensions(
                self.ptrSession, btype, &vec_out)

        cdef np.ndarray[np.float32_t, ndim=1] to_return = np.asarray(
            (vec_out.x, vec_out.y, vec_out.z), dtype=np.float32)

        return to_return

    def getBoundaryPoints(self, str boundaryType='PlayArea'):
        """Get the floor points which define the boundary."""
        pass  # TODO: make this work.

    def getConnectedControllers(self):
        """List of connected controllers.

        Returns
        -------
        list
            List of controller names. Check if a specific controller is
            connected by checking the membership of its name in the list.

        """
        cdef unsigned int result = ovr_capi.ovr_GetConnectedControllerTypes(
            self.ptrSession)

        cdef list ctrl_types = list()
        if (result & ovr_capi.ovrControllerType_XBox) == \
                ovr_capi.ovrControllerType_XBox:
            ctrl_types.append('Xbox')
        elif (result & ovr_capi.ovrControllerType_Remote) == \
                ovr_capi.ovrControllerType_Remote:
            ctrl_types.append('Remote')
        elif (result & ovr_capi.ovrControllerType_Touch) == \
                ovr_capi.ovrControllerType_Touch:
            ctrl_types.append('Touch')
        elif (result & ovr_capi.ovrControllerType_LTouch) == \
                ovr_capi.ovrControllerType_LTouch:
            ctrl_types.append('LeftTouch')
        elif (result & ovr_capi.ovrControllerType_RTouch) == \
                ovr_capi.ovrControllerType_RTouch:
            ctrl_types.append('RightTouch')

        return ctrl_types

    def getInputState(self, str controller):
        """Get the current state of a input device.

        Parameters
        ----------
        controller_type : str
            Controller name to poll. Valid names are: 'Xbox', 'Remote', 'Touch',
            'LeftTouch', and 'RightTouch'.

        Returns
        -------
        LibOVRControllerState
            Object storing controller state information.

        """
        cdef dict _controller_types = {
            'Xbox' : ovr_capi.ovrControllerType_XBox,
            'Remote' : ovr_capi.ovrControllerType_Remote,
            'Touch' : ovr_capi.ovrControllerType_Touch,
            'LeftTouch' : ovr_capi.ovrControllerType_LTouch,
            'RightTouch' : ovr_capi.ovrControllerType_RTouch}

        cdef LibOVRInputState to_return = LibOVRInputState()
        cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetInputState(
            self.ptrSession,
            <ovr_capi.ovrControllerType>_controller_types[controller],
            to_return.c_data)

        if self.debugMode:
            check_result(result)

        return to_return

    def setControllerVibration(self, str controller, str frequency, float amplitude):
        """Vibrate a controller.

        Vibration is constant at fixed frequency and amplitude. Vibration lasts
        2.5 seconds, so this function needs to be called more often than that
        for sustained vibration. Only controllers which support vibration can be
        used here.

        There are only two frequencies permitted 'high' (1 Hz) and 'low'
        (0.5 Hz), however, amplitude can vary from 0.0 to 1.0. Specifying
        frequency='off' stops vibration.

        Parameters
        ----------
        controller : str
            Controller name to vibrate. Valid names are: 'Xbox', 'Touch',
            'LeftTouch', and 'RightTouch'.
        frequency : str
            Vibration frequency. Valid values are: 'off', 'low', or 'high'.
        amplitude : float
            Vibration amplitude in the range of [0.0 and 1.0]. Values outside
            this range are clamped.

        Returns
        -------
        int
            Return value of API call 'ovr_SetControllerVibration'. Can return
            LIBOVR_SUCCESS_DEVICE_UNAVAILABLE if no device is present.

        """
        # get frequency associated with the string
        cdef float freq = 0.0
        if frequency == 'off':
            freq = 0.0
        elif frequency == 'low':
            freq = 0.5
        elif frequency == 'high':
            freq = 1.0

        cdef dict _controller_types = {
            'Xbox' : ovr_capi.ovrControllerType_XBox,
            'Touch' : ovr_capi.ovrControllerType_Touch,
            'LeftTouch' : ovr_capi.ovrControllerType_LTouch,
            'RightTouch' : ovr_capi.ovrControllerType_RTouch}

        cdef ovr_capi.ovrResult result = ovr_capi.ovr_SetControllerVibration(
            self.ptrSession,
            <ovr_capi.ovrControllerType>_controller_types[controller],
            freq,
            amplitude)

        if self.debugMode:
            check_result(result)

        return result

    def getSessionStatus(self):
        """Get the current session status.

        Returns
        -------
        LibOVRSessionStatus
            Object specifying the current state of the session.

        """

        cdef LibOVRSessionStatus to_return = LibOVRSessionStatus()
        cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetSessionStatus(
            self.ptrSession, to_return.c_data)

        return to_return


cdef class LibOVRPose(object):
    """Class for rigid body pose data for LibOVR.

    """
    cdef ovr_capi.ovrPosef* c_data
    cdef ovr_capi.ovrPosef c_ovrPosef  # internal data

    cdef np.ndarray _pos
    cdef np.ndarray _ori

    def __init__(self, ori=(0., 0., 0., 1.), pos=(0., 0., 0.)):
        """Constructor for LibOVRPose.

        Parameters
        ----------
        ori : tuple, list, or ndarray of float
            Orientation quaternion vector (x, y, z, w).
        pos : tuple, list, or ndarray of float
            Position vector (x, y, z).

        Notes
        -----
        Values for vectors are stored internally as 32-bit floating point
        numbers.

        """
        pass  # nop

    def __cinit__(self, ori=(0., 0., 0., 1.), pos=(0., 0., 0.)):
        self.c_data = &self.c_ovrPosef  # pointer to c_ovrPosef

        # numpy arrays for internal data
        self._pos = np.empty((3,), dtype=np.float32)
        self._ori = np.empty((4,), dtype=np.float32)

        self.c_data[0].Position.x = <float>pos[0]
        self.c_data[0].Position.y = <float>pos[1]
        self.c_data[0].Position.z = <float>pos[2]

        self.c_data[0].Orientation.x = <float>ori[0]
        self.c_data[0].Orientation.y = <float>ori[1]
        self.c_data[0].Orientation.z = <float>ori[2]
        self.c_data[0].Orientation.w = <float>ori[3]

    @property
    def pos(self):
        """Position vector X, Y, Z (`ndarray` of `float`).

        The returned object is a NumPy array which references data stored in an
        internal structure (ovrPosef). The array is conformal with the internal
        data's type (float32) and size (length 3).

        Examples
        --------
        Set the position of the pose manually::

            myPose.pos = [5., 6., 7.]

        Add 1 unit to the position's Z component::

            myPose.pos[2] += 1.

        """
        self._pos.data = <char*>&self.c_data.Position.x
        return self._pos

    @pos.setter
    def pos(self, object value):
        self.c_data[0].Position.x = <float>value[0]
        self.c_data[0].Position.y = <float>value[1]
        self.c_data[0].Position.z = <float>value[2]

    @property
    def ori(self):
        """Orientation quaternion X, Y, Z, W (`ndarray` of `float`).

        Components X, Y, Z are imaginary and W is real.

        The returned object is a NumPy array which references data stored in an
        internal structure (ovrPosef). The array is conformal with the internal
        data's type (float32) and size (length 3).

        Notes
        -----
            The orientation quaternion should be normalized.

        """
        self._ori.data = <char*>&self.c_data.Orientation.x
        return self._ori

    @ori.setter
    def ori(self, object value):
        self.c_data[0].Orientation.x = <float>value[0]
        self.c_data[0].Orientation.y = <float>value[1]
        self.c_data[0].Orientation.z = <float>value[2]
        self.c_data[0].Orientation.w = <float>value[3]

    def __mul__(LibOVRPose a, LibOVRPose b):
        """Multiplication operator (*) to combine poses."""
        cdef ovr_math.Posef pose_a = <ovr_math.Posef>a.c_data[0]
        cdef ovr_math.Posef pose_b = <ovr_math.Posef>b.c_data[0]
        cdef ovr_math.Posef pose_r = pose_a * pose_b

        cdef LibOVRPose to_return = \
            LibOVRPose(
                (pose_r.Rotation.x,
                 pose_r.Rotation.y,
                 pose_r.Rotation.z,
                 pose_r.Rotation.w),
                (pose_r.Translation.x,
                 pose_r.Translation.y,
                 pose_r.Translation.z),)

        return to_return

    def __invert__(self):
        """Invert operator (~) to invert a pose.

        """
        return self.inverted()

    def getYawPitchRoll(self):
        """Get the yaw, pitch, and roll of the orientation quaternion.

        Computed values are referenced relative to the world axes.

        """
        cdef float yaw, pitch, roll
        (<ovr_math.Posef>self.c_data[0]).Rotation.GetYawPitchRoll(
            &yaw, &pitch, &roll)
        cdef np.ndarray[np.float32_t, ndim=1] to_return = \
            np.array((yaw, pitch, roll), dtype=np.float32)

        return to_return

    def matrix4x4(self):
        """Convert this pose into a 4x4 transformation matrix.

        Returns
        -------
        ndarray
            4x4 transformation matrix.

        """
        cdef ovr_math.Matrix4f m_pose = ovr_math.Matrix4f(
            <ovr_math.Posef>self.c_data[0])

        cdef np.ndarray[np.float32_t, ndim=2] to_return = \
            np.zeros((4, 4), dtype=np.float32)

        # fast copy matrix to numpy array
        cdef float [:, :] mv = to_return
        cdef Py_ssize_t i, j
        i = j = 0
        for i in range(4):
            for j in range(4):
                mv[i, j] = m_pose.M[i][j]

        return to_return

    def matrix1D(self):
        """Convert this pose into a 1D (flattened) transform matrix.

        This will output an array suitable for use with OpenGL.

        Returns
        -------
        ndarray
            4x4 transformation matrix flattened to a 1D array assuming column
            major order with a 'float32' data type.

        """
        cdef ovr_math.Matrix4f m_pose = ovr_math.Matrix4f(
            <ovr_math.Posef>self.c_data[0])
        cdef np.ndarray[np.float32_t, ndim=1] to_return = \
            np.zeros((16,), dtype=np.float32)

        # fast copy matrix to numpy array
        cdef float [:] mv = to_return
        cdef Py_ssize_t i, j, k, N
        i = j = k = 0
        N = 4
        for i in range(N):
            for j in range(N):
                mv[k] = m_pose.M[j][i]  # row -> column major order
                k += 1

        return to_return

    def normalize(self):
        """Normalize this pose.

        """
        (<ovr_math.Posef>self.c_data[0]).Normalize()

    def inverted(self):
        """Get the inverse of the pose.

        Returns
        -------
        LibOVRPose
            Inverted pose.

        """
        cdef ovr_math.Quatf inv_ori = \
            (<ovr_math.Quatf>self.c_data[0].Orientation).Inverted()
        cdef ovr_math.Vector3f inv_pos = \
            (<ovr_math.Quatf>inv_ori).Rotate(
                -(<ovr_math.Vector3f>self.c_data[0].Position))
        cdef LibOVRPose to_return = \
            LibOVRPose(
                (self.c_data[0].Orientation.x,
                 self.c_data[0].Orientation.y,
                 self.c_data[0].Orientation.z,
                 self.c_data[0].Orientation.w),
                (self.c_data[0].Position.x,
                 self.c_data[0].Position.y,
                 self.c_data[0].Position.z))

    def rotate(self, object v):
        """Rotate a position vector.

        Parameters
        ----------
        v : tuple, list, or ndarray of float
            Vector to rotate.

        Returns
        -------
        ndarray
            Vector rotated by the pose's orientation.

        """
        cdef ovr_math.Vector3f pos_in = ovr_math.Vector3f(
            <float>v[0], <float>v[1], <float>v[2])
        cdef ovr_math.Vector3f rotated_pos = \
            (<ovr_math.Posef>self.c_data[0]).Rotate(pos_in)

        cdef np.ndarray[np.float32_t, ndim=1] to_return = \
            np.array((rotated_pos.x, rotated_pos.y, rotated_pos.z),
                     dtype=np.float32)

        return to_return

    def inverseRotate(self, object v):
        """Inverse rotate a position vector.

        Parameters
        ----------
        v : tuple, list, or ndarray of float
            Vector to rotate.

        Returns
        -------
        ndarray
            Vector rotated by the pose's inverse orientation.

        """
        cdef ovr_math.Vector3f pos_in = ovr_math.Vector3f(
            <float>v[0], <float>v[1], <float>v[2])
        cdef ovr_math.Vector3f inv_rotated_pos = \
            (<ovr_math.Posef>self.c_data[0]).InverseRotate(pos_in)

        cdef np.ndarray[np.float32_t, ndim=1] to_return = \
            np.array((inv_rotated_pos.x, inv_rotated_pos.y, inv_rotated_pos.z),
                     dtype=np.float32)

        return to_return

    def translate(self, object v):
        """Translate a position vector.

        Parameters
        ----------
        v : tuple, list, or ndarray of float
            Vector to translate (x, y, z).

        Returns
        -------
        ndarray
            Vector translated by the pose's position.

        """
        cdef ovr_math.Vector3f pos_in = ovr_math.Vector3f(
            <float>v[0], <float>v[1], <float>v[2])
        cdef ovr_math.Vector3f translated_pos = \
            (<ovr_math.Posef>self.c_data[0]).Translate(pos_in)

        cdef np.ndarray[np.float32_t, ndim=1] to_return = \
            np.array((translated_pos.x, translated_pos.y, translated_pos.z),
                     dtype=np.float32)

        return to_return

    def transform(self, object v):
        """Transform a position vector.

        Parameters
        ----------
        v : tuple, list, or ndarray of float
            Vector to transform (x, y, z).

        Returns
        -------
        ndarray
            Vector transformed by the pose's position and orientation.

        """
        cdef ovr_math.Vector3f pos_in = ovr_math.Vector3f(
            <float>v[0], <float>v[1], <float>v[2])
        cdef ovr_math.Vector3f transformed_pos = \
            (<ovr_math.Posef>self.c_data[0]).Transform(pos_in)

        cdef np.ndarray[np.float32_t, ndim=1] to_return = \
            np.array((transformed_pos.x, transformed_pos.y, transformed_pos.z),
                     dtype=np.float32)

        return to_return

    def inverseTransform(self, object v):
        """Inverse transform a position vector.

        Parameters
        ----------
        v : tuple, list, or ndarray of float
            Vector to transform (x, y, z).

        Returns
        -------
        ndarray
            Vector transformed by the inverse of the pose's position and
            orientation.

        """
        cdef ovr_math.Vector3f pos_in = ovr_math.Vector3f(
            <float>v[0], <float>v[1], <float>v[2])
        cdef ovr_math.Vector3f transformed_pos = \
            (<ovr_math.Posef>self.c_data[0]).InverseTransform(pos_in)

        cdef np.ndarray[np.float32_t, ndim=1] to_return = \
            np.array((transformed_pos.x, transformed_pos.y, transformed_pos.z),
                     dtype=np.float32)

        return to_return

    def transformNormal(self, object v):
        """Transform a normal vector.

        Parameters
        ----------
        v : tuple, list, or ndarray of float
            Vector to transform (x, y, z).

        Returns
        -------
        ndarray
            Vector transformed by the pose's position and orientation.

        """
        cdef ovr_math.Vector3f pos_in = ovr_math.Vector3f(
            <float>v[0], <float>v[1], <float>v[2])
        cdef ovr_math.Vector3f transformed_pos = \
            (<ovr_math.Posef>self.c_data[0]).TransformNormal(pos_in)

        cdef np.ndarray[np.float32_t, ndim=1] to_return = \
            np.array((transformed_pos.x, transformed_pos.y, transformed_pos.z),
                     dtype=np.float32)

        return to_return

    def inverseTransformNormal(self, object v):
        """Inverse transform a normal vector.

        Parameters
        ----------
        v : tuple, list, or ndarray of float
            Vector to transform (x, y, z).

        Returns
        -------
        ndarray
            Vector transformed by the pose's position and orientation.

        """
        cdef ovr_math.Vector3f pos_in = ovr_math.Vector3f(
            <float>v[0], <float>v[1], <float>v[2])
        cdef ovr_math.Vector3f transformed_pos = \
            (<ovr_math.Posef>self.c_data[0]).InverseTransformNormal(pos_in)

        cdef np.ndarray[np.float32_t, ndim=1] to_return = \
            np.array((transformed_pos.x, transformed_pos.y, transformed_pos.z),
                     dtype=np.float32)

        return to_return

    def apply(self, object v):
        """Apply a transform to a position vector.

        Parameters
        ----------
        v : tuple, list, or ndarray of float
            Vector to transform (x, y, z).

        Returns
        -------
        ndarray
            Vector transformed by the pose's position and orientation.

        """
        cdef ovr_math.Vector3f pos_in = ovr_math.Vector3f(
            <float>v[0], <float>v[1], <float>v[2])
        cdef ovr_math.Vector3f transformed_pos = \
            (<ovr_math.Posef>self.c_data[0]).Apply(pos_in)

        cdef np.ndarray[np.float32_t, ndim=1] to_return = \
            np.array((transformed_pos.x, transformed_pos.y, transformed_pos.z),
                     dtype=np.float32)

        return to_return

    def distanceTo(self, object v):
        """Distance to a point or pose from this pose.

        Parameters
        ----------
        v : tuple, list, ndarray, or LibOVRPose
            Vector to transform (x, y, z).

        Returns
        -------
        float
            Distance to a point or Pose.

        """
        cdef ovr_math.Vector3f pos_in

        if isinstance(v, LibOVRPose):
            pos_in = <ovr_math.Vector3f>((<LibOVRPose>v).c_data[0]).Position
        else:
            pos_in = ovr_math.Vector3f(<float>v[0], <float>v[1], <float>v[2])

        cdef float to_return = \
            (<ovr_math.Posef>self.c_data[0]).Translation.Distance(pos_in)


cdef class LibOVRPoseState(object):
    """Class for data about rigid body configuration with derivatives computed
    by the LibOVR runtime.

    """
    cdef ovr_capi.ovrPoseStatef* c_data
    cdef ovr_capi.ovrPoseStatef c_ovrPoseStatef

    cdef LibOVRPose _pose
    cdef np.ndarray _angular_vel
    cdef np.ndarray _linear_vel
    cdef np.ndarray _angular_acc
    cdef np.ndarray _linear_acc

    cdef int status_flags

    def __cinit__(self):
        self.c_data = &self.c_ovrPoseStatef  # pointer to ovrPoseStatef

        # the pose is accessed using a LibOVRPose object
        self._pose = LibOVRPose()
        self._pose.c_data = &self.c_data.ThePose

        # numpy arrays which view internal dat
        self._angular_vel = np.empty((3,), dtype=np.float32)
        self._linear_vel = np.empty((3,), dtype=np.float32)
        self._angular_acc = np.empty((3,), dtype=np.float32)
        self._linear_acc = np.empty((3,), dtype=np.float32)

    @property
    def thePose(self):
        """Body pose.

        Returns
        -------
        LibOVRPose
            Rigid body pose data with position and orientation information.

        """
        return self._pose

    @property
    def angularVelocity(self):
        """Angular velocity vector in radians/sec."""
        self._angular_vel.data = <char*>&self.c_data.AngularVelocity.x
        return self._angular_vel

    @angularVelocity.setter
    def angularVelocity(self, object value):
        """Angular velocity vector in radians/sec."""
        self.c_data[0].AngularVelocity.x = <float>value[0]
        self.c_data[0].AngularVelocity.y = <float>value[1]
        self.c_data[0].AngularVelocity.z = <float>value[2]

    @property
    def linearVelocity(self):
        """Linear velocity vector in meters/sec.

        This is only available if 'pos_tracked' is True.
        """
        self._linear_vel.data = <char*>&self.c_data.LinearVelocity.x
        return self._linear_vel

    @linearVelocity.setter
    def linearVelocity(self, object value):
        self.c_data[0].LinearVelocity.x = <float>value[0]
        self.c_data[0].LinearVelocity.y = <float>value[1]
        self.c_data[0].LinearVelocity.z = <float>value[2]

    @property
    def angularAcceleration(self):
        """Angular acceleration vector in radians/s^2."""
        self._angular_acc.data = <char*>&self.c_data.AngularAcceleration.x
        return self._angular_acc

    @angularAcceleration.setter
    def angularAcceleration(self, object value):
        self.c_data[0].AngularAcceleration.x = <float>value[0]
        self.c_data[0].AngularAcceleration.y = <float>value[1]
        self.c_data[0].AngularAcceleration.z = <float>value[2]

    @property
    def linearAcceleration(self):
        """Linear acceleration vector in meters/s^2."""
        self._linear_acc.data = <char*>&self.c_data.LinearAcceleration.x
        return self._linear_acc

    @linearAcceleration.setter
    def linearAcceleration(self, object value):
        self.c_data[0].LinearAcceleration.x = <float>value[0]
        self.c_data[0].LinearAcceleration.y = <float>value[1]
        self.c_data[0].LinearAcceleration.z = <float>value[2]

    @property
    def timeInSeconds(self):
        """Absolute time this data refers to in seconds."""
        return <double>self.c_data[0].TimeInSeconds

    @property
    def orientationTracked(self):
        """True if the orientation was tracked when sampled."""
        return <bint>((ovr_capi.ovrStatus_OrientationTracked &
             self.status_flags) == ovr_capi.ovrStatus_OrientationTracked)

    @property
    def positionTracked(self):
        """True if the position was tracked when sampled."""
        return <bint>((ovr_capi.ovrStatus_PositionTracked &
             self.status_flags) == ovr_capi.ovrStatus_PositionTracked)

    @property
    def fullyTracked(self):
        """True if position and orientation were tracked when sampled."""
        cdef int32_t full_tracking_flags = \
            ovr_capi.ovrStatus_OrientationTracked | \
            ovr_capi.ovrStatus_PositionTracked
        return <bint>((self.status_flags & full_tracking_flags) ==
                      full_tracking_flags)


cdef class LibOVRInputState(object):
    """Class for storing the input state of a controller.

    """
    cdef ovr_capi.ovrInputState* c_data
    cdef ovr_capi.ovrInputState c_ovrInputState

    def __cinit__(self):
        self.c_data = &self.c_ovrInputState

    def timeInSeconds(self):
        return <double>self.c_data.TimeInSeconds

    @property
    def buttons(self):
        """Button state as integer."""
        return self.c_data[0].Buttons

    @property
    def touches(self):
        """Touch state as integer."""
        return self.c_data[0].Touches

    @property
    def indexTrigger(self):
        """Index trigger values."""
        cdef float index_trigger_left = self.c_data[0].IndexTrigger[0]
        cdef float index_trigger_right = self.c_data[0].IndexTrigger[1]

        return index_trigger_left, index_trigger_right

    @property
    def handTrigger(self):
        """Hand trigger values."""
        cdef float hand_trigger_left = self.c_data[0].HandTrigger[0]
        cdef float hand_trigger_right = self.c_data[0].HandTrigger[1]

        return hand_trigger_left, hand_trigger_right

    @property
    def thumbstick(self):
        """Thhumstick values."""
        cdef float thumbstick_x0 = self.c_data[0].Thumbstick[0].x
        cdef float thumbstick_y0 = self.c_data[0].Thumbstick[0].y
        cdef float thumbstick_x1 = self.c_data[0].Thumbstick[1].x
        cdef float thumbstick_y1 = self.c_data[0].Thumbstick[1].y

        return (thumbstick_x0, thumbstick_y0), (thumbstick_x1, thumbstick_y1)

    @property
    def controllerType(self):
        """Controller type this object references."""
        cdef int ctrl_type = <int>self.c_data[0].ControllerType

        if ctrl_type == ovr_capi.ovrControllerType_XBox:
            return 'Xbox'
        elif ctrl_type == ovr_capi.ovrControllerType_Remote:
            return 'Remote'
        elif ctrl_type == ovr_capi.ovrControllerType_Touch:
            return 'Touch'
        elif ctrl_type == ovr_capi.ovrControllerType_LTouch:
            return 'LeftTouch'
        elif ctrl_type == ovr_capi.ovrControllerType_RTouch:
            return 'RightTouch'
        else:
            return None

    @property
    def indexTriggerNoDeadzone(self):
        cdef float index_trigger_left = self.c_data[0].IndexTriggerNoDeadzone[0]
        cdef float index_trigger_right = self.c_data[0].IndexTriggerNoDeadzone[
            1]

        return index_trigger_left, index_trigger_right

    @property
    def handTriggerNoDeadzone(self):
        cdef float hand_trigger_left = self.c_data[0].HandTriggerNoDeadzone[0]
        cdef float hand_trigger_right = self.c_data[0].HandTriggerNoDeadzone[1]

        return hand_trigger_left, hand_trigger_right

    @property
    def thumbstickNoDeadzone(self):
        cdef float thumbstick_x0 = self.c_data[0].ThumbstickNoDeadzone[0].x
        cdef float thumbstick_y0 = self.c_data[0].ThumbstickNoDeadzone[0].y
        cdef float thumbstick_x1 = self.c_data[0].ThumbstickNoDeadzone[1].x
        cdef float thumbstick_y1 = self.c_data[0].ThumbstickNoDeadzone[1].y

        return (thumbstick_x0, thumbstick_y0), (thumbstick_x1, thumbstick_y1)

    @property
    def indexTriggerRaw(self):
        cdef float index_trigger_left = self.c_data[0].IndexTriggerRaw[0]
        cdef float index_trigger_right = self.c_data[0].IndexTriggerRaw[1]

        return index_trigger_left, index_trigger_right

    @property
    def handTriggerRaw(self):
        cdef float hand_trigger_left = self.c_data[0].HandTriggerRaw[0]
        cdef float hand_trigger_right = self.c_data[0].HandTriggerRaw[1]

        return hand_trigger_left, hand_trigger_right

    @property
    def thumbstickRaw(self):
        cdef float thumbstick_x0 = self.c_data[0].ThumbstickRaw[0].x
        cdef float thumbstick_y0 = self.c_data[0].ThumbstickRaw[0].y
        cdef float thumbstick_x1 = self.c_data[0].ThumbstickRaw[1].x
        cdef float thumbstick_y1 = self.c_data[0].ThumbstickRaw[1].y

        return (thumbstick_x0, thumbstick_y0), (thumbstick_x1, thumbstick_y1)

    def getButtonPress(self, object buttons):
        """Check if buttons were pressed.

        Parameters
        ----------
        buttons : list, tuple, or str
            Name or list of buttons to test.

        Returns
        -------
        bool
            True if all buttons in 'buttons' were pressed.

        """
        cdef unsigned int button_bits = 0x00000000
        cdef int i, N
        if isinstance(buttons, str):  # don't loop if a string is specified
            button_bits |= _controller_buttons[buttons]
        elif isinstance(buttons, (tuple, list)):
            # loop over all names and combine them
            N = <int>len(buttons)
            for i in range(N):
                button_bits |= _controller_buttons[buttons[i]]

        cdef bint pressed = \
            (self.c_data[0].Buttons & button_bits) == button_bits

        return pressed

    def getTouches(self, object touches):
        """Check for touch states.

        Parameters
        ----------
        touches : list, tuple, or str
            Name or list of touches to test.

        Returns
        -------
        bool
            True if all touches in 'touches' are active.

        """
        cdef unsigned int touch_bits = 0x00000000
        cdef int i, N
        if isinstance(touches, str):  # don't loop if a string is specified
            touch_bits |= _touch_states[touches]
        elif isinstance(touches, (tuple, list)):
            # loop over all names and combine them
            N = <int> len(touches)
            for i in range(N):
                touch_bits |= _touch_states[touches[i]]

        # test if the button was pressed
        cdef bint touched = (self.c_data[0].Touches & touch_bits) == touch_bits

        return touched


cdef class LibOVRTrackerInfo(object):
    """Class for information about camera based tracking sensors.

    """
    cdef ovr_capi.ovrTrackerPose c_ovrTrackerPose
    cdef ovr_capi.ovrTrackerDesc c_ovrTrackerDesc

    cdef LibOVRPose _trackerPose
    cdef LibOVRPose _leveledPose

    def __cinit__(self):
        self._trackerPose.c_data = &self.c_ovrTrackerPose.Pose
        self._leveledPose.c_data = &self.c_ovrTrackerPose.LeveledPose

    @property
    def pose(self):
        """The pose of the sensor (`LibOVRPose`)."""
        return self._trackerPose

    @property
    def leveledPose(self):
        """Gravity aligned pose of the sensor (`LibOVRPose`)."""
        return self._leveledPose

    @property
    def isConnected(self):
        """True if the sensor is connected and available (`bool`)."""
        return <bint>((ovr_capi.ovrTracker_Connected &
             self.c_ovrTrackerPose.TrackerFlags) == ovr_capi.ovrTracker_Connected)

    @property
    def isPoseTracked(self):
        """True if the sensor has a valid pose (`bool`)."""
        return <bint>((ovr_capi.ovrTracker_PoseTracked &
             self.c_ovrTrackerPose.TrackerFlags) == ovr_capi.ovrTracker_PoseTracked)

    @property
    def frustum(self):
        """Frustum parameters of the sensor as an array (`ndarray`).

        Returns
        -------
        ndarray
            Frustum parameters [HFovInRadians, VFovInRadians, NearZInMeters,
            FarZInMeters].

        """
        cdef np.ndarray to_return = np.asarray([
            self.c_ovrTrackerDesc.FrustumHFovInRadians,
            self.c_ovrTrackerDesc.FrustumVFovInRadians,
            self.c_ovrTrackerDesc.FrustumNearZInMeters,
            self.c_ovrTrackerDesc.FrustumFarZInMeters],
            dtype=np.float32)

        return to_return

    @property
    def horizontalFOV(self):
        """Horizontal FOV of the sensor in radians (`float`)."""
        return self.c_ovrTrackerDesc.FrustumHFovInRadians

    @property
    def verticalFOV(self):
        """Vertical FOV of the sensor in radians (`float`)."""
        return self.c_ovrTrackerDesc.FrustumVFovInRadians

    @property
    def nearZ(self):
        """Near clipping plane of the sensor frustum in meters (`float`)."""
        return self.c_ovrTrackerDesc.FrustumNearZInMeters

    @property
    def farZ(self):
        """Far clipping plane of the sensor frustum in meters (`float`)."""
        return self.c_ovrTrackerDesc.FrustumFarZInMeters


cdef class LibOVRSessionStatus(object):
    """Class for session status information.

    """
    cdef ovr_capi.ovrSessionStatus* c_data
    cdef ovr_capi.ovrSessionStatus c_ovrSessionStatus

    def __cinit__(self):
        self.c_data = &self.c_ovrSessionStatus

    @property
    def isVisible(self):
        """True if the application has focus and visible in the HMD."""
        return self.c_data.IsVisible == ovr_capi.ovrTrue

    @property
    def hmdPresent(self):
        """True if the HMD is present."""
        return self.c_data.HmdPresent == ovr_capi.ovrTrue

    @property
    def hmdMounted(self):
        """True if the HMD is on the user's head."""
        return self.c_data.HmdMounted == ovr_capi.ovrTrue

    @property
    def displayLost(self):
        """True if the the display was lost."""
        return self.c_data.DisplayLost == ovr_capi.ovrTrue

    @property
    def shouldQuit(self):
        """True if the application was signaled to quit."""
        return self.c_data.ShouldQuit == ovr_capi.ovrTrue

    @property
    def shouldRecenter(self):
        """True if the application was signaled to re-center."""
        return self.c_data.ShouldRecenter == ovr_capi.ovrTrue

    @property
    def hasInputFocus(self):
        """True if the application has input focus."""
        return self.c_data.HasInputFocus == ovr_capi.ovrTrue

    @property
    def overlayPresent(self):
        """True if the system overlay is present."""
        return self.c_data.OverlayPresent == ovr_capi.ovrTrue

    @property
    def depthRequested(self):
        """True if the system requires a depth texture. Currently unused by
        PsychXR."""
        return self.c_data.DepthRequested == ovr_capi.ovrTrue


# cpdef object getInputState(str controller, object stateOut=None):
#     """Get a controller state as an object. If a 'InputStateData' object is
#     passed to 'state_out', that object will be updated.
#
#     :param controller: str
#     :param state_out: InputStateData or None
#     :return: InputStateData or None
#
#     """
#     cdef ovr_capi.ovrControllerType ctrl_type
#     if controller == 'xbox':
#         ctrl_type = ovr_capi.ovrControllerType_XBox
#     elif controller == 'remote':
#         ctrl_type = ovr_capi.ovrControllerType_Remote
#     elif controller == 'touch':
#         ctrl_type = ovr_capi.ovrControllerType_Touch
#     elif controller == 'left_touch':
#         ctrl_type = ovr_capi.ovrControllerType_LTouch
#     elif controller == 'right_touch':
#         ctrl_type = ovr_capi.ovrControllerType_RTouch
#
#     # create a controller state object and set its data
#     global _ptrSession_
#     cdef ovr_capi.ovrInputState*ptr_state
#     cdef LibOVRInputState to_return = LibOVRInputState()
#
#     if stateOut is None:
#         ptr_state = &(<LibOVRInputState> to_return).c_ovrInputState
#     else:
#         ptr_state = &(<LibOVRInputState> stateOut).c_ovrInputState
#
#     cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetInputState(
#         _ptrSession_,
#         ctrl_type,
#         ptr_state)
#
#     if stateOut is None:
#         return None
#
#     return to_return

# cpdef double pollController(str controller):
#     """Poll and update specified controller's state data. The time delta in
#     seconds between the current and previous controller state is returned.
#
#     :param controller: str or None
#     :return: double
#
#     """
#     global _ptrSession_, _ctrl_states_, _ctrl_states_prev_
#     cdef ovr_capi.ovrInputState*ptr_ctrl = NULL
#     cdef ovr_capi.ovrInputState*ptr_ctrl_prev = NULL
#
#     cdef ovr_capi.ovrControllerType ctrl_type
#     if controller == 'xbox':
#         ctrl_type = ovr_capi.ovrControllerType_XBox
#         ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.xbox]
#         ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.xbox]
#     elif controller == 'remote':
#         ctrl_type = ovr_capi.ovrControllerType_Remote
#         ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.remote]
#         ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.remote]
#     elif controller == 'touch':
#         ctrl_type = ovr_capi.ovrControllerType_Touch
#         ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.touch]
#         ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.touch]
#     elif controller == 'left_touch':
#         ctrl_type = ovr_capi.ovrControllerType_LTouch
#         ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.left_touch]
#         ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.left_touch]
#     elif controller == 'right_touch':
#         ctrl_type = ovr_capi.ovrControllerType_RTouch
#         ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.right_touch]
#         ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.right_touch]
#
#     # copy the previous control state
#     ptr_ctrl_prev[0] = ptr_ctrl[0]
#
#     # update the current controller state
#     cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetInputState(
#         _ptrSession_,
#         ctrl_type,
#         ptr_ctrl)
#
#     if debug_mode:
#         check_result(result)
#
#     # return the time delta between the last time the controller was polled
#     return ptr_ctrl[0].TimeInSeconds - ptr_ctrl_prev[0].TimeInSeconds

# cpdef double getControllerAbsTime(str controller):
#     """Get the absolute time the state of the specified controller was last
#     updated.
#
#     :param controller: str or None
#     :return: float
#
#     """
#     # get pointer to control state
#     global _ctrl_states_
#     cdef ovr_capi.ovrInputState*ptr_ctrl_state = NULL
#     if controller == 'xbox':
#         ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.xbox]
#     elif controller == 'remote':
#         ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.remote]
#     elif controller == 'touch':
#         ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.touch]
#     elif controller == 'left_touch':
#         ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.left_touch]
#     elif controller == 'right_touch':
#         ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.right_touch]
#
#     return ptr_ctrl_state[0].TimeInSeconds
#
# cpdef tuple getIndexTriggerValues(str controller, bint deadZone=False):
#     """Get index trigger values for a specified controller.
#
#     :param controller: str
#     :param deadZone: boolean
#     :return: tuple
#
#     """
#     # get pointer to control state
#     global _ctrl_states_
#     cdef ovr_capi.ovrInputState*ptr_ctrl_state = NULL
#     if controller == 'xbox':
#         ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.xbox]
#     elif controller == 'remote':
#         ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.remote]
#     elif controller == 'touch':
#         ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.touch]
#     elif controller == 'left_touch':
#         ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.left_touch]
#     elif controller == 'right_touch':
#         ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.right_touch]
#
#     cdef float index_trigger_left = 0.0
#     cdef float index_trigger_right = 0.0
#
#     # get the value with or without the deadzone
#     if not deadZone:
#         index_trigger_left = ptr_ctrl_state[0].IndexTriggerNoDeadzone[0]
#         index_trigger_right = ptr_ctrl_state[0].IndexTriggerNoDeadzone[1]
#     else:
#         index_trigger_left = ptr_ctrl_state[0].IndexTrigger[0]
#         index_trigger_right = ptr_ctrl_state[0].IndexTrigger[1]
#
#     return index_trigger_left, index_trigger_right
#
# cpdef tuple getHandTriggerValues(str controller, bint deadZone=False):
#     """Get hand trigger values for a specified controller.
#
#     :param controller: str
#     :param deadzone: boolean
#     :return: tuple
#
#     """
#     # get pointer to control state
#     global _ctrl_states_
#     cdef ovr_capi.ovrInputState*ptr_ctrl_state = NULL
#     if controller == 'xbox':
#         ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.xbox]
#     elif controller == 'remote':
#         ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.remote]
#     elif controller == 'touch':
#         ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.touch]
#     elif controller == 'left_touch':
#         ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.left_touch]
#     elif controller == 'right_touch':
#         ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.right_touch]
#
#     cdef float hand_trigger_left = 0.0
#     cdef float hand_trigger_right = 0.0
#
#     # get the value with or without the deadzone
#     if not deadZone:
#         hand_trigger_left = ptr_ctrl_state[0].HandTriggerNoDeadzone[0]
#         hand_trigger_right = ptr_ctrl_state[0].HandTriggerNoDeadzone[1]
#     else:
#         hand_trigger_left = ptr_ctrl_state[0].HandTrigger[0]
#         hand_trigger_right = ptr_ctrl_state[0].HandTrigger[1]
#
#     return hand_trigger_left, hand_trigger_right
#
# cdef float clip_input_range(float val):
#     """Constrain an analog input device's range between -1.0 and 1.0. This is
#     only accessible from module functions.
#
#     :param val: float
#     :return: float
#
#     """
#     if val > 1.0:
#         val = 1.0
#     elif val < 1.0:
#         val = 1.0
#
#     return val
#
# cpdef tuple getThumbstickValues(str controller, bint deadZone=False):
#     """Get thumbstick values for a specified controller.
#
#     :param controller:
#     :param dead_zone:
#     :return: tuple
#
#     """
#     # get pointer to control state
#     global _ctrl_states_
#     cdef ovr_capi.ovrInputState*ptr_ctrl = NULL
#     cdef ovr_capi.ovrInputState*ptr_ctrl_prev = NULL
#     if controller == 'xbox':
#         ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.xbox]
#         ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.xbox]
#     elif controller == 'remote':
#         ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.remote]
#         ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.remote]
#     elif controller == 'touch':
#         ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.touch]
#         ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.touch]
#     elif controller == 'left_touch':
#         ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.left_touch]
#         ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.left_touch]
#     elif controller == 'right_touch':
#         ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.right_touch]
#         ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.right_touch]
#
#     cdef float thumbstick0_x = 0.0
#     cdef float thumbstick0_y = 0.0
#     cdef float thumbstick1_x = 0.0
#     cdef float thumbstick1_y = 0.0
#
#     # get the value with or without the deadzone
#     if not deadZone:
#         thumbstick0_x = ptr_ctrl[0].Thumbstick[0].x
#         thumbstick0_y = ptr_ctrl[0].Thumbstick[0].y
#         thumbstick1_x = ptr_ctrl[0].Thumbstick[1].x
#         thumbstick1_y = ptr_ctrl[0].Thumbstick[1].y
#     else:
#         thumbstick0_x = ptr_ctrl[0].ThumbstickNoDeadzone[0].x
#         thumbstick0_y = ptr_ctrl[0].ThumbstickNoDeadzone[0].y
#         thumbstick1_x = ptr_ctrl[0].ThumbstickNoDeadzone[1].x
#         thumbstick1_y = ptr_ctrl[0].ThumbstickNoDeadzone[1].y
#
#     # clip range
#     thumbstick0_x = clip_input_range(thumbstick0_x)
#     thumbstick0_y = clip_input_range(thumbstick0_y)
#     thumbstick1_x = clip_input_range(thumbstick1_x)
#     thumbstick1_y = clip_input_range(thumbstick1_y)
#
#     return (thumbstick0_x, thumbstick0_y), (thumbstick1_x, thumbstick1_y)
#
# cpdef bint getButtons(str controller, object buttonNames,
#                       str trigger='continuous'):
#     """Get the state of a specified button for a given controller.
#
#     Buttons to test are specified using their string names. Argument
#     'button_names' accepts a single string or a list. If a list is specified,
#     the returned value will reflect whether all buttons were triggered at the
#     time the controller was polled last.
#
#     An optional trigger mode may be specified which defines the button's
#     activation criteria. Be default, trigger='continuous' which will return the
#     immediate state of the button is used. Using 'rising' will return True once
#     when the button is first pressed, whereas 'falling' will return True once
#     the button is released.
#
#     :param controller: str
#     :param buttonNames: str, tuple or list
#     :param trigger: str
#     :return: boolean
#
#     """
#     # get pointer to control state
#     global _ctrl_states_
#     cdef ovr_capi.ovrInputState*ptr_ctrl = NULL
#     cdef ovr_capi.ovrInputState*ptr_ctrl_prev = NULL
#     if controller == 'xbox':
#         ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.xbox]
#         ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.xbox]
#     elif controller == 'remote':
#         ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.remote]
#         ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.remote]
#     elif controller == 'touch':
#         ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.touch]
#         ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.touch]
#     elif controller == 'left_touch':
#         ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.left_touch]
#         ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.left_touch]
#     elif controller == 'right_touch':
#         ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.right_touch]
#         ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.right_touch]
#
#     cdef unsigned int button_bits = 0x00000000
#     cdef int i, N
#     if isinstance(buttonNames, str):  # don't loop if a string is specified
#         button_bits |= ctrl_button_lut[buttonNames]
#     elif isinstance(buttonNames, (tuple, list)):
#         # loop over all names and combine them
#         N = <int> len(buttonNames)
#         for i in range(N):
#             button_bits |= ctrl_button_lut[buttonNames[i]]
#
#     # test if the button was pressed
#     cdef bint pressed
#     if trigger == 'continuous':
#         pressed = (ptr_ctrl.Buttons & button_bits) == button_bits
#     elif trigger == 'rising' or trigger == 'pressed':
#         # rising edge, will trigger once when pressed
#         pressed = (ptr_ctrl.Buttons & button_bits) == button_bits and \
#                   (ptr_ctrl_prev.Buttons & button_bits) != button_bits
#     elif trigger == 'falling' or trigger == 'released':
#         # falling edge, will trigger once when released
#         pressed = (ptr_ctrl.Buttons & button_bits) != button_bits and \
#                   (ptr_ctrl_prev.Buttons & button_bits) == button_bits
#     else:
#         raise ValueError("Invalid trigger mode specified.")
#
#     return pressed
#
# cpdef bint getTouches(str controller, object touchNames,
#                       str trigger='continuous'):
#     """Get touches for a specified device.
#
#     Touches reveal information about the user's hand pose, for instance, whether
#     a pointing or pinching gesture is being made. Oculus Touch controllers are
#     required for this functionality.
#
#     Touch points to test are specified using their string names. Argument
#     'touch_names' accepts a single string or a list. If a list is specified,
#     the returned value will reflect whether all touches were triggered at the
#     time the controller was polled last.
#
#     :param controller: str
#     :param touchNames: str, tuple or list
#     :param trigger: str
#     :return: boolean
#
#     """
#     # get pointer to control state
#     global _ctrl_states_
#     cdef ovr_capi.ovrInputState*ptr_ctrl = NULL
#     cdef ovr_capi.ovrInputState*ptr_ctrl_prev = NULL
#     if controller == 'xbox':
#         ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.xbox]
#         ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.xbox]
#     elif controller == 'remote':
#         ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.remote]
#         ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.remote]
#     elif controller == 'touch':
#         ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.touch]
#         ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.touch]
#     elif controller == 'left_touch':
#         ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.left_touch]
#         ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.left_touch]
#     elif controller == 'right_touch':
#         ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.right_touch]
#         ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.right_touch]
#
#     cdef unsigned int touch_bits = 0x00000000
#     cdef int i, N
#     if isinstance(touchNames, str):  # don't loop if a string is specified
#         touch_bits |= ctrl_button_lut[touchNames]
#     elif isinstance(touchNames, (tuple, list)):
#         # loop over all names and combine them
#         N = <int> len(touchNames)
#         for i in range(N):
#             touch_bits |= ctrl_button_lut[touchNames[i]]
#
#     # test if the button was pressed
#     cdef bint touched
#     if trigger == 'continuous':
#         touched = (ptr_ctrl.Touches & touch_bits) == touch_bits
#     elif trigger == 'rising' or trigger == 'pressed':
#         # rising edge, will trigger once when pressed
#         touched = (ptr_ctrl.Touches & touch_bits) == touch_bits and \
#                   (ptr_ctrl_prev.Touches & touch_bits) != touch_bits
#     elif trigger == 'falling' or trigger == 'released':
#         # falling edge, will trigger once when released
#         touched = (ptr_ctrl.Touches & touch_bits) != touch_bits and \
#                   (ptr_ctrl_prev.Touches & touch_bits) == touch_bits
#     else:
#         raise ValueError("Invalid trigger mode specified.")
#
#     return touched
#
# # List of controller names that are available to the user. These are handled by
# # the SDK, additional joysticks, keyboards and mice must be accessed by some
# # other method.
# #
# controller_names = ['xbox', 'remote', 'touch', 'left_touch', 'right_touch']
#
# cpdef list getConnectedControllerTypes():
#     """Get a list of currently connected controllers. You can check if a
#     controller is attached by testing for its membership in the list using its
#     name.
#
#     :return: list
#
#     """
#     cdef unsigned int result = ovr_capi.ovr_GetConnectedControllerTypes(
#         _ptrSession_)
#
#     cdef list ctrl_types = list()
#     if (result & ovr_capi.ovrControllerType_XBox) == \
#             ovr_capi.ovrControllerType_XBox:
#         ctrl_types.append('xbox')
#     elif (result & ovr_capi.ovrControllerType_Remote) == \
#             ovr_capi.ovrControllerType_Remote:
#         ctrl_types.append('remote')
#     elif (result & ovr_capi.ovrControllerType_Touch) == \
#             ovr_capi.ovrControllerType_Touch:
#         ctrl_types.append('touch')
#     elif (result & ovr_capi.ovrControllerType_LTouch) == \
#             ovr_capi.ovrControllerType_LTouch:
#         ctrl_types.append('left_touch')
#     elif (result & ovr_capi.ovrControllerType_RTouch) == \
#             ovr_capi.ovrControllerType_RTouch:
#         ctrl_types.append('right_touch')
#
#     return ctrl_types

# -------------------------------
# Performance/Profiling Functions
# -------------------------------
#
# cdef class ovrPerfStatsPerCompositorFrame(object):
#     cdef ovr_capi.ovrPerfStatsPerCompositorFrame*c_data
#     cdef ovr_capi.ovrPerfStatsPerCompositorFrame  c_ovrPerfStatsPerCompositorFrame
#
#     def __cinit__(self, *args, **kwargs):
#         self.c_data = &self.c_ovrPerfStatsPerCompositorFrame
#
#     @property
#     def HmdVsyncIndex(self):
#         return self.c_data[0].HmdVsyncIndex
#
#     @property
#     def AppFrameIndex(self):
#         return self.c_data[0].AppFrameIndex
#
#     @property
#     def AppDroppedFrameCount(self):
#         return self.c_data[0].AppDroppedFrameCount
#
#     @property
#     def AppQueueAheadTime(self):
#         return self.c_data[0].AppQueueAheadTime
#
#     @property
#     def AppCpuElapsedTime(self):
#         return self.c_data[0].AppCpuElapsedTime
#
#     @property
#     def AppGpuElapsedTime(self):
#         return self.c_data[0].AppGpuElapsedTime
#
#     @property
#     def CompositorFrameIndex(self):
#         return self.c_data[0].CompositorFrameIndex
#
#     @property
#     def CompositorLatency(self):
#         return self.c_data[0].CompositorLatency
#
#     @property
#     def CompositorCpuElapsedTime(self):
#         return self.c_data[0].CompositorCpuElapsedTime
#
#     @property
#     def CompositorGpuElapsedTime(self):
#         return self.c_data[0].CompositorGpuElapsedTime
#
#     @property
#     def CompositorCpuStartToGpuEndElapsedTime(self):
#         return self.c_data[0].CompositorCpuStartToGpuEndElapsedTime
#
#     @property
#     def CompositorGpuEndToVsyncElapsedTime(self):
#         return self.c_data[0].CompositorGpuEndToVsyncElapsedTime
#
#
# cdef class ovrPerfStats(object):
#     cdef ovr_capi.ovrPerfStats*c_data
#     cdef ovr_capi.ovrPerfStats  c_ovrPerfStats
#     cdef list perf_stats
#
#     def __cinit__(self, *args, **kwargs):
#         self.c_data = &self.c_ovrPerfStats
#
#         # initialize performance stats list
#         self.perf_stats = list()
#         cdef int i, N
#         N = <int> ovr_capi.ovrMaxProvidedFrameStats
#         for i in range(N):
#             self.perf_stats.append(ovrPerfStatsPerCompositorFrame())
#             (<ovrPerfStatsPerCompositorFrame> self.perf_stats[i]).c_data[0] = \
#                 self.c_data[0].FrameStats[i]
#
#     @property
#     def FrameStatsCount(self):
#         return self.c_data[0].FrameStatsCount
#
#     @property
#     def AnyFrameStatsDropped(self):
#         return <bint> self.c_data[0].AnyFrameStatsDropped
#
#     @property
#     def FrameStats(self):
#         cdef int i, N
#         N = self.c_data[0].FrameStatsCount
#         for i in range(N):
#             (<ovrPerfStatsPerCompositorFrame> self.perf_stats[i]).c_data[0] = \
#                 self.c_data[0].FrameStats[i]
#
#         return self.perf_stats
#
#     @property
#     def AdaptiveGpuPerformanceScale(self):
#         return <bint> self.c_data[0].AdaptiveGpuPerformanceScale
#
#     @property
#     def AswIsAvailable(self):
#         return <bint> self.c_data[0].AswIsAvailable
#
# cpdef ovrPerfStats getFrameStats():
#     """Get most recent performance stats, returns an object with fields
#     corresponding to various performance stats reported by the SDK.
#
#     :return: dict
#
#     """
#     global _ptrSession_
#
#     cdef ovrPerfStats to_return = ovrPerfStats()
#     cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetPerfStats(
#         _ptrSession_,
#         &(<ovrPerfStats> to_return).c_data[0])
#
#     if debug_mode:
#         check_result(result)
#
#     return to_return

