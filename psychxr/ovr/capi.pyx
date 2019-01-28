# distutils: language=c++
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
# __all__ = [
#     'LibOVRSession', 'LibOVRPose', 'LibOVRPoseState', 'createSession', 'initialize',
#     'LibOVRTrackerInfo', 'LibOVRSessionStatus', 'isOculusServiceRunning',
#     'LIBOVR_SUCCESS', 'LIBOVR_UNQUALIFIED_SUCCESS', 'LIBOVR_FAILURE',
#     'isHmdConnected', 'LIBOVR_SUCCESS', 'LIBOVR_SUCCESS_NOT_VISIBLE',
#     'LIBOVR_SUCCESS_BOUNDARY_INVALID', 'LIBOVR_SUCCESS_DEVICE_UNAVAILABLE',
#     'LIBOVR_ERROR_TEXTURE_SWAP_CHAIN_FULL'
#     ]

from .cimport ovr_capi
from .cimport ovr_math
#from .math cimport *

from libc.stdint cimport int32_t, uint32_t
from libc.math cimport pow

cimport numpy as np
import numpy as np
import collections

import OpenGL as GL

# -----------------
# Initialize module
# -----------------
#
cdef ovr_capi.ovrInitParams _initParams  # initialization parameters
cdef ovr_capi.ovrSession _ptrSession  # session pointer
cdef ovr_capi.ovrGraphicsLuid _ptrLuid  # LUID
cdef ovr_capi.ovrHmdDesc _hmdDesc  # HMD information descriptor
cdef ovr_capi.ovrBoundaryLookAndFeel _boundryStyle
cdef ovr_capi.ovrTextureSwapChain[8] _swapChains
cdef ovr_capi.ovrMirrorTexture _mirrorTexture

# VR related data persistent across frames
cdef ovr_capi.ovrLayerEyeFov _eyeLayer
cdef ovr_capi.ovrEyeRenderDesc[2] _eyeRenderDesc

# texture swap chains, for eye views and mirror

# status and performance information
cdef ovr_capi.ovrSessionStatus _sessionStatus
cdef ovr_capi.ovrPerfStats _perfStats
cdef list compFrameStats

# error information
cdef ovr_capi.ovrErrorInfo _errorInfo  # store our last error here

# controller states
cdef ovr_capi.ovrInputState[5] _inputStates
cdef ovr_capi.ovrInputState[5] _prevInputState

# debug mode
cdef bint _debugMode

# geometric data
cdef ovr_math.Matrix4f[2] _eyeProjectionMatrix
cdef ovr_math.Matrix4f[2] _eyeViewMatrix

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

cdef dict _controller_type_enum = {
    "Xbox": ovr_capi.ovrControllerType_XBox,
    "Remote": ovr_capi.ovrControllerType_Remote,
    "Touch": ovr_capi.ovrControllerType_Touch,
    "LTouch": ovr_capi.ovrControllerType_LTouch,
    "RTouch": ovr_capi.ovrControllerType_RTouch
}

cdef ovr_capi.ovrControllerType* libovr_controller_enum = [
    ovr_capi.ovrControllerType_XBox,
    ovr_capi.ovrControllerType_Remote,
    ovr_capi.ovrControllerType_Touch,
    ovr_capi.ovrControllerType_LTouch,
    ovr_capi.ovrControllerType_RTouch
]

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

# return error code, not all of these are applicable
LIBOVR_ERROR_MEMORY_ALLOCATION_FAILURE = ovr_capi.ovrError_MemoryAllocationFailure
LIBOVR_ERROR_INVALID_SESSION = ovr_capi.ovrError_InvalidSession
LIBOVR_ERROR_TIMEOUT = ovr_capi.ovrError_Timeout
LIBOVR_ERROR_NOT_INITIALIZED = ovr_capi.ovrError_NotInitialized
LIBOVR_ERROR_INVALID_PARAMETER = ovr_capi.ovrError_InvalidParameter
LIBOVR_ERROR_SERVICE_ERROR = ovr_capi.ovrError_ServiceError
LIBOVR_ERROR_NO_HMD = ovr_capi.ovrError_NoHmd
LIBOVR_ERROR_UNSUPPORTED = ovr_capi.ovrError_Unsupported
LIBOVR_ERROR_DEVICE_UNAVAILABLE = ovr_capi.ovrError_DeviceUnavailable
LIBOVR_ERROR_INVALID_HEADSET_ORIENTATION = ovr_capi.ovrError_InvalidHeadsetOrientation
LIBOVR_ERROR_CLIENT_SKIPPED_DESTROY = ovr_capi.ovrError_ClientSkippedDestroy
LIBOVR_ERROR_CLIENT_SKIPPED_SHUTDOWN = ovr_capi.ovrError_ClientSkippedShutdown
LIBOVR_ERROR_SERVICE_DEADLOCK_DETECTED = ovr_capi.ovrError_ServiceDeadlockDetected
LIBOVR_ERROR_INVALID_OPERATION = ovr_capi.ovrError_InvalidOperation
LIBOVR_ERROR_INSUFFICENT_ARRAY_SIZE = ovr_capi.ovrError_InsufficientArraySize
LIBOVR_ERROR_NO_EXTERNAL_CAMERA_INFO = ovr_capi.ovrError_NoExternalCameraInfo
LIBOVR_ERROR_LOST_TRACKING = ovr_capi.ovrError_LostTracking
LIBOVR_ERROR_EXTERNAL_CAMERA_INITIALIZED_FAILED = ovr_capi.ovrError_ExternalCameraInitializedFailed
LIBOVR_ERROR_EXTERNAL_CAMERA_CAPTURE_FAILED = ovr_capi.ovrError_ExternalCameraCaptureFailed
LIBOVR_ERROR_EXTERNAL_CAMERA_NAME_LISTS_BUFFER_SIZE = ovr_capi.ovrError_ExternalCameraNameListsBufferSize
LIBOVR_ERROR_EXTERNAL_CAMERA_NAME_LISTS_MISMATCH = ovr_capi.ovrError_ExternalCameraNameListsMistmatch
LIBOVR_ERROR_EXTERNAL_CAMERA_NOT_CALIBRATED = ovr_capi.ovrError_ExternalCameraNotCalibrated
LIBOVR_ERROR_EXTERNAL_CAMERA_NAME_WRONG_SIZE = ovr_capi.ovrError_ExternalCameraNameWrongSize
LIBOVR_ERROR_AUDIO_DEVICE_NOT_FOUND = ovr_capi.ovrError_AudioDeviceNotFound
LIBOVR_ERROR_AUDIO_COM_ERROR = ovr_capi.ovrError_AudioComError
LIBOVR_ERROR_INITIALIZE = ovr_capi.ovrError_Initialize
LIBOVR_ERROR_LIB_LOAD = ovr_capi.ovrError_LibLoad
LIBOVR_ERROR_SERVICE_CONNECTION = ovr_capi.ovrError_ServiceConnection
LIBOVR_ERROR_SERVICE_VERSION = ovr_capi.ovrError_ServiceVersion
LIBOVR_ERROR_INCOMPATIBLE_OS = ovr_capi.ovrError_IncompatibleOS
LIBOVR_ERROR_DISPLAY_INIT = ovr_capi.ovrError_DisplayInit
LIBOVR_ERROR_SERVER_START = ovr_capi.ovrError_ServerStart
LIBOVR_ERROR_REINITIALIZATION = ovr_capi.ovrError_Reinitialization
LIBOVR_ERROR_MISMATCHED_ADAPTERS = ovr_capi.ovrError_MismatchedAdapters
LIBOVR_ERROR_LEAKING_RESOURCES = ovr_capi.ovrError_LeakingResources
LIBOVR_ERROR_CLIENT_VERSION = ovr_capi.ovrError_ClientVersion
LIBOVR_ERROR_OUT_OF_DATE_OS = ovr_capi.ovrError_OutOfDateOS
LIBOVR_ERROR_OUT_OF_DATE_GFX_DRIVER = ovr_capi.ovrError_OutOfDateGfxDriver
LIBOVR_ERROR_INCOMPATIBLE_OS = ovr_capi.ovrError_IncompatibleGPU
LIBOVR_ERROR_NO_VALID_VR_DISPLAY_SYSTEM = ovr_capi.ovrError_NoValidVRDisplaySystem
LIBOVR_ERROR_OBSOLETE = ovr_capi.ovrError_Obsolete
LIBOVR_ERROR_DISABLED_OR_DEFAULT_ADAPTER = ovr_capi.ovrError_DisabledOrDefaultAdapter
LIBOVR_ERROR_HYBRID_GRAPHICS_NOT_SUPPORTED = ovr_capi.ovrError_HybridGraphicsNotSupported
LIBOVR_ERROR_DISPLAY_MANAGER_INIT = ovr_capi.ovrError_DisplayManagerInit
LIBOVR_ERROR_TRACKER_DRIVER_INIT = ovr_capi.ovrError_TrackerDriverInit
LIBOVR_ERROR_LIB_SIGN_CHECK = ovr_capi.ovrError_LibSignCheck
LIBOVR_ERROR_LIB_PATH = ovr_capi.ovrError_LibPath
LIBOVR_ERROR_LIB_SYMBOLS = ovr_capi.ovrError_LibSymbols
LIBOVR_ERROR_REMOTE_SESSION = ovr_capi.ovrError_RemoteSession
LIBOVR_ERROR_INITIALIZE_VULKAN = ovr_capi.ovrError_InitializeVulkan
LIBOVR_ERROR_BLACKLISTED_GFX_DRIVER = ovr_capi.ovrError_BlacklistedGfxDriver
LIBOVR_ERROR_DISPLAY_LOST = ovr_capi.ovrError_DisplayLost
LIBOVR_ERROR_TEXTURE_SWAP_CHAIN_FULL = ovr_capi.ovrError_TextureSwapChainFull
LIBOVR_ERROR_TEXTURE_SWAP_CHAIN_INVALID = ovr_capi.ovrError_TextureSwapChainInvalid
LIBOVR_ERROR_GRAPHICS_DEVICE_RESET = ovr_capi.ovrError_GraphicsDeviceReset
LIBOVR_ERROR_DISPLAY_REMOVED = ovr_capi.ovrError_DisplayRemoved
LIBOVR_ERROR_CONTENT_PROTECTION_NOT_AVAILABLE = ovr_capi.ovrError_ContentProtectionNotAvailable
LIBOVR_ERROR_APPLICATION_VISIBLE = ovr_capi.ovrError_ApplicationInvisible
LIBOVR_ERROR_DISALLOWED = ovr_capi.ovrError_Disallowed
LIBOVR_ERROR_DISPLAY_PLUGGED_INCORRECTY = ovr_capi.ovrError_DisplayPluggedIncorrectly
LIBOVR_ERROR_DISPLAY_LIMIT_REACHED = ovr_capi.ovrError_DisplayLimitReached
LIBOVR_ERROR_RUNTIME_EXCEPTION = ovr_capi.ovrError_RuntimeException
LIBOVR_ERROR_NO_CALIBRATION = ovr_capi.ovrError_NoCalibration
LIBOVR_ERROR_OLD_VERSION = ovr_capi.ovrError_OldVersion
LIBOVR_ERROR_MISFORMATTED_BLOCK = ovr_capi.ovrError_MisformattedBlock

# misc constants
LIBOVR_EYE_LEFT = ovr_capi.ovrEye_Left
LIBOVR_EYE_RIGHT = ovr_capi.ovrEye_Right
LIBOVR_EYE_COUNT = ovr_capi.ovrEye_Count
LIBOVR_HAND_LEFT = ovr_capi.ovrHand_Left
LIBOVR_HAND_RIGHT = ovr_capi.ovrHand_Right
LIBOVR_HAND_COUNT = ovr_capi.ovrHand_Count

# swapchain handles
LIBOVR_TEXTURE_SWAP_CHAIN0 = 0
LIBOVR_TEXTURE_SWAP_CHAIN1 = 1
LIBOVR_TEXTURE_SWAP_CHAIN2 = 2
LIBOVR_TEXTURE_SWAP_CHAIN3 = 3
LIBOVR_TEXTURE_SWAP_CHAIN4 = 4
LIBOVR_TEXTURE_SWAP_CHAIN5 = 5
LIBOVR_TEXTURE_SWAP_CHAIN6 = 6
LIBOVR_TEXTURE_SWAP_CHAIN7 = 7

def LIBOVR_SUCCESS(int result):
    """Check if an API return indicates success."""
    return <bint>ovr_capi.OVR_SUCCESS(result)

def LIBOVR_UNQUALIFIED_SUCCESS(int result):
    """Check if an API return indicates unqualified success."""
    return <bint>ovr_capi.OVR_UNQUALIFIED_SUCCESS(result)

def LIBOVR_FAILURE(int result):
    """Check if an API return indicates failure (error)."""
    return <bint>ovr_capi.OVR_FAILURE(result)

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

def initialize(bint focusAware=False, int connectionTimeout=0):
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
    global _initParams
    _initParams.Flags = flags
    _initParams.RequestedMinorVersion = 25
    _initParams.LogCallback = NULL  # not used yet
    _initParams.ConnectionTimeoutMS = <uint32_t>connectionTimeout
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_Initialize(
        &_initParams)

    return result  # failed to initalize, return error code

def createSession():
    """Create a new session. Control is handed over to the application from
    Oculus Home.

    Starting a session will initialize and create a new session. Afterwards
    API functions will return valid values.

    Returns
    -------
    int
        Result of the 'ovr_Create' API call. A session was successfully
        created if the result is :data:`LIBOVR_SUCCESS`.

    """
    global _ptrSession
    global _ptrLuid
    global _eyeLayer
    global _hmdDesc
    global _eyeRenderDesc

    result = ovr_capi.ovr_Create(&_ptrSession, &_ptrLuid)
    check_result(result)
    if ovr_capi.OVR_FAILURE(result):
        return result  # failed to create session, return error code

    # if we got to this point, everything should be fine
    # get HMD descriptor
    _hmdDesc = ovr_capi.ovr_GetHmdDesc(_ptrSession)

    # configure the eye render descriptor to use the recommended FOV, this
    # can be changed later
    cdef Py_ssize_t i = 0
    for i in range(ovr_capi.ovrEye_Count):
        _eyeRenderDesc[i] = ovr_capi.ovr_GetRenderDesc(
            _ptrSession,
            <ovr_capi.ovrEyeType>i,
            _hmdDesc.DefaultEyeFov[i])

        _eyeLayer.Fov[i] = _eyeRenderDesc[i].Fov

    # prepare the render layer
    _eyeLayer.Header.Type = ovr_capi.ovrLayerType_EyeFov
    _eyeLayer.Header.Flags = \
        ovr_capi.ovrLayerFlag_TextureOriginAtBottomLeft | \
        ovr_capi.ovrLayerFlag_HighQuality
    _eyeLayer.ColorTexture[0] = _eyeLayer.ColorTexture[1] = NULL

    return result

def destroySession():
    """Destroy a session and free all resources associated with it.

    This destroys all texture swap chains and the session handle. This
    should be done when closing the application, prior to calling 'shutdown'
    or in the event of an error (such as the display being lost).

    """
    global _ptrSession
    global _swapChains
    global _mirrorTexture
    global _eyeLayer

    # free all swap chains
    cdef int i = 0
    for i in range(32):
        ovr_capi.ovr_DestroyTextureSwapChain(_ptrSession, _swapChains[i])
        _swapChains[i] = NULL

    # null eye textures in eye layer
    _eyeLayer.ColorTexture[0] = _eyeLayer.ColorTexture[1] = NULL

    # destroy the mirror texture
    if _mirrorTexture != NULL:
        ovr_capi.ovr_DestroyMirrorTexture(_ptrSession, _mirrorTexture)

    # destroy the current session and shutdown
    ovr_capi.ovr_Destroy(_ptrSession)

def shutdown():
    """End the current session.

    Clean-up routines are executed that destroy all swap chains and mirror
    texture buffers, afterwards control is returned to Oculus Home. This
    must be called after every successful 'initialize' call.

    """
    ovr_capi.ovr_Shutdown()

def getHmdInfo():
    """Get information about a given tracker.

    """
    return

def highQuality(bint enable):
    """Enable high quality mode."""
    global _eyeLayer
    if enable:
        _eyeLayer.Header.Flags |= ovr_capi.ovrLayerFlag_HighQuality
    else:
        _eyeLayer.Header.Flags &= ~ovr_capi.ovrLayerFlag_HighQuality

def headLocked(bint enable):
    """True when head-locked mode is enabled.

    This is disabled by default when a session is started. Head locking
    places the rendered image as a 'billboard' in front of the viewer.

    """
    global _eyeLayer
    if enable:
        _eyeLayer.Header.Flags |= ovr_capi.ovrLayerFlag_HeadLocked
    else:
        _eyeLayer.Header.Flags &= ~ovr_capi.ovrLayerFlag_HeadLocked

def getEyeRenderFOV(int eye):
    """Get the field-of-view to use for rendering.

    The FOV for a given eye are defined as a tuple of tangent angles (Up,
    Down, Left, Right). By default, this function will return the default
    FOVs after 'start' is called (see 'defaultEyeFOVs'). You can override
    these values using 'maxEyeFOVs' and 'symmetricEyeFOVs', or with
    custom values (see Examples below).

    Examples
    --------
    Setting eye render FOVs to symmetric (needed for mono rendering)::

        hmd.eyeRenderFOVs = hmd.symmetricEyeFOVs

    Getting the tangent angles::

        leftFov, rightFov = hmd.eyeRenderFOVs
        # left FOV tangent angles, do the same for the right
        upTan, downTan, leftTan, rightTan =  leftFov

    Using custom values::

        # Up, Down, Left, Right tan angles
        leftFov = [1.0, -1.0, -1.0, 1.0]
        rightFov = [1.0, -1.0, -1.0, 1.0]
        hmd.eyeRenderFOVs = leftFov, rightFov

    """
    global _eyeRenderDesc
    cdef np.ndarray to_return = np.asarray([
        _eyeRenderDesc[eye].Fov.UpTan,
        _eyeRenderDesc[eye].Fov.DownTan,
        _eyeRenderDesc[eye].Fov.LeftTan,
        _eyeRenderDesc[eye].Fov.RightTan],
        dtype=np.float32)

    return to_return

def setEyeRenderFOV(int eye, object fov):
    """Set the field-of-view of a given eye. This is used to compute the
    projection matrix.

    Parameters
    ----------
    eye : int
        Eye index.
    fov : tuple, list or ndarray of floats
    texelPerPixel : float

    """
    global _ptrSession
    global _eyeRenderDesc
    global _eyeLayer

    cdef ovr_capi.ovrFovPort fov_in
    fov_in.UpTan = <float>fov[0]
    fov_in.DownTan = <float>fov[1]
    fov_in.LeftTan = <float>fov[2]
    fov_in.RightTan = <float>fov[3]

    _eyeRenderDesc[<int>eye] = ovr_capi.ovr_GetRenderDesc(
        _ptrSession,
        <ovr_capi.ovrEyeType>eye,
        fov_in)

    # set in eye layer too
    _eyeLayer.Fov[eye] = _eyeRenderDesc[eye].Fov


def getDefaultEyeFOVs():
    """Default or recommended eye field-of-views (FOVs) provided by the API.

    Returns
    -------
    tuple of ndarray
        Pair of left and right eye FOVs specified as tangent angles [Up,
        Down, Left, Right].

    """
    global _hmdDesc
    cdef np.ndarray fovLeft = np.asarray([
        _hmdDesc.DefaultEyeFov[0].UpTan,
        _hmdDesc.DefaultEyeFov[0].DownTan,
        _hmdDesc.DefaultEyeFov[0].LeftTan,
        _hmdDesc.DefaultEyeFov[0].RightTan],
        dtype=np.float32)

    cdef np.ndarray fovRight = np.asarray([
        _hmdDesc.DefaultEyeFov[1].UpTan,
        _hmdDesc.DefaultEyeFov[1].DownTan,
        _hmdDesc.DefaultEyeFov[1].LeftTan,
        _hmdDesc.DefaultEyeFov[1].RightTan],
        dtype=np.float32)

    return fovLeft, fovRight


def getMaxEyeFOVs():
    """Maximum eye field-of-views (FOVs) provided by the API.

    Returns
    -------
    tuple of ndarray
        Pair of left and right eye FOVs specified as tangent angles in
        radians [Up, Down, Left, Right].

    """
    global _hmdDesc
    cdef np.ndarray[float, ndim=1] fov_left = np.asarray([
        _hmdDesc.MaxEyeFov[0].UpTan,
        _hmdDesc.MaxEyeFov[0].DownTan,
        _hmdDesc.MaxEyeFov[0].LeftTan,
        _hmdDesc.MaxEyeFov[0].RightTan],
        dtype=np.float32)

    cdef np.ndarray[float, ndim=1] fov_right = np.asarray([
        _hmdDesc.MaxEyeFov[1].UpTan,
        _hmdDesc.MaxEyeFov[1].DownTan,
        _hmdDesc.MaxEyeFov[1].LeftTan,
        _hmdDesc.MaxEyeFov[1].RightTan],
        dtype=np.float32)

    return fov_left, fov_right

def getSymmetricEyeFOVs(self):
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
    global _hmdDesc
    cdef ovr_capi.ovrFovPort fov_left = _hmdDesc.DefaultEyeFov[0]
    cdef ovr_capi.ovrFovPort fov_right = _hmdDesc.DefaultEyeFov[1]

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

def calcEyeBufferSizes(texelsPerPixel=1.0):
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
    global _ptrSession
    global _eyeRenderDesc

    cdef ovr_capi.ovrSizei sizeLeft = ovr_capi.ovr_GetFovTextureSize(
        _ptrSession,
        <ovr_capi.ovrEyeType>0,
        _eyeRenderDesc[0].Fov,
        <float>texelsPerPixel)

    cdef ovr_capi.ovrSizei sizeRight = ovr_capi.ovr_GetFovTextureSize(
        _ptrSession,
        <ovr_capi.ovrEyeType>1,
        _eyeRenderDesc[1].Fov,
        <float>texelsPerPixel)

    return (sizeLeft.w, sizeLeft.h), (sizeRight.w, sizeRight.h)

def getSwapChainLengthGL(int swapChain):
    """Get the length of a specified swap chain.

    Parameters
    ----------
    swapChain : int
        Swap chain handle to query. Must be a swap chain initialized by a
        previous call to 'createTextureSwapChainGL'. Index values can range
        between 0 and 31.

    Returns
    -------
    tuple of int
        Result of the 'ovr_GetTextureSwapChainLength' API call and the
        length of that swap chain.

    """
    cdef int outLength
    cdef ovr_capi.ovrResult result = 0
    global _swapChains
    global _ptrSession
    global _eyeLayer

    # check if there is a swap chain in the slot
    if _eyeLayer.ColorTexture[swapChain] == NULL:
        raise RuntimeError(
            "Cannot get swap chain length, NULL eye buffer texture.")

    # get the current texture index within the swap chain
    result = ovr_capi.ovr_GetTextureSwapChainLength(
        _ptrSession, _swapChains[swapChain], &outLength)

    return result, outLength

def getSwapChainCurrentIndex(swapChain):
    """Get the current buffer index within the swap chain.

    Parameters
    ----------
    swapChain : int
        Swap chain handle to query. Must be a swap chain initialized by a
        previous call to 'createTextureSwapChainGL'. Index values can range
        between 0 and 31.

    Returns
    -------
    tuple of int
        Result of the 'ovr_GetTextureSwapChainCurrentIndex' API call and the
        index of the buffer.

    """
    cdef int current_idx = 0
    cdef ovr_capi.ovrResult result = 0
    global _swapChains
    global _eyeLayer
    global _ptrSession

    # check if there is a swap chain in the slot
    if _eyeLayer.ColorTexture[swapChain] == NULL:
        raise RuntimeError(
            "Cannot get buffer ID, NULL eye buffer texture.")

    # get the current texture index within the swap chain
    result = ovr_capi.ovr_GetTextureSwapChainCurrentIndex(
        _ptrSession, _swapChains[swapChain], &current_idx)

    return result, current_idx

def getTextureSwapChainBufferGL(int swapChain, int index):
    """Get the texture buffer as an OpenGL name at a specific index in the
    swap chain for a given swapChain.

    Parameters
    ----------
    swapChain : int
        Swap chain handle to query. Must be a swap chain initialized by a
        previous call to 'createTextureSwapChainGL'. Index values can range
        between 0 and 31.
    index : int
        Index within the swap chain to retrieve its OpenGL texture name.

    Returns
    -------
    tuple of ints
        Result of the 'ovr_GetTextureSwapChainBufferGL' API call and the
        OpenGL texture buffer name. A OpenGL buffer name is invalid when 0,
        check the returned API call result for an error condition.

    Examples
    --------
    Get the OpenGL texture buffer name associated with the swap chain index::

    # get the current available index
    result, currentIdx = hmd.getSwapChainCurrentIndex(swapChain)

    # get the OpenGL buffer name
    result, texId = hmd.getTextureSwapChainBufferGL(swapChain, currentIdx)

    # bind the texture
    GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
        GL.GL_TEXTURE_2D, texId, 0)

    """
    cdef unsigned int tex_id = 0  # OpenGL texture handle
    global _swapChains
    global _eyeLayer
    global _ptrSession

    # get the next available texture ID from the swap chain
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetTextureSwapChainBufferGL(
        _ptrSession, _swapChains[swapChain], index, &tex_id)

    return result, tex_id

def createTextureSwapChainGL(swapChain, width, height, textureFormat='R8G8B8A8_UNORM_SRGB', levels=1):
    """Create a texture swap chain for eye image buffers.

    You can create up-to 32 swap chains, referenced by their index.

    Parameters
    ----------
    swapChain : int
        Index to initialize.
    textureFormat : str
        Texture format, valid texture formats are 'R8G8B8A8_UNORM',
        'R8G8B8A8_UNORM_SRGB', 'R16G16B16A16_FLOAT', and 'R11G11B10_FLOAT'.
    width : int
        Width of texture in pixels.
    height : int
        Height of texture in pixels.
    levels : int
        Mip levels to use, default is 1.

    Returns
    -------
    int
        The result of the 'ovr_CreateTextureSwapChainGL' API call.

    """
    global _swapChains
    global _ptrSession

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
            _ptrSession,
            &swapConfig,
            &_swapChains[swapChain])

    #_eyeLayer.ColorTexture[swapChain] = _swapChains[swapChain]

    return result

def setEyeColorTextureSwapChain(int eye, int swapChain):
    """Set the color texture swap chain for a given eye.

    Should be called after a successful 'createTextureSwapChainGL' call but
    before any rendering is done.

    Parameters
    ----------
    eye : int
        Eye index.
    swapChain : int
        Swap chain handle to query. Must be a swap chain initialized by a
        previous call to 'createTextureSwapChainGL'. Index values can range
        between 0 and 31.

    """
    global _swapChains
    global _eyeLayer

    _eyeLayer.ColorTexture[eye] = _swapChains[swapChain]

def createMirrorTexture(width, height, textureFormat='R8G8B8A8_UNORM_SRGB'):
    """Create a mirror texture.

    This displays the content of the rendered images being presented on the
    HMD. The image is automatically refreshed to reflect the current content
    on the display. This displays the post-distortion texture.

    Parameters
    ----------
    width : int
        Width of texture in pixels.
    height : int
        Height of texture in pixels.
    textureFormat : str
        Texture format. Valid texture formats are: 'R8G8B8A8_UNORM',
        'R8G8B8A8_UNORM_SRGB', 'R16G16B16A16_FLOAT', and 'R11G11B10_FLOAT'.

    Returns
    -------
    int
        Result of API call 'ovr_CreateMirrorTextureGL'.

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
    global _ptrSession
    global _mirrorTexture

    mirrorDesc.Format = ovr_capi.OVR_FORMAT_R8G8B8A8_UNORM_SRGB
    mirrorDesc.Width = <int>width
    mirrorDesc.Height = <int>height
    mirrorDesc.MiscFlags = ovr_capi.ovrTextureMisc_None
    mirrorDesc.MirrorOptions = ovr_capi.ovrMirrorOption_Default

    cdef ovr_capi.ovrResult result = ovr_capi.ovr_CreateMirrorTextureGL(
        _ptrSession, &mirrorDesc, &_mirrorTexture)

    return <int>result

def getMirrorTexture():
    """Mirror texture ID.

    Returns
    -------
    tuple of int
        Result of API call 'ovr_GetMirrorTextureBufferGL' and the mirror
        texture ID. A mirror texture ID = 0 is invalid.

    """
    cdef unsigned int mirror_id

    global _ptrSession
    global _mirrorTexture

    if _mirrorTexture == NULL:  # no texture created
        return None

    cdef ovr_capi.ovrResult result = \
        ovr_capi.ovr_GetMirrorTextureBufferGL(
            _ptrSession,
            _mirrorTexture,
            &mirror_id)

    return <int>result, <unsigned int>mirror_id


def getTrackedPoses(double absTime, bint latencyMarker=True):
    """Get the current poses of the head and hands.

    Parameters
    ----------
    absTime : float
        Absolute time in seconds which the tracking state refers to.
    latencyMarker : bool
        Insert a latency marker for motion-to-photon calculation.

    Returns
    -------
    tuple of LibOVRPoseState
        Pose state for the head, left and right hands.

    Examples
    --------
    Getting the head pose and calculating eye render poses::

        t = hmd.getPredictedDisplayTime()
        head, leftHand, rightHand = hmd.getTrackedPoses(t)

        # check if tracking
        if head.orientationTracked and head.positionTracked:
            hmd.calcEyePose(head)  # calculate eye poses

    """
    global _ptrSession
    global _eyeLayer

    cdef ovr_capi.ovrBool use_marker = \
        ovr_capi.ovrTrue if latencyMarker else ovr_capi.ovrFalse

    cdef ovr_capi.ovrTrackingState tracking_state = \
        ovr_capi.ovr_GetTrackingState(_ptrSession, absTime, use_marker)

    cdef LibOVRPoseState head_pose = LibOVRPoseState()
    head_pose.c_data[0] = tracking_state.HeadPose
    head_pose.status_flags = tracking_state.StatusFlags

    # for computing app photon-to-motion latency
    _eyeLayer.SensorSampleTime = tracking_state.HeadPose.TimeInSeconds

    cdef LibOVRPoseState left_hand_pose = LibOVRPoseState()
    left_hand_pose.c_data[0] = tracking_state.HandPoses[0]
    left_hand_pose.status_flags = tracking_state.HandStatusFlags[0]

    cdef LibOVRPoseState right_hand_pose = LibOVRPoseState()
    right_hand_pose.c_data[0] = tracking_state.HandPoses[1]
    right_hand_pose.status_flags = tracking_state.HandStatusFlags[1]

    return head_pose, left_hand_pose, right_hand_pose

def calcEyePoses(LibOVRPose headPose):
    """Calculate eye poses using a given pose state.

    Eye poses are derived from the head pose stored in the pose state and
    the HMD to eye poses reported by LibOVR. Calculated eye poses are stored
    and passed to the compositor when 'endFrame' is called for additional
    rendering.

    You can access the computed poses via the 'renderPoses' attribute.

    Parameters
    ----------
    headPose : LibOVRPose
        Head pose.

    Examples
    --------

    Compute the eye poses from tracker data::

        t = hmd.getPredictedDisplayTime()
        headPose, leftHandPose, rightHandPose = hmd.getTrackedPoses(t)

        # check if tracking
        if head.orientationTracked and head.positionTracked:
            hmd.calcEyePoses(head.thePose)  # calculate eye poses
        else:
            # do something ...

        # computed render poses appear here
        renderPoseLeft, renderPoseRight = hmd.renderPoses

    Use a custom head pose::

        headPose = LibOVRPose((0., 1.5, 0.))  # eyes 1.5 meters off the ground
        hmd.calcEyePoses(headPose)  # calculate eye poses

    """
    global _ptrSession
    global _eyeLayer
    global _eyeRenderDesc
    global _eyeViewMatrix

    cdef ovr_capi.ovrPosef[2] hmdToEyePoses
    hmdToEyePoses[0] = _eyeRenderDesc[0].HmdToEyePose
    hmdToEyePoses[1] = _eyeRenderDesc[1].HmdToEyePose

     # calculate the eye poses
    ovr_capi.ovr_CalcEyePoses2(
        headPose.c_data[0],
        hmdToEyePoses,
        _eyeLayer.RenderPose)

    # compute the eye transformation matrices from poses
    cdef ovr_math.Vector3f pos
    cdef ovr_math.Quatf ori
    cdef ovr_math.Vector3f up
    cdef ovr_math.Vector3f forward
    cdef ovr_math.Matrix4f rm

    cdef int eye = 0
    for eye in range(ovr_capi.ovrEye_Count):
        pos = <ovr_math.Vector3f>_eyeLayer.RenderPose[eye].Position
        ori = <ovr_math.Quatf>_eyeLayer.RenderPose[eye].Orientation

        if not ori.IsNormalized():  # make sure orientation is normalized
            ori.Normalize()

        rm = ovr_math.Matrix4f(ori)
        up = rm.Transform(ovr_math.Vector3f(0., 1., 0.))
        forward = rm.Transform(ovr_math.Vector3f(0., 0., -1.))
        _eyeViewMatrix[eye] = ovr_math.Matrix4f.LookAtRH(pos, pos + forward, up)

def getHmdToEyePoses(self):
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
    global _eyeRenderDesc
    cdef LibOVRPose leftHmdToEyePose = LibOVRPose()
    cdef LibOVRPose rightHmdToEyePose = LibOVRPose()

    leftHmdToEyePose.c_data[0] = _eyeRenderDesc[0].HmdToEyePose
    leftHmdToEyePose.c_data[1] = _eyeRenderDesc[1].HmdToEyePose

    return leftHmdToEyePose, rightHmdToEyePose

def setHmdToEyePoses(value):
    global _eyeRenderDesc
    _eyeRenderDesc[0].HmdToEyePose = (<LibOVRPose>value[0]).c_data[0]
    _eyeRenderDesc[1].HmdToEyePose = (<LibOVRPose>value[1]).c_data[0]

def getEyeRenderPoses():
    """Get eye render poses.

    Pose are those computed by the last 'calcEyePoses' call. Returned
    objects are copies of the data stored internally by the session
    instance. These poses are used to define the view matrix when rendering
    for each eye.

    Notes
    -----
        The returned LibOVRPose objects are copies of data stored internally
        by the session object. Setting renderPoses will recompute their
        transformation matrices.

    """
    global _eyeLayer

    cdef LibOVRPose left_eye_pose = LibOVRPose()
    cdef LibOVRPose right_eye_pose = LibOVRPose()

    left_eye_pose.c_data[0] = _eyeLayer.RenderPose[0]
    right_eye_pose.c_data[0] = _eyeLayer.RenderPose[1]

    return left_eye_pose, right_eye_pose

def setEyeRenderPoses(object value):

    global _eyeLayer
    global _eyeViewMatrix

    _eyeLayer.RenderPose[0] = (<LibOVRPose>value[0]).c_data[0]
    _eyeLayer.RenderPose[1] = (<LibOVRPose>value[1]).c_data[0]

    # re-compute the eye transformation matrices from poses
    cdef ovr_math.Vector3f pos
    cdef ovr_math.Quatf ori
    cdef ovr_math.Vector3f up
    cdef ovr_math.Vector3f forward
    cdef ovr_math.Matrix4f rm

    cdef int eye = 0
    for eye in range(ovr_capi.ovrEye_Count):
        pos = <ovr_math.Vector3f>_eyeLayer.RenderPose[eye].Position
        ori = <ovr_math.Quatf>_eyeLayer.RenderPose[eye].Orientation

        if not ori.IsNormalized():  # make sure orientation is normalized
            ori.Normalize()

        rm = ovr_math.Matrix4f(ori)
        up = rm.Transform(ovr_math.Vector3f(0., 1., 0.))
        forward = rm.Transform(ovr_math.Vector3f(0., 0., -1.))
        _eyeViewMatrix[eye] = ovr_math.Matrix4f.LookAtRH(pos, pos + forward, up)


def getEyeProjectionMatrix(int eye, float nearClip=0.1, float farClip=1000.0):
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
    global _eyeProjectionMatrix
    global _eyeRenderDesc

    _eyeProjectionMatrix[eye] = \
        <ovr_math.Matrix4f>ovr_capi.ovrMatrix4f_Projection(
            _eyeRenderDesc[eye].Fov,
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
            mv[i, j] = _eyeProjectionMatrix[eye].M[i][j]

    return to_return

# @property
# def eyeRenderViewports(self):
#     """Eye viewports."""
#
#     global _eyeLayer
#     self._viewport_left.data = <char*>&_eyeLayer.Viewport[0]
#     self._viewport_right.data = <char*>&_eyeLayer.Viewport[1]
#
#     return self._viewport_left, self._viewport_right

def getEyeRenderViewport(int eye):
    global _eyeLayer
    cdef np.ndarray to_return = np.asarray(
        [_eyeLayer.Viewport[eye].Pos.x,
         _eyeLayer.Viewport[eye].Pos.y,
         _eyeLayer.Viewport[eye].Size.w,
         _eyeLayer.Viewport[eye].Size.h],
        dtype=np.float32)

    return to_return

def setEyeRenderViewport(int eye, object values):
    global _eyeLayer
    _eyeLayer.Viewport[eye].Pos.x = <int>values[0]
    _eyeLayer.Viewport[eye].Pos.y = <int>values[1]
    _eyeLayer.Viewport[eye].Size.w = <int>values[2]
    _eyeLayer.Viewport[eye].Size.h = <int>values[3]

def getEyeViewMatrix(int eye, bint flatten=False):
    """Compute a view matrix for a specified eye.

    View matrices are derived from the eye render poses calculated by the
    last 'calcEyePoses' call or update to 'renderPoses'.

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
    global _eyeViewMatrix
    cdef np.ndarray to_return
    cdef Py_ssize_t i, j, k, N
    i = j = k = 0
    N = 4
    if flatten:
        to_return = np.zeros((16,), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                to_return[k] = _eyeViewMatrix[eye].M[j][i]
                k += 1
    else:
        to_return = np.zeros((4, 4), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                to_return[i, j] = _eyeViewMatrix[eye].M[i][j]

    return to_return

def getPredictedDisplayTime(unsigned int frame_index=0):
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
    global _ptrSession
    cdef double t_sec = ovr_capi.ovr_GetPredictedDisplayTime(
        _ptrSession,
        frame_index)

    return t_sec

@property
def timeInSeconds():
    """Absolute time in seconds.

    Returns
    -------
    float
        Time in seconds.

    """
    cdef double t_sec = ovr_capi.ovr_GetTimeInSeconds()

    return t_sec

def perfHudMode(str mode):
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
    global _ptrSession
    cdef int perfHudMode = 0

    try:
        perfHudMode = <int>_performance_hud_modes[mode]
    except KeyError:
        raise KeyError("Invalid performance HUD mode specified.")

    cdef ovr_capi.ovrBool ret = ovr_capi.ovr_SetInt(
        _ptrSession, b"PerfHudMode", perfHudMode)

def hidePerfHud(self):
    """Hide the performance HUD.

    This is a convenience function that is equivalent to calling
    'perf_hud_mode('Off').

    """
    global _ptrSession
    cdef ovr_capi.ovrBool ret = ovr_capi.ovr_SetInt(
        _ptrSession, b"PerfHudMode", ovr_capi.ovrPerfHud_Off)

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
    global _eyeLayer
    cdef ovr_capi.ovrRecti viewportRect
    viewportRect.Pos.x = <int>rect[0]
    viewportRect.Pos.y = <int>rect[1]
    viewportRect.Size.w = <int>rect[2]
    viewportRect.Size.h = <int>rect[3]

    _eyeLayer.Viewport[eye] = viewportRect

def getEyeViewport(eye):
    """Get the viewport for a given eye.

    Parameters
    ----------
    eye : int
        Which eye to set the viewport, where left=0 and right=1.

    """
    global _eyeLayer
    cdef ovr_capi.ovrRecti viewportRect = \
        _eyeLayer.Viewport[eye]
    cdef np.ndarray to_return = np.asarray(
        [viewportRect.Pos.x,
         viewportRect.Pos.y,
         viewportRect.Size.w,
         viewportRect.Size.h],
        dtype=np.float32)

    return to_return

def waitToBeginFrame(unsigned int frameIndex=0):
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
    global _ptrSession
    cdef ovr_capi.ovrResult result = \
        ovr_capi.ovr_WaitToBeginFrame(_ptrSession, frameIndex)

    return <int>result

def beginFrame(unsigned int frameIndex=0):
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
    global _ptrSession
    cdef ovr_capi.ovrResult result = \
        ovr_capi.ovr_BeginFrame(_ptrSession, frameIndex)

    return <int>result

def commitSwapChain(int eye):
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
    global _swapChains
    global _ptrSession
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_CommitTextureSwapChain(
        _ptrSession,
        _swapChains[eye])

    return <int>result

def endFrame(unsigned int frameIndex=0):
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
    global _ptrSession
    global _eyeLayer

    cdef ovr_capi.ovrLayerHeader* layers = &_eyeLayer.Header
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_EndFrame(
        _ptrSession,
        frameIndex,
        NULL,
        &layers,
        <unsigned int>1)

    return result

def resetFrameStats():
    """Reset frame statistics.

    Returns
    -------
    int
        Error code returned by 'ovr_ResetPerfStats'.

    """
    global _ptrSession
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_ResetPerfStats(_ptrSession)

    return result

def getTrackingOriginType():
    """Tracking origin type.

    The tracking origin type specifies where the origin is placed when
    computing the pose of tracked objects (i.e. the head and touch
    controllers.) Valid values are 'floor' and 'eye'.

    """
    global _ptrSession
    cdef ovr_capi.ovrTrackingOrigin origin = \
        ovr_capi.ovr_GetTrackingOriginType(_ptrSession)

    if origin == ovr_capi.ovrTrackingOrigin_FloorLevel:
        return 'floor'
    elif origin == ovr_capi.ovrTrackingOrigin_EyeLevel:
        return 'eye'

def setTrackingOriginType(str value):
    cdef ovr_capi.ovrResult result
    global _ptrSession
    if value == 'floor':
        result = ovr_capi.ovr_SetTrackingOriginType(
            _ptrSession, ovr_capi.ovrTrackingOrigin_FloorLevel)
    elif value == 'eye':
        result = ovr_capi.ovr_SetTrackingOriginType(
            _ptrSession, ovr_capi.ovrTrackingOrigin_EyeLevel)

    return result

def recenterTrackingOrigin():
    """Recenter the tracking origin.

    Returns
    -------
    None

    """
    global _ptrSession
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_RecenterTrackingOrigin(
        _ptrSession)

    return result

def getTrackerInfo(int trackerIndex):
    """Get information about a given tracker.

    Parameters
    ----------
    trackerIndex : int
        The index of the sensor to query. Valid values are between 0 and
        '~LibOVRSession.trackerCount'.

    """
    cdef LibOVRTrackerInfo to_return = LibOVRTrackerInfo()
    global _ptrSession

    # set the descriptor data
    to_return.c_ovrTrackerDesc = ovr_capi.ovr_GetTrackerDesc(
        _ptrSession, <unsigned int>trackerIndex)
    # get the tracker pose
    to_return.c_ovrTrackerPose = ovr_capi.ovr_GetTrackerPose(
        _ptrSession, <unsigned int>trackerIndex)

    return to_return

def refreshPerformanceStats():
    """Refresh performance statistics.

    Should be called after 'endFrame'.

    """
    global _ptrSession
    global _perfStats
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetPerfStats(
        _ptrSession,
        &_perfStats)

    # clear
    cdef list compFrameStats = list()

    cdef int statIdx = 0
    cdef int numStats = _perfStats.FrameStatsCount
    for statIdx in range(numStats):
        frameStat = LibOVRCompFramePerfStat()
        frameStat.c_data[0] = _perfStats.FrameStats[statIdx]
        compFrameStats.append(frameStat)

    return result

# @property
# def maxProvidedFrameStats(self):
#     """Maximum number of frame stats provided."""
#     return 5
#
# @property
# def getFrameStatsCount(self):
#     """Number of frame stats available."""
#     return self.perfStats.FrameStatsCount
#
# @property
# def anyFrameStatsDropped(self):
#     """Have any frame stats been dropped?"""
#     return self.perfStats.AnyFrameStatsDropped
#
# @property
# def adaptiveGpuPerformanceScale(self):
#     """Adaptive GPU performance scaling factor."""
#     return self.perfStats.AdaptiveGpuPerformanceScale
#
# @property
# def aswIsAvailable(self):
#     """Is ASW available?"""
#     return self.perfStats.AswIsAvailable
#
# @property
# def frameStats(self):
#     """Get all frame compositior frame statistics."""
#     return self.compFrameStats

def getLastErrorInfo():
    """Get the last error code and information string reported by the API.

    This function can be used when implementing custom error handlers.

    Returns
    -------
    tuple of int, str
        Tuple of the API call result and error string.

    """
    cdef ovr_capi.ovrErrorInfo lastErrorInfo  # store our last error here
    ovr_capi.ovr_GetLastErrorInfo(&lastErrorInfo)

    cdef ovr_capi.ovrResult result = lastErrorInfo.Result
    cdef str errorString = lastErrorInfo.ErrorString.decode("utf-8")

    return <int>result, errorString

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
    global _boundryStyle
    global _ptrSession

    cdef ovr_capi.ovrColorf color
    color.r = <float>red
    color.g = <float>green
    color.b = <float>blue

    _boundryStyle.Color = color

    cdef ovr_capi.ovrResult result = ovr_capi.ovr_SetBoundaryLookAndFeel(
        _ptrSession,
        &_boundryStyle)

    return result

def resetBoundaryColor():
    """Reset the boundary color to system default.

    """
    global _ptrSession
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_ResetBoundaryLookAndFeel(
        _ptrSession)

    return result

def getBoundryVisible(self):
    """Check if the Guardian boundary is visible.

    The boundary is drawn by the compositor which overlays the extents of
    the physical space where the user can safely move.

    """
    global _ptrSession
    cdef ovr_capi.ovrBool is_visible
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetBoundaryVisible(
        _ptrSession, &is_visible)

    return result, is_visible

def showBoundary():
    """Show the boundary.

    The boundary is drawn by the compositor which overlays the extents of
    the physical space where the user can safely move.

    """
    global _ptrSession
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_RequestBoundaryVisible(
        _ptrSession, ovr_capi.ovrTrue)

    return result

def hideBoundary():
    """Hide the boundry."""
    global _ptrSession
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_RequestBoundaryVisible(
        _ptrSession, ovr_capi.ovrFalse)

    return result

def getBoundaryDimensions(str boundaryType='PlayArea'):
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
    global _ptrSession
    cdef ovr_capi.ovrBoundaryType btype
    if boundaryType == 'PlayArea':
        btype = ovr_capi.ovrBoundary_PlayArea
    elif boundaryType == 'Outer':
        btype = ovr_capi.ovrBoundary_Outer
    else:
        raise ValueError("Invalid boundary type specified.")

    cdef ovr_capi.ovrVector3f vec_out
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetBoundaryDimensions(
            _ptrSession, btype, &vec_out)

    cdef np.ndarray[np.float32_t, ndim=1] to_return = np.asarray(
        (vec_out.x, vec_out.y, vec_out.z), dtype=np.float32)

    return result, to_return

def getBoundaryPoints(self, str boundaryType='PlayArea'):
    """Get the floor points which define the boundary."""
    pass  # TODO: make this work.

def getConnectedControllers():
    """List of connected controllers.

    Returns
    -------
    tuple of int and list
        List of connected controller names. Check if a specific controller
        is available by checking the membership of its name in the list.

    Examples
    --------

    Check if the Xbox gamepad is connected::

        hasGamepad = "Xbox" in hmd.getConnectedControllers()

    Check if the left and right touch controllers are both paired::

        connected = hmd.getConnectedControllers()
        isPaired = 'LeftTouch' in connected and 'RightTouch' in connected

    """
    global _ptrSession
    cdef unsigned int result = ovr_capi.ovr_GetConnectedControllerTypes(
        _ptrSession)

    cdef list controllerTypes = list()
    if (result & ovr_capi.ovrControllerType_XBox) == \
            ovr_capi.ovrControllerType_XBox:
        controllerTypes.append('Xbox')
    elif (result & ovr_capi.ovrControllerType_Remote) == \
            ovr_capi.ovrControllerType_Remote:
        controllerTypes.append('Remote')
    elif (result & ovr_capi.ovrControllerType_Touch) == \
            ovr_capi.ovrControllerType_Touch:
        controllerTypes.append('Touch')
    elif (result & ovr_capi.ovrControllerType_LTouch) == \
            ovr_capi.ovrControllerType_LTouch:
        controllerTypes.append('LeftTouch')
    elif (result & ovr_capi.ovrControllerType_RTouch) == \
            ovr_capi.ovrControllerType_RTouch:
        controllerTypes.append('RightTouch')

    return result, controllerTypes

def refreshInputState(str controller):
    """Refresh the input state of a controller.

    Parameters
    ----------
    controller : str
        Controller name to poll. Valid names are: 'Xbox', 'Remote', 'Touch',
        'LeftTouch', and 'RightTouch'.

    """
    global _prevInputState
    global _inputStates
    global _ptrSession

    # convert the string to an index
    cdef dict idx = {'Xbox' : 0, 'Remote' : 1, 'Touch' : 2, 'LeftTouch' : 3,
                     'RightTouch' : 4}

    # pointer to the current and previous input state
    cdef ovr_capi.ovrInputState* previousInputState = \
        &_prevInputState[idx[controller]]
    cdef ovr_capi.ovrInputState* currentInputState = \
        &_inputStates[idx[controller]]

    # copy the current input state into the previous before updating
    previousInputState[0] = currentInputState[0]

    # get the current input state
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetInputState(
        _ptrSession,
        _controller_type_enum[controller],  # get the enum for the controller
        currentInputState)

    return result

def getButtons(str controller, object buttonNames, str testState='continuous'):
    """Get the state of a specified button for a given controller.

    Buttons to test are specified using their string names. Argument
    'buttonNames' accepts a single string or a list. If a list is specified,
    the returned value will reflect whether all buttons were triggered at
    the time the controller was polled last.

    An optional trigger mode may be specified which defines the button's
    activation criteria. By default, testState='continuous' will return the
    immediate state of the button. Using 'rising' (and 'pressed') will
    return True once when the button transitions to being pressed, whereas
    'falling' (and 'released') will return True once the button is released.

    Parameters
    ----------
    controller : str
        Controller name to poll. Valid names are: 'Xbox', 'Remote', 'Touch',
        'LeftTouch', and 'RightTouch'.
    buttonNames : tuple of str, list of str, or str
        Button names to test for state changes.
    testState : str
        State to test buttons for. Valid states are 'rising', 'falling',
        'continuous', 'pressed', and 'released'.

    Returns
    -------
    bool
        Result of the button press.

    """
    global _prevInputState
    global _inputStates

    # convert the string to an index
    cdef dict idx = {'Xbox' : 0, 'Remote' : 1, 'Touch' : 2, 'LeftTouch' : 3,
                     'RightTouch' : 4}

    # pointer to the current and previous input state
    cdef unsigned int curButtons = \
        _inputStates[idx[controller]].Buttons
    cdef unsigned int prvButtons = \
        _prevInputState[idx[controller]].Buttons

    # generate a bit mask for testing button presses
    cdef unsigned int buttonBits = 0x00000000
    cdef int i, N
    if isinstance(buttonNames, str):  # don't loop if a string is specified
        buttonBits |= ctrl_button_lut[buttonNames]
    elif isinstance(buttonNames, (tuple, list)):
        # loop over all names and combine them
        N = <int>len(buttonNames)
        for i in range(N):
            buttonBits |= ctrl_button_lut[buttonNames[i]]

    # test if the button was pressed
    cdef bint stateResult = False
    if testState == 'continuous':
        stateResult = (curButtons & buttonBits) == buttonBits
    elif testState == 'rising' or testState == 'pressed':
        # rising edge, will trigger once when pressed
        stateResult = (curButtons & buttonBits) == buttonBits and \
                      (prvButtons & buttonBits) != buttonBits
    elif testState == 'falling' or testState == 'released':
        # falling edge, will trigger once when released
        stateResult = (curButtons & buttonBits) != buttonBits and \
                      (prvButtons & buttonBits) == buttonBits
    else:
        raise ValueError("Invalid trigger mode specified.")

    return stateResult

def getTouches(str controller, object touchNames, str testState='continuous'):
    """Get touches for a specified device.

    Touches reveal information about the user's hand pose, for instance,
    whether a pointing or pinching gesture is being made. Oculus Touch
    controllers are required for this functionality.

    Touch points to test are specified using their string names. Argument
    'touch_names' accepts a single string or a list. If a list is specified,
    the returned value will reflect whether all touches were triggered at
    the time the controller was polled last.

    """
    # convert the string to an index
    cdef dict idx = {'Xbox' : 0, 'Remote' : 1, 'Touch' : 2, 'LeftTouch' : 3,
                     'RightTouch' : 4}

    global _prevInputState
    global _inputStates

    # pointer to the current and previous input state
    cdef unsigned int curTouches = \
        _inputStates[idx[controller]].Touches
    cdef unsigned int prvTouches = \
        _prevInputState[idx[controller]].Touches

    # generate a bit mask for testing button presses
    cdef unsigned int touchBits = 0x00000000
    cdef int i, N
    if isinstance(touchNames, str):  # don't loop if a string is specified
        touchBits |= ctrl_touch_lut[touchNames]
    elif isinstance(touchNames, (tuple, list)):
        # loop over all names and combine them
        N = <int>len(touchNames)
        for i in range(N):
            touchBits |= ctrl_touch_lut[touchNames[i]]

    # test if the button was pressed
    cdef bint stateResult = False
    if testState == 'continuous':
        stateResult = (curTouches & touchBits) == touchBits
    elif testState == 'rising' or testState == 'pressed':
        # rising edge, will trigger once when pressed
        stateResult = (curTouches & touchBits) == touchBits and \
                      (prvTouches & touchBits) != touchBits
    elif testState == 'falling' or testState == 'released':
        # falling edge, will trigger once when released
        stateResult = (curTouches & touchBits) != touchBits and \
                      (prvTouches & touchBits) == touchBits
    else:
        raise ValueError("Invalid trigger mode specified.")

    return stateResult

def getThumbstickValues(str controller, bint deadzone=False):
    """Get thumbstick values."""
    cdef dict idx = {'Xbox' : 0, 'Touch' : 2, 'LeftTouch' : 3,
                     'RightTouch' : 4}

    global _inputStates

    # pointer to the current and previous input state
    cdef ovr_capi.ovrInputState* currentInputState = \
        &_inputStates[idx[controller]]

    cdef float thumbstick_x0 = 0.0
    cdef float thumbstick_y0 = 0.0
    cdef float thumbstick_x1 = 0.0
    cdef float thumbstick_y1 = 0.0

    if deadzone:
        thumbstick_x0 = currentInputState[0].Thumbstick[0].x
        thumbstick_y0 = currentInputState[0].Thumbstick[0].y
        thumbstick_x1 = currentInputState[0].Thumbstick[1].x
        thumbstick_y1 = currentInputState[0].Thumbstick[1].y
    else:
        thumbstick_x0 = currentInputState[0].ThumbstickNoDeadzone[0].x
        thumbstick_y0 = currentInputState[0].ThumbstickNoDeadzone[0].y
        thumbstick_x1 = currentInputState[0].ThumbstickNoDeadzone[1].x
        thumbstick_y1 = currentInputState[0].ThumbstickNoDeadzone[1].y

    return (thumbstick_x0, thumbstick_y0), (thumbstick_x1, thumbstick_y1)

def getIndexTriggerValues(str controller, bint deadzone=False):
    """Get index trigger values."""
    # convert the string to an index
    cdef dict idx = {'Xbox' : 0, 'Remote' : 1, 'Touch' : 2, 'LeftTouch' : 3,
                     'RightTouch' : 4}

    global _inputStates

    # pointer to the current and previous input state
    cdef ovr_capi.ovrInputState* currentInputState = \
        &_inputStates[idx[controller]]

    cdef float indexTriggerLeft = 0.0
    cdef float indexTriggerRight = 0.0

    if deadzone:
        indexTriggerLeft = currentInputState[0].IndexTrigger[0]
        indexTriggerRight = currentInputState[0].IndexTrigger[1]
    else:
        indexTriggerLeft = currentInputState[0].IndexTriggerNoDeadzone[0]
        indexTriggerRight = currentInputState[0].IndexTriggerNoDeadzone[1]

    return indexTriggerLeft, indexTriggerRight

def getHandTriggerValues(str controller, bint deadzone=False):
    """Get hand trigger values."""
    # convert the string to an index
    cdef dict idx = {'Xbox' : 0, 'Touch' : 2, 'LeftTouch' : 3,
                     'RightTouch' : 4}

    global _inputStates

    # pointer to the current and previous input state
    cdef ovr_capi.ovrInputState* currentInputState = \
        &_inputStates[idx[controller]]

    cdef float indexTriggerLeft = 0.0
    cdef float indexTriggerRight = 0.0

    if deadzone:
        indexTriggerLeft = currentInputState[0].HandTrigger[0]
        indexTriggerRight = currentInputState[0].HandTrigger[1]
    else:
        indexTriggerLeft = currentInputState[0].HandTriggerNoDeadzone[0]
        indexTriggerRight = currentInputState[0].HandTriggerNoDeadzone[1]

    return indexTriggerLeft, indexTriggerRight

def setControllerVibration(str controller, str frequency, float amplitude):
    """Vibrate a controller.

    Vibration is constant at fixed frequency and amplitude. Vibration lasts
    2.5 seconds, so this function needs to be called more often than that
    for sustained vibration. Only controllers which support vibration can be
    used here.

    There are only two frequencies permitted 'high' and 'low', however,
    amplitude can vary from 0.0 to 1.0. Specifying frequency='off' stops
    vibration.

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
    global _ptrSession

    # get frequency associated with the string
    cdef float freq = 0.0
    if frequency == 'off':
        freq = amplitude = 0.0
    elif frequency == 'low':
        freq = 0.5
    elif frequency == 'high':
        freq = 1.0
    else:
        raise RuntimeError("Invalid frequency specified.")

    cdef dict _controller_types = {
        'Xbox' : ovr_capi.ovrControllerType_XBox,
        'Touch' : ovr_capi.ovrControllerType_Touch,
        'LeftTouch' : ovr_capi.ovrControllerType_LTouch,
        'RightTouch' : ovr_capi.ovrControllerType_RTouch}

    cdef ovr_capi.ovrResult result = ovr_capi.ovr_SetControllerVibration(
        _ptrSession,
        <ovr_capi.ovrControllerType>_controller_types[controller],
        freq,
        amplitude)

    return result

def getSessionStatus():
    """Get the current session status.

    Returns
    -------
    LibOVRSessionStatus
        Object specifying the current state of the session.

    """
    global _ptrSession
    cdef LibOVRSessionStatus to_return = LibOVRSessionStatus()
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetSessionStatus(
        _ptrSession, to_return.c_data)

    return to_return

# cdef class LibOVRSession(object):
#     """Session object for LibOVR.
#
#     This class provides an interface for LibOVR sessions. Once initialized,
#     the LibOVRSession instance provides configuration, data acquisition (e.g.
#     sensors, inputs, etc.), and control of the HMD via attributes and methods.
#
#     LibOVR API functions which return matrix and vector data types are converted
#     to Numpy arrays.
#
#     """
#     # Session related pointers and information
#     cdef ovr_capi.ovrInitParams initParams  # initialization parameters
#     cdef ovr_capi.ovrSession ptrSession  # session pointer
#     #cdef ovr_capi.ovrGraphicsLuid ptrLuid  # LUID
#     cdef ovr_capi.ovrHmdDesc hmdDesc  # HMD information descriptor
#     cdef ovr_capi.ovrBoundaryLookAndFeel boundryStyle
#
#     # VR related data persistent across frames
#     cdef ovr_capi.ovrLayerEyeFov eyeLayer
#     cdef ovr_capi.ovrEyeRenderDesc[2] eyeRenderDesc
#
#     # texture swap chains, for eye views and mirror
#
#     # status and performance information
#     cdef ovr_capi.ovrSessionStatus sessionStatus
#     cdef ovr_capi.ovrPerfStats perfStats
#     cdef list compFrameStats
#
#     # error information
#     cdef ovr_capi.ovrErrorInfo errorInfo  # store our last error here
#
#     # controller states
#     cdef ovr_capi.ovrInputState[5] inputStates
#     cdef ovr_capi.ovrInputState[5] prevInputState
#
#     # debug mode
#     cdef bint debugMode
#
#     # view objects
#     cdef np.ndarray _viewport_left
#     cdef np.ndarray _viewport_right
#
#     # geometric data
#     cdef ovr_math.Matrix4f[2] eyeProjectionMatrix
#     cdef ovr_math.Matrix4f[2] eyeViewMatrix
#
#     def __init__(self, raiseErrors=False, timeout=100, *args, **kwargs):
#         """Constructor for LibOVRSession.
#
#         Parameters
#         ----------
#         raiseErrors : bool
#             Raise exceptions when LibOVR functions return errors. If False,
#             returned values of some methods will need to be checked for error
#             conditions.
#         timeout : int
#             Connection timeout in milliseconds.
#
#         """
#         pass
#
#     def __cinit__(self, bint raiseErrors=False, int timeout=100, *args, **kwargs):
#         self.debugMode = raiseErrors
#
#         # view objects
#         self._viewport_left = np.empty((4,), dtype=np.int)
#         self._viewport_right = np.empty((4,), dtype=np.int)
#
#         self.compFrameStats = list()
#
#         # check if the driver and service are available
#         cdef ovr_capi.ovrDetectResult result = \
#             ovr_capi.ovr_Detect(<int>timeout)
#
#         if not result.IsOculusServiceRunning:
#             raise RuntimeError("Oculus service is not running, it may be "
#                                "disabled or not installed.")
#
#         if not result.IsOculusHMDConnected:
#             raise RuntimeError("No Oculus HMD connected! Check connections "
#                                "and try again.")
#
#     def __dealloc__(self):
#         pass
#
#     @property
#     def userHeight(self):
#         """User's calibrated height in meters.
#
#         Getter
#         ------
#         float
#             Distance from floor to the top of the user's head in meters reported
#             by LibOVR. If not set, the default value is 1.778 meters.
#
#         """
#         cdef float to_return = ovr_capi.ovr_GetFloat(
#             self.ptrSession,
#             b"PlayerHeight",
#             <float> 1.778)
#
#         return to_return
#
#     @property
#     def eyeHeight(self):
#         """Calibrated eye height from floor in meters.
#
#         Getter
#         ------
#         float
#             Distance from floor to the user's eye level in meters.
#
#         """
#         cdef float to_return = ovr_capi.ovr_GetFloat(
#             self.ptrSession,
#             b"EyeHeight",
#             <float> 1.675)
#
#         return to_return
#
#     @property
#     def neckEyeDist(self):
#         """Distance from the neck to eyes in meters.
#
#         Getter
#         ------
#         float
#             Distance in meters.
#
#         """
#         cdef float vals[2]
#
#         cdef unsigned int ret = ovr_capi.ovr_GetFloatArray(
#             self.ptrSession,
#             b"NeckEyeDistance",
#             vals,
#             <unsigned int>2)
#
#         return <float> vals[0], <float> vals[1]
#
#     @property
#     def eyeToNoseDist(self):
#         """Distance between the nose and eyes in meters.
#
#         Getter
#         ------
#         float
#             Distance in meters.
#
#         """
#         cdef float vals[2]
#
#         cdef unsigned int ret = ovr_capi.ovr_GetFloatArray(
#             self.ptrSession,
#             b"EyeToNoseDist",
#             vals,
#             <unsigned int> 2)
#
#         return <float>vals[0], <float> vals[1]
#
#     @property
#     def productName(self):
#         """Get the product name for this device.
#
#         Getter
#         ------
#         str
#             Product name string (utf-8).
#
#         """
#         return self.hmdDesc.ProductName.decode('utf-8')
#
#     @property
#     def manufacturerName(self):
#         """Get the device manufacturer name.
#
#         Getter
#         ------
#         str
#             Manufacturer name string (utf-8).
#
#         """
#         return self.hmdDesc.Manufacturer.decode('utf-8')
#
#     @property
#     def serialNumber(self):
#         """Get the device serial number.
#
#         Getter
#         ------
#         str
#             Serial number (utf-8).
#
#         """
#         return self.hmdDesc.SerialNumber.decode('utf-8')
#
#     @property
#     def screenSize(self):
#         """Horizontal and vertical resolution of the display in pixels.
#
#         Getter
#         ------
#         ndarray of int
#             Resolution of the display [w, h].
#
#         """
#         return np.asarray(
#             (self.hmdDesc.Resolution.w, self.hmdDesc.Resolution.h),
#             dtype=int)
#
#     @property
#     def refreshRate(self):
#         """Nominal refresh rate in Hertz of the display.
#
#         Getter
#         ------
#         float
#             Refresh rate in Hz.
#
#         """
#         return <float>self.hmdDesc.DisplayRefreshRate
#
#     @property
#     def hid(self):
#         """USB human interface device class identifiers.
#
#         Getter
#         ------
#         tuple
#             USB HIDs (vendor, product).
#
#         """
#         return <int>self.hmdDesc.VendorId, <int>self.hmdDesc.ProductId
#
#     @property
#     def firmwareVersion(self):
#         """Firmware version for this device.
#
#         Getter
#         ------
#         tuple
#             Firmware version (major, minor).
#
#         """
#         return <int>self.hmdDesc.FirmwareMajor, <int>self.hmdDesc.FirmwareMinor
#
#     @property
#     def versionString(self):
#         """LibOVRRT version as a string.
#
#         Getter
#         ------
#         str
#             Runtime version information as a UTF-8 encoded string.
#
#         """
#         cdef const char* version = ovr_capi.ovr_GetVersionString()
#         return version.decode('utf-8')  # already UTF-8?
#
#     def initialize(self, bint focusAware=False, int connectionTimeout=0):
#         """Initialize the session.
#
#         Parameters
#         ----------
#         focusAware : bool
#             Client is focus aware.
#         connectionTimeout : bool
#             Timeout in milliseconds for connecting to the server.
#
#         Returns
#         -------
#         int
#             Return code of the LibOVR API call 'ovr_Initialize'. Returns
#             LIBOVR_SUCCESS if completed without errors. In the event of an
#             error, possible return values are:
#
#             - :data:`LIBOVR_ERROR_INITIALIZE`: Initialization error.
#             - :data:`LIBOVR_ERROR_LIB_LOAD`:  Failed to load LibOVRRT.
#             - :data:`LIBOVR_ERROR_LIB_VERSION`:  LibOVRRT version incompatible.
#             - :data:`LIBOVR_ERROR_SERVICE_CONNECTION`:  Cannot connect to OVR service.
#             - :data:`LIBOVR_ERROR_SERVICE_VERSION`: OVR service version is incompatible.
#             - :data:`LIBOVR_ERROR_INCOMPATIBLE_OS`: Operating system version is incompatible.
#             - :data:`LIBOVR_ERROR_DISPLAY_INIT`: Unable to initialize the HMD.
#             - :data:`LIBOVR_ERROR_SERVER_START`:  Cannot start a server.
#             - :data:`LIBOVR_ERROR_REINITIALIZATION`: Reinitialized with a different version.
#
#         Raises
#         ------
#         RuntimeError
#             Raised if 'debugMode' is True and the API call to
#             'ovr_Initialize' returns an error.
#
#         """
#         cdef int32_t flags = ovr_capi.ovrInit_RequestVersion
#         if focusAware is True:
#             flags |= ovr_capi.ovrInit_FocusAware
#
#         #if debug is True:
#         #    flags |= ovr_capi.ovrInit_Debug
#
#         self.initParams.Flags = flags
#         self.initParams.RequestedMinorVersion = 32
#         self.initParams.LogCallback = NULL  # not used yet
#         self.initParams.ConnectionTimeoutMS = <uint32_t>connectionTimeout
#         cdef ovr_capi.ovrResult result = ovr_capi.ovr_Initialize(
#             &self.initParams)
#
#         if self.debugMode:
#             check_result(result)
#
#         return result  # failed to initalize, return error code
#
#     @property
#     def isReady(self):
#         """True if a session has been started.
#
#         This should return True if 'createSession' was previously called and was
#         successful. Will return false after 'destroySession' is called.
#
#         """
#         return self.ptrSession != NULL
#
#     # def createSession(self):
#     #     """Create a new session. Control is handed over to the application from
#     #     Oculus Home.
#     #
#     #     Starting a session will initialize and create a new session. Afterwards
#     #     API functions will return valid values.
#     #
#     #     Returns
#     #     -------
#     #     int
#     #         Result of the 'ovr_Create' API call. A session was successfully
#     #         created if the result is :data:`LIBOVR_SUCCESS`.
#     #
#     #     """
#     #     result = ovr_capi.ovr_Create(&self.ptrSession, &self.ptrLuid)
#     #     check_result(result)
#     #     if ovr_capi.OVR_FAILURE(result):
#     #         return result  # failed to create session, return error code
#     #
#     #     # if we got to this point, everything should be fine
#     #     # get HMD descriptor
#     #     self.hmdDesc = ovr_capi.ovr_GetHmdDesc(self.ptrSession)
#     #
#     #     # configure the eye render descriptor to use the recommended FOV, this
#     #     # can be changed later
#     #     cdef Py_ssize_t i = 0
#     #     for i in range(ovr_capi.ovrEye_Count):
#     #         self.eyeRenderDesc[i] = ovr_capi.ovr_GetRenderDesc(
#     #             self.ptrSession,
#     #             <ovr_capi.ovrEyeType>i,
#     #             self.hmdDesc.DefaultEyeFov[i])
#     #
#     #         self.eyeLayer.Fov[i] = self.eyeRenderDesc[i].Fov
#     #
#     #     # prepare the render layer
#     #     self.eyeLayer.Header.Type = ovr_capi.ovrLayerType_EyeFov
#     #     self.eyeLayer.Header.Flags = \
#     #         ovr_capi.ovrLayerFlag_TextureOriginAtBottomLeft | \
#     #         ovr_capi.ovrLayerFlag_HighQuality
#     #     self.eyeLayer.ColorTexture[0] = self.eyeLayer.ColorTexture[1] = NULL
#     #
#     #     return result
#
#     def destroySession(self):
#         """Destroy a session and free all resources associated with it.
#
#         This destroys all texture swap chains and the session handle. This
#         should be done when closing the application, prior to calling 'shutdown'
#         or in the event of an error (such as the display being lost).
#
#         """
#         global swapChains
#         global mirrorTexture
#         # free all swap chains
#         cdef int i = 0
#         for i in range(32):
#             ovr_capi.ovr_DestroyTextureSwapChain(
#                 self.ptrSession, swapChains[i])
#             swapChains[i] = NULL
#
#         # null eye textures in eye layer
#         self.eyeLayer.ColorTexture[0] = self.eyeLayer.ColorTexture[1] = NULL
#
#         # destroy the mirror texture
#         if mirrorTexture != NULL:
#             ovr_capi.ovr_DestroyMirrorTexture(self.ptrSession, mirrorTexture)
#
#         # destroy the current session and shutdown
#         ovr_capi.ovr_Destroy(self.ptrSession)
#
#     def shutdown(self):
#         """End the current session.
#
#         Clean-up routines are executed that destroy all swap chains and mirror
#         texture buffers, afterwards control is returned to Oculus Home. This
#         must be called after every successful 'initialize' call.
#
#         """
#         ovr_capi.ovr_Shutdown()
#
#     @property
#     def debugMode(self):
#         """Enable session debugging. Python exceptions are raised if the LibOVR
#         API returns an error. If 'debugMode=False', API errors will be silently
#         ignored.
#
#         """
#         return self.debugMode
#
#     @debugMode.setter
#     def debugMode(self, value):
#         self.debugMode = value
#
#     @property
#     def highQuality(self):
#         """True when high quality mode is enabled.
#
#         The distortion compositor applies 4x anisotropic texture filtering which
#         reduces the visibility of artifacts, particularly in the periphery.
#
#         This is enabled by default when a session is started.
#
#         """
#         return (self.eyeLayer.Header.Flags &
#                 ovr_capi.ovrLayerFlag_HighQuality) == \
#                ovr_capi.ovrLayerFlag_HighQuality
#
#     @highQuality.setter
#     def highQuality(self, value):
#         if value:
#             self.eyeLayer.Header.Flags |= ovr_capi.ovrLayerFlag_HighQuality
#         else:
#             self.eyeLayer.Header.Flags &= ~ovr_capi.ovrLayerFlag_HighQuality
#
#     @property
#     def headLocked(self):
#         """True when head-locked mode is enabled.
#
#         This is disabled by default when a session is started. Head locking
#         places the rendered image as a 'billboard' in front of the viewer.
#
#         """
#         return (self.eyeLayer.Header.Flags &
#                 ovr_capi.ovrLayerFlag_HeadLocked) == \
#                ovr_capi.ovrLayerFlag_HeadLocked
#
#     @headLocked.setter
#     def headLocked(self, value):
#         if value:
#             self.eyeLayer.Header.Flags |= ovr_capi.ovrLayerFlag_HeadLocked
#         else:
#             self.eyeLayer.Header.Flags &= ~ovr_capi.ovrLayerFlag_HeadLocked
#
#     @property
#     def trackerCount(self):
#         """Number of connected trackers."""
#         cdef unsigned int trackerCount = ovr_capi.ovr_GetTrackerCount(
#             self.ptrSession)
#
#         return <int>trackerCount
#
#     @property
#     def defaultEyeFOVs(self):
#         """Default or recommended eye field-of-views (FOVs) provided by the API.
#
#         Returns
#         -------
#         tuple of ndarray
#             Pair of left and right eye FOVs specified as tangent angles [Up,
#             Down, Left, Right].
#
#         """
#         cdef np.ndarray fovLeft = np.asarray([
#             self.hmdDesc.DefaultEyeFov[0].UpTan,
#             self.hmdDesc.DefaultEyeFov[0].DownTan,
#             self.hmdDesc.DefaultEyeFov[0].LeftTan,
#             self.hmdDesc.DefaultEyeFov[0].RightTan],
#             dtype=np.float32)
#
#         cdef np.ndarray fovRight = np.asarray([
#             self.hmdDesc.DefaultEyeFov[1].UpTan,
#             self.hmdDesc.DefaultEyeFov[1].DownTan,
#             self.hmdDesc.DefaultEyeFov[1].LeftTan,
#             self.hmdDesc.DefaultEyeFov[1].RightTan],
#             dtype=np.float32)
#
#         return fovLeft, fovRight
#
#     @property
#     def maxEyeFOVs(self):
#         """Maximum eye field-of-views (FOVs) provided by the API.
#
#         Returns
#         -------
#         tuple of ndarray
#             Pair of left and right eye FOVs specified as tangent angles in
#             radians [Up, Down, Left, Right].
#
#         """
#         cdef np.ndarray[float, ndim=1] fov_left = np.asarray([
#             self.hmdDesc.MaxEyeFov[0].UpTan,
#             self.hmdDesc.MaxEyeFov[0].DownTan,
#             self.hmdDesc.MaxEyeFov[0].LeftTan,
#             self.hmdDesc.MaxEyeFov[0].RightTan],
#             dtype=np.float32)
#
#         cdef np.ndarray[float, ndim=1] fov_right = np.asarray([
#             self.hmdDesc.MaxEyeFov[1].UpTan,
#             self.hmdDesc.MaxEyeFov[1].DownTan,
#             self.hmdDesc.MaxEyeFov[1].LeftTan,
#             self.hmdDesc.MaxEyeFov[1].RightTan],
#             dtype=np.float32)
#
#         return fov_left, fov_right
#
#     @property
#     def symmetricEyeFOVs(self):
#         """Symmetric field-of-views (FOVs) for mono rendering.
#
#         By default, the Rift uses off-axis FOVs. These frustum parameters make
#         it difficult to converge monoscopic stimuli.
#
#         Returns
#         -------
#         tuple of ndarray of float
#             Pair of left and right eye FOVs specified as tangent angles in
#             radians [Up, Down, Left, Right]. Both FOV objects will have the same
#             values.
#
#         """
#         cdef ovr_capi.ovrFovPort fov_left = self.hmdDesc.DefaultEyeFov[0]
#         cdef ovr_capi.ovrFovPort fov_right = self.hmdDesc.DefaultEyeFov[1]
#
#         cdef ovr_capi.ovrFovPort fov_max
#         fov_max.UpTan = maxf(fov_left.UpTan, fov_right.Uptan)
#         fov_max.DownTan = maxf(fov_left.DownTan, fov_right.DownTan)
#         fov_max.LeftTan = maxf(fov_left.LeftTan, fov_right.LeftTan)
#         fov_max.RightTan = maxf(fov_left.RightTan, fov_right.RightTan)
#
#         cdef float tan_half_fov_horz = maxf(fov_max.LeftTan, fov_max.RightTan)
#         cdef float tan_half_fov_vert = maxf(fov_max.DownTan, fov_max.UpTan)
#
#         cdef ovr_capi.ovrFovPort fov_both
#         fov_both.LeftTan = fov_both.RightTan = tan_half_fov_horz
#         fov_both.UpTan = fov_both.DownTan = tan_half_fov_horz
#
#         cdef np.ndarray[float, ndim=1] fov_left_out = np.asarray([
#             fov_both.UpTan,
#             fov_both.DownTan,
#             fov_both.LeftTan,
#             fov_both.RightTan],
#             dtype=np.float32)
#
#         cdef np.ndarray[float, ndim=1] fov_right_out = np.asarray([
#             fov_both.UpTan,
#             fov_both.DownTan,
#             fov_both.LeftTan,
#             fov_both.RightTan],
#             dtype=np.float32)
#
#         return fov_left_out, fov_right_out
#
#     @property
#     def eyeRenderFOVs(self):
#         """Field-of-view to use for rendering.
#
#         The FOV for a given eye are defined as a tuple of tangent angles (Up,
#         Down, Left, Right). By default, this function will return the default
#         FOVs after 'start' is called (see 'defaultEyeFOVs'). You can override
#         these values using 'maxEyeFOVs' and 'symmetricEyeFOVs', or with
#         custom values (see Examples below).
#
#         Examples
#         --------
#         Setting eye render FOVs to symmetric (needed for mono rendering)::
#
#             hmd.eyeRenderFOVs = hmd.symmetricEyeFOVs
#
#         Getting the tangent angles::
#
#             leftFov, rightFov = hmd.eyeRenderFOVs
#             # left FOV tangent angles, do the same for the right
#             upTan, downTan, leftTan, rightTan =  leftFov
#
#         Using custom values::
#
#             # Up, Down, Left, Right tan angles
#             leftFov = [1.0, -1.0, -1.0, 1.0]
#             rightFov = [1.0, -1.0, -1.0, 1.0]
#             hmd.eyeRenderFOVs = leftFov, rightFov
#
#         """
#         cdef np.ndarray left_fov = np.asarray([
#             self.eyeRenderDesc[0].Fov.UpTan,
#             self.eyeRenderDesc[0].Fov.DownTan,
#             self.eyeRenderDesc[0].Fov.LeftTan,
#             self.eyeRenderDesc[0].Fov.RightTan],
#             dtype=np.float32)
#
#         cdef np.ndarray right_fov = np.asarray([
#             self.eyeRenderDesc[1].Fov.UpTan,
#             self.eyeRenderDesc[1].Fov.DownTan,
#             self.eyeRenderDesc[1].Fov.LeftTan,
#             self.eyeRenderDesc[1].Fov.RightTan],
#             dtype=np.float32)
#
#         return left_fov, right_fov
#
#     @eyeRenderFOVs.setter
#     def eyeRenderFOVs(self, object value):
#         cdef ovr_capi.ovrFovPort fov_in
#         cdef int i = 0
#         for i in range(ovr_capi.ovrEye_Count):
#             fov_in.UpTan = <float>value[i][0]
#             fov_in.DownTan = <float>value[i][1]
#             fov_in.LeftTan = <float>value[i][2]
#             fov_in.RightTan = <float>value[i][3]
#
#             self.eyeRenderDesc[i] = ovr_capi.ovr_GetRenderDesc(
#                 self.ptrSession,
#                 <ovr_capi.ovrEyeType>i,
#                 fov_in)
#
#             self.eyeLayer.Fov[i] = self.eyeRenderDesc[i].Fov
#
#     def getEyeRenderFOV(self, int eye):
#         """Get the field-of-view of a given eye used to compute the projection
#         matrix.
#
#         Returns
#         -------
#         tuple of ndarray
#             Eye FOVs specified as tangent angles [Up, Down, Left, Right].
#
#         """
#         cdef np.ndarray to_return = np.asarray([
#             self.eyeRenderDesc[eye].Fov.UpTan,
#             self.eyeRenderDesc[eye].Fov.DownTan,
#             self.eyeRenderDesc[eye].Fov.LeftTan,
#             self.eyeRenderDesc[eye].Fov.RightTan],
#             dtype=np.float32)
#
#         return to_return
#
#     def setEyeRenderFOV(self, int eye, object fov):
#         """Set the field-of-view of a given eye. This is used to compute the
#         projection matrix.
#
#         Parameters
#         ----------
#         eye : int
#             Eye index.
#         fov : tuple, list or ndarray of floats
#         texelPerPixel : float
#
#         """
#         cdef ovr_capi.ovrFovPort fov_in
#         fov_in.UpTan = <float>fov[0]
#         fov_in.DownTan = <float>fov[1]
#         fov_in.LeftTan = <float>fov[2]
#         fov_in.RightTan = <float>fov[3]
#
#         self.eyeRenderDesc[<int>eye] = ovr_capi.ovr_GetRenderDesc(
#             self.ptrSession,
#             <ovr_capi.ovrEyeType>eye,
#             fov_in)
#
#         # set in eye layer too
#         self.eyeLayer.Fov[eye] = self.eyeRenderDesc[eye].Fov
#
#     def calcEyeBufferSizes(self, texelsPerPixel=1.0):
#         """Get the recommended buffer (texture) sizes for eye buffers.
#
#         Should be called after 'eye_render_fovs' is set. Returns left and
#         right buffer resolutions (w, h). The values can be used when configuring
#         a framebuffer for rendering to the HMD eye buffers.
#
#         Parameters
#         ----------
#         texelsPerPixel : float
#             Display pixels per texture pixels at the center of the display.
#             Use a value less than 1.0 to improve performance at the cost of
#             resolution. Specifying a larger texture is possible, but not
#             recommended by the manufacturer.
#
#         Returns
#         -------
#         tuple of tuples
#             Buffer widths and heights (w, h) for each eye.
#
#         Examples
#         --------
#         Getting the buffer sizes::
#
#             hmd.eyeRenderFOVs = hmd.defaultEyeFOVs  # set the FOV
#             leftBufferSize, rightBufferSize = hmd.calcEyeBufferSizes()
#             left_w, left_h = leftBufferSize
#             right_w, right_h = rightBufferSize
#             # combined size if using a single texture buffer for both eyes
#             w, h = left_w + right_w, max(left_h, right_h)
#             # make the texture ...
#
#         Notes
#         -----
#         This function returns the recommended texture resolution for each eye.
#         If you are using a single buffer for both eyes, that buffer should be
#         as wide as the combined width of both returned size.
#
#         """
#         cdef ovr_capi.ovrSizei sizeLeft = ovr_capi.ovr_GetFovTextureSize(
#             self.ptrSession,
#             <ovr_capi.ovrEyeType>0,
#             self.eyeRenderDesc[0].Fov,
#             <float>texelsPerPixel)
#
#         cdef ovr_capi.ovrSizei sizeRight = ovr_capi.ovr_GetFovTextureSize(
#             self.ptrSession,
#             <ovr_capi.ovrEyeType>1,
#             self.eyeRenderDesc[1].Fov,
#             <float>texelsPerPixel)
#
#         return (sizeLeft.w, sizeLeft.h), (sizeRight.w, sizeRight.h)
#
#     def getSwapChainLengthGL(self, int swapChain):
#         """Get the length of a specified swap chain.
#
#         Parameters
#         ----------
#         swapChain : int
#             Swap chain handle to query. Must be a swap chain initialized by a
#             previous call to 'createTextureSwapChainGL'. Index values can range
#             between 0 and 31.
#
#         Returns
#         -------
#         tuple of int
#             Result of the 'ovr_GetTextureSwapChainLength' API call and the
#             length of that swap chain.
#
#         """
#         cdef int outLength
#         cdef ovr_capi.ovrResult result = 0
#         global swapChains
#
#         # check if there is a swap chain in the slot
#         if self.eyeLayer.ColorTexture[swapChain] == NULL:
#             raise RuntimeError(
#                 "Cannot get swap chain length, NULL eye buffer texture.")
#
#         # get the current texture index within the swap chain
#         result = ovr_capi.ovr_GetTextureSwapChainLength(
#             self.ptrSession, swapChains[swapChain], &outLength)
#
#         return result, outLength
#
#     def getSwapChainCurrentIndex(self, swapChain):
#         """Get the current buffer index within the swap chain.
#
#         Parameters
#         ----------
#         swapChain : int
#             Swap chain handle to query. Must be a swap chain initialized by a
#             previous call to 'createTextureSwapChainGL'. Index values can range
#             between 0 and 31.
#
#         Returns
#         -------
#         tuple of int
#             Result of the 'ovr_GetTextureSwapChainCurrentIndex' API call and the
#             index of the buffer.
#
#         """
#         cdef int current_idx = 0
#         cdef ovr_capi.ovrResult result = 0
#         global swapChains
#
#         # check if there is a swap chain in the slot
#         if self.eyeLayer.ColorTexture[swapChain] == NULL:
#             raise RuntimeError(
#                 "Cannot get buffer ID, NULL eye buffer texture.")
#
#         # get the current texture index within the swap chain
#         result = ovr_capi.ovr_GetTextureSwapChainCurrentIndex(
#             self.ptrSession, swapChains[swapChain], &current_idx)
#
#         return result, current_idx
#
#     def getTextureSwapChainBufferGL(self, int swapChain, int index):
#         """Get the texture buffer as an OpenGL name at a specific index in the
#         swap chain for a given swapChain.
#
#         Parameters
#         ----------
#         swapChain : int
#             Swap chain handle to query. Must be a swap chain initialized by a
#             previous call to 'createTextureSwapChainGL'. Index values can range
#             between 0 and 31.
#         index : int
#             Index within the swap chain to retrieve its OpenGL texture name.
#
#         Returns
#         -------
#         tuple of ints
#             Result of the 'ovr_GetTextureSwapChainBufferGL' API call and the
#             OpenGL texture buffer name. A OpenGL buffer name is invalid when 0,
#             check the returned API call result for an error condition.
#
#         Examples
#         --------
#         Get the OpenGL texture buffer name associated with the swap chain index::
#
#         # get the current available index
#         result, currentIdx = hmd.getSwapChainCurrentIndex(swapChain)
#
#         # get the OpenGL buffer name
#         result, texId = hmd.getTextureSwapChainBufferGL(swapChain, currentIdx)
#
#         # bind the texture
#         GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
#             GL.GL_TEXTURE_2D, texId, 0)
#
#         """
#         cdef unsigned int tex_id = 0  # OpenGL texture handle
#         global swapChains
#
#         # get the next available texture ID from the swap chain
#         cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetTextureSwapChainBufferGL(
#             self.ptrSession, swapChains[swapChain], index, &tex_id)
#
#         return result, tex_id
#
#     def createTextureSwapChainGL(self, swapChain, width, height, textureFormat='R8G8B8A8_UNORM_SRGB', levels=1):
#         """Create a texture swap chain for eye image buffers.
#
#         You can create up-to 32 swap chains, referenced by their index.
#
#         Parameters
#         ----------
#         swapChain : int
#             Index to initialize.
#         textureFormat : str
#             Texture format, valid texture formats are 'R8G8B8A8_UNORM',
#             'R8G8B8A8_UNORM_SRGB', 'R16G16B16A16_FLOAT', and 'R11G11B10_FLOAT'.
#         width : int
#             Width of texture in pixels.
#         height : int
#             Height of texture in pixels.
#         levels : int
#             Mip levels to use, default is 1.
#
#         Returns
#         -------
#         int
#             The result of the 'ovr_CreateTextureSwapChainGL' API call.
#
#         """
#         global swapChains
#         # configure the texture
#         cdef ovr_capi.ovrTextureSwapChainDesc swapConfig
#         swapConfig.Type = ovr_capi.ovrTexture_2D
#         swapConfig.Format = _supported_texture_formats[textureFormat]
#         swapConfig.ArraySize = 1
#         swapConfig.Width = <int>width
#         swapConfig.Height = <int>height
#         swapConfig.MipLevels = <int>levels
#         swapConfig.SampleCount = 1
#         swapConfig.StaticImage = ovr_capi.ovrFalse
#         swapConfig.MiscFlags = ovr_capi.ovrTextureMisc_None
#         swapConfig.BindFlags = ovr_capi.ovrTextureBind_None
#
#         # create the swap chain
#         cdef ovr_capi.ovrResult result = \
#             ovr_capi.ovr_CreateTextureSwapChainGL(
#                 self.ptrSession,
#                 &swapConfig,
#                 &swapChains[swapChain])
#
#         #self.eyeLayer.ColorTexture[swapChain] = self.swapChains[swapChain]
#
#         return result
#
#     def setEyeColorTextureSwapChain(self, int eye, int swapChain):
#         """Set the color texture swap chain for a given eye.
#
#         Should be called after a successful 'createTextureSwapChainGL' call but
#         before any rendering is done.
#
#         Parameters
#         ----------
#         eye : int
#             Eye index.
#         swapChain : int
#             Swap chain handle to query. Must be a swap chain initialized by a
#             previous call to 'createTextureSwapChainGL'. Index values can range
#             between 0 and 31.
#
#         """
#         global swapChains
#         self.eyeLayer.ColorTexture[eye] = swapChains[swapChain]
#
#     def createMirrorTexture(
#             self,
#             width,
#             height,
#             textureFormat='R8G8B8A8_UNORM_SRGB'):
#         """Create a mirror texture.
#
#         This displays the content of the rendered images being presented on the
#         HMD. The image is automatically refreshed to reflect the current content
#         on the display. This displays the post-distortion texture.
#
#         Parameters
#         ----------
#         width : int
#             Width of texture in pixels.
#         height : int
#             Height of texture in pixels.
#         textureFormat : str
#             Texture format. Valid texture formats are: 'R8G8B8A8_UNORM',
#             'R8G8B8A8_UNORM_SRGB', 'R16G16B16A16_FLOAT', and 'R11G11B10_FLOAT'.
#
#         Returns
#         -------
#         int
#             Result of API call 'ovr_CreateMirrorTextureGL'.
#
#         """
#         # additional options
#         #cdef unsigned int mirror_options = ovr_capi.ovrMirrorOption_Default
#         # set the mirror texture mode
#         #if mirrorMode == 'Default':
#         #    mirror_options = <ovr_capi.ovrMirrorOptions>ovr_capi.ovrMirrorOption_Default
#         #elif mirrorMode == 'PostDistortion':
#         #    mirror_options = <ovr_capi.ovrMirrorOptions>ovr_capi.ovrMirrorOption_PostDistortion
#         #elif mirrorMode == 'LeftEyeOnly':
#         #    mirror_options = <ovr_capi.ovrMirrorOptions>ovr_capi.ovrMirrorOption_LeftEyeOnly
#         #elif mirrorMode == 'RightEyeOnly':
#         #    mirror_options = <ovr_capi.ovrMirrorOptions>ovr_capi.ovrMirrorOption_RightEyeOnly
#         #else:
#         #    raise RuntimeError("Invalid 'mirrorMode' mode specified.")
#
#         #if include_guardian:
#         #    mirror_options |= ovr_capi.ovrMirrorOption_IncludeGuardian
#         #if include_notifications:
#         #    mirror_options |= ovr_capi.ovrMirrorOption_IncludeNotifications
#         #if include_system_gui:
#         #    mirror_options |= ovr_capi.ovrMirrorOption_IncludeSystemGui
#
#         # create the descriptor
#         cdef ovr_capi.ovrMirrorTextureDesc mirrorDesc
#         global swapChains
#
#         mirrorDesc.Format = ovr_capi.OVR_FORMAT_R8G8B8A8_UNORM_SRGB
#         mirrorDesc.Width = <int>width
#         mirrorDesc.Height = <int>height
#         mirrorDesc.MiscFlags = ovr_capi.ovrTextureMisc_None
#         mirrorDesc.MirrorOptions = ovr_capi.ovrMirrorOption_Default
#
#         cdef ovr_capi.ovrResult result = ovr_capi.ovr_CreateMirrorTextureGL(
#             self.ptrSession, &mirrorDesc, &mirrorTexture)
#
#         if self.debugMode:
#             check_result(result)
#
#         return <int>result
#
#     @property
#     def getMirrorTexture(self):
#         """Mirror texture ID.
#
#         Returns
#         -------
#         tuple of int
#             Result of API call 'ovr_GetMirrorTextureBufferGL' and the mirror
#             texture ID. A mirror texture ID = 0 is invalid.
#
#         """
#         cdef unsigned int mirror_id
#         global swapChains
#         if mirrorTexture is NULL:  # no texture created
#             return None
#
#         cdef ovr_capi.ovrResult result = \
#             ovr_capi.ovr_GetMirrorTextureBufferGL(
#                 self.ptrSession,
#                 mirrorTexture,
#                 &mirror_id)
#
#         return <int>result, <unsigned int>mirror_id
#
#     def autoSetup(self, int numEyeBuffers=1, object mirrorSize=(800, 600), ):
#         """Automatically setup the session.
#
#         Parameters
#         ----------
#         numEyeBuffers : int
#             Number of eye buffers to use. If 1, a single buffer is used for both
#             eye textures. If 2, separate swap chains will be used for each eye.
#             The viewports will be configured accordingly.
#         mirrorSize : tuple of ints
#             Width and height of the mirror texture to create.
#
#         """
#         pass
#
#     def getTrackedPoses(self, double absTime, bint latencyMarker=True):
#         """Get the current poses of the head and hands.
#
#         Parameters
#         ----------
#         absTime : float
#             Absolute time in seconds which the tracking state refers to.
#         latencyMarker : bool
#             Insert a latency marker for motion-to-photon calculation.
#
#         Returns
#         -------
#         tuple of LibOVRPoseState
#             Pose state for the head, left and right hands.
#
#         Examples
#         --------
#         Getting the head pose and calculating eye render poses::
#
#             t = hmd.getPredictedDisplayTime()
#             head, leftHand, rightHand = hmd.getTrackedPoses(t)
#
#             # check if tracking
#             if head.orientationTracked and head.positionTracked:
#                 hmd.calcEyePose(head)  # calculate eye poses
#
#         """
#         cdef ovr_capi.ovrBool use_marker = \
#             ovr_capi.ovrTrue if latencyMarker else ovr_capi.ovrFalse
#
#         cdef ovr_capi.ovrTrackingState tracking_state = \
#             ovr_capi.ovr_GetTrackingState(self.ptrSession, absTime, use_marker)
#
#         cdef LibOVRPoseState head_pose = LibOVRPoseState()
#         head_pose.c_data[0] = tracking_state.HeadPose
#         head_pose.status_flags = tracking_state.StatusFlags
#
#         # for computing app photon-to-motion latency
#         self.eyeLayer.SensorSampleTime = tracking_state.HeadPose.TimeInSeconds
#
#         cdef LibOVRPoseState left_hand_pose = LibOVRPoseState()
#         left_hand_pose.c_data[0] = tracking_state.HandPoses[0]
#         left_hand_pose.status_flags = tracking_state.HandStatusFlags[0]
#
#         cdef LibOVRPoseState right_hand_pose = LibOVRPoseState()
#         right_hand_pose.c_data[0] = tracking_state.HandPoses[1]
#         right_hand_pose.status_flags = tracking_state.HandStatusFlags[1]
#
#         return head_pose, left_hand_pose, right_hand_pose
#
#     def calcEyePoses(self, LibOVRPose headPose):
#         """Calculate eye poses using a given pose state.
#
#         Eye poses are derived from the head pose stored in the pose state and
#         the HMD to eye poses reported by LibOVR. Calculated eye poses are stored
#         and passed to the compositor when 'endFrame' is called for additional
#         rendering.
#
#         You can access the computed poses via the 'renderPoses' attribute.
#
#         Parameters
#         ----------
#         headPose : LibOVRPose
#             Head pose.
#
#         Examples
#         --------
#
#         Compute the eye poses from tracker data::
#
#             t = hmd.getPredictedDisplayTime()
#             headPose, leftHandPose, rightHandPose = hmd.getTrackedPoses(t)
#
#             # check if tracking
#             if head.orientationTracked and head.positionTracked:
#                 hmd.calcEyePoses(head.thePose)  # calculate eye poses
#             else:
#                 # do something ...
#
#             # computed render poses appear here
#             renderPoseLeft, renderPoseRight = hmd.renderPoses
#
#         Use a custom head pose::
#
#             headPose = LibOVRPose((0., 1.5, 0.))  # eyes 1.5 meters off the ground
#             hmd.calcEyePoses(headPose)  # calculate eye poses
#
#         """
#         cdef ovr_capi.ovrPosef[2] hmdToEyePoses
#         hmdToEyePoses[0] = self.eyeRenderDesc[0].HmdToEyePose
#         hmdToEyePoses[1] = self.eyeRenderDesc[1].HmdToEyePose
#
#          # calculate the eye poses
#         ovr_capi.ovr_CalcEyePoses2(
#             headPose.c_data[0],
#             hmdToEyePoses,
#             self.eyeLayer.RenderPose)
#
#         # compute the eye transformation matrices from poses
#         cdef ovr_math.Vector3f pos
#         cdef ovr_math.Quatf ori
#         cdef ovr_math.Vector3f up
#         cdef ovr_math.Vector3f forward
#         cdef ovr_math.Matrix4f rm
#
#         cdef int eye = 0
#         for eye in range(ovr_capi.ovrEye_Count):
#             pos = <ovr_math.Vector3f>self.eyeLayer.RenderPose[eye].Position
#             ori = <ovr_math.Quatf>self.eyeLayer.RenderPose[eye].Orientation
#
#             if not ori.IsNormalized():  # make sure orientation is normalized
#                 ori.Normalize()
#
#             rm = ovr_math.Matrix4f(ori)
#             up = rm.Transform(ovr_math.Vector3f(0., 1., 0.))
#             forward = rm.Transform(ovr_math.Vector3f(0., 0., -1.))
#
#             self.eyeViewMatrix[eye] = ovr_math.Matrix4f.LookAtRH(
#                 pos, pos + forward, up)
#
#     @property
#     def hmdToEyePoses(self):
#         """HMD to eye poses (`tuple` of `LibOVRPose`).
#
#         These are the prototype eye poses specified by LibOVR, defined only
#         after 'start' is called. These poses are transformed by the head pose
#         by 'calcEyePoses' to get 'eyeRenderPoses'.
#
#         Notes
#         -----
#             The horizontal (x-axis) separation of the eye poses are determined
#             by the configured lens spacing (slider adjustment). This spacing is
#             supposed to correspond to the actual inter-ocular distance (IOD) of
#             the user. You can get the IOD used for rendering by adding up the
#             absolute values of the x-components of the eye poses, or by
#             multiplying the value of 'eyeToNoseDist' by two. Furthermore, the
#             IOD values can be altered, prior to calling 'calcEyePoses', to
#             override the values specified by LibOVR.
#
#         Returns
#         -------
#         tuple of LibOVRPose
#             Copies of the HMD to eye poses for the left and right eye.
#
#         """
#         cdef LibOVRPose leftHmdToEyePose = LibOVRPose()
#         cdef LibOVRPose rightHmdToEyePose = LibOVRPose()
#
#         leftHmdToEyePose.c_data[0] = self.eyeRenderDesc[0].HmdToEyePose
#         leftHmdToEyePose.c_data[1] = self.eyeRenderDesc[1].HmdToEyePose
#
#         return leftHmdToEyePose, rightHmdToEyePose
#
#     @hmdToEyePoses.setter
#     def hmdToEyePoses(self, value):
#         self.eyeRenderDesc[0].HmdToEyePose = (<LibOVRPose>value[0]).c_data[0]
#         self.eyeRenderDesc[1].HmdToEyePose = (<LibOVRPose>value[1]).c_data[0]
#
#     @property
#     def renderPoses(self):
#         """Eye render poses.
#
#         Pose are those computed by the last 'calcEyePoses' call. Returned
#         objects are copies of the data stored internally by the session
#         instance. These poses are used to define the view matrix when rendering
#         for each eye.
#
#         Notes
#         -----
#             The returned LibOVRPose objects are copies of data stored internally
#             by the session object. Setting renderPoses will recompute their
#             transformation matrices.
#
#         """
#         cdef LibOVRPose left_eye_pose = LibOVRPose()
#         cdef LibOVRPose right_eye_pose = LibOVRPose()
#
#         left_eye_pose.c_data[0] = self.eyeLayer.RenderPose[0]
#         right_eye_pose.c_data[0] = self.eyeLayer.RenderPose[1]
#
#         return left_eye_pose, right_eye_pose
#
#     @renderPoses.setter
#     def renderPoses(self, object value):
#         self.eyeLayer.RenderPose[0] = (<LibOVRPose>value[0]).c_data[0]
#         self.eyeLayer.RenderPose[1] = (<LibOVRPose>value[1]).c_data[0]
#
#         # re-compute the eye transformation matrices from poses
#         cdef ovr_math.Vector3f pos
#         cdef ovr_math.Quatf ori
#         cdef ovr_math.Vector3f up
#         cdef ovr_math.Vector3f forward
#         cdef ovr_math.Matrix4f rm
#
#         cdef int eye = 0
#         for eye in range(ovr_capi.ovrEye_Count):
#             pos = <ovr_math.Vector3f>self.eyeLayer.RenderPose[eye].Position
#             ori = <ovr_math.Quatf>self.eyeLayer.RenderPose[eye].Orientation
#
#             if not ori.IsNormalized():  # make sure orientation is normalized
#                 ori.Normalize()
#
#             rm = ovr_math.Matrix4f(ori)
#             up = rm.Transform(ovr_math.Vector3f(0., 1., 0.))
#             forward = rm.Transform(ovr_math.Vector3f(0., 0., -1.))
#
#             self.eyeViewMatrix[eye] = ovr_math.Matrix4f.LookAtRH(
#                 pos, pos + forward, up)
#
#     def getMirrorTexture(self):
#         """Get the mirror texture name.
#
#         Returns
#         -------
#         tuple of int
#             Result of the 'ovr_GetMirrorTextureBufferGL' API call and OpenGL
#             texture name.
#
#         """
#         global swapChains
#         cdef unsigned int mirror_id
#         cdef ovr_capi.ovrResult result = \
#             ovr_capi.ovr_GetMirrorTextureBufferGL(
#                 self.ptrSession,
#                 mirrorTexture,
#                 &mirror_id)
#
#         return <int>result, <unsigned int>mirror_id
#
#     def getTextureSwapChainBufferGL(self, int eye):
#         """Get the next available swap chain buffer for a specified eye.
#
#         Parameters
#         ----------
#         eye : int
#             Swap chain belonging to a given eye to get the texture ID.
#
#         Returns
#         -------
#         int
#             OpenGL texture handle.
#
#         """
#         cdef int current_idx = 0
#         cdef unsigned int tex_id = 0
#         cdef ovr_capi.ovrResult result = 0
#
#         # check if there is a swap chain in the slot
#         if self.eyeLayer.ColorTexture[eye] == NULL:
#             raise RuntimeError(
#                 "Cannot get buffer ID, NULL eye buffer texture.")
#
#         global _swapChains
#
#         # get the current texture index within the swap chain
#         result = ovr_capi.ovr_GetTextureSwapChainCurrentIndex(
#             self.ptrSession, _swapChains[eye], &current_idx)
#
#         if self.debugMode:
#             check_result(result)
#
#         # get the next available texture ID from the swap chain
#         result = ovr_capi.ovr_GetTextureSwapChainBufferGL(
#             self.ptrSession, _swapChains[eye], current_idx, &tex_id)
#
#         if self.debugMode:
#             check_result(result)
#
#         return tex_id
#
#     def getEyeProjectionMatrix(self, int eye, float nearClip=0.1, float farClip=1000.0):
#         """Compute the projection matrix.
#
#         The projection matrix is computed by the runtime using the eye FOV
#         parameters set with '~ovr.LibOVRSession.setEyeRenderFov' calls.
#
#         Parameters
#         ----------
#         eye : int
#             Eye index.
#         nearClip : float
#             Near clipping plane in meters.
#         farClip : float
#             Far clipping plane in meters.
#
#         Returns
#         -------
#         ndarray of floats
#             4x4 projection matrix.
#
#         """
#         self.eyeProjectionMatrix[eye] = \
#             <ovr_math.Matrix4f>ovr_capi.ovrMatrix4f_Projection(
#                 self.eyeRenderDesc[eye].Fov,
#                 nearClip,
#                 farClip,
#                 ovr_capi.ovrProjection_ClipRangeOpenGL)
#
#         cdef np.ndarray to_return = np.zeros((4, 4), dtype=np.float32)
#
#         # fast copy matrix to numpy array
#         cdef float [:, :] mv = to_return
#         cdef Py_ssize_t i, j
#         i = j = 0
#         for i in range(4):
#             for j in range(4):
#                 mv[i, j] = self.eyeProjectionMatrix[eye].M[i][j]
#
#         return to_return
#
#     @property
#     def eyeRenderViewports(self):
#         """Eye viewports."""
#         self._viewport_left.data = <char*>&self.eyeLayer.Viewport[0]
#         self._viewport_right.data = <char*>&self.eyeLayer.Viewport[1]
#
#         return self._viewport_left, self._viewport_right
#
#     @eyeRenderViewports.setter
#     def eyeRenderViewports(self, object values):
#         cdef int i = 0
#         for i in range(ovr_capi.ovrEye_Count):
#             self.eyeLayer.Viewport[i].Pos.x = <int>values[i][0]
#             self.eyeLayer.Viewport[i].Pos.y = <int>values[i][1]
#             self.eyeLayer.Viewport[i].Size.w = <int>values[i][2]
#             self.eyeLayer.Viewport[i].Size.h = <int>values[i][3]
#
#     def getEyeViewMatrix(self, int eye, bint flatten=False):
#         """Compute a view matrix for a specified eye.
#
#         View matrices are derived from the eye render poses calculated by the
#         last 'calcEyePoses' call or update to 'renderPoses'.
#
#         Parameters
#         ----------
#         eye : int
#             Eye index.
#         flatten : bool
#             Flatten the matrix into a 1D vector. This will create an array
#             suitable for use with OpenGL functions accepting column-major, 4x4
#             matrices as a length 16 vector of floats.
#
#         Returns
#         -------
#         ndarray
#             4x4 view matrix (16x1 if flatten=True).
#
#         """
#         cdef np.ndarray to_return
#         cdef Py_ssize_t i, j, k, N
#         i = j = k = 0
#         N = 4
#         if flatten:
#             to_return = np.zeros((16,), dtype=np.float32)
#             for i in range(N):
#                 for j in range(N):
#                     to_return[k] = self.eyeViewMatrix[eye].M[j][i]
#                     k += 1
#         else:
#             to_return = np.zeros((4, 4), dtype=np.float32)
#             for i in range(N):
#                 for j in range(N):
#                     to_return[i, j] = self.eyeViewMatrix[eye].M[i][j]
#
#         return to_return
#
#     def getPredictedDisplayTime(self, unsigned int frame_index=0):
#         """Get the predicted time a frame will be displayed.
#
#         Parameters
#         ----------
#         frame_index : int
#             Frame index.
#
#         Returns
#         -------
#         float
#             Absolute frame mid-point time for the given frame index in seconds.
#
#         """
#         cdef double t_sec = ovr_capi.ovr_GetPredictedDisplayTime(
#             self.ptrSession,
#             frame_index)
#
#         return t_sec
#
#     @property
#     def timeInSeconds(self):
#         """Absolute time in seconds.
#
#         Returns
#         -------
#         float
#             Time in seconds.
#
#         """
#         cdef double t_sec = ovr_capi.ovr_GetTimeInSeconds()
#
#         return t_sec
#
#     def perfHudMode(self, str mode):
#         """Display a performance information HUD.
#
#         Parameters
#         ----------
#         mode : str
#             Performance HUD mode to present. Valid mode strings are:
#             'PerfSummary', 'LatencyTiming', 'AppRenderTiming',
#             'CompRenderTiming', 'AswStats', 'VersionInfo' and 'Off'. Specifying
#             'Off' hides the performance HUD.
#
#         Warning
#         -------
#         The performance HUD remains visible until 'Off' is specified, even after
#         the application quits.
#
#         """
#         cdef int perfHudMode = 0
#
#         try:
#             perfHudMode = <int>_performance_hud_modes[mode]
#         except KeyError:
#             raise KeyError("Invalid performance HUD mode specified.")
#
#         cdef ovr_capi.ovrBool ret = ovr_capi.ovr_SetInt(
#             self.ptrSession, b"PerfHudMode", perfHudMode)
#
#     def hidePerfHud(self):
#         """Hide the performance HUD.
#
#         This is a convenience function that is equivalent to calling
#         'perf_hud_mode('Off').
#
#         """
#         cdef ovr_capi.ovrBool ret = ovr_capi.ovr_SetInt(
#             self.ptrSession, b"PerfHudMode", ovr_capi.ovrPerfHud_Off)
#
#     @property
#     def perfHudModes(self):
#         """List of valid performance HUD modes."""
#         return [*_performance_hud_modes]
#
#     def setEyeViewport(self, eye, rect):
#         """Set the viewport for a given eye.
#
#         Parameters
#         ----------
#         eye : int
#             Which eye to set the viewport, where left=0 and right=1.
#         rect : ndarray, list or tuple of float
#             Rectangle specifying the viewport's position and dimensions on the
#             eye buffer.
#
#         """
#         cdef ovr_capi.ovrRecti viewportRect
#         viewportRect.Pos.x = <int>rect[0]
#         viewportRect.Pos.y = <int>rect[1]
#         viewportRect.Size.w = <int>rect[2]
#         viewportRect.Size.h = <int>rect[3]
#
#         self.eyeLayer.Viewport[eye] = viewportRect
#
#     def getEyeViewport(self, eye):
#         """Get the viewport for a given eye.
#
#         Parameters
#         ----------
#         eye : int
#             Which eye to set the viewport, where left=0 and right=1.
#
#         """
#         cdef ovr_capi.ovrRecti viewportRect = \
#             self.eyeLayer.Viewport[eye]
#         cdef np.ndarray to_return = np.asarray(
#             [viewportRect.Pos.x,
#              viewportRect.Pos.y,
#              viewportRect.Size.w,
#              viewportRect.Size.h],
#             dtype=np.float32)
#
#         return to_return
#
#     def waitToBeginFrame(self, unsigned int frameIndex=0):
#         """Wait until a buffer is available and frame rendering can begin. Must
#         be called before 'beginFrame'.
#
#         Parameters
#         ----------
#         frameIndex : int
#             The target frame index.
#
#         Returns
#         -------
#         int
#             Return code of the LibOVR API call 'ovr_WaitToBeginFrame'. Returns
#             LIBOVR_SUCCESS if completed without errors. May return
#             LIBOVR_ERROR_DISPLAY_LOST if the device was removed, rendering the
#             current session invalid.
#
#         Raises
#         ------
#         RuntimeError
#             Raised if 'debugMode' is True and the API call to
#             'ovr_WaitToBeginFrame' returns an error.
#
#         """
#         cdef ovr_capi.ovrResult result = \
#             ovr_capi.ovr_WaitToBeginFrame(self.ptrSession, frameIndex)
#
#         return <int>result
#
#     def beginFrame(self, unsigned int frameIndex=0):
#         """Begin rendering the frame. Must be called prior to drawing and
#         'endFrame'.
#
#         Parameters
#         ----------
#         frameIndex : int
#             The target frame index.
#
#         Returns
#         -------
#         int
#             Error code returned by 'ovr_BeginFrame'.
#
#         """
#         cdef ovr_capi.ovrResult result = \
#             ovr_capi.ovr_BeginFrame(self.ptrSession, frameIndex)
#
#         return <int>result
#
#     def commitSwapChain(self, int eye):
#         """Commit changes to a given eye's texture swap chain. When called, the
#         runtime is notified that the texture is ready for use, and the swap
#         chain index is incremented.
#
#         Parameters
#         ----------
#         eye : int
#             Eye buffer index.
#
#         Returns
#         -------
#         int
#             Error code returned by API call 'ovr_CommitTextureSwapChain'. Will
#             return :data:`LIBOVR_SUCCESS` if successful. Returns error code
#             :data:`LIBOVR_ERROR_TEXTURE_SWAP_CHAIN_FULL` if called too many
#             times without calling 'endFrame'.
#
#         Raises
#         ------
#         RuntimeError
#             Raised if 'debugMode' is True and the API call to
#             'ovr_CommitTextureSwapChain' returns an error.
#
#         Warning
#         -------
#             No additional drawing operations are permitted once the texture is
#             committed until the SDK dereferences it, making it available again.
#
#         """
#         global swapChains
#         cdef ovr_capi.ovrResult result = ovr_capi.ovr_CommitTextureSwapChain(
#             self.ptrSession,
#             swapChains[eye])
#
#         if self.debugMode:
#             check_result(result)
#
#         return <int>result
#
#     def endFrame(self, unsigned int frameIndex=0):
#         """Call when rendering a frame has completed. Buffers which have been
#         committed are passed to the compositor for distortion.
#
#         Parameters
#         ----------
#         frameIndex : int
#             The target frame index.
#
#         Returns
#         -------
#         int
#             Error code returned by API call 'ovr_EndFrame'. Check against
#             LIBOVR_SUCCESS, LIBOVR_SUCCESS_NOT_VISIBLE,
#             LIBOVR_SUCCESS_BOUNDARY_INVALID, LIBOVR_SUCCESS_DEVICE_UNAVAILABLE.
#
#         Raises
#         ------
#         RuntimeError
#             Raised if 'debugMode' is True and the API call to 'ovr_EndFrame'
#             returns an error.
#
#         """
#         cdef ovr_capi.ovrLayerHeader* layers = &self.eyeLayer.Header
#         cdef ovr_capi.ovrResult result = ovr_capi.ovr_EndFrame(
#             self.ptrSession,
#             frameIndex,
#             NULL,
#             &layers,
#             <unsigned int>1)
#
#         if self.debugMode:
#             check_result(result)
#
#         return result
#
#     def resetFrameStats(self):
#         """Reset frame statistics.
#
#         Returns
#         -------
#         int
#             Error code returned by 'ovr_ResetPerfStats'.
#
#         """
#         cdef ovr_capi.ovrResult result = ovr_capi.ovr_ResetPerfStats(
#             self.ptrSession)
#
#         return result
#
#     @property
#     def trackingOriginType(self):
#         """Tracking origin type.
#
#         The tracking origin type specifies where the origin is placed when
#         computing the pose of tracked objects (i.e. the head and touch
#         controllers.) Valid values are 'floor' and 'eye'.
#
#         """
#         cdef ovr_capi.ovrTrackingOrigin origin = \
#             ovr_capi.ovr_GetTrackingOriginType(self.ptrSession)
#
#         if origin == ovr_capi.ovrTrackingOrigin_FloorLevel:
#             return 'floor'
#         elif origin == ovr_capi.ovrTrackingOrigin_EyeLevel:
#             return 'eye'
#
#
#     @trackingOriginType.setter
#     def trackingOriginType(self, str value):
#         cdef ovr_capi.ovrResult result
#         if value == 'floor':
#             result = ovr_capi.ovr_SetTrackingOriginType(
#                 self.ptrSession, ovr_capi.ovrTrackingOrigin_FloorLevel)
#         elif value == 'eye':
#             result = ovr_capi.ovr_SetTrackingOriginType(
#                 self.ptrSession, ovr_capi.ovrTrackingOrigin_EyeLevel)
#
#         if self.debugMode:
#             check_result(result)
#
#     def recenterTrackingOrigin(self):
#         """Recenter the tracking origin.
#
#         Returns
#         -------
#         None
#
#         """
#         cdef ovr_capi.ovrResult result = ovr_capi.ovr_RecenterTrackingOrigin(
#             self.ptrSession)
#
#         if self.debugMode:
#             check_result(result)
#
#     def getTrackerInfo(self, int trackerIndex):
#         """Get information about a given tracker.
#
#         Parameters
#         ----------
#         trackerIndex : int
#             The index of the sensor to query. Valid values are between 0 and
#             '~LibOVRSession.trackerCount'.
#
#         """
#         cdef LibOVRTrackerInfo to_return = LibOVRTrackerInfo()
#
#         # set the descriptor data
#         to_return.c_ovrTrackerDesc = ovr_capi.ovr_GetTrackerDesc(
#             self.ptrSession, <unsigned int>trackerIndex)
#         # get the tracker pose
#         to_return.c_ovrTrackerPose = ovr_capi.ovr_GetTrackerPose(
#             self.ptrSession, <unsigned int>trackerIndex)
#
#         return to_return
#
#     def refreshPerformanceStats(self):
#         """Refresh performance statistics.
#
#         Should be called after 'endFrame'.
#
#         """
#         cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetPerfStats(
#             self.ptrSession,
#             &self.perfStats)
#
#         # clear
#         self.compFrameStats = list()
#
#         cdef int statIdx = 0
#         cdef int numStats = self.perfStats.FrameStatsCount
#         for statIdx in range(numStats):
#             frameStat = LibOVRCompFramePerfStat()
#             frameStat.c_data[0] = self.perfStats.FrameStats[statIdx]
#             self.compFrameStats.append(frameStat)
#
#         return result
#
#     @property
#     def maxProvidedFrameStats(self):
#         """Maximum number of frame stats provided."""
#         return 5
#
#     @property
#     def frameStatsCount(self):
#         """Number of frame stats available."""
#         return self.perfStats.FrameStatsCount
#
#     @property
#     def anyFrameStatsDropped(self):
#         """Have any frame stats been dropped?"""
#         return self.perfStats.AnyFrameStatsDropped
#
#     @property
#     def adaptiveGpuPerformanceScale(self):
#         """Adaptive GPU performance scaling factor."""
#         return self.perfStats.AdaptiveGpuPerformanceScale
#
#     @property
#     def aswIsAvailable(self):
#         """Is ASW available?"""
#         return self.perfStats.AswIsAvailable
#
#     @property
#     def frameStats(self):
#         """Get all frame compositior frame statistics."""
#         return self.compFrameStats
#
#     def getLastErrorInfo(self):
#         """Get the last error code and information string reported by the API.
#
#         This function can be used when implementing custom error handlers.
#
#         Returns
#         -------
#         tuple of int, str
#             Tuple of the API call result and error string.
#
#         """
#         cdef ovr_capi.ovrErrorInfo lastErrorInfo  # store our last error here
#         ovr_capi.ovr_GetLastErrorInfo(&lastErrorInfo)
#
#         cdef ovr_capi.ovrResult result = lastErrorInfo.Result
#         cdef str errorString = lastErrorInfo.ErrorString.decode("utf-8")
#
#         return <int>result, errorString
#
#     def setBoundaryColor(self, red, green, blue):
#         """Set the boundary color.
#
#         The boundary is drawn by the compositor which overlays the extents of
#         the physical space where the user can safely move.
#
#         Parameters
#         ----------
#         red : float
#             Red component of the color from 0.0 to 1.0.
#         green : float
#             Green component of the color from 0.0 to 1.0.
#         blue : float
#             Blue component of the color from 0.0 to 1.0.
#
#         """
#         cdef ovr_capi.ovrColorf color
#         color.r = <float>red
#         color.g = <float>green
#         color.b = <float>blue
#
#         self.boundryStyle.Color = color
#
#         cdef ovr_capi.ovrResult result = ovr_capi.ovr_SetBoundaryLookAndFeel(
#             self.ptrSession,
#             &self.boundryStyle)
#
#         if self.debugMode:
#             check_result(result)
#
#     def resetBoundaryColor(self):
#         """Reset the boundary color to system default.
#
#         """
#         cdef ovr_capi.ovrResult result = ovr_capi.ovr_ResetBoundaryLookAndFeel(
#             self.ptrSession)
#
#         if self.debugMode:
#             check_result(result)
#
#     @property
#     def isBoundryVisible(self):
#         """Check if the Guardian boundary is visible.
#
#         The boundary is drawn by the compositor which overlays the extents of
#         the physical space where the user can safely move.
#
#         """
#         cdef ovr_capi.ovrBool is_visible
#         cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetBoundaryVisible(
#             self.ptrSession, &is_visible)
#
#         if self.debugMode:
#             check_result(result)
#
#         return <bint> is_visible
#
#     def showBoundary(self):
#         """Show the boundary.
#
#         The boundary is drawn by the compositor which overlays the extents of
#         the physical space where the user can safely move.
#
#         """
#         cdef ovr_capi.ovrResult result = ovr_capi.ovr_RequestBoundaryVisible(
#             self.ptrSession, ovr_capi.ovrTrue)
#
#         return result
#
#     def hideBoundary(self):
#         """Hide the boundry."""
#         cdef ovr_capi.ovrResult result = ovr_capi.ovr_RequestBoundaryVisible(
#             self.ptrSession, ovr_capi.ovrFalse)
#
#         return result
#
#     def getBoundaryDimensions(self, str boundaryType='PlayArea'):
#         """Get the dimensions of the boundary.
#
#         Parameters
#         ----------
#         boundaryType : str
#             Boundary type, can be 'PlayArea' or 'Outer'.
#
#         Returns
#         -------
#         ndarray
#             Dimensions of the boundary meters [x, y, z].
#
#         """
#         cdef ovr_capi.ovrBoundaryType btype
#         if boundaryType == 'PlayArea':
#             btype = ovr_capi.ovrBoundary_PlayArea
#         elif boundaryType == 'Outer':
#             btype = ovr_capi.ovrBoundary_Outer
#         else:
#             raise ValueError("Invalid boundary type specified.")
#
#         cdef ovr_capi.ovrVector3f vec_out
#         cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetBoundaryDimensions(
#                 self.ptrSession, btype, &vec_out)
#
#         cdef np.ndarray[np.float32_t, ndim=1] to_return = np.asarray(
#             (vec_out.x, vec_out.y, vec_out.z), dtype=np.float32)
#
#         return to_return
#
#     def getBoundaryPoints(self, str boundaryType='PlayArea'):
#         """Get the floor points which define the boundary."""
#         pass  # TODO: make this work.
#
#     def getConnectedControllers(self):
#         """List of connected controllers.
#
#         Returns
#         -------
#         tuple of int and list
#             List of connected controller names. Check if a specific controller
#             is available by checking the membership of its name in the list.
#
#         Examples
#         --------
#
#         Check if the Xbox gamepad is connected::
#
#             hasGamepad = "Xbox" in hmd.getConnectedControllers()
#
#         Check if the left and right touch controllers are both paired::
#
#             connected = hmd.getConnectedControllers()
#             isPaired = 'LeftTouch' in connected and 'RightTouch' in connected
#
#         """
#         cdef unsigned int result = ovr_capi.ovr_GetConnectedControllerTypes(
#             self.ptrSession)
#
#         cdef list controllerTypes = list()
#         if (result & ovr_capi.ovrControllerType_XBox) == \
#                 ovr_capi.ovrControllerType_XBox:
#             controllerTypes.append('Xbox')
#         elif (result & ovr_capi.ovrControllerType_Remote) == \
#                 ovr_capi.ovrControllerType_Remote:
#             controllerTypes.append('Remote')
#         elif (result & ovr_capi.ovrControllerType_Touch) == \
#                 ovr_capi.ovrControllerType_Touch:
#             controllerTypes.append('Touch')
#         elif (result & ovr_capi.ovrControllerType_LTouch) == \
#                 ovr_capi.ovrControllerType_LTouch:
#             controllerTypes.append('LeftTouch')
#         elif (result & ovr_capi.ovrControllerType_RTouch) == \
#                 ovr_capi.ovrControllerType_RTouch:
#             controllerTypes.append('RightTouch')
#
#         return controllerTypes
#
#     def refreshInputState(self, str controller):
#         """Refresh the input state of a controller.
#
#         Parameters
#         ----------
#         controller : str
#             Controller name to poll. Valid names are: 'Xbox', 'Remote', 'Touch',
#             'LeftTouch', and 'RightTouch'.
#
#         """
#         # convert the string to an index
#         cdef dict idx = {'Xbox' : 0, 'Remote' : 1, 'Touch' : 2, 'LeftTouch' : 3,
#                          'RightTouch' : 4}
#
#         # pointer to the current and previous input state
#         cdef ovr_capi.ovrInputState* previousInputState = \
#             &self.prevInputState[idx[controller]]
#         cdef ovr_capi.ovrInputState* currentInputState = \
#             &self.inputStates[idx[controller]]
#
#         # copy the current input state into the previous before updating
#         previousInputState[0] = currentInputState[0]
#
#         # get the current input state
#         cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetInputState(
#             self.ptrSession,
#             _controller_type_enum[controller],  # get the enum for the controller
#             currentInputState)
#
#         if self.debugMode:
#             check_result(result)
#
#         return result
#
#     def getButtons(self, str controller, object buttonNames, str testState='continuous'):
#         """Get the state of a specified button for a given controller.
#
#         Buttons to test are specified using their string names. Argument
#         'buttonNames' accepts a single string or a list. If a list is specified,
#         the returned value will reflect whether all buttons were triggered at
#         the time the controller was polled last.
#
#         An optional trigger mode may be specified which defines the button's
#         activation criteria. By default, testState='continuous' will return the
#         immediate state of the button. Using 'rising' (and 'pressed') will
#         return True once when the button transitions to being pressed, whereas
#         'falling' (and 'released') will return True once the button is released.
#
#         Parameters
#         ----------
#         controller : str
#             Controller name to poll. Valid names are: 'Xbox', 'Remote', 'Touch',
#             'LeftTouch', and 'RightTouch'.
#         buttonNames : tuple of str, list of str, or str
#             Button names to test for state changes.
#         testState : str
#             State to test buttons for. Valid states are 'rising', 'falling',
#             'continuous', 'pressed', and 'released'.
#
#         Returns
#         -------
#         bool
#             Result of the button press.
#
#         """
#         # convert the string to an index
#         cdef dict idx = {'Xbox' : 0, 'Remote' : 1, 'Touch' : 2, 'LeftTouch' : 3,
#                          'RightTouch' : 4}
#
#         # pointer to the current and previous input state
#         cdef unsigned int curButtons = \
#             self.inputStates[idx[controller]].Buttons
#         cdef unsigned int prvButtons = \
#             self.prevInputState[idx[controller]].Buttons
#
#         # generate a bit mask for testing button presses
#         cdef unsigned int buttonBits = 0x00000000
#         cdef int i, N
#         if isinstance(buttonNames, str):  # don't loop if a string is specified
#             buttonBits |= ctrl_button_lut[buttonNames]
#         elif isinstance(buttonNames, (tuple, list)):
#             # loop over all names and combine them
#             N = <int>len(buttonNames)
#             for i in range(N):
#                 buttonBits |= ctrl_button_lut[buttonNames[i]]
#
#         # test if the button was pressed
#         cdef bint stateResult = False
#         if testState == 'continuous':
#             stateResult = (curButtons & buttonBits) == buttonBits
#         elif testState == 'rising' or testState == 'pressed':
#             # rising edge, will trigger once when pressed
#             stateResult = (curButtons & buttonBits) == buttonBits and \
#                           (prvButtons & buttonBits) != buttonBits
#         elif testState == 'falling' or testState == 'released':
#             # falling edge, will trigger once when released
#             stateResult = (curButtons & buttonBits) != buttonBits and \
#                           (prvButtons & buttonBits) == buttonBits
#         else:
#             raise ValueError("Invalid trigger mode specified.")
#
#         return stateResult
#
#     def getTouches(self, str controller, object touchNames, str testState='continuous'):
#         """Get touches for a specified device.
#
#         Touches reveal information about the user's hand pose, for instance,
#         whether a pointing or pinching gesture is being made. Oculus Touch
#         controllers are required for this functionality.
#
#         Touch points to test are specified using their string names. Argument
#         'touch_names' accepts a single string or a list. If a list is specified,
#         the returned value will reflect whether all touches were triggered at
#         the time the controller was polled last.
#
#         """
#         # convert the string to an index
#         cdef dict idx = {'Xbox' : 0, 'Remote' : 1, 'Touch' : 2, 'LeftTouch' : 3,
#                          'RightTouch' : 4}
#
#         # pointer to the current and previous input state
#         cdef unsigned int curTouches = \
#             self.inputStates[idx[controller]].Touches
#         cdef unsigned int prvTouches = \
#             self.prevInputState[idx[controller]].Touches
#
#         # generate a bit mask for testing button presses
#         cdef unsigned int touchBits = 0x00000000
#         cdef int i, N
#         if isinstance(touchNames, str):  # don't loop if a string is specified
#             touchBits |= ctrl_touch_lut[touchNames]
#         elif isinstance(touchNames, (tuple, list)):
#             # loop over all names and combine them
#             N = <int>len(touchNames)
#             for i in range(N):
#                 touchBits |= ctrl_touch_lut[touchNames[i]]
#
#         # test if the button was pressed
#         cdef bint stateResult = False
#         if testState == 'continuous':
#             stateResult = (curTouches & touchBits) == touchBits
#         elif testState == 'rising' or testState == 'pressed':
#             # rising edge, will trigger once when pressed
#             stateResult = (curTouches & touchBits) == touchBits and \
#                           (prvTouches & touchBits) != touchBits
#         elif testState == 'falling' or testState == 'released':
#             # falling edge, will trigger once when released
#             stateResult = (curTouches & touchBits) != touchBits and \
#                           (prvTouches & touchBits) == touchBits
#         else:
#             raise ValueError("Invalid trigger mode specified.")
#
#         return stateResult
#
#     def getThumbstickValues(self, str controller, bint deadzone=False):
#         """Get thumbstick values."""
#         cdef dict idx = {'Xbox' : 0, 'Touch' : 2, 'LeftTouch' : 3,
#                          'RightTouch' : 4}
#
#         # pointer to the current and previous input state
#         cdef ovr_capi.ovrInputState* currentInputState = \
#             &self.inputStates[idx[controller]]
#
#         cdef float thumbstick_x0 = 0.0
#         cdef float thumbstick_y0 = 0.0
#         cdef float thumbstick_x1 = 0.0
#         cdef float thumbstick_y1 = 0.0
#
#         if deadzone:
#             thumbstick_x0 = currentInputState[0].Thumbstick[0].x
#             thumbstick_y0 = currentInputState[0].Thumbstick[0].y
#             thumbstick_x1 = currentInputState[0].Thumbstick[1].x
#             thumbstick_y1 = currentInputState[0].Thumbstick[1].y
#         else:
#             thumbstick_x0 = currentInputState[0].ThumbstickNoDeadzone[0].x
#             thumbstick_y0 = currentInputState[0].ThumbstickNoDeadzone[0].y
#             thumbstick_x1 = currentInputState[0].ThumbstickNoDeadzone[1].x
#             thumbstick_y1 = currentInputState[0].ThumbstickNoDeadzone[1].y
#
#         return (thumbstick_x0, thumbstick_y0), (thumbstick_x1, thumbstick_y1)
#
#     def getIndexTriggerValues(self, str controller, bint deadzone=False):
#         """Get index trigger values."""
#         # convert the string to an index
#         cdef dict idx = {'Xbox' : 0, 'Remote' : 1, 'Touch' : 2, 'LeftTouch' : 3,
#                          'RightTouch' : 4}
#
#         # pointer to the current and previous input state
#         cdef ovr_capi.ovrInputState* currentInputState = \
#             &self.inputStates[idx[controller]]
#
#         cdef float indexTriggerLeft = 0.0
#         cdef float indexTriggerRight = 0.0
#
#         if deadzone:
#             indexTriggerLeft = currentInputState[0].IndexTrigger[0]
#             indexTriggerRight = currentInputState[0].IndexTrigger[1]
#         else:
#             indexTriggerLeft = currentInputState[0].IndexTriggerNoDeadzone[0]
#             indexTriggerRight = currentInputState[0].IndexTriggerNoDeadzone[1]
#
#         return indexTriggerLeft, indexTriggerRight
#
#     def getHandTriggerValues(self, str controller, bint deadzone=False):
#         """Get hand trigger values."""
#         # convert the string to an index
#         cdef dict idx = {'Xbox' : 0, 'Touch' : 2, 'LeftTouch' : 3,
#                          'RightTouch' : 4}
#
#         # pointer to the current and previous input state
#         cdef ovr_capi.ovrInputState* currentInputState = \
#             &self.inputStates[idx[controller]]
#
#         cdef float indexTriggerLeft = 0.0
#         cdef float indexTriggerRight = 0.0
#
#         if deadzone:
#             indexTriggerLeft = currentInputState[0].HandTrigger[0]
#             indexTriggerRight = currentInputState[0].HandTrigger[1]
#         else:
#             indexTriggerLeft = currentInputState[0].HandTriggerNoDeadzone[0]
#             indexTriggerRight = currentInputState[0].HandTriggerNoDeadzone[1]
#
#         return indexTriggerLeft, indexTriggerRight
#
#     def setControllerVibration(self, str controller, str frequency, float amplitude):
#         """Vibrate a controller.
#
#         Vibration is constant at fixed frequency and amplitude. Vibration lasts
#         2.5 seconds, so this function needs to be called more often than that
#         for sustained vibration. Only controllers which support vibration can be
#         used here.
#
#         There are only two frequencies permitted 'high' and 'low', however,
#         amplitude can vary from 0.0 to 1.0. Specifying frequency='off' stops
#         vibration.
#
#         Parameters
#         ----------
#         controller : str
#             Controller name to vibrate. Valid names are: 'Xbox', 'Touch',
#             'LeftTouch', and 'RightTouch'.
#         frequency : str
#             Vibration frequency. Valid values are: 'off', 'low', or 'high'.
#         amplitude : float
#             Vibration amplitude in the range of [0.0 and 1.0]. Values outside
#             this range are clamped.
#
#         Returns
#         -------
#         int
#             Return value of API call 'ovr_SetControllerVibration'. Can return
#             LIBOVR_SUCCESS_DEVICE_UNAVAILABLE if no device is present.
#
#         """
#         # get frequency associated with the string
#         cdef float freq = 0.0
#         if frequency == 'off':
#             freq = 0.0
#         elif frequency == 'low':
#             freq = 0.5
#         elif frequency == 'high':
#             freq = 1.0
#         else:
#             raise RuntimeError("Invalid frequency specified.")
#
#         cdef dict _controller_types = {
#             'Xbox' : ovr_capi.ovrControllerType_XBox,
#             'Touch' : ovr_capi.ovrControllerType_Touch,
#             'LeftTouch' : ovr_capi.ovrControllerType_LTouch,
#             'RightTouch' : ovr_capi.ovrControllerType_RTouch}
#
#         cdef ovr_capi.ovrResult result = ovr_capi.ovr_SetControllerVibration(
#             self.ptrSession,
#             <ovr_capi.ovrControllerType>_controller_types[controller],
#             freq,
#             amplitude)
#
#         if self.debugMode:
#             check_result(result)
#
#         return result
#
#     def getSessionStatus(self):
#         """Get the current session status.
#
#         Returns
#         -------
#         LibOVRSessionStatus
#             Object specifying the current state of the session.
#
#         """
#
#         cdef LibOVRSessionStatus to_return = LibOVRSessionStatus()
#         cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetSessionStatus(
#             self.ptrSession, to_return.c_data)
#
#         return to_return


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
        pos : tuple, list, or ndarray of float
            Position vector (x, y, z).
        ori : tuple, list, or ndarray of float
            Orientation quaternion vector (x, y, z, w).

        Notes
        -----
        Values for vectors are stored internally as 32-bit floating point
        numbers.

        """
        pass  # nop

    def __cinit__(self, pos=(0., 0., 0.), ori=(0., 0., 0., 1.)):
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
        cdef ovr_math.Posef pose_r = \
            <ovr_math.Posef>a.c_data[0] * <ovr_math.Posef>b.c_data[0]

        cdef LibOVRPose to_return = \
            LibOVRPose(
                (pose_r.Translation.x,
                 pose_r.Translation.y,
                 pose_r.Translation.z),
                (pose_r.Rotation.x,
                 pose_r.Rotation.y,
                 pose_r.Rotation.z,
                 pose_r.Rotation.w))

        return to_return

    def __invert__(self):
        """Invert operator (~) to invert a pose.

        """
        return self.inverted()

    @property
    def posOri(self):
        return self.pos, self.ori

    @posOri.setter
    def posOri(self, value):
        self.setPosOri(value[0], value[1])

    def getPosOri(self):
        """Get position and orientation."""
        return self.pos, self.ori

    def setPosOri(self, object pos, object ori):
        """Set the position and orientation."""
        self.c_data[0].Position.x = <float>pos[0]
        self.c_data[0].Position.y = <float>pos[1]
        self.c_data[0].Position.z = <float>pos[2]

        self.c_data[0].Orientation.x = <float>ori[0]
        self.c_data[0].Orientation.y = <float>ori[1]
        self.c_data[0].Orientation.z = <float>ori[2]
        self.c_data[0].Orientation.w = <float>ori[3]

    def getYawPitchRoll(self, LibOVRPose refPose=None):
        """Get the yaw, pitch, and roll of the orientation quaternion.

        Parameters
        ----------
        refPose : LibOVRPose or None
            Reference pose to compute angles relative to. If None is specified,
            computed values are referenced relative to the world axes.

        Returns
        -------
        ndarray of floats
            Yaw, pitch, and roll of the pose in degrees.

        """
        cdef float yaw, pitch, roll
        cdef ovr_math.Posef inPose = <ovr_math.Posef>self.c_data[0]
        cdef ovr_math.Posef invRef

        if refPose is not None:
            invRef = (<ovr_math.Posef>refPose.c_data[0]).Inverted()
            inPose = invRef * inPose

        inPose.Rotation.GetYawPitchRoll(&yaw, &pitch, &roll)

        cdef np.ndarray[np.float32_t, ndim=1] to_return = \
            np.array((yaw, pitch, roll), dtype=np.float32)

        return to_return

    def getMatrix4x4(self, bint inverse=False):
        """Convert this pose into a 4x4 transformation matrix.

        Parameters
        ----------
        inverse : bool
            If True, return the inverse of the matrix.

        Returns
        -------
        ndarray
            4x4 transformation matrix.

        """
        cdef ovr_math.Matrix4f m_pose = ovr_math.Matrix4f(
            <ovr_math.Posef>self.c_data[0])

        if inverse:
            m_pose.InvertHomogeneousTransform()

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

    def getMatrix1D(self, bint inverse=False):
        """Convert this pose into a 1D (flattened) transform matrix.

        This will output an array suitable for use with OpenGL.

        Parameters
        ----------
        inverse : bool
            If True, return the inverse of the matrix.

        Returns
        -------
        ndarray
            4x4 transformation matrix flattened to a 1D array assuming column
            major order with a 'float32' data type.

        """
        cdef ovr_math.Matrix4f m_pose = ovr_math.Matrix4f(
            <ovr_math.Posef>self.c_data[0])

        if inverse:
            m_pose.InvertHomogeneousTransform()

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
                (self.c_data[0].Position.x,
                 self.c_data[0].Position.y,
                 self.c_data[0].Position.z),
                (self.c_data[0].Orientation.x,
                 self.c_data[0].Orientation.y,
                 self.c_data[0].Orientation.z,
                 self.c_data[0].Orientation.w))

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

    def raycastSphere(self, object targetPose, float radius=0.5, object rayDir=(0., 0., -1.), float maxRange=0.0):
        """Raycast to a sphere.

        Project an invisible ray of finite or infinite length from this pose in
        rayDir and check if it intersects with the targetPose bounding sphere.

        Specifying maxRange as >0.0 casts a ray of finite length in world
        units. The distance between the target and ray origin position are
        checked prior to casting the ray; automatically failing if the ray can
        never reach the edge of the bounding sphere centered about targetPose.
        This avoids having to do the costly transformations required for
        picking.

        This raycast implementation can only determine if contact is being made
        with the object's bounding sphere, not where on the object the ray
        intersects. This method might not work for irregular or elongated
        objects since bounding spheres may not approximate those shapes well. In
        such cases, one may use multiple spheres at different locations and
        radii to pick the same object.

        Parameters
        ----------
        targetPose : tuple, list, or ndarray of floats
            Coordinates of the center of the trarget sphere (x, y, z).
        radius : float
            The radius of the target.
        rayDir : tuple, list, or ndarray of floats
            Vector indicating the direction for the ray (default is -Z).
        maxRange : float
            The maximum range of the ray. Ray testing will fail automatically if
            the target is out of range. The ray has infinite length if None is
            specified. Ray is infinite if maxRange=0.0.

        Returns
        -------
        bool
            True if the ray intersects anywhere on the bounding sphere, False in
            every other condition.

        """
        cdef ovr_math.Vector3f targetPos = ovr_math.Vector3f(
            <float>targetPose[0], <float>targetPose[1], <float>targetPose[2])
        cdef ovr_math.Vector3f _rayDir = ovr_math.Vector3f(
            <float>rayDir[0], <float>rayDir[1], <float>rayDir[2])
        cdef ovr_math.Posef originPos = <ovr_math.Posef>self.c_data[0]

        # if the ray is finite, does it ever touch the edge of the sphere?
        cdef float targetDist
        if maxRange != 0.0:
            targetDist = targetPos.Distance(originPos.Translation) - radius
            if targetDist > maxRange:
                return False

        # put the target in the caster's local coordinate system
        cdef ovr_math.Vector3f offset = -originPos.InverseTransform(targetPos)

        # find the discriminant
        cdef float desc = pow(_rayDir.Dot(offset), 2.0) - \
               (offset.Dot(offset) - pow(radius, 2.0))

        # one or more roots? if so we are touching the sphere
        return desc >= 0.0

    def interp(self, LibOVRPose toPose, float s, bint fast=False):
        """Interpolate between poses.

        Linear interpolation is used on position (Lerp) while the orientation
        has spherical linear interpolation (Slerp) applied.

        Parameters
        ----------
        toPose : LibOVRPose
            End pose.
        s : float
            Interpolation factor between in interval 0.0 and 1.0.
        fast : bool
            If True, use fast interpolation which is quicker but less accurate
            over larger distances.

        Returns
        -------
        LibOVRPose
            Interpolated pose at 's'.

        """
        cdef ovr_math.Posef _toPose = <ovr_math.Posef>toPose.c_data[0]
        cdef ovr_math.Posef interp

        if not fast:
            interp = (<ovr_math.Posef>self.c_data[0]).Lerp(_toPose, s)
        else:
            interp = (<ovr_math.Posef>self.c_data[0]).FastLerp(_toPose, s)

        cdef LibOVRPose to_return = \
            LibOVRPose(
                (interp.Translation.x,
                 interp.Translation.y,
                 interp.Translation.z),
                (interp.Rotation.x,
                 interp.Rotation.y,
                 interp.Rotation.z,
                 interp.Rotation.w))

        return to_return

    cdef toarray(self, float* arr):
        """Copy position and orientation data to an array. 
        
        This function provides an interface to exchange pose data between API 
        specific classes.
        
        Parameters
        ----------
        a : float* 
            Pointer to the first element of the array.
        
        """
        arr[0] = self.c_data[0].Position.x
        arr[1] = self.c_data[0].Position.y
        arr[2] = self.c_data[0].Position.z
        arr[3] = self.c_data[0].Orientation.x
        arr[4] = self.c_data[0].Orientation.y
        arr[5] = self.c_data[0].Orientation.z
        arr[6] = self.c_data[0].Orientation.w


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


cdef class LibOVRTrackerInfo(object):
    """Class for information about camera based tracking sensors.

    """
    cdef ovr_capi.ovrTrackerPose* c_data
    cdef ovr_capi.ovrTrackerPose c_ovrTrackerPose
    cdef ovr_capi.ovrTrackerDesc c_ovrTrackerDesc

    cdef LibOVRPose _pose
    cdef LibOVRPose _leveledPose

    def __cinit__(self):
        self._pose = LibOVRPose()
        self._leveledPose = LibOVRPose()

    @property
    def pose(self):
        """The pose of the sensor (`LibOVRPose`)."""
        self._pose.c_data[0] = self.c_ovrTrackerPose.Pose

        return self._pose

    @property
    def leveledPose(self):
        """Gravity aligned pose of the sensor (`LibOVRPose`)."""
        self._leveledPose.c_data[0] = self.c_ovrTrackerPose.LeveledPose

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


# -------------------------------
# Performance/Profiling Functions
# -------------------------------
#
cdef class LibOVRCompFramePerfStat(object):
    cdef ovr_capi.ovrPerfStatsPerCompositorFrame* c_data
    cdef ovr_capi.ovrPerfStatsPerCompositorFrame c_ovrPerfStatsPerCompositorFrame

    def __cinit__(self, *args, **kwargs):
        self.c_data = &self.c_ovrPerfStatsPerCompositorFrame

    @property
    def hmdVsyncIndex(self):
        return self.c_data[0].HmdVsyncIndex

    @property
    def appFrameIndex(self):
        return self.c_data[0].AppFrameIndex

    @property
    def appDroppedFrameCount(self):
        return self.c_data[0].AppDroppedFrameCount

    @property
    def appQueueAheadTime(self):
        return self.c_data[0].AppQueueAheadTime

    @property
    def appCpuElapsedTime(self):
        return self.c_data[0].AppCpuElapsedTime

    @property
    def appGpuElapsedTime(self):
        return self.c_data[0].AppGpuElapsedTime

    @property
    def compositorFrameIndex(self):
        return self.c_data[0].CompositorFrameIndex

    @property
    def compositorLatency(self):
        return self.c_data[0].CompositorLatency

    @property
    def compositorCpuElapsedTime(self):
        return self.c_data[0].CompositorCpuElapsedTime

    @property
    def compositorGpuElapsedTime(self):
        return self.c_data[0].CompositorGpuElapsedTime

    @property
    def compositorCpuStartToGpuEndElapsedTime(self):
        return self.c_data[0].CompositorCpuStartToGpuEndElapsedTime

    @property
    def compositorGpuEndToVsyncElapsedTime(self):
        return self.c_data[0].CompositorGpuEndToVsyncElapsedTime

