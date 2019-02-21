# distutils: language=c++
#  =============================================================================
#  Python Interface Module for LibOVR
#  =============================================================================
#
#  _libovr.pyx
#
#  Copyright 2019 Matthew Cutone <cutonem(a)yorku.ca> and Laurie M. Wilcox
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
"""This extension module exposes the LibOVR API to Python using the official,
proprietary Oculus PC SDK.

This extension module makes use of the official Oculus PC SDK. A C/C++ interface
for tracking, rendering, and VR math for Oculus products. The Oculus PC SDK is
Copyright (c) Facebook Technologies, LLC and its affiliates. All rights
reserved. You must accept the 'EULA', 'Terms of Use' and 'Privacy Policy'
associated with the Oculus PC SDK to use this module in your software (which you
did when you downloaded the SDK to build this module, didn't ya?), if not see
https://www.oculus.com/legal/terms-of-service/ to access those documents.

"""
# ------------------
# Module information
#-------------------
#
__author__ = "Matthew D. Cutone"
__credits__ = ["Laurie M. Wilcox"]
__copyright__ = "Copyright 2019 Matthew D. Cutone"
__license__ = "MIT"
__version__ = "0.2.0"
__status__ = "Beta"
__maintainer__ = "Matthew D. Cutone"
__email__ = "cutonem@yorku.ca"

# ----------------
# Exported objects
# ----------------
#
__all__ = [
    'LIBOVR_SUCCESS',
    'LIBOVR_SUCCESS_NOT_VISIBLE',
    'LIBOVR_SUCCESS_DEVICE_UNAVAILABLE',
    'LIBOVR_SUCCESS_BOUNDARY_INVALID',
    'LIBOVR_ERROR_MEMORY_ALLOCATION_FAILURE',
    'LIBOVR_ERROR_INVALID_SESSION',
    'LIBOVR_ERROR_TIMEOUT',
    'LIBOVR_ERROR_NOT_INITIALIZED',
    'LIBOVR_ERROR_INVALID_PARAMETER',
    'LIBOVR_ERROR_SERVICE_ERROR',
    'LIBOVR_ERROR_NO_HMD',
    'LIBOVR_ERROR_UNSUPPORTED',
    'LIBOVR_ERROR_DEVICE_UNAVAILABLE',
    'LIBOVR_ERROR_INVALID_HEADSET_ORIENTATION',
    'LIBOVR_ERROR_CLIENT_SKIPPED_DESTROY',
    'LIBOVR_ERROR_CLIENT_SKIPPED_SHUTDOWN',
    'LIBOVR_ERROR_SERVICE_DEADLOCK_DETECTED',
    'LIBOVR_ERROR_INSUFFICENT_ARRAY_SIZE',
    'LIBOVR_ERROR_NO_EXTERNAL_CAMERA_INFO',
    'LIBOVR_ERROR_LOST_TRACKING',
    'LIBOVR_ERROR_EXTERNAL_CAMERA_INITIALIZED_FAILED',
    'LIBOVR_ERROR_EXTERNAL_CAMERA_CAPTURE_FAILED',
    'LIBOVR_ERROR_EXTERNAL_CAMERA_NAME_LISTS_BUFFER_SIZE',
    'LIBOVR_ERROR_EXTERNAL_CAMERA_NAME_LISTS_MISMATCH',
    'LIBOVR_ERROR_EXTERNAL_CAMERA_NOT_CALIBRATED',
    'LIBOVR_ERROR_EXTERNAL_CAMERA_NAME_WRONG_SIZE',
    'LIBOVR_ERROR_AUDIO_DEVICE_NOT_FOUND',
    'LIBOVR_ERROR_AUDIO_COM_ERROR',
    'LIBOVR_ERROR_INITIALIZE',
    'LIBOVR_ERROR_LIB_LOAD',
    'LIBOVR_ERROR_SERVICE_CONNECTION',
    'LIBOVR_ERROR_SERVICE_VERSION',
    'LIBOVR_ERROR_INCOMPATIBLE_OS',
    'LIBOVR_ERROR_DISPLAY_INIT',
    'LIBOVR_ERROR_SERVER_START',
    'LIBOVR_ERROR_REINITIALIZATION',
    'LIBOVR_ERROR_MISMATCHED_ADAPTERS',
    'LIBOVR_ERROR_LEAKING_RESOURCES',
    'LIBOVR_ERROR_CLIENT_VERSION',
    'LIBOVR_ERROR_OUT_OF_DATE_OS',
    'LIBOVR_ERROR_OUT_OF_DATE_GFX_DRIVER',
    'LIBOVR_ERROR_INCOMPATIBLE_OS',
    'LIBOVR_ERROR_NO_VALID_VR_DISPLAY_SYSTEM',
    'LIBOVR_ERROR_OBSOLETE',
    'LIBOVR_ERROR_DISABLED_OR_DEFAULT_ADAPTER',
    'LIBOVR_ERROR_HYBRID_GRAPHICS_NOT_SUPPORTED',
    'LIBOVR_ERROR_DISPLAY_MANAGER_INIT',
    'LIBOVR_ERROR_TRACKER_DRIVER_INIT',
    'LIBOVR_ERROR_LIB_SIGN_CHECK',
    'LIBOVR_ERROR_LIB_PATH',
    'LIBOVR_ERROR_LIB_SYMBOLS',
    'LIBOVR_ERROR_REMOTE_SESSION',
    'LIBOVR_ERROR_INITIALIZE_VULKAN',
    'LIBOVR_ERROR_BLACKLISTED_GFX_DRIVER',
    'LIBOVR_ERROR_DISPLAY_LOST',
    'LIBOVR_ERROR_TEXTURE_SWAP_CHAIN_FULL',
    'LIBOVR_ERROR_TEXTURE_SWAP_CHAIN_INVALID',
    'LIBOVR_ERROR_GRAPHICS_DEVICE_RESET',
    'LIBOVR_ERROR_DISPLAY_REMOVED',
    'LIBOVR_ERROR_CONTENT_PROTECTION_NOT_AVAILABLE',
    'LIBOVR_ERROR_APPLICATION_VISIBLE',
    'LIBOVR_ERROR_DISALLOWED',
    'LIBOVR_ERROR_DISPLAY_PLUGGED_INCORRECTY',
    'LIBOVR_ERROR_DISPLAY_LIMIT_REACHED',
    'LIBOVR_ERROR_RUNTIME_EXCEPTION',
    'LIBOVR_ERROR_NO_CALIBRATION',
    'LIBOVR_ERROR_OLD_VERSION',
    'LIBOVR_ERROR_MISFORMATTED_BLOCK',
    'LIBOVR_EYE_LEFT',
    'LIBOVR_EYE_RIGHT',
    'LIBOVR_EYE_COUNT',
    'LIBOVR_HAND_LEFT',
    'LIBOVR_HAND_RIGHT',
    'LIBOVR_HAND_COUNT',
    'LIBOVR_CONTROLLER_TYPE_XBOX',
    'LIBOVR_CONTROLLER_TYPE_REMOTE',
    'LIBOVR_CONTROLLER_TYPE_TOUCH',
    'LIBOVR_CONTROLLER_TYPE_LTOUCH',
    'LIBOVR_CONTROLLER_TYPE_RTOUCH',
    'LIBOVR_BUTTON_A',
    'LIBOVR_BUTTON_B',
    'LIBOVR_BUTTON_RTHUMB',
    'LIBOVR_BUTTON_RSHOULDER',
    'LIBOVR_BUTTON_X',
    'LIBOVR_BUTTON_Y',
    'LIBOVR_BUTTON_LTHUMB',
    'LIBOVR_BUTTON_LSHOULDER',
    'LIBOVR_BUTTON_UP',
    'LIBOVR_BUTTON_DOWN',
    'LIBOVR_BUTTON_LEFT',
    'LIBOVR_BUTTON_RIGHT',
    'LIBOVR_BUTTON_ENTER',
    'LIBOVR_BUTTON_BACK',
    'LIBOVR_BUTTON_VOLUP',
    'LIBOVR_BUTTON_VOLDOWN',
    'LIBOVR_BUTTON_HOME',
    'LIBOVR_BUTTON_PRIVATE',
    'LIBOVR_BUTTON_RMASK',
    'LIBOVR_BUTTON_LMASK',
    'LIBOVR_TEXTURE_SWAP_CHAIN0',
    'LIBOVR_TEXTURE_SWAP_CHAIN1',
    'LIBOVR_TEXTURE_SWAP_CHAIN2',
    'LIBOVR_TEXTURE_SWAP_CHAIN3',
    'LIBOVR_TEXTURE_SWAP_CHAIN4',
    'LIBOVR_TEXTURE_SWAP_CHAIN5',
    'LIBOVR_TEXTURE_SWAP_CHAIN6',
    'LIBOVR_TEXTURE_SWAP_CHAIN7',
    'LIBOVR_FORMAT_R8G8B8A8_UNORM',
    'LIBOVR_FORMAT_R8G8B8A8_UNORM_SRGB',
    'LIBOVR_FORMAT_R16G16B16A16_FLOAT',
    'LIBOVR_FORMAT_R11G11B10_FLOAT',
    'LIBOVR_FORMAT_D16_UNORM',
    'LIBOVR_FORMAT_D24_UNORM_S8_UINT',
    'LIBOVR_FORMAT_D32_FLOAT',
    'LIBOVR_MAX_PROVIDED_FRAME_STATS',
    'LIBOVR_TRACKED_DEVICE_TYPE_HMD',
    'LIBOVR_TRACKED_DEVICE_TYPE_LTOUCH',
    'LIBOVR_TRACKED_DEVICE_TYPE_RTOUCH',
    'LIBOVR_TRACKED_DEVICE_TYPE_TOUCH',
    'LIBOVR_TRACKED_DEVICE_TYPE_OBJECT0',
    'LIBOVR_TRACKED_DEVICE_TYPE_OBJECT1',
    'LIBOVR_TRACKED_DEVICE_TYPE_OBJECT2',
    'LIBOVR_TRACKED_DEVICE_TYPE_OBJECT3',
    'LibOVRPose',
    'LibOVRPoseState',
    'LibOVRTrackerInfo',
    'LibOVRSessionStatus',
    'LibOVRHmdInfo',
    'LibOVRFrameStat',
    'success',
    'unqualifedSuccess',
    'failure',
    'isOculusServiceRunning',
    'isHmdConnected',
    'getHmdInfo',
    'getUserHeight',
    'getEyeHeight',
    'getNeckEyeDist',
    'getEyeToNoseDist',
    'initialize',
    'create',
    'destroyTextureSwapChain',
    'destroyMirrorTexture',
    'destroy',
    'shutdown',
    'getGraphicsLUID',
    'setHighQuality',
    'setHeadLocked',
    'getPixelsPerTanAngleAtCenter',
    'getDistortedViewport',
    'getEyeRenderFov',
    'setEyeRenderFov',
    'calcEyeBufferSize',
    'getTextureSwapChainLengthGL',
    'getTextureSwapChainCurrentIndex',
    'getTextureSwapChainBufferGL',
    'createTextureSwapChainGL',
    'setEyeColorTextureSwapChain',
    'createMirrorTexture',
    'getMirrorTexture',
    'getTrackedPoses',
    #'getDevicePoses',
    'calcEyePoses',
    'getHmdToEyePoses',
    'setHmdToEyePoses',
    'getEyeRenderPoses',
    'setEyeRenderPoses',
    'getEyeProjectionMatrix',
    'getEyeRenderViewport',
    'setEyeRenderViewport',
    'getEyeViewMatrix',
    'getPredictedDisplayTime',
    'timeInSeconds',
    'perfHudMode',
    'hidePerfHud',
    'perfHudModes',
    'waitToBeginFrame',
    'beginFrame',
    'commitTextureSwapChain',
    'endFrame',
    'resetFrameStats',
    'getTrackingOriginType',
    'setTrackingOriginType',
    'recenterTrackingOrigin',
    'getTrackerCount',
    'getTrackerInfo',
    'refreshPerformanceStats',
    'updatePerfStats',
    'getAdaptiveGpuPerformanceScale',
    'getFrameStatsCount',
    'anyFrameStatsDropped',
    'checkAswIsAvailable',
    'getVisibleProcessId',
    'checkAppLastFrameDropped',
    'checkCompLastFrameDropped',
    'getFrameStats',
    'getLastErrorInfo',
    'setBoundaryColor',
    'resetBoundaryColor',
    'getBoundaryVisible',
    'showBoundary',
    'hideBoundary',
    'getBoundaryDimensions',
    'getConnectedControllerTypes',
    'updateInputState',
    'getInputTime',
    'getButton',
    'getTouch',
    'getThumbstickValues',
    'getIndexTriggerValues',
    'getHandTriggerValues',
    'setControllerVibration',
    'getSessionStatus'
    #'anyPointInFrustum'
]

from .cimport libovr_capi
from .cimport libovr_math

from libc.stdint cimport int32_t, uint32_t
from libc.stdlib cimport malloc, free
from libc.math cimport pow

cimport numpy as np
import numpy as np

# -----------------
# Initialize module
# -----------------
#
cdef libovr_capi.ovrInitParams _initParams  # initialization parameters
cdef libovr_capi.ovrSession _ptrSession  # session pointer
cdef libovr_capi.ovrGraphicsLuid _gfxLuid  # LUID
cdef libovr_capi.ovrHmdDesc _hmdDesc  # HMD information descriptor
cdef libovr_capi.ovrBoundaryLookAndFeel _boundryStyle
cdef libovr_capi.ovrTextureSwapChain[8] _swapChains
cdef libovr_capi.ovrMirrorTexture _mirrorTexture

# VR related data persistent across frames
cdef libovr_capi.ovrLayerEyeFov _eyeLayer
cdef libovr_capi.ovrEyeRenderDesc[2] _eyeRenderDesc
cdef libovr_capi.ovrTrackingState _trackingState
cdef libovr_capi.ovrViewScaleDesc _viewScale

# prepare the render layer
_eyeLayer.Header.Type = libovr_capi.ovrLayerType_EyeFov
_eyeLayer.Header.Flags = \
    libovr_capi.ovrLayerFlag_TextureOriginAtBottomLeft | \
    libovr_capi.ovrLayerFlag_HighQuality
_eyeLayer.ColorTexture[0] = _eyeLayer.ColorTexture[1] = NULL

# status and performance information
cdef libovr_capi.ovrSessionStatus _sessionStatus
cdef libovr_capi.ovrPerfStats _frameStats
cdef libovr_capi.ovrPerfStatsPerCompositorFrame _lastFrameStats
cdef list compFrameStats

# error information
cdef libovr_capi.ovrErrorInfo _errorInfo  # store our last error here

# controller states
cdef libovr_capi.ovrInputState[5] _inputStates
cdef libovr_capi.ovrInputState[5] _prevInputState

# debug mode
cdef bint _debugMode

# geometric data
cdef float[2] _nearClip
cdef float[2] _farClip
cdef libovr_math.Matrix4f[2] _eyeProjectionMatrix
cdef libovr_math.Matrix4f[2] _eyeViewMatrix
cdef libovr_math.Matrix4f[2] _eyeViewProjectionMatrix

# Function to check for errors returned by OVRLib functions
#
cdef libovr_capi.ovrErrorInfo _last_error_info_  # store our last error here
def check_result(result):
    if libovr_capi.OVR_FAILURE(result):
        libovr_capi.ovr_GetLastErrorInfo(&_last_error_info_)
        raise RuntimeError(
            str(result) + ": " + _last_error_info_.ErrorString.decode("utf-8"))

# helper functions
cdef float maxf(float a, float b):
    return a if a >= b else b

# Color texture formats supported by OpenGL, can be used for creating swap
# chains.
#
cdef dict _supported_texture_formats = {
    "R8G8B8A8_UNORM": libovr_capi.OVR_FORMAT_R8G8B8A8_UNORM,
    "R8G8B8A8_UNORM_SRGB": libovr_capi.OVR_FORMAT_R8G8B8A8_UNORM_SRGB,
    "R16G16B16A16_FLOAT": libovr_capi.OVR_FORMAT_R16G16B16A16_FLOAT,
    "R11G11B10_FLOAT": libovr_capi.OVR_FORMAT_R11G11B10_FLOAT
}

# Performance HUD modes
#
cdef dict _performance_hud_modes = {
    "Off" : libovr_capi.ovrPerfHud_Off,
    "PerfSummary": libovr_capi.ovrPerfHud_PerfSummary,
    "AppRenderTiming" : libovr_capi.ovrPerfHud_AppRenderTiming,
    "LatencyTiming" : libovr_capi.ovrPerfHud_LatencyTiming,
    "CompRenderTiming" : libovr_capi.ovrPerfHud_CompRenderTiming,
    "AswStats" : libovr_capi.ovrPerfHud_AswStats,
    "VersionInfo" : libovr_capi.ovrPerfHud_VersionInfo
}

# mirror texture options
#
cdef dict _mirror_texture_options = {
    "Default" : libovr_capi.ovrMirrorOption_Default,
    "PostDistortion" : libovr_capi.ovrMirrorOption_PostDistortion,
    "LeftEyeOnly" : libovr_capi.ovrMirrorOption_LeftEyeOnly,
    "RightEyeOnly" : libovr_capi.ovrMirrorOption_RightEyeOnly,
    "IncludeGuardian" : libovr_capi.ovrMirrorOption_IncludeGuardian,
    "IncludeNotifications" : libovr_capi.ovrMirrorOption_IncludeNotifications,
    "IncludeSystemGui" : libovr_capi.ovrMirrorOption_IncludeSystemGui
}

# controller enums associated with each string identifier
#
cdef dict _controller_type_enum = {
    "Xbox": libovr_capi.ovrControllerType_XBox,
    "Remote": libovr_capi.ovrControllerType_Remote,
    "Touch": libovr_capi.ovrControllerType_Touch,
    "LeftTouch": libovr_capi.ovrControllerType_LTouch,
    "RightTouch": libovr_capi.ovrControllerType_RTouch
}

# Button values
#
cdef dict _controller_buttons = {
    "A": libovr_capi.ovrButton_A,
    "B": libovr_capi.ovrButton_B,
    "RThumb": libovr_capi.ovrButton_RThumb,
    "RShoulder": libovr_capi.ovrButton_RShoulder,
    "X": libovr_capi.ovrButton_X,
    "Y": libovr_capi.ovrButton_Y,
    "LThumb": libovr_capi.ovrButton_LThumb,
    "LShoulder": libovr_capi.ovrButton_LShoulder,
    "Up": libovr_capi.ovrButton_Up,
    "Down": libovr_capi.ovrButton_Down,
    "Left": libovr_capi.ovrButton_Left,
    "Right": libovr_capi.ovrButton_Right,
    "Enter": libovr_capi.ovrButton_Enter,
    "Back": libovr_capi.ovrButton_Back,
    "VolUp": libovr_capi.ovrButton_VolUp,
    "VolDown": libovr_capi.ovrButton_VolDown,
    "Home": libovr_capi.ovrButton_Home,
    "Private": libovr_capi.ovrButton_Private,
    "RMask": libovr_capi.ovrButton_RMask,
    "LMask": libovr_capi.ovrButton_LMask}

LIBOVR_BUTTON_A = libovr_capi.ovrButton_A
LIBOVR_BUTTON_B = libovr_capi.ovrButton_B
LIBOVR_BUTTON_RTHUMB = libovr_capi.ovrButton_RThumb
LIBOVR_BUTTON_RSHOULDER = libovr_capi.ovrButton_RShoulder
LIBOVR_BUTTON_X = libovr_capi.ovrButton_X
LIBOVR_BUTTON_Y = libovr_capi.ovrButton_Y
LIBOVR_BUTTON_LTHUMB = libovr_capi.ovrButton_LThumb
LIBOVR_BUTTON_LSHOULDER = libovr_capi.ovrButton_LShoulder
LIBOVR_BUTTON_UP = libovr_capi.ovrButton_Up
LIBOVR_BUTTON_DOWN = libovr_capi.ovrButton_Down
LIBOVR_BUTTON_LEFT = libovr_capi.ovrButton_Left
LIBOVR_BUTTON_RIGHT = libovr_capi.ovrButton_Right
LIBOVR_BUTTON_ENTER = libovr_capi.ovrButton_Enter
LIBOVR_BUTTON_BACK = libovr_capi.ovrButton_Back
LIBOVR_BUTTON_VOLUP = libovr_capi.ovrButton_VolUp
LIBOVR_BUTTON_VOLDOWN = libovr_capi.ovrButton_VolDown
LIBOVR_BUTTON_HOME = libovr_capi.ovrButton_Home
LIBOVR_BUTTON_PRIVATE = libovr_capi.ovrButton_Private
LIBOVR_BUTTON_RMASK = libovr_capi.ovrButton_RMask
LIBOVR_BUTTON_LMASK = libovr_capi.ovrButton_LMask

# Touch states
#
cdef dict _touch_states = {
    "A": libovr_capi.ovrTouch_A,
    "B": libovr_capi.ovrTouch_B,
    "RThumb": libovr_capi.ovrTouch_RThumb,
    "RThumbRest": libovr_capi.ovrTouch_RThumbRest,
    "RIndexTrigger": libovr_capi.ovrTouch_RThumb,
    "X": libovr_capi.ovrTouch_X,
    "Y": libovr_capi.ovrTouch_Y,
    "LThumb": libovr_capi.ovrTouch_LThumb,
    "LThumbRest": libovr_capi.ovrTouch_LThumbRest,
    "LIndexTrigger": libovr_capi.ovrTouch_LIndexTrigger,
    "RIndexPointing": libovr_capi.ovrTouch_RIndexPointing,
    "RThumbUp": libovr_capi.ovrTouch_RThumbUp,
    "LIndexPointing": libovr_capi.ovrTouch_LIndexPointing,
    "LThumbUp": libovr_capi.ovrTouch_LThumbUp}

LIBOVR_TOUCH_A = libovr_capi.ovrTouch_A
LIBOVR_TOUCH_B = libovr_capi.ovrTouch_B
LIBOVR_TOUCH_RTHUMB = libovr_capi.ovrTouch_RThumb
LIBOVR_TOUCH_RTHUMBREST = libovr_capi.ovrTouch_RThumbRest
LIBOVR_TOUCH_X = libovr_capi.ovrTouch_X
LIBOVR_TOUCH_Y = libovr_capi.ovrTouch_Y
LIBOVR_TOUCH_LTHUMB = libovr_capi.ovrTouch_LThumb
LIBOVR_TOUCH_RTHUMBREST = libovr_capi.ovrTouch_LThumbRest
LIBOVR_TOUCH_LINDEXTRIGGER = libovr_capi.ovrTouch_LIndexTrigger
LIBOVR_TOUCH_RINDEXPOINTING = libovr_capi.ovrTouch_RIndexPointing
LIBOVR_TOUCH_RTHUMBUP = libovr_capi.ovrTouch_RThumbUp
LIBOVR_TOUCH_LINDEXPOINTING = libovr_capi.ovrTouch_LIndexPointing
LIBOVR_TOUCH_LTHUMBUP = libovr_capi.ovrTouch_LThumbUp

# Controller types
#
cdef dict _controller_types = {
    'Xbox' : libovr_capi.ovrControllerType_XBox,
    'Remote' : libovr_capi.ovrControllerType_Remote,
    'Touch' : libovr_capi.ovrControllerType_Touch,
    'LeftTouch' : libovr_capi.ovrControllerType_LTouch,
    'RightTouch' : libovr_capi.ovrControllerType_RTouch}

# ---------
# Constants
# ---------
#
# controller types
LIBOVR_CONTROLLER_TYPE_XBOX = libovr_capi.ovrControllerType_XBox
LIBOVR_CONTROLLER_TYPE_REMOTE = libovr_capi.ovrControllerType_Remote
LIBOVR_CONTROLLER_TYPE_TOUCH = libovr_capi.ovrControllerType_Touch
LIBOVR_CONTROLLER_TYPE_LTOUCH = libovr_capi.ovrControllerType_LTouch
LIBOVR_CONTROLLER_TYPE_RTOUCH = libovr_capi.ovrControllerType_RTouch

# return success codes, values other than 'LIBOVR_SUCCESS' are conditional
LIBOVR_SUCCESS = libovr_capi.ovrSuccess
LIBOVR_SUCCESS_NOT_VISIBLE = libovr_capi.ovrSuccess_NotVisible
LIBOVR_SUCCESS_DEVICE_UNAVAILABLE = libovr_capi.ovrSuccess_DeviceUnavailable
LIBOVR_SUCCESS_BOUNDARY_INVALID = libovr_capi.ovrSuccess_BoundaryInvalid

# return error code, not all of these are applicable
LIBOVR_ERROR_MEMORY_ALLOCATION_FAILURE = libovr_capi.ovrError_MemoryAllocationFailure
LIBOVR_ERROR_INVALID_SESSION = libovr_capi.ovrError_InvalidSession
LIBOVR_ERROR_TIMEOUT = libovr_capi.ovrError_Timeout
LIBOVR_ERROR_NOT_INITIALIZED = libovr_capi.ovrError_NotInitialized
LIBOVR_ERROR_INVALID_PARAMETER = libovr_capi.ovrError_InvalidParameter
LIBOVR_ERROR_SERVICE_ERROR = libovr_capi.ovrError_ServiceError
LIBOVR_ERROR_NO_HMD = libovr_capi.ovrError_NoHmd
LIBOVR_ERROR_UNSUPPORTED = libovr_capi.ovrError_Unsupported
LIBOVR_ERROR_DEVICE_UNAVAILABLE = libovr_capi.ovrError_DeviceUnavailable
LIBOVR_ERROR_INVALID_HEADSET_ORIENTATION = libovr_capi.ovrError_InvalidHeadsetOrientation
LIBOVR_ERROR_CLIENT_SKIPPED_DESTROY = libovr_capi.ovrError_ClientSkippedDestroy
LIBOVR_ERROR_CLIENT_SKIPPED_SHUTDOWN = libovr_capi.ovrError_ClientSkippedShutdown
LIBOVR_ERROR_SERVICE_DEADLOCK_DETECTED = libovr_capi.ovrError_ServiceDeadlockDetected
LIBOVR_ERROR_INVALID_OPERATION = libovr_capi.ovrError_InvalidOperation
LIBOVR_ERROR_INSUFFICENT_ARRAY_SIZE = libovr_capi.ovrError_InsufficientArraySize
LIBOVR_ERROR_NO_EXTERNAL_CAMERA_INFO = libovr_capi.ovrError_NoExternalCameraInfo
LIBOVR_ERROR_LOST_TRACKING = libovr_capi.ovrError_LostTracking
LIBOVR_ERROR_EXTERNAL_CAMERA_INITIALIZED_FAILED = libovr_capi.ovrError_ExternalCameraInitializedFailed
LIBOVR_ERROR_EXTERNAL_CAMERA_CAPTURE_FAILED = libovr_capi.ovrError_ExternalCameraCaptureFailed
LIBOVR_ERROR_EXTERNAL_CAMERA_NAME_LISTS_BUFFER_SIZE = libovr_capi.ovrError_ExternalCameraNameListsBufferSize
LIBOVR_ERROR_EXTERNAL_CAMERA_NAME_LISTS_MISMATCH = libovr_capi.ovrError_ExternalCameraNameListsMistmatch
LIBOVR_ERROR_EXTERNAL_CAMERA_NOT_CALIBRATED = libovr_capi.ovrError_ExternalCameraNotCalibrated
LIBOVR_ERROR_EXTERNAL_CAMERA_NAME_WRONG_SIZE = libovr_capi.ovrError_ExternalCameraNameWrongSize
LIBOVR_ERROR_AUDIO_DEVICE_NOT_FOUND = libovr_capi.ovrError_AudioDeviceNotFound
LIBOVR_ERROR_AUDIO_COM_ERROR = libovr_capi.ovrError_AudioComError
LIBOVR_ERROR_INITIALIZE = libovr_capi.ovrError_Initialize
LIBOVR_ERROR_LIB_LOAD = libovr_capi.ovrError_LibLoad
LIBOVR_ERROR_SERVICE_CONNECTION = libovr_capi.ovrError_ServiceConnection
LIBOVR_ERROR_SERVICE_VERSION = libovr_capi.ovrError_ServiceVersion
LIBOVR_ERROR_INCOMPATIBLE_OS = libovr_capi.ovrError_IncompatibleOS
LIBOVR_ERROR_DISPLAY_INIT = libovr_capi.ovrError_DisplayInit
LIBOVR_ERROR_SERVER_START = libovr_capi.ovrError_ServerStart
LIBOVR_ERROR_REINITIALIZATION = libovr_capi.ovrError_Reinitialization
LIBOVR_ERROR_MISMATCHED_ADAPTERS = libovr_capi.ovrError_MismatchedAdapters
LIBOVR_ERROR_LEAKING_RESOURCES = libovr_capi.ovrError_LeakingResources
LIBOVR_ERROR_CLIENT_VERSION = libovr_capi.ovrError_ClientVersion
LIBOVR_ERROR_OUT_OF_DATE_OS = libovr_capi.ovrError_OutOfDateOS
LIBOVR_ERROR_OUT_OF_DATE_GFX_DRIVER = libovr_capi.ovrError_OutOfDateGfxDriver
LIBOVR_ERROR_INCOMPATIBLE_OS = libovr_capi.ovrError_IncompatibleGPU
LIBOVR_ERROR_NO_VALID_VR_DISPLAY_SYSTEM = libovr_capi.ovrError_NoValidVRDisplaySystem
LIBOVR_ERROR_OBSOLETE = libovr_capi.ovrError_Obsolete
LIBOVR_ERROR_DISABLED_OR_DEFAULT_ADAPTER = libovr_capi.ovrError_DisabledOrDefaultAdapter
LIBOVR_ERROR_HYBRID_GRAPHICS_NOT_SUPPORTED = libovr_capi.ovrError_HybridGraphicsNotSupported
LIBOVR_ERROR_DISPLAY_MANAGER_INIT = libovr_capi.ovrError_DisplayManagerInit
LIBOVR_ERROR_TRACKER_DRIVER_INIT = libovr_capi.ovrError_TrackerDriverInit
LIBOVR_ERROR_LIB_SIGN_CHECK = libovr_capi.ovrError_LibSignCheck
LIBOVR_ERROR_LIB_PATH = libovr_capi.ovrError_LibPath
LIBOVR_ERROR_LIB_SYMBOLS = libovr_capi.ovrError_LibSymbols
LIBOVR_ERROR_REMOTE_SESSION = libovr_capi.ovrError_RemoteSession
LIBOVR_ERROR_INITIALIZE_VULKAN = libovr_capi.ovrError_InitializeVulkan
LIBOVR_ERROR_BLACKLISTED_GFX_DRIVER = libovr_capi.ovrError_BlacklistedGfxDriver
LIBOVR_ERROR_DISPLAY_LOST = libovr_capi.ovrError_DisplayLost
LIBOVR_ERROR_TEXTURE_SWAP_CHAIN_FULL = libovr_capi.ovrError_TextureSwapChainFull
LIBOVR_ERROR_TEXTURE_SWAP_CHAIN_INVALID = libovr_capi.ovrError_TextureSwapChainInvalid
LIBOVR_ERROR_GRAPHICS_DEVICE_RESET = libovr_capi.ovrError_GraphicsDeviceReset
LIBOVR_ERROR_DISPLAY_REMOVED = libovr_capi.ovrError_DisplayRemoved
LIBOVR_ERROR_CONTENT_PROTECTION_NOT_AVAILABLE = libovr_capi.ovrError_ContentProtectionNotAvailable
LIBOVR_ERROR_APPLICATION_VISIBLE = libovr_capi.ovrError_ApplicationInvisible
LIBOVR_ERROR_DISALLOWED = libovr_capi.ovrError_Disallowed
LIBOVR_ERROR_DISPLAY_PLUGGED_INCORRECTY = libovr_capi.ovrError_DisplayPluggedIncorrectly
LIBOVR_ERROR_DISPLAY_LIMIT_REACHED = libovr_capi.ovrError_DisplayLimitReached
LIBOVR_ERROR_RUNTIME_EXCEPTION = libovr_capi.ovrError_RuntimeException
LIBOVR_ERROR_NO_CALIBRATION = libovr_capi.ovrError_NoCalibration
LIBOVR_ERROR_OLD_VERSION = libovr_capi.ovrError_OldVersion
LIBOVR_ERROR_MISFORMATTED_BLOCK = libovr_capi.ovrError_MisformattedBlock

# misc constants
LIBOVR_EYE_LEFT = libovr_capi.ovrEye_Left
LIBOVR_EYE_RIGHT = libovr_capi.ovrEye_Right
LIBOVR_EYE_COUNT = libovr_capi.ovrEye_Count
LIBOVR_HAND_LEFT = libovr_capi.ovrHand_Left
LIBOVR_HAND_RIGHT = libovr_capi.ovrHand_Right
LIBOVR_HAND_COUNT = libovr_capi.ovrHand_Count

# swapchain handles, more than enough for now
LIBOVR_TEXTURE_SWAP_CHAIN0 = 0
LIBOVR_TEXTURE_SWAP_CHAIN1 = 1
LIBOVR_TEXTURE_SWAP_CHAIN2 = 2
LIBOVR_TEXTURE_SWAP_CHAIN3 = 3
LIBOVR_TEXTURE_SWAP_CHAIN4 = 4
LIBOVR_TEXTURE_SWAP_CHAIN5 = 5
LIBOVR_TEXTURE_SWAP_CHAIN6 = 6
LIBOVR_TEXTURE_SWAP_CHAIN7 = 7

# texture formats, color and depth
LIBOVR_FORMAT_R8G8B8A8_UNORM = libovr_capi.OVR_FORMAT_R8G8B8A8_UNORM
LIBOVR_FORMAT_R8G8B8A8_UNORM_SRGB = libovr_capi.OVR_FORMAT_R8G8B8A8_UNORM_SRGB
LIBOVR_FORMAT_R16G16B16A16_FLOAT =  libovr_capi.OVR_FORMAT_R16G16B16A16_FLOAT
LIBOVR_FORMAT_R11G11B10_FLOAT = libovr_capi.OVR_FORMAT_R11G11B10_FLOAT
LIBOVR_FORMAT_D16_UNORM = libovr_capi.OVR_FORMAT_D16_UNORM
LIBOVR_FORMAT_D24_UNORM_S8_UINT = libovr_capi.OVR_FORMAT_D24_UNORM_S8_UINT
LIBOVR_FORMAT_D32_FLOAT = libovr_capi.OVR_FORMAT_D32_FLOAT

# performance
LIBOVR_MAX_PROVIDED_FRAME_STATS = libovr_capi.ovrMaxProvidedFrameStats

# tracked device types
LIBOVR_TRACKED_DEVICE_TYPE_HMD = libovr_capi.ovrTrackedDevice_HMD
LIBOVR_TRACKED_DEVICE_TYPE_LTOUCH = libovr_capi.ovrTrackedDevice_LTouch
LIBOVR_TRACKED_DEVICE_TYPE_RTOUCH = libovr_capi.ovrTrackedDevice_RTouch
LIBOVR_TRACKED_DEVICE_TYPE_TOUCH = libovr_capi.ovrTrackedDevice_Touch
LIBOVR_TRACKED_DEVICE_TYPE_OBJECT0 = libovr_capi.ovrTrackedDevice_Object0
LIBOVR_TRACKED_DEVICE_TYPE_OBJECT1 = libovr_capi.ovrTrackedDevice_Object1
LIBOVR_TRACKED_DEVICE_TYPE_OBJECT2 = libovr_capi.ovrTrackedDevice_Object2
LIBOVR_TRACKED_DEVICE_TYPE_OBJECT3 = libovr_capi.ovrTrackedDevice_Object3


# ------------------------------------------------------------------------------
# Classes and extension types
#

cdef class LibOVRPose(object):
    """Class for rigid body pose data for LibOVR.

    Parameters
    ----------
    pos : tuple, list, or ndarray of float
        Position vector (x, y, z).
    ori : tuple, list, or ndarray of float
        Orientation quaternion vector (x, y, z, w).

    Attributes
    ----------
    pos : ndarray
        Position vector [X, Y, Z] (read-only).
    ori : ndarray
        Orientation quaternion [X, Y, Z, W] (read-only).
    posOri : tuple of ndarray
        Combined position and orientation (read-only).
    at : ndarray
        Forward vector of this pose (-Z is forward) (read-only).
    up : ndarray
        Up vector of this pose (+Y is up) (read-only).

    """
    cdef libovr_capi.ovrPosef* c_data
    cdef libovr_capi.ovrPosef c_ovrPosef  # internal data

    def __init__(self, pos=(0., 0., 0.), ori=(0., 0., 0., 1.)):
        pass  # nop

    def __cinit__(self, pos=(0., 0., 0.), ori=(0., 0., 0., 1.)):
        self.c_data = &self.c_ovrPosef  # pointer to c_ovrPosef

        self.c_data[0].Position.x = <float>pos[0]
        self.c_data[0].Position.y = <float>pos[1]
        self.c_data[0].Position.z = <float>pos[2]

        self.c_data[0].Orientation.x = <float>ori[0]
        self.c_data[0].Orientation.y = <float>ori[1]
        self.c_data[0].Orientation.z = <float>ori[2]
        self.c_data[0].Orientation.w = <float>ori[3]

    def __mul__(LibOVRPose a, LibOVRPose b):
        """Multiplication operator (*) to combine poses."""
        cdef libovr_math.Posef pose_r = \
            <libovr_math.Posef>a.c_data[0] * <libovr_math.Posef>b.c_data[0]

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
        """Invert operator (~) to invert a pose."""
        return self.inverted()

    def __eq__(self, LibOVRPose other):
        """Equality operator (==) for two poses.

        The tolerance of the comparison is defined by the Oculus SDK as 1e-5.

        """
        return <bint>(<libovr_math.Posef>other.c_data[0]).IsEqual(
            <libovr_math.Posef>other.c_data[0], <float>1e-5)

    def __ne__(self, LibOVRPose other):
        """Inequality operator (!=) for two poses.

        The tolerance of the comparison is defined by the Oculus SDK as 1e-5.

        """
        return not <bint>(<libovr_math.Posef>other.c_data[0]).IsEqual(
            <libovr_math.Posef>other.c_data[0], <float>1e-5)

    def __str__(self):
        return \
            "LibOVRPose(({px}, {py}, {pz}), ({rx}, {ry}, {rz}, {rw}))".format(
                px=self.c_data[0].Position.x,
                py=self.c_data[0].Position.y,
                pz=self.c_data[0].Position.z,
                rx=self.c_data[0].Orientation.x,
                ry=self.c_data[0].Orientation.y,
                rz=self.c_data[0].Orientation.z,
                rw=self.c_data[0].Orientation.w)

    def setIdentity(self):
        """Clear this pose's translation and orientation."""
        (<libovr_math.Posef>self.c_data[0]).SetIdentity()

    @property
    def pos(self):
        return self.getPos()

    def getPos(self, object outVector=None):
        """Position vector X, Y, Z (`ndarray` of `float`).

        The returned object is a NumPy array which contains a copy of the data
        stored in an internal structure (ovrPosef). The array is conformal with
        the internal data's type (float32) and size (length 3).

        Parameters
        ----------
        outVector : ndarray or None
            Option array to write values to. If None, the function will return
            a new array. Must have a float32 data type.

        Returns
        -------
        ndarray or None

        Examples
        --------

        Get the position coordinates::

            x, y, z = myPose.getPos()  # Python float literals
            # ... or ...
            pos = myPose.getPos()  # NumPy array shape=(3,) and dtype=float32

        Write the position to an existing array by specifying `outVector`::

            position = numpy.zeros((3,), dtype=numpy.float32)  # mind the dtype!
            myPose.getPos(position)  # position now contains myPose.pos

        You can also pass a view/slice to `outVector`::

            coords = numpy.zeros((100,3,), dtype=numpy.float32)  # big array
            myPose.getPos(coords[42,:])  # row 42

        Notes
        -----
        Q: Why is there no property setter for `pos`?
        A: It confused people that setting values of the returned array didn't
        update anything.

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if outVector is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = outVector

        toReturn[0] = self.c_data[0].Position.x
        toReturn[1] = self.c_data[0].Position.y
        toReturn[2] = self.c_data[0].Position.z

        if outVector is None:
            return toReturn

    def setPos(self, object pos):
        self.c_data[0].Position.x = <float>pos[0]
        self.c_data[0].Position.y = <float>pos[1]
        self.c_data[0].Position.z = <float>pos[2]

    @property
    def ori(self):
        return self.getOri()

    def getOri(self, object outVector=None):
        """Orientation quaternion X, Y, Z, W (`ndarray` of `float`).

        Components X, Y, Z are imaginary and W is real.

        The returned object is a NumPy array which references data stored in an
        internal structure (ovrPosef). The array is conformal with the internal
        data's type (float32) and size (length 3).

        Parameters
        ----------
        outVector : ndarray or None
            Option array to write values to. If None, the function will return
            a new array. Must have a float32 data type.

        Returns
        -------
        ndarray or None

        Notes
        -----
            The orientation quaternion should be normalized.

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if outVector is None:
            toReturn = np.zeros((4,), dtype=np.float32)
        else:
            toReturn = outVector

        toReturn[0] = self.c_data[0].Orientation.x
        toReturn[1] = self.c_data[0].Orientation.y
        toReturn[2] = self.c_data[0].Orientation.z
        toReturn[3] = self.c_data[0].Orientation.w

        if outVector is None:
            return toReturn

    def setOri(self, object ori):
        self.c_data[0].Orientation.x = <float>ori[0]
        self.c_data[0].Orientation.y = <float>ori[1]
        self.c_data[0].Orientation.z = <float>ori[2]
        self.c_data[0].Orientation.w = <float>ori[3]

    @property
    def posOri(self):
        return self.pos, self.ori

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

    @property
    def at(self):
        return self.getAt()

    def getAt(self, object outVector=None):
        """Get the 'at' vector for this pose.

        Parameters
        ----------
        outVector : ndarray or None
            Option array to write values to. If None, the function will return
            a new array. Must have a float32 data type.

        Returns
        -------
        ndarray or None
            The vector for 'at' if `outVector`=None. Returns None if `outVector`
            was specified.

        Notes
        -----
        It's better to use the 'at' property if you are not supplying an output
        array. However, `getAt` will have the same effect as the property if
        `outVector`=None.

        Examples
        --------

        Setting the listener orientation for 3D positional audio (PyOpenAL)::

            myListener.set_orientation((*myPose.getAt(), *myPose.getUp()))

        See Also
        --------
        getUp : Get the 'up' vector.

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if outVector is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = outVector

        cdef libovr_math.Vector3f at = \
            (<libovr_math.Quatf>self.c_data[0].Orientation).Rotate(
                libovr_math.Vector3f(0.0, 0.0, -1.0))

        toReturn[0] = at.x
        toReturn[1] = at.y
        toReturn[2] = at.z

        if outVector is None:
            return toReturn

    @property
    def up(self):
        return self.getUp()

    def getUp(self, object outVector=None):
        """Get the 'up' vector for this pose.

        Parameters
        ----------
        outVector : ndarray, optional
            Option array to write values to. If None, the function will return
            a new array. Must have a float32 data type and a length of 3.

        Returns
        -------
        ndarray or None
            The vector for 'up' if `outVector`=None. Returns None if `outVector`
            was specified.

        Notes
        -----
        It's better to use the 'up' property if you are not supplying an output
        array. However, `getUp` will have the same effect as the property if
        outVector=None.

        Examples
        --------

        Using the 'up' vector with gluLookAt::

            up = myPose.getUp()  # myPose.up also works
            center = myPose.pos
            target = targetPose.pos  # some target pose
            gluLookAt(*(*up, *center, *target))

        See Also
        --------
        getAt : Get the 'at' vector.

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if outVector is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = outVector

        cdef libovr_math.Vector3f up = \
            (<libovr_math.Quatf>self.c_data[0].Orientation).Rotate(
                libovr_math.Vector3f(0.0, 1.0, 0.0))

        toReturn[0] = up.x
        toReturn[1] = up.y
        toReturn[2] = up.z

        if outVector is None:
            return toReturn

    def getYawPitchRoll(self, LibOVRPose refPose=None):
        """Get the yaw, pitch, and roll of the orientation quaternion.

        Parameters
        ----------
        refPose : LibOVRPose, optional
            Reference pose to compute angles relative to. If None is specified,
            computed values are referenced relative to the world axes.

        Returns
        -------
        ndarray of floats
            Yaw, pitch, and roll of the pose in degrees.

        """
        cdef float yaw, pitch, roll
        cdef libovr_math.Posef inPose = <libovr_math.Posef>self.c_data[0]
        cdef libovr_math.Posef invRef

        if refPose is not None:
            invRef = (<libovr_math.Posef>refPose.c_data[0]).Inverted()
            inPose = invRef * inPose

        inPose.Rotation.GetYawPitchRoll(&yaw, &pitch, &roll)

        cdef np.ndarray[np.float32_t, ndim=1] to_return = \
            np.array((yaw, pitch, roll), dtype=np.float32)

        return to_return

    def getTransformMatrix(self, bint inverse=False):
        """Convert this pose into a 4x4 transformation matrix.

        Parameters
        ----------
        inverse : bool, optional
            If True, return the inverse of the matrix.

        Returns
        -------
        ndarray
            4x4 transformation matrix.

        """
        cdef libovr_math.Matrix4f m_pose = libovr_math.Matrix4f(
            <libovr_math.Posef>self.c_data[0])

        if inverse:
            m_pose.InvertHomogeneousTransform()  # faster than Invert() here

        cdef np.ndarray[np.float32_t, ndim=2] to_return = \
            np.zeros((4, 4), dtype=np.float32)

        # fast copy matrix to numpy array
        cdef float [:, :] mv = to_return
        cdef Py_ssize_t i, j
        cdef Py_ssize_t N = 4
        i = j = 0
        for i in range(N):
            for j in range(N):
                mv[i, j] = m_pose.M[i][j]

        return to_return

    def normalize(self):
        """Normalize this pose.

        """
        (<libovr_math.Posef>self.c_data[0]).Normalize()

    def inverted(self):
        """Get the inverse of the pose.

        Returns
        -------
        `LibOVRPose`
            Inverted pose.

        """
        cdef libovr_math.Quatf inv_ori = \
            (<libovr_math.Quatf>self.c_data[0].Orientation).Inverted()
        cdef libovr_math.Vector3f inv_pos = \
            (<libovr_math.Quatf>inv_ori).Rotate(
                -(<libovr_math.Vector3f>self.c_data[0].Position))
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
        cdef libovr_math.Vector3f pos_in = libovr_math.Vector3f(
            <float>v[0], <float>v[1], <float>v[2])
        cdef libovr_math.Vector3f rotated_pos = \
            (<libovr_math.Posef>self.c_data[0]).Rotate(pos_in)

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
        cdef libovr_math.Vector3f pos_in = libovr_math.Vector3f(
            <float>v[0], <float>v[1], <float>v[2])
        cdef libovr_math.Vector3f invRotatedPos = \
            (<libovr_math.Posef>self.c_data[0]).InverseRotate(pos_in)

        cdef np.ndarray[np.float32_t, ndim=1] to_return = \
            np.array((invRotatedPos.x, invRotatedPos.y, invRotatedPos.z),
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
        cdef libovr_math.Vector3f pos_in = libovr_math.Vector3f(
            <float>v[0], <float>v[1], <float>v[2])
        cdef libovr_math.Vector3f translated_pos = \
            (<libovr_math.Posef>self.c_data[0]).Translate(pos_in)

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
        cdef libovr_math.Vector3f pos_in = libovr_math.Vector3f(
            <float>v[0], <float>v[1], <float>v[2])
        cdef libovr_math.Vector3f transformed_pos = \
            (<libovr_math.Posef>self.c_data[0]).Transform(pos_in)

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
        cdef libovr_math.Vector3f pos_in = libovr_math.Vector3f(
            <float>v[0], <float>v[1], <float>v[2])
        cdef libovr_math.Vector3f transformed_pos = \
            (<libovr_math.Posef>self.c_data[0]).InverseTransform(pos_in)

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
        cdef libovr_math.Vector3f pos_in = libovr_math.Vector3f(
            <float>v[0], <float>v[1], <float>v[2])
        cdef libovr_math.Vector3f transformed_pos = \
            (<libovr_math.Posef>self.c_data[0]).TransformNormal(pos_in)

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
        cdef libovr_math.Vector3f pos_in = libovr_math.Vector3f(
            <float>v[0], <float>v[1], <float>v[2])
        cdef libovr_math.Vector3f transformed_pos = \
            (<libovr_math.Posef>self.c_data[0]).InverseTransformNormal(pos_in)

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
        cdef libovr_math.Vector3f pos_in = libovr_math.Vector3f(
            <float>v[0], <float>v[1], <float>v[2])
        cdef libovr_math.Vector3f transformed_pos = \
            (<libovr_math.Posef>self.c_data[0]).Apply(pos_in)

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
        cdef libovr_math.Vector3f pos_in

        if isinstance(v, LibOVRPose):
            pos_in = <libovr_math.Vector3f>((<LibOVRPose>v).c_data[0]).Position
        else:
            pos_in = libovr_math.Vector3f(<float>v[0], <float>v[1], <float>v[2])

        cdef float to_return = \
            (<libovr_math.Posef>self.c_data[0]).Translation.Distance(pos_in)

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
            Coordinates of the center of the target sphere (x, y, z).
        radius : float, optional
            The radius of the target.
        rayDir : tuple, list, or ndarray of floats, optional
            Vector indicating the direction for the ray (default is -Z).
        maxRange : float, optional
            The maximum range of the ray. Ray testing will fail automatically if
            the target is out of range. The ray has infinite length if None is
            specified. Ray is infinite if `maxRange`=0.0.

        Returns
        -------
        bool
            True if the ray intersects anywhere on the bounding sphere, False in
            every other condition.

        """
        cdef libovr_math.Vector3f targetPos = libovr_math.Vector3f(
            <float>targetPose[0], <float>targetPose[1], <float>targetPose[2])
        cdef libovr_math.Vector3f _rayDir = libovr_math.Vector3f(
            <float>rayDir[0], <float>rayDir[1], <float>rayDir[2])
        cdef libovr_math.Posef originPos = <libovr_math.Posef>self.c_data[0]

        # if the ray is finite, does it ever touch the edge of the sphere?
        cdef float targetDist
        if maxRange != 0.0:
            targetDist = targetPos.Distance(originPos.Translation) - radius
            if targetDist > maxRange:
                return False

        # put the target in the ray caster's local coordinate system
        cdef libovr_math.Vector3f offset = -originPos.InverseTransform(targetPos)

        # find the discriminant, this is based on the method described here:
        # http://antongerdelan.net/opengl/raycasting.html
        cdef float desc = <float>pow(_rayDir.Dot(offset), 2.0) - \
               (offset.Dot(offset) - <float>pow(radius, 2.0))

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
        fast : bool, optional
            If True, use fast interpolation which is quicker but less accurate
            over larger distances.

        Returns
        -------
        LibOVRPose
            Interpolated pose at `s`.

        """
        cdef libovr_math.Posef _toPose = <libovr_math.Posef>toPose.c_data[0]
        cdef libovr_math.Posef interp

        if not fast:
            interp = (<libovr_math.Posef>self.c_data[0]).Lerp(_toPose, s)
        else:
            interp = (<libovr_math.Posef>self.c_data[0]).FastLerp(_toPose, s)

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
        arr : float* 
            Pointer to the first element of the array.
        
        """
        arr[0] = self.c_data[0].Position.x
        arr[1] = self.c_data[0].Position.y
        arr[2] = self.c_data[0].Position.z
        arr[3] = self.c_data[0].Orientation.x
        arr[4] = self.c_data[0].Orientation.y
        arr[5] = self.c_data[0].Orientation.z
        arr[6] = self.c_data[0].Orientation.w

    #cdef fromarray(self, float* arr):
    #    pass


cdef class LibOVRPoseState(object):
    """Class for data about rigid body configuration with derivatives computed
    by the LibOVR runtime.

    """
    cdef libovr_capi.ovrPoseStatef* c_data
    cdef libovr_capi.ovrPoseStatef c_ovrPoseStatef

    cdef LibOVRPose _pose

    cdef int status_flags

    def __cinit__(self):
        self.c_data = &self.c_ovrPoseStatef  # pointer to ovrPoseStatef

        # the pose is accessed using a LibOVRPose object
        self._pose = LibOVRPose()
        self._pose.c_data = &self.c_data.ThePose

    @property
    def pose(self):
        """Rigid body pose.

        """
        return self._pose

    @pose.setter
    def pose(self, LibOVRPose value):
        self._pose.c_data[0] = value.c_data[0]  # copy into

    @property
    def angularVelocity(self):
        """Angular velocity vector in radians/sec."""
        return np.array((self.c_data.AngularVelocity.x,
                         self.c_data.AngularVelocity.y,
                         self.c_data.AngularVelocity.z), dtype=np.float32)

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
        return np.array((self.c_data.LinearVelocity.x,
                         self.c_data.LinearVelocity.y,
                         self.c_data.LinearVelocity.z), dtype=np.float32)

    @linearVelocity.setter
    def linearVelocity(self, object value):
        self.c_data[0].LinearVelocity.x = <float>value[0]
        self.c_data[0].LinearVelocity.y = <float>value[1]
        self.c_data[0].LinearVelocity.z = <float>value[2]

    @property
    def angularAcceleration(self):
        """Angular acceleration vector in radians/s^2."""
        return np.array((self.c_data.AngularAcceleration.x,
                         self.c_data.AngularAcceleration.y,
                         self.c_data.AngularAcceleration.z), dtype=np.float32)

    @angularAcceleration.setter
    def angularAcceleration(self, object value):
        self.c_data[0].AngularAcceleration.x = <float>value[0]
        self.c_data[0].AngularAcceleration.y = <float>value[1]
        self.c_data[0].AngularAcceleration.z = <float>value[2]

    @property
    def linearAcceleration(self):
        """Linear acceleration vector in meters/s^2."""
        return np.array((self.c_data.LinearAcceleration.x,
                         self.c_data.LinearAcceleration.y,
                         self.c_data.LinearAcceleration.z), dtype=np.float32)

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
        return <bint>((libovr_capi.ovrStatus_OrientationTracked &
             self.status_flags) == libovr_capi.ovrStatus_OrientationTracked)

    @property
    def positionTracked(self):
        """True if the position was tracked when sampled."""
        return <bint>((libovr_capi.ovrStatus_PositionTracked &
             self.status_flags) == libovr_capi.ovrStatus_PositionTracked)

    @property
    def fullyTracked(self):
        """True if position and orientation were tracked when sampled."""
        cdef int32_t full_tracking_flags = \
            libovr_capi.ovrStatus_OrientationTracked | \
            libovr_capi.ovrStatus_PositionTracked
        return <bint>((self.status_flags & full_tracking_flags) ==
                      full_tracking_flags)

    def timeIntegrate(self, float dt):
        """Time integrate rigid body motion derivatives referenced by the
        current pose.

        Parameters
        ----------
        dt : float
            Time delta in seconds.

        Returns
        -------
        LibOVRPose

        """
        cdef libovr_math.Posef res = \
            (<libovr_math.Posef>self.c_data[0].ThePose).TimeIntegrate(
                <libovr_math.Vector3f>self.c_data[0].LinearVelocity,
                <libovr_math.Vector3f>self.c_data[0].AngularVelocity,
                <libovr_math.Vector3f>self.c_data[0].LinearAcceleration,
                <libovr_math.Vector3f>self.c_data[0].AngularAcceleration,
                dt)
        cdef LibOVRPose toReturn = LibOVRPose(
            (res.Translation.x, res.Translation.y, res.Translation.z),
            (res.Rotation.x, res.Rotation.y, res.Rotation.z, res.Rotation.w))

        return toReturn


cdef class LibOVRTrackerInfo(object):
    """Class for information about camera based tracking sensors.

    """
    cdef libovr_capi.ovrTrackerPose* c_data
    cdef libovr_capi.ovrTrackerPose c_ovrTrackerPose
    cdef libovr_capi.ovrTrackerDesc c_ovrTrackerDesc

    cdef LibOVRPose _pose
    cdef LibOVRPose _leveledPose

    cdef unsigned int _trackerIndex

    def __cinit__(self):
        self._pose = LibOVRPose()
        self._leveledPose = LibOVRPose()
        self._trackerIndex = 0

    @property
    def trackerIndex(self):
        """Tracker index this objects refers to."""
        return self._trackerIndex

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
        return <bint>((libovr_capi.ovrTracker_Connected &
             self.c_ovrTrackerPose.TrackerFlags) == libovr_capi.ovrTracker_Connected)

    @property
    def isPoseTracked(self):
        """True if the sensor has a valid pose (`bool`)."""
        return <bint>((libovr_capi.ovrTracker_PoseTracked &
             self.c_ovrTrackerPose.TrackerFlags) == libovr_capi.ovrTracker_PoseTracked)

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
    def horizontalFov(self):
        """Horizontal FOV of the sensor in radians (`float`)."""
        return self.c_ovrTrackerDesc.FrustumHFovInRadians

    @property
    def verticalFov(self):
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
    cdef libovr_capi.ovrSessionStatus* c_data
    cdef libovr_capi.ovrSessionStatus c_ovrSessionStatus

    def __cinit__(self):
        self.c_data = &self.c_ovrSessionStatus

    @property
    def isVisible(self):
        """True if the application has focus and visible in the HMD."""
        return self.c_data.IsVisible == libovr_capi.ovrTrue

    @property
    def hmdPresent(self):
        """True if the HMD is present."""
        return self.c_data.HmdPresent == libovr_capi.ovrTrue

    @property
    def hmdMounted(self):
        """True if the HMD is on the user's head."""
        return self.c_data.HmdMounted == libovr_capi.ovrTrue

    @property
    def displayLost(self):
        """True if the the display was lost."""
        return self.c_data.DisplayLost == libovr_capi.ovrTrue

    @property
    def shouldQuit(self):
        """True if the application was signaled to quit."""
        return self.c_data.ShouldQuit == libovr_capi.ovrTrue

    @property
    def shouldRecenter(self):
        """True if the application was signaled to re-center."""
        return self.c_data.ShouldRecenter == libovr_capi.ovrTrue

    @property
    def hasInputFocus(self):
        """True if the application has input focus."""
        return self.c_data.HasInputFocus == libovr_capi.ovrTrue

    @property
    def overlayPresent(self):
        """True if the system overlay is present."""
        return self.c_data.OverlayPresent == libovr_capi.ovrTrue

    @property
    def depthRequested(self):
        """True if the system requires a depth texture. Currently unused by
        PsychXR."""
        return self.c_data.DepthRequested == libovr_capi.ovrTrue


cdef class LibOVRHmdInfo(object):
    """Class for HMD information returned by 'getHmdInfo'."""

    cdef libovr_capi.ovrHmdDesc* c_data
    cdef libovr_capi.ovrHmdDesc c_ovrHmdDesc

    def __cinit__(self, *args, **kwargs):
        self.c_data = &self.c_ovrHmdDesc

    @property
    def productName(self):
        """Get the product name for this device.

        Returns
        -------
        str
            Product name string (utf-8).

        """
        return self.c_data[0].ProductName.decode('utf-8')

    @property
    def manufacturer(self):
        """Get the device manufacturer name.

        Returns
        -------
        str
            Manufacturer name string (utf-8).

        """
        return self.c_data[0].Manufacturer.decode('utf-8')

    @property
    def serialNumber(self):
        """Get the device serial number.

        Returns
        -------
        str
            Serial number (utf-8).

        """
        return self.c_data[0].SerialNumber.decode('utf-8')

    @property
    def resolution(self):
        """Horizontal and vertical resolution of the display in pixels.

        Returns
        -------
        ndarray of int
            Resolution of the display [w, h].

        """
        return np.asarray((self.c_data[0].Resolution.w,
                           self.c_data[0].Resolution.h), dtype=int)

    @property
    def refreshRate(self):
        """Nominal refresh rate in Hertz of the display.

        Returns
        -------
        float
            Refresh rate in Hz.

        """
        return <float>self.c_data[0].DisplayRefreshRate

    @property
    def hid(self):
        """USB human interface device class identifiers.

        Returns
        -------
        tuple
            USB HIDs (vendor, product).

        """
        return <int>self.c_data[0].VendorId, <int>self.c_data[0].ProductId

    @property
    def firmwareVersion(self):
        """Firmware version for this device.

        Returns
        -------
        tuple
            Firmware version (major, minor).

        """
        return <int>self.c_data[0].FirmwareMajor, \
               <int>self.c_data[0].FirmwareMinor

    @property
    def defaultEyeFov(self):
        """Default or recommended eye field-of-views (FOVs) provided by the API.

        Returns
        -------
        tuple of ndarray
            Pair of left and right eye FOVs specified as tangent angles [Up,
            Down, Left, Right].

        """
        cdef np.ndarray fovLeft = np.asarray([
            self.c_data[0].DefaultEyeFov[0].UpTan,
            self.c_data[0].DefaultEyeFov[0].DownTan,
            self.c_data[0].DefaultEyeFov[0].LeftTan,
            self.c_data[0].DefaultEyeFov[0].RightTan],
            dtype=np.float32)

        cdef np.ndarray fovRight = np.asarray([
            self.c_data[0].DefaultEyeFov[1].UpTan,
            self.c_data[0].DefaultEyeFov[1].DownTan,
            self.c_data[0].DefaultEyeFov[1].LeftTan,
            self.c_data[0].DefaultEyeFov[1].RightTan],
            dtype=np.float32)

        return fovLeft, fovRight

    @property
    def maxEyeFov(self):
        """Maximum eye field-of-views (FOVs) provided by the API.

        Returns
        -------
        tuple of ndarray
            Pair of left and right eye FOVs specified as tangent angles in
            radians [Up, Down, Left, Right].

        """
        cdef np.ndarray[float, ndim=1] fov_left = np.asarray([
            self.c_data[0].MaxEyeFov[0].UpTan,
            self.c_data[0].MaxEyeFov[0].DownTan,
            self.c_data[0].MaxEyeFov[0].LeftTan,
            self.c_data[0].MaxEyeFov[0].RightTan],
            dtype=np.float32)

        cdef np.ndarray[float, ndim=1] fov_right = np.asarray([
            self.c_data[0].MaxEyeFov[1].UpTan,
            self.c_data[0].MaxEyeFov[1].DownTan,
            self.c_data[0].MaxEyeFov[1].LeftTan,
            self.c_data[0].MaxEyeFov[1].RightTan],
            dtype=np.float32)

        return fov_left, fov_right

    @property
    def symmetricEyeFov(self):
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
        cdef libovr_capi.ovrFovPort fov_left = self.c_data[0].DefaultEyeFov[0]
        cdef libovr_capi.ovrFovPort fov_right = self.c_data[0].DefaultEyeFov[1]

        cdef libovr_capi.ovrFovPort fov_max
        fov_max.UpTan = maxf(fov_left.UpTan, fov_right.UpTan)
        fov_max.DownTan = maxf(fov_left.DownTan, fov_right.DownTan)
        fov_max.LeftTan = maxf(fov_left.LeftTan, fov_right.LeftTan)
        fov_max.RightTan = maxf(fov_left.RightTan, fov_right.RightTan)

        cdef float tan_half_fov_horz = maxf(fov_max.LeftTan, fov_max.RightTan)
        cdef float tan_half_fov_vert = maxf(fov_max.DownTan, fov_max.UpTan)

        cdef libovr_capi.ovrFovPort fov_both
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


cdef class LibOVRFrameStat(object):
    cdef libovr_capi.ovrPerfStatsPerCompositorFrame* c_data
    cdef libovr_capi.ovrPerfStatsPerCompositorFrame c_ovrPerfStatsPerCompositorFrame

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


def success(int result):
    """Check if an API return indicates success."""
    return <bint>libovr_capi.OVR_SUCCESS(result)

def unqualifedSuccess(int result):
    """Check if an API return indicates unqualified success."""
    return <bint>libovr_capi.OVR_UNQUALIFIED_SUCCESS(result)

def failure(int result):
    """Check if an API return indicates failure (error)."""
    return <bint>libovr_capi.OVR_FAILURE(result)

def isOculusServiceRunning(int timeoutMS=100):
    """Check if the Oculus Runtime is loaded and running.

    Parameters
    ----------
    timeout_ms : int
        Timeout in milliseconds.

    Returns
    -------
    bool

    """
    cdef libovr_capi.ovrDetectResult result = libovr_capi.ovr_Detect(
        timeoutMS)

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
    cdef libovr_capi.ovrDetectResult result = libovr_capi.ovr_Detect(
        timeout_ms)

    return <bint>result.IsOculusHMDConnected

def getHmdInfo():
    """Get HMD information.

    Returns
    -------
    LibOVRHmdInfo
        HMD information.

    """
    global _hmdDesc
    cdef LibOVRHmdInfo toReturn = LibOVRHmdInfo()
    toReturn.c_data[0] = _hmdDesc

    return toReturn

def getUserHeight():
    """User's calibrated height in meters.

    Returns
    -------
    float
        Distance from floor to the top of the user's head in meters reported
        by LibOVR. If not set, the default value is 1.778 meters.

    """
    global _ptrSession
    cdef float to_return = libovr_capi.ovr_GetFloat(
        _ptrSession,
        b"PlayerHeight",
        <float> 1.778)

    return to_return

def getEyeHeight():
    """Calibrated eye height from floor in meters.

    Returns
    -------
    float
        Distance from floor to the user's eye level in meters.

    """
    global _ptrSession
    cdef float to_return = libovr_capi.ovr_GetFloat(
        _ptrSession,
        b"EyeHeight",
        <float> 1.675)

    return to_return

def getNeckEyeDist():
    """Distance from the neck to eyes in meters.

    Returns
    -------
    float
        Distance in meters.

    """
    global _ptrSession
    cdef float vals[2]

    cdef unsigned int ret = libovr_capi.ovr_GetFloatArray(
        _ptrSession,
        b"NeckEyeDistance",
        vals,
        <unsigned int>2)

    return <float> vals[0], <float> vals[1]

def getEyeToNoseDist():
    """Distance between the nose and eyes in meters.

    Returns
    -------
    float
        Distance in meters.

    """
    global _ptrSession
    cdef float vals[2]

    cdef unsigned int ret = libovr_capi.ovr_GetFloatArray(
        _ptrSession,
        b"EyeToNoseDist",
        vals,
        <unsigned int> 2)

    return <float>vals[0], <float> vals[1]

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

    """
    cdef int32_t flags = libovr_capi.ovrInit_RequestVersion
    if focusAware is True:
        flags |= libovr_capi.ovrInit_FocusAware

    #if debug is True:
    #    flags |= libovr_capi.ovrInit_Debug
    global _initParams
    _initParams.Flags = flags
    _initParams.RequestedMinorVersion = libovr_capi.OVR_MINOR_VERSION
    _initParams.LogCallback = NULL  # not used yet
    _initParams.ConnectionTimeoutMS = <uint32_t>connectionTimeout
    cdef libovr_capi.ovrResult result = libovr_capi.ovr_Initialize(
        &_initParams)

    return result  # failed to initalize, return error code

def create():
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
    global _gfxLuid
    global _eyeLayer
    global _hmdDesc
    global _eyeRenderDesc

    result = libovr_capi.ovr_Create(&_ptrSession, &_gfxLuid)
    check_result(result)
    if libovr_capi.OVR_FAILURE(result):
        return result  # failed to create session, return error code

    # if we got to this point, everything should be fine
    # get HMD descriptor
    _hmdDesc = libovr_capi.ovr_GetHmdDesc(_ptrSession)

    # configure the eye render descriptor to use the recommended FOV, this
    # can be changed later
    cdef Py_ssize_t i = 0
    for i in range(libovr_capi.ovrEye_Count):
        _eyeRenderDesc[i] = libovr_capi.ovr_GetRenderDesc(
            _ptrSession,
            <libovr_capi.ovrEyeType>i,
            _hmdDesc.DefaultEyeFov[i])

        _eyeLayer.Fov[i] = _eyeRenderDesc[i].Fov

    return result

def destroyTextureSwapChain(int swapChain):
    """Destroy a texture swap chain."""
    global _ptrSession
    global _swapChains
    libovr_capi.ovr_DestroyTextureSwapChain(_ptrSession, _swapChains[swapChain])
    _swapChains[swapChain] = NULL

def destroyMirrorTexture():
    """Destroy the mirror texture."""
    global _ptrSession
    global _mirrorTexture
    if _mirrorTexture != NULL:
        libovr_capi.ovr_DestroyMirrorTexture(_ptrSession, _mirrorTexture)

def destroy():
    """Destroy a session.
    """
    global _ptrSession
    global _eyeLayer
    # null eye textures in eye layer
    _eyeLayer.ColorTexture[0] = _eyeLayer.ColorTexture[1] = NULL

    # destroy the current session and shutdown
    libovr_capi.ovr_Destroy(_ptrSession)

def shutdown():
    """End the current session.

    Clean-up routines are executed that destroy all swap chains and mirror
    texture buffers, afterwards control is returned to Oculus Home. This
    must be called after every successful 'initialize' call.

    """
    libovr_capi.ovr_Shutdown()

def getGraphicsLUID():
    """The graphics device LUID."""
    global _gfxLuid
    return _gfxLuid.Reserved.decode('utf-8')

def setHighQuality(bint enable):
    """Enable high quality mode.
    """
    global _eyeLayer
    if enable:
        _eyeLayer.Header.Flags |= libovr_capi.ovrLayerFlag_HighQuality
    else:
        _eyeLayer.Header.Flags &= ~libovr_capi.ovrLayerFlag_HighQuality

def setHeadLocked(bint enable):
    """True when head-locked mode is enabled.

    This is disabled by default when a session is started. Enable this if you
    are considering to use custom head poses.

    """
    global _eyeLayer
    if enable:
        _eyeLayer.Header.Flags |= libovr_capi.ovrLayerFlag_HeadLocked
    else:
        _eyeLayer.Header.Flags &= ~libovr_capi.ovrLayerFlag_HeadLocked

def getPixelsPerTanAngleAtCenter(int eye):
    """Get pixels per tan angle at te center of the display.

    You must call 'setEyeRenderFov' first for values to be valid.

    """
    global _eyeRenderDesc

    cdef libovr_capi.ovrVector2f toReturn = \
        _eyeRenderDesc[eye].PixelsPerTanAngleAtCenter

    return toReturn.x, toReturn.y

def getDistortedViewport(int eye):
    """Get the distorted viewport.

    You must call 'setEyeRenderFov' first for values to be valid.

    """
    cdef libovr_capi.ovrRecti distVp = _eyeRenderDesc[eye].DistortedViewport

    cdef np.ndarray toReturn = np.asarray([
        distVp.Pos.x,
        distVp.Pos.x,
        distVp.Size.w,
        distVp.Size.h],
        dtype=np.int)

    return toReturn

def getEyeRenderFov(int eye):
    """Get the field-of-view to use for rendering.

    The FOV for a given eye are defined as a tuple of tangent angles (Up,
    Down, Left, Right). By default, this function will return the default
    FOVs after 'start' is called (see 'defaultEyeFOVs'). You can override
    these values using 'maxEyeFOVs' and 'symmetricEyeFOVs', or with
    custom values (see Examples below).

    Parameters
    ----------
    eye : int
        Eye index.

    Returns
    -------
    ndarray of floats
        Eye FOV tangent angles [UpTan, DownTan, LeftTan, RightTan].

    Examples
    --------
    Getting the tangent angles::

        leftFov = libovr.getEyeRenderFOV(libovr.LIBOVR_EYE_LEFT)
        # left FOV tangent angles, do the same for the right
        upTan, downTan, leftTan, rightTan =  leftFov

    """
    global _eyeRenderDesc
    cdef np.ndarray to_return = np.asarray([
        _eyeRenderDesc[eye].Fov.UpTan,
        _eyeRenderDesc[eye].Fov.DownTan,
        _eyeRenderDesc[eye].Fov.LeftTan,
        _eyeRenderDesc[eye].Fov.RightTan],
        dtype=np.float32)

    return to_return

def setEyeRenderFov(int eye, object fov):
    """Set the field-of-view of a given eye. This is used to compute the
    projection matrix.

    Parameters
    ----------
    eye : int
        Eye index. Values are `LIBOVR_EYE_LEFT` and `LIBOVR_EYE_RIGHT`.
    fov : tuple, list or ndarray of floats
        Eye FOV tangent angles [UpTan, DownTan, LeftTan, RightTan].

    Examples
    --------

    Setting eye render FOVs to symmetric (needed for mono rendering)::

        leftFov, rightFov = libovr.getSymmetricEyeFOVs()
        libovr.setEyeRenderFOV(libovr.LIBOVR_EYE_LEFT, leftFov)
        libovr.setEyeRenderFOV(libovr.LIBOVR_EYE_RIGHT, rightFov)

    Using custom values::

        # Up, Down, Left, Right tan angles
        libovr.setEyeRenderFOV(libovr.LIBOVR_EYE_LEFT, [1.0, -1.0, -1.0, 1.0])

    """
    global _ptrSession
    global _eyeRenderDesc
    global _eyeLayer

    cdef libovr_capi.ovrFovPort fov_in
    fov_in.UpTan = <float>fov[0]
    fov_in.DownTan = <float>fov[1]
    fov_in.LeftTan = <float>fov[2]
    fov_in.RightTan = <float>fov[3]

    _eyeRenderDesc[<int>eye] = libovr_capi.ovr_GetRenderDesc(
        _ptrSession,
        <libovr_capi.ovrEyeType>eye,
        fov_in)

    # set in eye layer too
    _eyeLayer.Fov[eye] = _eyeRenderDesc[eye].Fov

def calcEyeBufferSize(int eye, float texelsPerPixel=1.0):
    """Get the recommended buffer (texture) sizes for eye buffers.

    Should be called after 'setEyerenderFovs'. Returns buffer resolutions in
    pixels (w, h). The values can be used when configuring a framebuffer or swap
    chain for rendering.

    Parameters
    ----------
    eye: int
        Eye index. Use either :data:LIBOVR_EYE_LEFT or :data:LIBOVR_EYE_RIGHT.
    texelsPerPixel : float
        Display pixels per texture pixels at the center of the display. Use a
        value less than 1.0 to improve performance at the cost of resolution.
        Specifying a larger texture is possible, but not recommended by the
        manufacturer.

    Returns
    -------
    tuple of tuples
        Buffer widths and heights (w, h) for each eye.

    Examples
    --------
    Getting the buffer sizes::

        # eye FOVs must be set first!
        leftFov, rightFov = libovr.getDefaultEyeFOVs()
        libovr.setEyeRenderFOV(libovr.LIBOVR_EYE_LEFT, leftFov)
        libovr.setEyeRenderFOV(libovr.LIBOVR_EYE_RIGHT, rightFov)

        leftBufferSize, rightBufferSize = libovr.calcEyeBufferSize()
        leftW leftH = leftBufferSize
        rightW, rightH = rightBufferSize
        # combined size if using a single texture buffer for both eyes
        w, h = leftW + rightW, max(leftH, rightH)
        # make the texture ...

    Notes
    -----
    This function returns the recommended texture resolution for a specified
    eye. If you are using a single buffer for both eyes, that buffer should be
    as wide as the combined width of both eye's values.

    """
    global _ptrSession
    global _eyeRenderDesc

    cdef libovr_capi.ovrSizei buffSize = libovr_capi.ovr_GetFovTextureSize(
        _ptrSession,
        <libovr_capi.ovrEyeType>0,
        _eyeRenderDesc[0].Fov,
        <float>texelsPerPixel)

    return buffSize.w, buffSize.h

def getTextureSwapChainLengthGL(int swapChain):
    """Get the length of a specified swap chain.

    Parameters
    ----------
    swapChain : int
        Swap chain handle to query. Must be a swap chain initialized by a
        previous call to 'createTextureSwapChainGL'.

    Returns
    -------
    tuple of int
        Result of the 'ovr_GetTextureSwapChainLength' API call and the
        length of that swap chain.

    """
    cdef int outLength
    cdef libovr_capi.ovrResult result = 0
    global _swapChains
    global _ptrSession
    global _eyeLayer

    # check if there is a swap chain in the slot
    if _eyeLayer.ColorTexture[swapChain] == NULL:
        raise RuntimeError(
            "Cannot get swap chain length, NULL eye buffer texture.")

    # get the current texture index within the swap chain
    result = libovr_capi.ovr_GetTextureSwapChainLength(
        _ptrSession, _swapChains[swapChain], &outLength)

    return result, outLength

def getTextureSwapChainCurrentIndex(int swapChain):
    """Get the current buffer index within the swap chain.

    Parameters
    ----------
    swapChain : int
        Swap chain handle to query. Must be a swap chain initialized by a
        previous call to 'createTextureSwapChainGL'.

    Returns
    -------
    tuple of int
        Result of the 'ovr_GetTextureSwapChainCurrentIndex' API call and the
        index of the buffer.

    """
    cdef int current_idx = 0
    cdef libovr_capi.ovrResult result = 0
    global _swapChains
    global _eyeLayer
    global _ptrSession

    # check if there is a swap chain in the slot
    if _eyeLayer.ColorTexture[swapChain] == NULL:
        raise RuntimeError(
            "Cannot get buffer ID, NULL eye buffer texture.")

    # get the current texture index within the swap chain
    result = libovr_capi.ovr_GetTextureSwapChainCurrentIndex(
        _ptrSession, _swapChains[swapChain], &current_idx)

    return result, current_idx

def getTextureSwapChainBufferGL(int swapChain, int index):
    """Get the texture buffer as an OpenGL name at a specific index in the
    swap chain for a given swapChain.

    Parameters
    ----------
    swapChain : int
        Swap chain handle to query. Must be a swap chain initialized by a
        previous call to 'createTextureSwapChainGL'.
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
        swapChain = libovr.LIBOVR_TEXTURE_SWAP_CHAIN0
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
    cdef libovr_capi.ovrResult result = libovr_capi.ovr_GetTextureSwapChainBufferGL(
        _ptrSession, _swapChains[swapChain], index, &tex_id)

    return result, tex_id

def createTextureSwapChainGL(int swapChain, int width, int height, int textureFormat=LIBOVR_FORMAT_R8G8B8A8_UNORM_SRGB, int levels=1):
    """Create a texture swap chain for eye image buffers.

    You can create up-to 32 swap chains, referenced by their index.

    Parameters
    ----------
    swapChain : int
        Swap chain handle to initialize, usually 'LIBOVR_SWAP_CHAIN*'.
    width : int
        Width of texture in pixels.
    height : int
        Height of texture in pixels.
    textureFormat : int
        Texture format to use. Valid color texture formats are:
            - :data:`LIBOVR_FORMAT_R8G8B8A8_UNORM`
            - :data:`LIBOVR_FORMAT_R8G8B8A8_UNORM_SRGB`
            - :data:`LIBOVR_FORMAT_R16G16B16A16_FLOAT`
            - :data:`LIBOVR_FORMAT_R11G11B10_FLOAT`
        Depth texture formats:
            - :data:`LIBOVR_FORMAT_D16_UNORM`
            - :data:`LIBOVR_FORMAT_D24_UNORM_S8_UINT`
            - :data:`LIBOVR_FORMAT_D32_FLOAT`

    Other Parameters
    ----------------
    levels : int
        Mip levels to use, default is 1.

    Returns
    -------
    int
        The result of the 'ovr_CreateTextureSwapChainGL' API call.

    Examples
    --------

    Create a texture swap chain::

        result = libovr.createTextureSwapChainGL(libovr.LIBOVR_TEXTURE_SWAP_CHAIN0,
            texWidth, texHeight, libovr.LIBOVR_FORMAT_R8G8B8A8_UNORM)
        # set the swap chain for each eye buffer
        for eye in range(libovr.LIBOVR_EYE_COUNT):
            hmd.setEyeColorTextureSwapChain(eye, libovr.LIBOVR_TEXTURE_SWAP_CHAIN0)

    """
    global _swapChains
    global _ptrSession

    # configure the texture
    cdef libovr_capi.ovrTextureSwapChainDesc swapConfig
    swapConfig.Type = libovr_capi.ovrTexture_2D
    swapConfig.Format = <libovr_capi.ovrTextureFormat>textureFormat
    swapConfig.ArraySize = 1
    swapConfig.Width = <int>width
    swapConfig.Height = <int>height
    swapConfig.MipLevels = <int>levels
    swapConfig.SampleCount = 1
    swapConfig.StaticImage = libovr_capi.ovrFalse  # always buffered
    swapConfig.MiscFlags = libovr_capi.ovrTextureMisc_None
    swapConfig.BindFlags = libovr_capi.ovrTextureBind_None

    # create the swap chain
    cdef libovr_capi.ovrResult result = \
        libovr_capi.ovr_CreateTextureSwapChainGL(
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
        previous call to 'createTextureSwapChainGL'.

    Examples
    --------

    Associate the swap chain with both eyes (single buffer for stereo views)::

        libovr.setEyeColorTextureSwapChain(
            libovr.LIBOVR_EYE_LEFT, libovr.LIBOVR_TEXTURE_SWAP_CHAIN0)
        libovr.setEyeColorTextureSwapChain(
            libovr.LIBOVR_EYE_RIGHT, libovr.LIBOVR_TEXTURE_SWAP_CHAIN0)

        # same as above but with a loop
        for eye in range(libovr.LIBOVR_EYE_COUNT):
            libovr.setEyeColorTextureSwapChain(eye, libovr.LIBOVR_TEXTURE_SWAP_CHAIN0)

    Associate a swap chain with each eye (separate buffer for stereo views)::

        libovr.setEyeColorTextureSwapChain(
            libovr.LIBOVR_EYE_LEFT, libovr.LIBOVR_TEXTURE_SWAP_CHAIN0)
        libovr.setEyeColorTextureSwapChain(
            libovr.LIBOVR_EYE_RIGHT, libovr.LIBOVR_TEXTURE_SWAP_CHAIN1)

        # with a loop ...
        for eye in range(libovr.LIBOVR_EYE_COUNT):
            libovr.setEyeColorTextureSwapChain(
                eye, libovr.LIBOVR_TEXTURE_SWAP_CHAIN0 + eye)

    """
    global _swapChains
    global _eyeLayer

    _eyeLayer.ColorTexture[eye] = _swapChains[swapChain]

def createMirrorTexture(int width, int height, int textureFormat=LIBOVR_FORMAT_R8G8B8A8_UNORM_SRGB):
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
    textureFormat : int
        Color texture format to use, valid texture formats are:
            - :data:`LIBOVR_FORMAT_R8G8B8A8_UNORM`
            - :data:`LIBOVR_FORMAT_R8G8B8A8_UNORM_SRGB`
            - :data:`LIBOVR_FORMAT_R16G16B16A16_FLOAT`
            - :data:`LIBOVR_FORMAT_R11G11B10_FLOAT`

    Returns
    -------
    int
        Result of API call 'ovr_CreateMirrorTextureGL'.

    """
    # additional options
    #cdef unsigned int mirror_options = libovr_capi.ovrMirrorOption_Default
    # set the mirror texture mode
    #if mirrorMode == 'Default':
    #    mirror_options = <libovr_capi.ovrMirrorOptions>libovr_capi.ovrMirrorOption_Default
    #elif mirrorMode == 'PostDistortion':
    #    mirror_options = <libovr_capi.ovrMirrorOptions>libovr_capi.ovrMirrorOption_PostDistortion
    #elif mirrorMode == 'LeftEyeOnly':
    #    mirror_options = <libovr_capi.ovrMirrorOptions>libovr_capi.ovrMirrorOption_LeftEyeOnly
    #elif mirrorMode == 'RightEyeOnly':
    #    mirror_options = <libovr_capi.ovrMirrorOptions>libovr_capi.ovrMirrorOption_RightEyeOnly
    #else:
    #    raise RuntimeError("Invalid 'mirrorMode' mode specified.")

    #if include_guardian:
    #    mirror_options |= libovr_capi.ovrMirrorOption_IncludeGuardian
    #if include_notifications:
    #    mirror_options |= libovr_capi.ovrMirrorOption_IncludeNotifications
    #if include_system_gui:
    #    mirror_options |= libovr_capi.ovrMirrorOption_IncludeSystemGui

    # create the descriptor
    cdef libovr_capi.ovrMirrorTextureDesc mirrorDesc
    global _ptrSession
    global _mirrorTexture

    mirrorDesc.Format = <libovr_capi.ovrTextureFormat>textureFormat
    mirrorDesc.Width = <int>width
    mirrorDesc.Height = <int>height
    mirrorDesc.MiscFlags = libovr_capi.ovrTextureMisc_None
    mirrorDesc.MirrorOptions = libovr_capi.ovrMirrorOption_Default

    cdef libovr_capi.ovrResult result = libovr_capi.ovr_CreateMirrorTextureGL(
        _ptrSession, &mirrorDesc, &_mirrorTexture)

    return <int>result

def getMirrorTexture():
    """Mirror texture ID.

    Returns
    -------
    tuple of int
        Result of API call 'ovr_GetMirrorTextureBufferGL' and the mirror
        texture ID. A mirror texture ID = 0 is invalid.

    Examples
    --------

    Getting the mirror texture for use::

        # get the mirror texture
        result, mirrorTexId = libovr.getMirrorTexture()
        if libovr.LIBOVR_FAILURE(result):
            # raise error ...
        # bind the mirror texture texture to the framebuffer
        GL.glFramebufferTexture2D(
            GL.GL_READ_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D, mirrorTexId, 0)

    """
    cdef unsigned int mirror_id

    global _ptrSession
    global _mirrorTexture

    if _mirrorTexture == NULL:  # no texture created
        return None

    cdef libovr_capi.ovrResult result = \
        libovr_capi.ovr_GetMirrorTextureBufferGL(
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
        trackedPoses = hmd.getTrackedPoses(t)
        head = trackedPoses['Head']

        # check if tracking
        if head.orientationTracked and head.positionTracked:
            hmd.calcEyePose(head.thePose)  # calculate eye poses

    """
    global _ptrSession
    global _eyeLayer
    global _trackingState

    cdef libovr_capi.ovrBool use_marker = \
        libovr_capi.ovrTrue if latencyMarker else libovr_capi.ovrFalse

    _trackingState = libovr_capi.ovr_GetTrackingState(
        _ptrSession, absTime, use_marker)

    cdef LibOVRPoseState head_pose = LibOVRPoseState()
    head_pose.c_data[0] = _trackingState.HeadPose
    head_pose.status_flags = _trackingState.StatusFlags

    # for computing app photon-to-motion latency
    _eyeLayer.SensorSampleTime = _trackingState.HeadPose.TimeInSeconds

    cdef LibOVRPoseState left_hand_pose = LibOVRPoseState()
    left_hand_pose.c_data[0] = _trackingState.HandPoses[0]
    left_hand_pose.status_flags = _trackingState.HandStatusFlags[0]

    cdef LibOVRPoseState right_hand_pose = LibOVRPoseState()
    right_hand_pose.c_data[0] = _trackingState.HandPoses[1]
    right_hand_pose.status_flags = _trackingState.HandStatusFlags[1]

    cdef dict toReturn = {'Head': head_pose,
                          'LeftHand': left_hand_pose,
                          'RightHand': right_hand_pose}

    return toReturn
#
# def getDevicePoses(object deviceTypes, double absTime, bint latencyMarker=True):
#     """Get tracked device poses.
#
#     Each pose in the returned array matches the device type at each index
#     specified in 'deviceTypes'. You need to call this function to get the poses
#     for 'objects', which are additional touch controllers.
#
#     Parameters
#     ----------
#     deviceTypes : `list` or `tuple` of `int`
#         List of device types. Valid device types are:
#
#         - LIBOVR_TRACKED_DEVICE_TYPE_HMD: The head or HMD.
#         - LIBOVR_TRACKED_DEVICE_TYPE_LTOUCH: Left touch controller or hand.
#         - LIBOVR_TRACKED_DEVICE_TYPE_RTOUCH: Right touch controller or hand.
#         - LIBOVR_TRACKED_DEVICE_TYPE_TOUCH: Both touch controllers.
#
#         Up to four additional touch controllers can be paired and tracked, they
#         are assigned types:
#
#         - LIBOVR_TRACKED_DEVICE_TYPE_OBJECT0
#         - LIBOVR_TRACKED_DEVICE_TYPE_OBJECT1
#         - LIBOVR_TRACKED_DEVICE_TYPE_OBJECT2
#         - LIBOVR_TRACKED_DEVICE_TYPE_OBJECT3
#
#     absTime : `float`
#         Absolute time in seconds poses refer to.
#     latencyMarker: `bool`
#         Insert a marker for motion-to-photon latency calculation. Set this to
#         False if 'getTrackedPoses' was previously called and a latency marker
#         was set there.
#
#     Returns
#     -------
#     tuple
#         Return code (`int`) of the 'ovr_GetDevicePoses' API call and list of
#         tracked device poses (`list` of `LibOVRPoseState`).
#
#     Examples
#     --------
#
#     Get HMD and touch controller poses::
#
#         deviceTypes = (ovr.LIBOVR_TRACKED_DEVICE_TYPE_HMD,
#                        ovr.LIBOVR_TRACKED_DEVICE_TYPE_LTOUCH,
#                        ovr.LIBOVR_TRACKED_DEVICE_TYPE_RTOUCH)
#         headPose, leftHandPose, rightHandPose = ovr.getDevicePoses(
#             deviceTypes, absTime)
#
#     """
#     # nop if args indicate no devices
#     if deviceTypes is None:
#         return None
#
#     global _ptrSession
#     global _eyeLayer
#     #global _devicePoses
#
#     # for computing app photon-to-motion latency
#     if latencyMarker:
#         _eyeLayer.SensorSampleTime = absTime
#
#     # allocate arrays to store pose types and poses
#     cdef int count = <int>len(deviceTypes)
#     cdef libovr_capi.ovrTrackedDeviceType* devices = \
#         <libovr_capi.ovrTrackedDeviceType*>malloc(
#             count * sizeof(libovr_capi.ovrTrackedDeviceType))
#     if not devices:
#         raise MemoryError("Failed to allocate array 'devices'.")
#
#     cdef int i = 0
#     for i in range(count):
#         devices[i] = <libovr_capi.ovrTrackedDeviceType>deviceTypes[i]
#
#     cdef libovr_capi.ovrPoseStatef* devicePoses = \
#         <libovr_capi.ovrPoseStatef*>malloc(
#             count * sizeof(libovr_capi.ovrPoseStatef))
#     if not devicePoses:
#         raise MemoryError("Failed to allocate array 'devicePoses'.")
#
#     # get the device poses
#     cdef libovr_capi.ovrResult result = libovr_capi.ovr_GetDevicePoses(
#         _ptrSession,
#         &devices,
#         count,
#         absTime,
#         &devicePoses)
#
#     # build list of device poses
#     cdef list outPoses = list()
#     cdef LibOVRPoseState thisPose
#     for i in range(count):
#         thisPose = LibOVRPoseState()  # new
#         thisPose.c_data[0] = devicePoses[i]
#         outPoses.append(thisPose)
#
#     # free the arrays
#     free(devices)
#     free(devicePoses)
#
#     return result, outPoses

def calcEyePoses(LibOVRPose headPose):
    """Calculate eye poses using a given pose state.

    Eye poses are derived from the head pose stored in the pose state and
    the HMD to eye poses reported by LibOVR. Calculated eye poses are stored
    and passed to the compositor when 'endFrame' is called for additional
    rendering.

    You can access the computed poses via the 'getEyeRenderPose' function.

    Parameters
    ----------
    headPose : LibOVRPose
        Head pose.

    Examples
    --------

    Compute the eye poses from tracker data::

        t = hmd.getPredictedDisplayTime()
        trackedPoses = hmd.getTrackedPoses(t)

        head = trackedPoses['Head']

        # check if tracking
        if head.orientationTracked and head.positionTracked:
            hmd.calcEyePoses(head.thePose)  # calculate eye poses
        else:
            # do something ...

        # computed render poses appear here
        renderPoseLeft, renderPoseRight = hmd.getEyeRenderPoses()

    Use a custom head pose::

        headPose = LibOVRPose((0., 1.5, 0.))  # eyes 1.5 meters off the ground
        hmd.calcEyePoses(headPose)  # calculate eye poses

    """
    global _ptrSession
    global _eyeLayer
    global _eyeRenderDesc
    global _eyeViewMatrix
    global _eyeProjectionMatrix
    global _eyeViewProjectionMatrix

    cdef libovr_capi.ovrPosef[2] hmdToEyePoses
    hmdToEyePoses[0] = _eyeRenderDesc[0].HmdToEyePose
    hmdToEyePoses[1] = _eyeRenderDesc[1].HmdToEyePose

     # calculate the eye poses
    libovr_capi.ovr_CalcEyePoses2(
        headPose.c_data[0],
        hmdToEyePoses,
        _eyeLayer.RenderPose)

    # compute the eye transformation matrices from poses
    cdef libovr_math.Vector3f pos
    cdef libovr_math.Quatf ori
    cdef libovr_math.Vector3f up
    cdef libovr_math.Vector3f forward
    cdef libovr_math.Matrix4f rm

    cdef int eye = 0
    for eye in range(libovr_capi.ovrEye_Count):
        pos = <libovr_math.Vector3f>_eyeLayer.RenderPose[eye].Position
        ori = <libovr_math.Quatf>_eyeLayer.RenderPose[eye].Orientation

        if not ori.IsNormalized():  # make sure orientation is normalized
            ori.Normalize()

        rm = libovr_math.Matrix4f(ori)
        up = rm.Transform(libovr_math.Vector3f(0., 1., 0.))
        forward = rm.Transform(libovr_math.Vector3f(0., 0., -1.))
        _eyeViewMatrix[eye] = \
            libovr_math.Matrix4f.LookAtRH(pos, pos + forward, up)
        _eyeViewProjectionMatrix[eye] = \
            _eyeProjectionMatrix[eye] * _eyeViewMatrix[eye]

def getHmdToEyePoses():
    """HMD to eye poses.

    These are the prototype eye poses specified by LibOVR, defined only
    after 'start' is called. These poses are transformed by the head pose
    by 'calcEyePoses' to get 'getEyeRenderPoses'.

    Notes
    -----
    The horizontal (x-axis) separation of the eye poses are determined by the
    configured lens spacing (slider adjustment). This spacing is supposed to
    correspond to the actual inter-ocular distance (IOD) of the user. You can
    get the IOD used for rendering by adding up the absolute values of the
    x-components of the eye poses, or by multiplying the value of
    'getEyeToNoseDist()' by two. Furthermore, the IOD values can be altered,
    prior to calling 'calcEyePoses', to override the values specified by LibOVR.

    Returns
    -------
    tuple of LibOVRPose
        Copies of the HMD to eye poses for the left and right eye.

    """
    global _eyeRenderDesc
    cdef LibOVRPose leftHmdToEyePose = LibOVRPose()
    cdef LibOVRPose rightHmdToEyePose = LibOVRPose()

    leftHmdToEyePose.c_data[0] = _eyeRenderDesc[0].HmdToEyePose
    rightHmdToEyePose.c_data[0] = _eyeRenderDesc[1].HmdToEyePose

    return leftHmdToEyePose, rightHmdToEyePose

def setHmdToEyePoses(value):
    """Set the HMD eye poses."""
    global _eyeRenderDesc
    _eyeRenderDesc[0].HmdToEyePose = (<LibOVRPose>value[0]).c_data[0]
    _eyeRenderDesc[1].HmdToEyePose = (<LibOVRPose>value[1]).c_data[0]

def getEyeRenderPoses():
    """Get eye render poses.

    Pose are those computed by the last 'calcEyePoses' call. Returned
    objects are copies of the data stored internally by the session
    instance. These poses are used to define the view matrix when rendering
    for each eye.

    Returns
    -------
    tuple of LibOVRPose
        Copies of the HMD to eye poses for the left and right eye.

    Notes
    -----
    The returned LibOVRPose objects are copies of data stored internally by the
    session object. Setting renderPoses will recompute their transformation
    matrices.

    """
    global _eyeLayer

    cdef LibOVRPose left_eye_pose = LibOVRPose()
    cdef LibOVRPose right_eye_pose = LibOVRPose()

    left_eye_pose.c_data[0] = _eyeLayer.RenderPose[0]
    right_eye_pose.c_data[0] = _eyeLayer.RenderPose[1]

    return left_eye_pose, right_eye_pose

def setEyeRenderPoses(object value):
    """Set eye render poses."""

    global _eyeLayer
    global _eyeViewMatrix
    global _eyeViewProjectionMatrix

    _eyeLayer.RenderPose[0] = (<LibOVRPose>value[0]).c_data[0]
    _eyeLayer.RenderPose[1] = (<LibOVRPose>value[1]).c_data[0]

    # re-compute the eye transformation matrices from poses
    cdef libovr_math.Vector3f pos
    cdef libovr_math.Quatf ori
    cdef libovr_math.Vector3f up
    cdef libovr_math.Vector3f forward
    cdef libovr_math.Matrix4f rm

    cdef int eye = 0
    for eye in range(libovr_capi.ovrEye_Count):
        pos = <libovr_math.Vector3f>_eyeLayer.RenderPose[eye].Position
        ori = <libovr_math.Quatf>_eyeLayer.RenderPose[eye].Orientation

        if not ori.IsNormalized():  # make sure orientation is normalized
            ori.Normalize()

        rm = libovr_math.Matrix4f(ori)
        up = rm.Transform(libovr_math.Vector3f(0., 1., 0.))
        forward = rm.Transform(libovr_math.Vector3f(0., 0., -1.))
        _eyeViewMatrix[eye] = \
            libovr_math.Matrix4f.LookAtRH(pos, pos + forward, up)
        _eyeViewProjectionMatrix[eye] = \
            _eyeProjectionMatrix[eye] * _eyeViewMatrix[eye]

def getEyeProjectionMatrix(int eye, float nearClip=0.01, float farClip=1000.0):
    """Compute the projection matrix.

    The projection matrix is computed by the runtime using the eye FOV
    parameters set with '~libovr.LibOVRSession.setEyeRenderFov' calls.

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
    #global _nearClip
    #lobal _farClip

    #_nearClip = nearClip
    #_farClip = farClip

    _eyeProjectionMatrix[eye] = \
        <libovr_math.Matrix4f>libovr_capi.ovrMatrix4f_Projection(
            _eyeRenderDesc[eye].Fov,
            nearClip,
            farClip,
            libovr_capi.ovrProjection_ClipRangeOpenGL)

    cdef np.ndarray to_return = np.zeros((4, 4), dtype=np.float32)

    # fast copy matrix to numpy array
    cdef float [:, :] mv = to_return
    cdef Py_ssize_t i, j
    cdef Py_ssize_t N = 4
    i = j = 0
    for i in range(N):
        for j in range(N):
            mv[i, j] = _eyeProjectionMatrix[eye].M[i][j]

    return to_return

def getEyeRenderViewport(int eye):
    """Get the eye render viewport.

    The viewport defines the region on the swap texture a given eye's image is
    drawn to.

    Parameters
    ----------
    eye : int
        The eye index.

    Returns
    -------
    ndarray of ints
        Viewport rectangle [x, y, w, h].

    """
    global _eyeLayer
    cdef np.ndarray to_return = np.asarray(
        [_eyeLayer.Viewport[eye].Pos.x,
         _eyeLayer.Viewport[eye].Pos.y,
         _eyeLayer.Viewport[eye].Size.w,
         _eyeLayer.Viewport[eye].Size.h],
        dtype=np.int)

    return to_return

def setEyeRenderViewport(int eye, object values):
    """Set the eye render viewport.

    The viewport defines the region on the swap texture a given eye's image is
    drawn to.

    Parameters
    ----------
    eye : int
        The eye index.
    ndarray, list, or tuple of ints
        Viewport rectangle [x, y, w, h].

    Examples
    --------

    Setting the viewports for both eyes on a single swap chain buffer::

        # Calculate the optimal eye buffer sizes for the FOVs, these will define the
        # dimensions of the render target.
        leftBufferSize, rightBufferSize = libovr.calcEyeBufferSizes()
        # Define the viewports, which specifies the region on the render target a
        # eye's image will be drawn to and accessed from. Viewports are rectangles
        # defined like [x, y, w, h]. The x-position of the rightViewport is offset
        # by the width of the left viewport.
        leftViewport = [0, 0, leftBufferSize[0], leftBufferSize[1]]
        rightViewport = [leftBufferSize[0], 0, rightBufferSize[0], rightBufferSize[1]]
        # set both viewports
        libovr.setEyeRenderViewport(libovr.LIBOVR_EYE_LEFT, leftViewport)
        libovr.setEyeRenderViewport(libovr.LIBOVR_EYE_RIGHT, rightViewport)

    """
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

def getPredictedDisplayTime(unsigned int frameIndex=0):
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
    cdef double t_sec = libovr_capi.ovr_GetPredictedDisplayTime(
        _ptrSession,
        frameIndex)

    return t_sec

def timeInSeconds():
    """Absolute time in seconds.

    Returns
    -------
    float
        Time in seconds.

    """
    cdef double t_sec = libovr_capi.ovr_GetTimeInSeconds()

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

    cdef libovr_capi.ovrBool ret = libovr_capi.ovr_SetInt(
        _ptrSession, b"PerfHudMode", perfHudMode)

def hidePerfHud():
    """Hide the performance HUD.

    This is a convenience function that is equivalent to calling
    'perf_hud_mode('Off').

    """
    global _ptrSession
    cdef libovr_capi.ovrBool ret = libovr_capi.ovr_SetInt(
        _ptrSession, b"PerfHudMode", libovr_capi.ovrPerfHud_Off)

def perfHudModes():
    """List of valid performance HUD modes."""
    return [*_performance_hud_modes]

# def getEyeViewport(int eye):
#     """Get the viewport for a given eye.
#
#     Parameters
#     ----------
#     eye : int
#         Which eye to set the viewport, where left=0 and right=1.
#
#     """
#     global _eyeLayer
#     cdef libovr_capi.ovrRecti viewportRect = \
#         _eyeLayer.Viewport[eye]
#     cdef np.ndarray to_return = np.asarray(
#         [viewportRect.Pos.x,
#          viewportRect.Pos.y,
#          viewportRect.Size.w,
#          viewportRect.Size.h],
#         dtype=np.float32)
#
#     return to_return
#
# def setEyeViewport(int eye, object rect):
#     """Set the viewport for a given eye.
#
#     Parameters
#     ----------
#     eye : int
#         Which eye to set the viewport, where left=0 and right=1.
#     rect : ndarray, list or tuple of float
#         Rectangle specifying the viewport's position and dimensions on the
#         eye buffer.
#
#     """
#     global _eyeLayer
#     cdef libovr_capi.ovrRecti viewportRect
#     viewportRect.Pos.x = <int>rect[0]
#     viewportRect.Pos.y = <int>rect[1]
#     viewportRect.Size.w = <int>rect[2]
#     viewportRect.Size.h = <int>rect[3]
#
#     _eyeLayer.Viewport[eye] = viewportRect

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
        :data:`LIBOVR_SUCCESS` if completed without errors. May return
        :data:`LIBOVR_ERROR_DISPLAY_LOST` if the device was removed, rendering
        the current session invalid.

    """
    global _ptrSession
    cdef libovr_capi.ovrResult result = \
        libovr_capi.ovr_WaitToBeginFrame(_ptrSession, frameIndex)

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
    cdef libovr_capi.ovrResult result = \
        libovr_capi.ovr_BeginFrame(_ptrSession, frameIndex)

    return <int>result

def commitTextureSwapChain(int eye):
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

    Warning
    -------

    No additional drawing operations are permitted once the texture is committed
    until the SDK dereferences it, making it available again.

    """
    global _swapChains
    global _ptrSession
    cdef libovr_capi.ovrResult result = libovr_capi.ovr_CommitTextureSwapChain(
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
        :data:`LIBOVR_SUCCESS`, :data:`LIBOVR_SUCCESS_NOT_VISIBLE`,
        :data:`LIBOVR_SUCCESS_BOUNDARY_INVALID`,
        :data:`LIBOVR_SUCCESS_DEVICE_UNAVAILABLE`.

    Raises
    ------
    RuntimeError
        Raised if 'debugMode' is True and the API call to 'ovr_EndFrame'
        returns an error.

    """
    global _ptrSession
    global _eyeLayer

    cdef libovr_capi.ovrLayerHeader* layers = &_eyeLayer.Header
    cdef libovr_capi.ovrResult result = libovr_capi.ovr_EndFrame(
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
    cdef libovr_capi.ovrResult result = libovr_capi.ovr_ResetPerfStats(_ptrSession)

    return result

def getTrackingOriginType():
    """Tracking origin type.

    The tracking origin type specifies where the origin is placed when computing
    the pose of tracked objects (i.e. the head and touch controllers.) Valid
    values are 'floor' and 'eye'.

    """
    global _ptrSession
    cdef libovr_capi.ovrTrackingOrigin origin = \
        libovr_capi.ovr_GetTrackingOriginType(_ptrSession)

    if origin == libovr_capi.ovrTrackingOrigin_FloorLevel:
        return 'floor'
    elif origin == libovr_capi.ovrTrackingOrigin_EyeLevel:
        return 'eye'

def setTrackingOriginType(str value):
    cdef libovr_capi.ovrResult result
    global _ptrSession
    if value == 'floor':
        result = libovr_capi.ovr_SetTrackingOriginType(
            _ptrSession, libovr_capi.ovrTrackingOrigin_FloorLevel)
    elif value == 'eye':
        result = libovr_capi.ovr_SetTrackingOriginType(
            _ptrSession, libovr_capi.ovrTrackingOrigin_EyeLevel)

    return result

def recenterTrackingOrigin():
    """Recenter the tracking origin.

    Returns
    -------
    None

    """
    global _ptrSession
    cdef libovr_capi.ovrResult result = libovr_capi.ovr_RecenterTrackingOrigin(
        _ptrSession)

    return result

def getTrackerCount():
    """Get the number of attached trackers."""
    global _ptrSession
    cdef unsigned int trackerCount = libovr_capi.ovr_GetTrackerCount(
        _ptrSession)

    return <int>trackerCount

def getTrackerInfo(int trackerIndex):
    """Get information about a given tracker.

    Parameters
    ----------
    trackerIndex : int
        The index of the sensor to query. Valid values are between 0 and
        'getTrackerCount()'.

    """
    cdef LibOVRTrackerInfo to_return = LibOVRTrackerInfo()
    global _ptrSession

    # set the tracker index
    to_return._trackerIndex = <unsigned int>trackerIndex

    # set the descriptor data
    to_return.c_ovrTrackerDesc = libovr_capi.ovr_GetTrackerDesc(
        _ptrSession, <unsigned int>trackerIndex)
    # get the tracker pose
    to_return.c_ovrTrackerPose = libovr_capi.ovr_GetTrackerPose(
        _ptrSession, <unsigned int>trackerIndex)

    return to_return

def refreshPerformanceStats():
    """Refresh performance statistics.

    Should be called after 'endFrame'.

    """
    global _ptrSession
    global _frameStats
    cdef libovr_capi.ovrResult result = libovr_capi.ovr_GetPerfStats(
        _ptrSession,
        &_frameStats)

    # clear
    cdef list compFrameStats = list()

    cdef int statIdx = 0
    cdef int numStats = _frameStats.FrameStatsCount
    for statIdx in range(numStats):
        frameStat = LibOVRFrameStat()
        frameStat.c_data[0] = _frameStats.FrameStats[statIdx]
        compFrameStats.append(frameStat)

    return result

def updatePerfStats():
    """Update performance stats.

    Returns
    -------
    int
        Result of the 'ovr_GetPerfStats' LibOVR API call.

    """
    global _ptrSession
    global _frameStats
    global _lastFrameStats

    if _frameStats.FrameStatsCount > 0:
        if _frameStats.FrameStats[0].HmdVsyncIndex > 0:
            # copy last frame stats
            _lastFrameStats = _frameStats.FrameStats[0]

    cdef libovr_capi.ovrResult result = libovr_capi.ovr_GetPerfStats(
        _ptrSession, &_frameStats)

    return result

def getAdaptiveGpuPerformanceScale():
    """Get the adaptive GPU performance scale.

    Returns
    -------
    float

    """
    global _frameStats
    return _frameStats.AdaptiveGpuPerformanceScale

def getFrameStatsCount():
    """Get the number of queued compositor statistics.

    Returns
    -------
    int

    """
    global _frameStats
    return _frameStats.FrameStatsCount

def anyFrameStatsDropped():
    """Check if frame stats were dropped.

    This occurs when 'updatePerfStats' is called fewer than once every 5 frames.

    Returns
    -------
    bool
        True if frame statistics were dropped.

    """
    global _frameStats
    return <bint>_frameStats.AnyFrameStatsDropped

def checkAswIsAvailable():
    """Check if ASW is available.

    Returns
    -------
    bool

    """
    global _frameStats
    return <bint>_frameStats.AswIsAvailable

def getVisibleProcessId():
    """Process ID which the performance stats are currently being polled.

    Returns
    -------
    int
        Process ID.

    """
    global _frameStats
    return <int>_frameStats.VisibleProcessId

def checkAppLastFrameDropped():
    """Check if the application dropped a frame.

    Returns
    -------
    bool
        True if the application missed the HMD's flip deadline last frame.

    """
    global _lastFrameStats
    global _frameStats

    if _frameStats.FrameStatsCount > 0:
        if _frameStats.FrameStats[0].HmdVsyncIndex > 0:
            return _frameStats.FrameStats[0].AppDroppedFrameCount > \
                   _lastFrameStats.AppDroppedFrameCount

    return False

def checkCompLastFrameDropped():
    """Check if the compositor dropped a frame.

    Returns
    -------
    bool
        True if the compositor missed the HMD's flip deadline last frame.

    """
    global _lastFrameStats
    global _frameStats

    if _frameStats.FrameStatsCount > 0:
        if _frameStats.FrameStats[0].HmdVsyncIndex > 0:
            return _frameStats.FrameStats[0].CompositorDroppedFrameCount > \
                   _lastFrameStats.CompositorDroppedFrameCount

    return False

# def getFrameStats():
#     """Get a list of frame stats."""
#     global _perfStats
#
#     cdef list toReturn = list()
#     cdef LibOVRFramePerfStat stat
#     cdef Py_ssize_t N = <Py_ssize_t>_perfStats.FrameStatsCount
#     cdef Py_ssize_t i = 0
#     for i in range(N):
#         stat = LibOVRFramePerfStat()
#         stat.c_data[0] = _perfStats.FrameStats[i]
#         toReturn.append(stat)
#
#     return toReturn

def getFrameStats(int frameStatIndex=0):
    """Get detailed compositor frame statistics.

    Parameters
    ----------
    frameStatIndex : int (default 0)
        Frame statistics index to retrieve.

    Returns
    -------
    LibOVRFrameStat
        Frame statistics from the compositor.

    Notes
    -----
    If 'updatePerfStats' was called less than once per frame, more than one
    frame statistic will be available. Check 'getFrameStatsCount' for the number
    of queued stats and use an index >0 to access them.

    """
    global _frameStats

    if 0 > frameStatIndex >= _frameStats.FrameStatsCount:
        raise IndexError("Frame stats index out of range.")

    cdef LibOVRFrameStat stat = LibOVRFrameStat()
    stat.c_data[0] = _frameStats.FrameStats[0]

    return stat

def getLastErrorInfo():
    """Get the last error code and information string reported by the API.

    This function can be used when implementing custom error handlers.

    Returns
    -------
    tuple of int, str
        Tuple of the API call result and error string.

    """
    cdef libovr_capi.ovrErrorInfo lastErrorInfo  # store our last error here
    libovr_capi.ovr_GetLastErrorInfo(&lastErrorInfo)

    cdef libovr_capi.ovrResult result = lastErrorInfo.Result
    cdef str errorString = lastErrorInfo.ErrorString.decode("utf-8")

    return <int>result, errorString

def setBoundaryColor(float red, float green, float blue):
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

    cdef libovr_capi.ovrColorf color
    color.r = <float>red
    color.g = <float>green
    color.b = <float>blue

    _boundryStyle.Color = color

    cdef libovr_capi.ovrResult result = libovr_capi.ovr_SetBoundaryLookAndFeel(
        _ptrSession,
        &_boundryStyle)

    return result

def resetBoundaryColor():
    """Reset the boundary color to system default.

    """
    global _ptrSession
    cdef libovr_capi.ovrResult result = libovr_capi.ovr_ResetBoundaryLookAndFeel(
        _ptrSession)

    return result

def getBoundaryVisible():
    """Check if the Guardian boundary is visible.

    The boundary is drawn by the compositor which overlays the extents of
    the physical space where the user can safely move.

    """
    global _ptrSession
    cdef libovr_capi.ovrBool is_visible
    cdef libovr_capi.ovrResult result = libovr_capi.ovr_GetBoundaryVisible(
        _ptrSession, &is_visible)

    return result, is_visible

def showBoundary():
    """Show the boundary.

    The boundary is drawn by the compositor which overlays the extents of
    the physical space where the user can safely move.

    """
    global _ptrSession
    cdef libovr_capi.ovrResult result = libovr_capi.ovr_RequestBoundaryVisible(
        _ptrSession, libovr_capi.ovrTrue)

    return result

def hideBoundary():
    """Hide the boundry."""
    global _ptrSession
    cdef libovr_capi.ovrResult result = libovr_capi.ovr_RequestBoundaryVisible(
        _ptrSession, libovr_capi.ovrFalse)

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
    cdef libovr_capi.ovrBoundaryType btype
    if boundaryType == 'PlayArea':
        btype = libovr_capi.ovrBoundary_PlayArea
    elif boundaryType == 'Outer':
        btype = libovr_capi.ovrBoundary_Outer
    else:
        raise ValueError("Invalid boundary type specified.")

    cdef libovr_capi.ovrVector3f vec_out
    cdef libovr_capi.ovrResult result = libovr_capi.ovr_GetBoundaryDimensions(
            _ptrSession, btype, &vec_out)

    cdef np.ndarray[np.float32_t, ndim=1] to_return = np.asarray(
        (vec_out.x, vec_out.y, vec_out.z), dtype=np.float32)

    return result, to_return

#def getBoundaryPoints(str boundaryType='PlayArea'):
#    """Get the floor points which define the boundary."""
#    pass  # TODO: make this work.

def getConnectedControllerTypes():
    """Get connected controller types.

    Returns
    -------
    int
        Connected controller types ORed together.

    Examples
    --------

    Check if the Xbox gamepad is connected::

        connected = libovr.getConnectedControllerTypes()
        isConnected = (connected & libovr.LIBOVR_CONTROLLER_TYPE_XBOX) == \
            libovr.LIBOVR_CONTROLLER_TYPE_XBOX

    """
    global _ptrSession
    cdef unsigned int result = libovr_capi.ovr_GetConnectedControllerTypes(
        _ptrSession)

    return result

def updateInputState(int controller):
    """Refresh the input state of a controller.

    Parameters
    ----------
    controller : int
        Controller name. Valid values are:
            - :data:`LIBOVR_CONTROLLER_TYPE_XBOX` : XBox gamepad.
            - :data:`LIBOVR_CONTROLLER_TYPE_REMOTE` : Oculus Remote.
            - :data:`LIBOVR_CONTROLLER_TYPE_TOUCH` : Combined Touch controllers.
            - :data:`LIBOVR_CONTROLLER_TYPE_LTOUCH` : Left Touch controller.
            - :data:`LIBOVR_CONTROLLER_TYPE_RTOUCH` : Right Touch controller.

    """
    global _prevInputState
    global _inputStates
    global _ptrSession

    # get the controller index in the states array
    cdef int idx
    if controller == LIBOVR_CONTROLLER_TYPE_XBOX:
        idx = 0
    elif controller == LIBOVR_CONTROLLER_TYPE_REMOTE:
        idx = 1
    elif controller == LIBOVR_CONTROLLER_TYPE_TOUCH:
        idx = 2
    elif controller == LIBOVR_CONTROLLER_TYPE_LTOUCH:
        idx = 3
    elif controller == LIBOVR_CONTROLLER_TYPE_RTOUCH:
        idx = 4
    else:
        raise ValueError("Invalid controller type specified.")

    # pointer to the current and previous input state
    cdef libovr_capi.ovrInputState* previousInputState = \
        &_prevInputState[idx]
    cdef libovr_capi.ovrInputState* currentInputState = \
        &_inputStates[idx]

    # copy the current input state into the previous before updating
    previousInputState[0] = currentInputState[0]

    # get the current input state
    cdef libovr_capi.ovrResult result = libovr_capi.ovr_GetInputState(
        _ptrSession,
        <libovr_capi.ovrControllerType>controller,
        currentInputState)

    return result, currentInputState.TimeInSeconds

def getInputTime(int controller):
    """Get the time a controller was last polled.

    Parameters
    ----------
    controller : int
        Controller name. Valid values are:
            - :data:`LIBOVR_CONTROLLER_TYPE_XBOX` : XBox gamepad.
            - :data:`LIBOVR_CONTROLLER_TYPE_REMOTE` : Oculus Remote.
            - :data:`LIBOVR_CONTROLLER_TYPE_TOUCH` : Combined Touch controllers.
            - :data:`LIBOVR_CONTROLLER_TYPE_LTOUCH` : Left Touch controller.
            - :data:`LIBOVR_CONTROLLER_TYPE_RTOUCH` : Right Touch controller.

    Returns
    -------
    float
        The absolute time the controller was last polled.

    """
    global _prevInputState
    global _inputStates
    global _ptrSession

    # get the controller index in the states array
    cdef int idx
    if controller == LIBOVR_CONTROLLER_TYPE_XBOX:
        idx = 0
    elif controller == LIBOVR_CONTROLLER_TYPE_REMOTE:
        idx = 1
    elif controller == LIBOVR_CONTROLLER_TYPE_TOUCH:
        idx = 2
    elif controller == LIBOVR_CONTROLLER_TYPE_LTOUCH:
        idx = 3
    elif controller == LIBOVR_CONTROLLER_TYPE_RTOUCH:
        idx = 4
    else:
        raise ValueError("Invalid controller type specified.")

    # pointer to the current and previous input state
    cdef libovr_capi.ovrInputState* currentInputState = &_inputStates[idx]

    return currentInputState.TimeInSeconds

def getButton(int controller, int button, str testState='continuous'):
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
    controller : int
        Controller name. Valid values are:

        - :data:`LIBOVR_CONTROLLER_TYPE_XBOX` : XBox gamepad.
        - :data:`LIBOVR_CONTROLLER_TYPE_REMOTE` : Oculus Remote.
        - :data:`LIBOVR_CONTROLLER_TYPE_TOUCH` : Combined Touch controllers.
        - :data:`LIBOVR_CONTROLLER_TYPE_LTOUCH` : Left Touch controller.
        - :data:`LIBOVR_CONTROLLER_TYPE_RTOUCH` : Right Touch controller.

    button : int
        Button to check. Values can be ORed together to test for multiple button
        presses. If a given controller does not have a particular button, False
        will always be returned. Valid button values are:

        - :data:`LIBOVR_BUTTON_A`
        - :data:`LIBOVR_BUTTON_B`
        - :data:`LIBOVR_BUTTON_RTHUMB`
        - :data:`LIBOVR_BUTTON_RSHOULDER`
        - :data:`LIBOVR_BUTTON_X`
        - :data:`LIBOVR_BUTTON_Y`
        - :data:`LIBOVR_BUTTON_LTHUMB`
        - :data:`LIBOVR_BUTTON_LSHOULDER`
        - :data:`LIBOVR_BUTTON_UP`
        - :data:`LIBOVR_BUTTON_DOWN`
        - :data:`LIBOVR_BUTTON_LEFT`
        - :data:`LIBOVR_BUTTON_RIGHT`
        - :data:`LIBOVR_BUTTON_ENTER`
        - :data:`LIBOVR_BUTTON_BACK`
        - :data:`LIBOVR_BUTTON_VOLUP`
        - :data:`LIBOVR_BUTTON_VOLDOWN`
        - :data:`LIBOVR_BUTTON_HOME`
        - :data:`LIBOVR_BUTTON_PRIVATE`
        - :data:`LIBOVR_BUTTON_RMASK`
        - :data:`LIBOVR_BUTTON_LMASK`

    testState : str
        State to test buttons for. Valid states are 'rising', 'falling',
        'continuous', 'pressed', and 'released'.

    Returns
    -------
    tuple of bool and float
        Result of the button press and the time in seconds it was polled.

    Examples
    --------
    Check if the 'X' button on the touch controllers was pressed::

        isPressed = libovr.getButtons(libovr.LIBOVR_CONTROLLER_TYPE_TOUCH,
            libovr.LIBOVR_BUTTON_X, 'pressed')

    Test for multiple buttons ('X' and 'Y') released::

        buttons = libovr.LIBOVR_BUTTON_X | libovr.LIBOVR_BUTTON_Y
        controller = libovr.LIBOVR_CONTROLLER_TYPE_TOUCH
        isReleased = libovr.getButtons(controller, buttons, 'released')

    """
    global _prevInputState
    global _inputStates

    # get the controller index in the states array
    cdef int idx
    if controller == LIBOVR_CONTROLLER_TYPE_XBOX:
        idx = 0
    elif controller == LIBOVR_CONTROLLER_TYPE_REMOTE:
        idx = 1
    elif controller == LIBOVR_CONTROLLER_TYPE_TOUCH:
        idx = 2
    elif controller == LIBOVR_CONTROLLER_TYPE_LTOUCH:
        idx = 3
    elif controller == LIBOVR_CONTROLLER_TYPE_RTOUCH:
        idx = 4
    else:
        raise ValueError("Invalid controller type specified.")

    # get the time the controller was polled
    cdef double t_sec = _inputStates[idx].TimeInSeconds

    # pointer to the current and previous input state
    cdef unsigned int curButtons = _inputStates[idx].Buttons
    cdef unsigned int prvButtons = _prevInputState[idx].Buttons

    # test if the button was pressed
    cdef bint stateResult = False
    if testState == 'continuous':
        stateResult = (curButtons & button) == button
    elif testState == 'rising' or testState == 'pressed':
        # rising edge, will trigger once when pressed
        stateResult = (curButtons & button) == button and \
                      (prvButtons & button) != button
    elif testState == 'falling' or testState == 'released':
        # falling edge, will trigger once when released
        stateResult = (curButtons & button) != button and \
                      (prvButtons & button) == button
    else:
        raise ValueError("Invalid trigger mode specified.")

    return stateResult, t_sec

def getTouch(str controller, object touch, str testState='continuous'):
    """Get touches for a specified device.

    Touches reveal information about the user's hand pose, for instance,
    whether a pointing or pinching gesture is being made. Oculus Touch
    controllers are required for this functionality.

    Touch points to test are specified using their string names. Argument
    'touch_names' accepts a single string or a list. If a list is specified,
    the returned value will reflect whether all touches were triggered at
    the time the controller was polled last.

    Parameters
    ----------
    controller : int
        Controller name. Valid values are:

        - :data:`LIBOVR_CONTROLLER_TYPE_XBOX` : XBox gamepad.
        - :data:`LIBOVR_CONTROLLER_TYPE_REMOTE` : Oculus Remote.
        - :data:`LIBOVR_CONTROLLER_TYPE_TOUCH` : Combined Touch controllers.
        - :data:`LIBOVR_CONTROLLER_TYPE_LTOUCH` : Left Touch controller.
        - :data:`LIBOVR_CONTROLLER_TYPE_RTOUCH` : Right Touch controller.

    button : int
        Button to check. Values can be ORed together to test for multiple
        touches. If a given controller does not have a particular button, False
        will always be returned. Valid button values are:

        - :data:`LIBOVR_TOUCH_A`
        - :data:`LIBOVR_TOUCH_B`
        - :data:`LIBOVR_TOUCH_RTHUMB`
        - :data:`LIBOVR_TOUCH_RSHOULDER`
        - :data:`LIBOVR_TOUCH_X`
        - :data:`LIBOVR_TOUCH_Y`
        - :data:`LIBOVR_TOUCH_LTHUMB`
        - :data:`LIBOVR_TOUCH_LSHOULDER`
        - :data:`LIBOVR_TOUCH_LINDEXTRIGGER`
        - :data:`LIBOVR_TOUCH_LINDEXTRIGGER`
        - :data:`LIBOVR_TOUCH_RINDEXPOINTING`
        - :data:`LIBOVR_TOUCH_RTHUMBUP`
        - :data:`LIBOVR_TOUCH_LINDEXPOINTING`
        - :data:`LIBOVR_TOUCH_LTHUMBUP`

    testState : str
        State to test touches for. Valid states are 'rising', 'falling',
        'continuous', 'pressed', and 'released'.

    Returns
    -------
    tuple of bool and float
        Result of the touches and the time in seconds it was polled.

    Notes
    -----
    Not every controller type supports touch.

    """
    global _prevInputState
    global _inputStates

    # get the controller index in the states array
    cdef int idx
    if controller == LIBOVR_CONTROLLER_TYPE_XBOX:
        idx = 0
    elif controller == LIBOVR_CONTROLLER_TYPE_REMOTE:
        idx = 1
    elif controller == LIBOVR_CONTROLLER_TYPE_TOUCH:
        idx = 2
    elif controller == LIBOVR_CONTROLLER_TYPE_LTOUCH:
        idx = 3
    elif controller == LIBOVR_CONTROLLER_TYPE_RTOUCH:
        idx = 4
    else:
        raise ValueError("Invalid controller type specified.")

    # get the time the controller was polled
    cdef double t_sec = _inputStates[idx].TimeInSeconds

    # pointer to the current and previous input state
    cdef unsigned int curTouches = _inputStates[idx].Touches
    cdef unsigned int prvTouches = _prevInputState[idx].Touches

    # test if the button was pressed
    cdef bint stateResult = False
    if testState == 'continuous':
        stateResult = (curTouches & touch) == touch
    elif testState == 'rising' or testState == 'pressed':
        # rising edge, will trigger once when pressed
        stateResult = (curTouches & touch) == touch and \
                      (prvTouches & touch) != touch
    elif testState == 'falling' or testState == 'released':
        # falling edge, will trigger once when released
        stateResult = (curTouches & touch) != touch and \
                      (prvTouches & touch) == touch
    else:
        raise ValueError("Invalid trigger mode specified.")

    return stateResult, t_sec

def getThumbstickValues(int controller, bint deadzone=False):
    """Get analog thumbstick values.

    Get the values indicating the displacement of the controller's analog
    thumbsticks. Returns two tuples for the up-down and left-right of each
    stick. Values range from -1 to 1.

    Parameters
    ----------
    controller : int
        Controller name. Valid values are:

        - :data:`LIBOVR_CONTROLLER_TYPE_XBOX` : XBox gamepad.
        - :data:`LIBOVR_CONTROLLER_TYPE_REMOTE` : Oculus Remote.
        - :data:`LIBOVR_CONTROLLER_TYPE_TOUCH` : Combined Touch controllers.
        - :data:`LIBOVR_CONTROLLER_TYPE_LTOUCH` : Left Touch controller.
        - :data:`LIBOVR_CONTROLLER_TYPE_RTOUCH` : Right Touch controller.

    deadzone : bool
        Apply a deadzone if True.

    Returns
    -------
    tuple
        Thumbstick values.

    Examples
    --------
    Get the thumbstick values with deadzone for the touch controllers::

        ovr.updateInputState()  # get most recent input state
        leftThumbStick, rightThumbStick = ovr.getThumbstickValues(
            ovr.LIBOVR_CONTROLLER_TYPE_TOUCH, deadzone=True)
        x, y = rightThumbStick  # left-right, up-down values for right stick

    """
    global _inputStates

    # get the controller index in the states array
    cdef int idx
    if controller == LIBOVR_CONTROLLER_TYPE_XBOX:
        idx = 0
    elif controller == LIBOVR_CONTROLLER_TYPE_REMOTE:
        idx = 1
    elif controller == LIBOVR_CONTROLLER_TYPE_TOUCH:
        idx = 2
    elif controller == LIBOVR_CONTROLLER_TYPE_LTOUCH:
        idx = 3
    elif controller == LIBOVR_CONTROLLER_TYPE_RTOUCH:
        idx = 4
    else:
        raise ValueError("Invalid controller type specified.")

    # pointer to the current and previous input state
    cdef libovr_capi.ovrInputState* currentInputState = &_inputStates[idx]

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

def getIndexTriggerValues(int controller, bint deadzone=False):
    """Get analog index trigger values.

    Get the values indicating the displacement of the controller's analog
    thumbsticks. Returns two tuples for the up-down and left-right of each
    stick. Values range from -1 to 1.

    Parameters
    ----------
    controller : int
        Controller name. Valid values are:

        - :data:`LIBOVR_CONTROLLER_TYPE_XBOX` : XBox gamepad.
        - :data:`LIBOVR_CONTROLLER_TYPE_REMOTE` : Oculus Remote.
        - :data:`LIBOVR_CONTROLLER_TYPE_TOUCH` : Combined Touch controllers.
        - :data:`LIBOVR_CONTROLLER_TYPE_LTOUCH` : Left Touch controller.
        - :data:`LIBOVR_CONTROLLER_TYPE_RTOUCH` : Right Touch controller.

    Returns
    -------
    tuple
        Trigger values.

    """
    # convert the string to an index
    global _inputStates

    # get the controller index in the states array
    cdef int idx
    if controller == LIBOVR_CONTROLLER_TYPE_XBOX:
        idx = 0
    elif controller == LIBOVR_CONTROLLER_TYPE_REMOTE:
        idx = 1
    elif controller == LIBOVR_CONTROLLER_TYPE_TOUCH:
        idx = 2
    elif controller == LIBOVR_CONTROLLER_TYPE_LTOUCH:
        idx = 3
    elif controller == LIBOVR_CONTROLLER_TYPE_RTOUCH:
        idx = 4
    else:
        raise ValueError("Invalid controller type specified.")

    # pointer to the current and previous input state
    cdef libovr_capi.ovrInputState* currentInputState = &_inputStates[idx]

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
    global _inputStates

    # get the controller index in the states array
    cdef int idx
    if controller == LIBOVR_CONTROLLER_TYPE_XBOX:
        idx = 0
    elif controller == LIBOVR_CONTROLLER_TYPE_REMOTE:
        idx = 1
    elif controller == LIBOVR_CONTROLLER_TYPE_TOUCH:
        idx = 2
    elif controller == LIBOVR_CONTROLLER_TYPE_LTOUCH:
        idx = 3
    elif controller == LIBOVR_CONTROLLER_TYPE_RTOUCH:
        idx = 4
    else:
        raise ValueError("Invalid controller type specified.")

    # pointer to the current and previous input state
    cdef libovr_capi.ovrInputState* currentInputState = &_inputStates[idx]

    cdef float indexTriggerLeft = 0.0
    cdef float indexTriggerRight = 0.0

    if deadzone:
        indexTriggerLeft = currentInputState[0].HandTrigger[0]
        indexTriggerRight = currentInputState[0].HandTrigger[1]
    else:
        indexTriggerLeft = currentInputState[0].HandTriggerNoDeadzone[0]
        indexTriggerRight = currentInputState[0].HandTriggerNoDeadzone[1]

    return indexTriggerLeft, indexTriggerRight

def setControllerVibration(int controller, str frequency, float amplitude):
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
    controller : int
        Controller name. Valid values are:
        - :data:`LIBOVR_CONTROLLER_TYPE_XBOX` : XBox gamepad.
        - :data:`LIBOVR_CONTROLLER_TYPE_REMOTE` : Oculus Remote.
        - :data:`LIBOVR_CONTROLLER_TYPE_TOUCH` : Combined Touch controllers.
        - :data:`LIBOVR_CONTROLLER_TYPE_LTOUCH` : Left Touch controller.
        - :data:`LIBOVR_CONTROLLER_TYPE_RTOUCH` : Right Touch controller.
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

    cdef libovr_capi.ovrResult result = libovr_capi.ovr_SetControllerVibration(
        _ptrSession,
        <libovr_capi.ovrControllerType>controller,
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
    cdef libovr_capi.ovrResult result = libovr_capi.ovr_GetSessionStatus(
        _ptrSession, to_return.c_data)

    return to_return

# def testPointsInEyeFrustums(object points, object out=None):
#     """Test if points are within each eye's frustum.
#
#     This function uses current view and projection matrix in the computation.
#
#     Parameters
#     ----------
#     points : tuple, list, or ndarray
#         2D array of points to test. Each coordinate should be in format
#         [x, y ,z], where dimensions are in meters. Passing a NumPy ndarray with
#         dtype=float32 and ndim=2 will avoid copying.
#
#     out : ndarray
#         Optional array to write test results. Must have the same shape as
#         'points' and dtype=bool. If None, the function will create and return an
#         appropriate array with results.
#
#     Returns
#     -------
#     ndarray
#         Nx2 array of results. The row index of the returned array contains the
#         test results for the coordinates with the matching row index in
#         'points'. The results for the left and right eye are stored in the first
#         and second column, respectively.
#
#     """
#     global _nearClip
#     global _farClip
#
#     cdef np.ndarray[np.float32_t, ndim=2] pointsIn = \
#         np.array(points, dtype=np.float32, ndmin=2, copy=False)
#
#     cdef np.ndarray[np.uint8_t, ndim=2] testOut
#     if testOut is not None:
#         testOut = out
#     else:
#         testOut = np.zeros_like(out, dtype=np.uint8_t)
#
#     assert testOut.shape == pointsIn.shape
#
#     cdef float[:,:] mvPoints = pointsIn
#     cdef np.uint8_t[:,:] mvResult = testOut
#
#     # intermediates
#     cdef libovr_math.Vector4f vecIn
#     cdef libovr_math.Vector4f pointHCS
#     cdef libovr_math.Vector3f pointNDC
#
#     # loop over all points specified
#     cdef Py_ssize_t eye = 0
#     cdef Py_ssize_t pt = 0
#     cdef Py_ssize_t N = mvPoints.shape[0]
#     for pt in range(N):
#         for eye in range(libovr_capi.ovrEye_Count):
#             vecIn.x = mvPoints[pt, 0]
#             vecIn.y = mvPoints[pt, 1]
#             vecIn.z = mvPoints[pt, 2]
#             vecIn.w = 1.0
#             pointHCS = _eyeViewProjectionMatrix[eye].Transform(vecIn)
#
#             # too close to the singularity for perspective division or behind
#             # the viewer, fail automatically
#             if pointHCS.w < 0.0001:
#                 continue
#
#             # perspective division XYZ / W
#             pointNDC.x = pointHCS.x / pointHCS.w
#             pointNDC.y = pointHCS.y / pointHCS.w
#             pointNDC.z = pointHCS.z / pointHCS.w
#
#             # check if outside [-1:1] in any NDC dimension
#             if -1.0 < pointNDC.x < 1.0 and -1.0 < pointNDC.y < 1.0 and -1.0 < pointNDC.z < 1.0:
#                 mvResult[pt, eye] = 1
#
#     return testOut.astype(dtype=np.bool)
#
# def anyPointInFrustum(object points):
#     """Check if any of the specified points in world/scene coordinates are
#     within the viewing frustum of either eye.
#
#     This can be used to determine whether or not something should be drawn by
#     specifying its position, mesh or bounding box vertices. The function will
#     return True immediately when it comes across a point that falls within
#     either eye's frustum.
#
#     Parameters
#     ----------
#     points : tuple, list, or ndarray
#         2D array of points to test. Each coordinate should be in format
#         [x, y ,z], where dimensions are in meters. Passing a NumPy ndarray with
#         dtype=float32 and ndim=2 will avoid copying.
#
#     Returns
#     -------
#     bool
#         True if any point specified falls inside a viewing frustum.
#
#     Examples
#     --------
#     Test if points fall within a viewing frustum::
#
#         points = [[1.2, -0.2, -5.6], [-0.01, 0.0, -10.0]]
#         isVisible = libovr.testPointsInFrustum(points)
#
#     """
#     # eventually we're going to move this function if we decide to support more
#     # HMDs. This really isn't something specific to LibOVR.
#
#     # input values to 2D memory view
#     cdef np.ndarray[np.float32_t, ndim=2] pointsIn = \
#         np.array(points, dtype=np.float32, ndmin=2, copy=False)
#
#     if pointsIn.shape[1] != 3:
#         raise ValueError("Invalid number of columns, must be 3.")
#
#     cdef float[:,:] mvPoints = pointsIn
#
#     # intermediates
#     cdef libovr_math.Vector4f vecIn
#     cdef libovr_math.Vector4f pointHCS
#     cdef libovr_math.Vector3f pointNDC
#
#     # loop over all points specified
#     cdef Py_ssize_t eye = 0
#     cdef Py_ssize_t pt = 0
#     cdef Py_ssize_t N = mvPoints.shape[0]
#     for pt in range(N):
#         for eye in range(libovr_capi.ovrEye_Count):
#             vecIn.x = mvPoints[pt, 0]
#             vecIn.y = mvPoints[pt, 1]
#             vecIn.z = mvPoints[pt, 2]
#             vecIn.w = 1.0
#             pointHCS = _eyeViewProjectionMatrix[eye].Transform(vecIn)
#
#             # too close to the singularity for perspective division or behind
#             # the viewer, fail automatically
#             if pointHCS.w < 0.0001:
#                 return False
#
#             # perspective division XYZ / W
#             pointNDC.x = pointHCS.x / pointHCS.w
#             pointNDC.y = pointHCS.y / pointHCS.w
#             pointNDC.z = pointHCS.z / pointHCS.w
#
#             # check if outside [-1:1] in any NDC dimension
#             if -1.0 < pointNDC.x < 1.0 and -1.0 < pointNDC.y < 1.0 and -1.0 < pointNDC.z < 1.0:
#                 return True
#
#     return False