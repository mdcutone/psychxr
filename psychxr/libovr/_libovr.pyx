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
# ------------------------------------------------------------------------------
# Module information
#
__author__ = "Matthew D. Cutone"
__credits__ = ["Laurie M. Wilcox"]
__copyright__ = "Copyright 2019 Matthew D. Cutone"
__license__ = "MIT"
__version__ = "0.2.0"
__status__ = "Beta"
__maintainer__ = "Matthew D. Cutone"
__email__ = "cutonem@yorku.ca"

# ------------------------------------------------------------------------------
# Exported objects
#
__all__ = [
    'SUCCESS',
    'SUCCESS_NOT_VISIBLE',
    'SUCCESS_DEVICE_UNAVAILABLE',
    'SUCCESS_BOUNDARY_INVALID',
    'ERROR_MEMORY_ALLOCATION_FAILURE',
    'ERROR_INVALID_SESSION',
    'ERROR_TIMEOUT',
    'ERROR_NOT_INITIALIZED',
    'ERROR_INVALID_PARAMETER',
    'ERROR_SERVICE_ERROR',
    'ERROR_NO_HMD',
    'ERROR_UNSUPPORTED',
    'ERROR_DEVICE_UNAVAILABLE',
    'ERROR_INVALID_HEADSET_ORIENTATION',
    'ERROR_CLIENT_SKIPPED_DESTROY',
    'ERROR_CLIENT_SKIPPED_SHUTDOWN',
    'ERROR_SERVICE_DEADLOCK_DETECTED',
    'ERROR_INSUFFICENT_ARRAY_SIZE',
    'ERROR_NO_EXTERNAL_CAMERA_INFO',
    'ERROR_LOST_TRACKING',
    'ERROR_EXTERNAL_CAMERA_INITIALIZED_FAILED',
    'ERROR_EXTERNAL_CAMERA_CAPTURE_FAILED',
    'ERROR_EXTERNAL_CAMERA_NAME_LISTS_BUFFER_SIZE',
    'ERROR_EXTERNAL_CAMERA_NAME_LISTS_MISMATCH',
    'ERROR_EXTERNAL_CAMERA_NOT_CALIBRATED',
    'ERROR_EXTERNAL_CAMERA_NAME_WRONG_SIZE',
    'ERROR_AUDIO_DEVICE_NOT_FOUND',
    'ERROR_AUDIO_COM_ERROR',
    'ERROR_INITIALIZE',
    'ERROR_LIB_LOAD',
    'ERROR_SERVICE_CONNECTION',
    'ERROR_SERVICE_VERSION',
    'ERROR_INCOMPATIBLE_OS',
    'ERROR_DISPLAY_INIT',
    'ERROR_SERVER_START',
    'ERROR_REINITIALIZATION',
    'ERROR_MISMATCHED_ADAPTERS',
    'ERROR_LEAKING_RESOURCES',
    'ERROR_CLIENT_VERSION',
    'ERROR_OUT_OF_DATE_OS',
    'ERROR_OUT_OF_DATE_GFX_DRIVER',
    'ERROR_INCOMPATIBLE_OS',
    'ERROR_NO_VALID_VR_DISPLAY_SYSTEM',
    'ERROR_OBSOLETE',
    'ERROR_DISABLED_OR_DEFAULT_ADAPTER',
    'ERROR_HYBRID_GRAPHICS_NOT_SUPPORTED',
    'ERROR_DISPLAY_MANAGER_INIT',
    'ERROR_TRACKER_DRIVER_INIT',
    'ERROR_LIB_SIGN_CHECK',
    'ERROR_LIB_PATH',
    'ERROR_LIB_SYMBOLS',
    'ERROR_REMOTE_SESSION',
    'ERROR_INITIALIZE_VULKAN',
    'ERROR_BLACKLISTED_GFX_DRIVER',
    'ERROR_DISPLAY_LOST',
    'ERROR_TEXTURE_SWAP_CHAIN_FULL',
    'ERROR_TEXTURE_SWAP_CHAIN_INVALID',
    'ERROR_GRAPHICS_DEVICE_RESET',
    'ERROR_DISPLAY_REMOVED',
    'ERROR_CONTENT_PROTECTION_NOT_AVAILABLE',
    'ERROR_APPLICATION_VISIBLE',
    'ERROR_DISALLOWED',
    'ERROR_DISPLAY_PLUGGED_INCORRECTY',
    'ERROR_DISPLAY_LIMIT_REACHED',
    'ERROR_RUNTIME_EXCEPTION',
    'ERROR_NO_CALIBRATION',
    'ERROR_OLD_VERSION',
    'ERROR_MISFORMATTED_BLOCK',
    'EYE_LEFT',
    'EYE_RIGHT',
    'EYE_COUNT',
    'HAND_LEFT',
    'HAND_RIGHT',
    'HAND_COUNT',
    'KEY_USER',
    'KEY_NAME',
    'KEY_GENDER',
    'DEFAULT_GENDER',
    'KEY_PLAYER_HEIGHT',
    'DEFAULT_PLAYER_HEIGHT',
    'KEY_EYE_HEIGHT',
    'DEFAULT_EYE_HEIGHT',
    'KEY_NECK_TO_EYE_DISTANCE',
    'DEFAULT_NECK_TO_EYE_HORIZONTAL',
    'DEFAULT_NECK_TO_EYE_VERTICAL',
    'KEY_EYE_TO_NOSE_DISTANCE',
    'DEBUG_HUD_STEREO_MODE',
    'DEBUG_HUD_STEREO_GUIDE_INFO_ENABLE',
    'DEBUG_HUD_STEREO_GUIDE_SIZE',
    'DEBUG_HUD_STEREO_GUIDE_POSITION',
    'DEBUG_HUD_STEREO_GUIDE_YAWPITCHROLL',
    'DEBUG_HUD_STEREO_GUIDE_COLOR',
    'CONTROLLER_TYPE_XBOX',
    'CONTROLLER_TYPE_REMOTE',
    'CONTROLLER_TYPE_TOUCH',
    'CONTROLLER_TYPE_LTOUCH',
    'CONTROLLER_TYPE_RTOUCH',
    'CONTROLLER_TYPE_OBJECT0',
    'CONTROLLER_TYPE_OBJECT1',
    'CONTROLLER_TYPE_OBJECT2',
    'CONTROLLER_TYPE_OBJECT3',
    'BUTTON_A',
    'BUTTON_B',
    'BUTTON_RTHUMB',
    'BUTTON_RSHOULDER',
    'BUTTON_X',
    'BUTTON_Y',
    'BUTTON_LTHUMB',
    'BUTTON_LSHOULDER',
    'BUTTON_UP',
    'BUTTON_DOWN',
    'BUTTON_LEFT',
    'BUTTON_RIGHT',
    'BUTTON_ENTER',
    'BUTTON_BACK',
    'BUTTON_VOLUP',
    'BUTTON_VOLDOWN',
    'BUTTON_HOME',
    'BUTTON_PRIVATE',
    'BUTTON_RMASK',
    'BUTTON_LMASK',
    'TEXTURE_SWAP_CHAIN0',
    'TEXTURE_SWAP_CHAIN1',
    'TEXTURE_SWAP_CHAIN2',
    'TEXTURE_SWAP_CHAIN3',
    'TEXTURE_SWAP_CHAIN4',
    'TEXTURE_SWAP_CHAIN5',
    'TEXTURE_SWAP_CHAIN6',
    'TEXTURE_SWAP_CHAIN7',
    'TEXTURE_SWAP_CHAIN8',
    'TEXTURE_SWAP_CHAIN9',
    'TEXTURE_SWAP_CHAIN10',
    'TEXTURE_SWAP_CHAIN11',
    'TEXTURE_SWAP_CHAIN12',
    'TEXTURE_SWAP_CHAIN13',
    'TEXTURE_SWAP_CHAIN14',
    'TEXTURE_SWAP_CHAIN15',
    'FORMAT_R8G8B8A8_UNORM',
    'FORMAT_R8G8B8A8_UNORM_SRGB',
    'FORMAT_R16G16B16A16_FLOAT',
    'FORMAT_R11G11B10_FLOAT',
    'FORMAT_D16_UNORM',
    'FORMAT_D24_UNORM_S8_UINT',
    'FORMAT_D32_FLOAT',
    'MAX_PROVIDED_FRAME_STATS',
    'TRACKED_DEVICE_TYPE_HMD',
    'TRACKED_DEVICE_TYPE_LTOUCH',
    'TRACKED_DEVICE_TYPE_RTOUCH',
    'TRACKED_DEVICE_TYPE_TOUCH',
    'TRACKED_DEVICE_TYPE_OBJECT0',
    'TRACKED_DEVICE_TYPE_OBJECT1',
    'TRACKED_DEVICE_TYPE_OBJECT2',
    'TRACKED_DEVICE_TYPE_OBJECT3',
    'TRACKING_ORIGIN_EYE_LEVEL',
    'TRACKING_ORIGIN_FLOOR_LEVEL',
    'PRODUCT_VERSION',
    'MAJOR_VERSION',
    'MINOR_VERSION',
    'PATCH_VERSION',
    'BUILD_NUMBER',
    'DLL_COMPATIBLE_VERSION',
    'MIN_REQUESTABLE_MINOR_VERSION',
    'FEATURE_VERSION',
    'STATUS_ORIENTATION_TRACKED',
    'STATUS_POSITION_TRACKED',
    'PERF_HUD_MODE',
    'PERF_HUD_OFF',
    'PERF_HUD_PERF_SUMMARY',
    'PERF_HUD_LATENCY_TIMING',
    'PERF_HUD_APP_RENDER_TIMING',
    'PERF_HUD_COMP_RENDER_TIMING',
    'PERF_HUD_ASW_STATS',
    'PERF_HUD_VERSION_INFO',
    'DEBUG_HUD_STEREO_MODE_OFF',
    'DEBUG_HUD_STEREO_MODE_QUAD',
    'DEBUG_HUD_STEREO_MODE_QUAD_WITH_CROSSHAIR',
    'DEBUG_HUD_STEREO_MODE_CROSSHAIR_AT_INFINITY',
    'BOUNDARY_PLAY_AREA',
    'BOUNDARY_OUTER',
    'LAYER_FLAG_HIGH_QUALITY',
    'LAYER_FLAG_TEXTURE_ORIGIN_AT_BOTTOM_LEFT',
    'LAYER_FLAG_HEAD_LOCKED',
    'LibOVRPose',
    'LibOVRPoseState',
    'LibOVRTrackerInfo',
    'LibOVRHmdInfo',
    'LibOVRSessionStatus',
    'success',
    'unqualifiedSuccess',
    'failure',
    'setBool',
    'getBool',
    'setInt',
    'getInt',
    'setFloat',
    'getFloat',
    'getFloatArray',
    'setFloatArray',
    'setString',
    'getString',
    'isOculusServiceRunning',
    'isHmdConnected',
    'getHmdInfo',
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
    'getTanAngleToRenderTargetNDC',
    'getDistortedViewport',
    'getEyeRenderFov',
    'setEyeRenderFov',
    'getEyeAspectRatio',
    'getEyeHorizontalFovRadians',
    'getEyeVerticalFovRadians',
    'getEyeFocalLength',
    'calcEyeBufferSize',
    'getTextureSwapChainLengthGL',
    'getTextureSwapChainCurrentIndex',
    'getTextureSwapChainBufferGL',
    'createTextureSwapChainGL',
    'setEyeColorTextureSwapChain',
    'createMirrorTexture',
    'getMirrorTexture',
    'getTrackingState',
    'getDevicePoses',
    'calcEyePoses',
    'getHmdToEyePose',
    'setHmdToEyePose',
    'getEyeRenderPose',
    'setEyeRenderPose',
    'getEyeHorizontalFovRadians',
    'getEyeVerticalFovRadians',
    'getEyeFocalLength',
    'getEyeProjectionMatrix',
    'getEyeRenderViewport',
    'setEyeRenderViewport',
    'getEyeViewMatrix',
    'getPredictedDisplayTime',
    'timeInSeconds',
    'waitToBeginFrame',
    'beginFrame',
    'commitTextureSwapChain',
    'endFrame',
    'resetFrameStats',
    'getTrackingOriginType',
    'setTrackingOriginType',
    'recenterTrackingOrigin',
    'specifyTrackingOrigin',
    'clearShouldRecenterFlag',
    'getTrackerCount',
    'getTrackerInfo',
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
    'getButton',
    'getTouch',
    'getThumbstickValues',
    'getIndexTriggerValues',
    'getHandTriggerValues',
    'setControllerVibration',
    'getSessionStatus',
    'getLayerEyeFovFlags',
    'setLayerEyeFovFlags'
    #'anyPointInFrustum'
]


from .cimport libovr_capi as capi
from .cimport libovr_math

from libc.stdint cimport int32_t, uint32_t
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.math cimport pow, tan

cimport numpy as np
import numpy as np
np.import_array()

import collections
import warnings

# ------------------------------------------------------------------------------
# Initialize module
#
cdef capi.ovrInitParams _initParams  # initialization parameters
cdef capi.ovrSession _ptrSession  # session pointer
cdef capi.ovrGraphicsLuid _gfxLuid  # LUID
cdef capi.ovrHmdDesc _hmdDesc  # HMD information descriptor
cdef capi.ovrBoundaryLookAndFeel _boundryStyle
cdef capi.ovrTextureSwapChain[16] _swapChains
cdef capi.ovrMirrorTexture _mirrorTexture

# VR related data persistent across frames
cdef capi.ovrLayerEyeFov _eyeLayer
cdef capi.ovrEyeRenderDesc[2] _eyeRenderDesc
cdef capi.ovrTrackingState _trackingState
cdef capi.ovrViewScaleDesc _viewScale

# prepare the render layer
_eyeLayer.Header.Type = capi.ovrLayerType_EyeFov
_eyeLayer.Header.Flags = \
    capi.ovrLayerFlag_TextureOriginAtBottomLeft | \
    capi.ovrLayerFlag_HighQuality
_eyeLayer.ColorTexture[0] = _eyeLayer.ColorTexture[1] = NULL

# status and performance information
cdef capi.ovrSessionStatus _sessionStatus
cdef capi.ovrPerfStats _frameStats
cdef capi.ovrPerfStatsPerCompositorFrame _lastFrameStats
cdef list compFrameStats

# error information
cdef capi.ovrErrorInfo _errorInfo  # store our last error here

# controller states
cdef capi.ovrInputState[9] _inputStates
cdef capi.ovrInputState[9] _prevInputState

# debug mode
cdef bint _debugMode

# geometric data
#cdef float[2] _nearClip
#cdef float[2] _farClip
cdef libovr_math.Matrix4f[2] _eyeProjectionMatrix
cdef libovr_math.Matrix4f[2] _eyeViewMatrix
cdef libovr_math.Matrix4f[2] _eyeViewProjectionMatrix

# Function to check for errors returned by OVRLib functions
#
cdef capi.ovrErrorInfo _last_error_info_  # store our last error here
def check_result(result):
    if capi.OVR_FAILURE(result):
        capi.ovr_GetLastErrorInfo(&_last_error_info_)
        raise RuntimeError(
            str(result) + ": " + _last_error_info_.ErrorString.decode("utf-8"))

# helper functions
cdef float maxf(float a, float b):
    return a if a >= b else b

# ------------------------------------------------------------------------------
# Version checker

if capi.OVR_MAJOR_VERSION != 1 or capi.OVR_MINOR_VERSION != 38:
    # raise a warning if the version of the Oculus SDK may be incompatible
    warnings.warn(
        "PsychXR was built using version {major}.{minor} of the Oculus PC SDK "
        "however 1.38 is recommended. This might be perfectly fine if there "
        "aren't any API breaking changes.".format(
            major=capi.OVR_MAJOR_VERSION, minor=capi.OVR_MINOR_VERSION),
        RuntimeWarning)

# ------------------------------------------------------------------------------
# Constants

# button types
BUTTON_A = capi.ovrButton_A
BUTTON_B = capi.ovrButton_B
BUTTON_RTHUMB = capi.ovrButton_RThumb
BUTTON_RSHOULDER = capi.ovrButton_RShoulder
BUTTON_X = capi.ovrButton_X
BUTTON_Y = capi.ovrButton_Y
BUTTON_LTHUMB = capi.ovrButton_LThumb
BUTTON_LSHOULDER = capi.ovrButton_LShoulder
BUTTON_UP = capi.ovrButton_Up
BUTTON_DOWN = capi.ovrButton_Down
BUTTON_LEFT = capi.ovrButton_Left
BUTTON_RIGHT = capi.ovrButton_Right
BUTTON_ENTER = capi.ovrButton_Enter
BUTTON_BACK = capi.ovrButton_Back
BUTTON_VOLUP = capi.ovrButton_VolUp
BUTTON_VOLDOWN = capi.ovrButton_VolDown
BUTTON_HOME = capi.ovrButton_Home
BUTTON_PRIVATE = capi.ovrButton_Private
BUTTON_RMASK = capi.ovrButton_RMask
BUTTON_LMASK = capi.ovrButton_LMask

# touch types
TOUCH_A = capi.ovrTouch_A
TOUCH_B = capi.ovrTouch_B
TOUCH_RTHUMB = capi.ovrTouch_RThumb
TOUCH_RTHUMBREST = capi.ovrTouch_RThumbRest
TOUCH_X = capi.ovrTouch_X
TOUCH_Y = capi.ovrTouch_Y
TOUCH_LTHUMB = capi.ovrTouch_LThumb
TOUCH_LTHUMBREST = capi.ovrTouch_LThumbRest
TOUCH_LINDEXTRIGGER = capi.ovrTouch_LIndexTrigger
TOUCH_RINDEXPOINTING = capi.ovrTouch_RIndexPointing
TOUCH_RTHUMBUP = capi.ovrTouch_RThumbUp
TOUCH_LINDEXPOINTING = capi.ovrTouch_LIndexPointing
TOUCH_LTHUMBUP = capi.ovrTouch_LThumbUp

# controller types
CONTROLLER_TYPE_NONE = capi.ovrControllerType_None
CONTROLLER_TYPE_XBOX = capi.ovrControllerType_XBox
CONTROLLER_TYPE_REMOTE = capi.ovrControllerType_Remote
CONTROLLER_TYPE_TOUCH = capi.ovrControllerType_Touch
CONTROLLER_TYPE_LTOUCH = capi.ovrControllerType_LTouch
CONTROLLER_TYPE_RTOUCH = capi.ovrControllerType_RTouch
CONTROLLER_TYPE_OBJECT0 = capi.ovrControllerType_Object0
CONTROLLER_TYPE_OBJECT1 = capi.ovrControllerType_Object1
CONTROLLER_TYPE_OBJECT2 = capi.ovrControllerType_Object2
CONTROLLER_TYPE_OBJECT3 = capi.ovrControllerType_Object3

# return success codes, values other than 'SUCCESS' are conditional
SUCCESS = capi.ovrSuccess
SUCCESS_NOT_VISIBLE = capi.ovrSuccess_NotVisible
SUCCESS_DEVICE_UNAVAILABLE = capi.ovrSuccess_DeviceUnavailable
SUCCESS_BOUNDARY_INVALID = capi.ovrSuccess_BoundaryInvalid

# return error code, not all of these are applicable
ERROR_MEMORY_ALLOCATION_FAILURE = capi.ovrError_MemoryAllocationFailure
ERROR_INVALID_SESSION = capi.ovrError_InvalidSession
ERROR_TIMEOUT = capi.ovrError_Timeout
ERROR_NOT_INITIALIZED = capi.ovrError_NotInitialized
ERROR_INVALID_PARAMETER = capi.ovrError_InvalidParameter
ERROR_SERVICE_ERROR = capi.ovrError_ServiceError
ERROR_NO_HMD = capi.ovrError_NoHmd
ERROR_UNSUPPORTED = capi.ovrError_Unsupported
ERROR_DEVICE_UNAVAILABLE = capi.ovrError_DeviceUnavailable
ERROR_INVALID_HEADSET_ORIENTATION = capi.ovrError_InvalidHeadsetOrientation
ERROR_CLIENT_SKIPPED_DESTROY = capi.ovrError_ClientSkippedDestroy
ERROR_CLIENT_SKIPPED_SHUTDOWN = capi.ovrError_ClientSkippedShutdown
ERROR_SERVICE_DEADLOCK_DETECTED = capi.ovrError_ServiceDeadlockDetected
ERROR_INVALID_OPERATION = capi.ovrError_InvalidOperation
ERROR_INSUFFICENT_ARRAY_SIZE = capi.ovrError_InsufficientArraySize
ERROR_NO_EXTERNAL_CAMERA_INFO = capi.ovrError_NoExternalCameraInfo
ERROR_LOST_TRACKING = capi.ovrError_LostTracking
ERROR_EXTERNAL_CAMERA_INITIALIZED_FAILED = capi.ovrError_ExternalCameraInitializedFailed
ERROR_EXTERNAL_CAMERA_CAPTURE_FAILED = capi.ovrError_ExternalCameraCaptureFailed
ERROR_EXTERNAL_CAMERA_NAME_LISTS_BUFFER_SIZE = capi.ovrError_ExternalCameraNameListsBufferSize
ERROR_EXTERNAL_CAMERA_NAME_LISTS_MISMATCH = capi.ovrError_ExternalCameraNameListsMistmatch
ERROR_EXTERNAL_CAMERA_NOT_CALIBRATED = capi.ovrError_ExternalCameraNotCalibrated
ERROR_EXTERNAL_CAMERA_NAME_WRONG_SIZE = capi.ovrError_ExternalCameraNameWrongSize
ERROR_AUDIO_DEVICE_NOT_FOUND = capi.ovrError_AudioDeviceNotFound
ERROR_AUDIO_COM_ERROR = capi.ovrError_AudioComError
ERROR_INITIALIZE = capi.ovrError_Initialize
ERROR_LIB_LOAD = capi.ovrError_LibLoad
ERROR_SERVICE_CONNECTION = capi.ovrError_ServiceConnection
ERROR_SERVICE_VERSION = capi.ovrError_ServiceVersion
ERROR_INCOMPATIBLE_OS = capi.ovrError_IncompatibleOS
ERROR_DISPLAY_INIT = capi.ovrError_DisplayInit
ERROR_SERVER_START = capi.ovrError_ServerStart
ERROR_REINITIALIZATION = capi.ovrError_Reinitialization
ERROR_MISMATCHED_ADAPTERS = capi.ovrError_MismatchedAdapters
ERROR_LEAKING_RESOURCES = capi.ovrError_LeakingResources
ERROR_CLIENT_VERSION = capi.ovrError_ClientVersion
ERROR_OUT_OF_DATE_OS = capi.ovrError_OutOfDateOS
ERROR_OUT_OF_DATE_GFX_DRIVER = capi.ovrError_OutOfDateGfxDriver
ERROR_INCOMPATIBLE_OS = capi.ovrError_IncompatibleGPU
ERROR_NO_VALID_VR_DISPLAY_SYSTEM = capi.ovrError_NoValidVRDisplaySystem
ERROR_OBSOLETE = capi.ovrError_Obsolete
ERROR_DISABLED_OR_DEFAULT_ADAPTER = capi.ovrError_DisabledOrDefaultAdapter
ERROR_HYBRID_GRAPHICS_NOT_SUPPORTED = capi.ovrError_HybridGraphicsNotSupported
ERROR_DISPLAY_MANAGER_INIT = capi.ovrError_DisplayManagerInit
ERROR_TRACKER_DRIVER_INIT = capi.ovrError_TrackerDriverInit
ERROR_LIB_SIGN_CHECK = capi.ovrError_LibSignCheck
ERROR_LIB_PATH = capi.ovrError_LibPath
ERROR_LIB_SYMBOLS = capi.ovrError_LibSymbols
ERROR_REMOTE_SESSION = capi.ovrError_RemoteSession
ERROR_INITIALIZE_VULKAN = capi.ovrError_InitializeVulkan
ERROR_BLACKLISTED_GFX_DRIVER = capi.ovrError_BlacklistedGfxDriver
ERROR_DISPLAY_LOST = capi.ovrError_DisplayLost
ERROR_TEXTURE_SWAP_CHAIN_FULL = capi.ovrError_TextureSwapChainFull
ERROR_TEXTURE_SWAP_CHAIN_INVALID = capi.ovrError_TextureSwapChainInvalid
ERROR_GRAPHICS_DEVICE_RESET = capi.ovrError_GraphicsDeviceReset
ERROR_DISPLAY_REMOVED = capi.ovrError_DisplayRemoved
ERROR_CONTENT_PROTECTION_NOT_AVAILABLE = capi.ovrError_ContentProtectionNotAvailable
ERROR_APPLICATION_VISIBLE = capi.ovrError_ApplicationInvisible
ERROR_DISALLOWED = capi.ovrError_Disallowed
ERROR_DISPLAY_PLUGGED_INCORRECTY = capi.ovrError_DisplayPluggedIncorrectly
ERROR_DISPLAY_LIMIT_REACHED = capi.ovrError_DisplayLimitReached
ERROR_RUNTIME_EXCEPTION = capi.ovrError_RuntimeException
ERROR_NO_CALIBRATION = capi.ovrError_NoCalibration
ERROR_OLD_VERSION = capi.ovrError_OldVersion
ERROR_MISFORMATTED_BLOCK = capi.ovrError_MisformattedBlock

# misc constants
EYE_LEFT = capi.ovrEye_Left
EYE_RIGHT = capi.ovrEye_Right
EYE_COUNT = capi.ovrEye_Count
HAND_LEFT = capi.ovrHand_Left
HAND_RIGHT = capi.ovrHand_Right
HAND_COUNT = capi.ovrHand_Count

# texture formats, color and depth
FORMAT_R8G8B8A8_UNORM = capi.OVR_FORMAT_R8G8B8A8_UNORM
FORMAT_R8G8B8A8_UNORM_SRGB = capi.OVR_FORMAT_R8G8B8A8_UNORM_SRGB
FORMAT_R16G16B16A16_FLOAT =  capi.OVR_FORMAT_R16G16B16A16_FLOAT
FORMAT_R11G11B10_FLOAT = capi.OVR_FORMAT_R11G11B10_FLOAT
FORMAT_D16_UNORM = capi.OVR_FORMAT_D16_UNORM
FORMAT_D24_UNORM_S8_UINT = capi.OVR_FORMAT_D24_UNORM_S8_UINT
FORMAT_D32_FLOAT = capi.OVR_FORMAT_D32_FLOAT

# performance
MAX_PROVIDED_FRAME_STATS = capi.ovrMaxProvidedFrameStats

# tracked device types
TRACKED_DEVICE_TYPE_HMD = capi.ovrTrackedDevice_HMD
TRACKED_DEVICE_TYPE_LTOUCH = capi.ovrTrackedDevice_LTouch
TRACKED_DEVICE_TYPE_RTOUCH = capi.ovrTrackedDevice_RTouch
TRACKED_DEVICE_TYPE_TOUCH = capi.ovrTrackedDevice_Touch
TRACKED_DEVICE_TYPE_OBJECT0 = capi.ovrTrackedDevice_Object0
TRACKED_DEVICE_TYPE_OBJECT1 = capi.ovrTrackedDevice_Object1
TRACKED_DEVICE_TYPE_OBJECT2 = capi.ovrTrackedDevice_Object2
TRACKED_DEVICE_TYPE_OBJECT3 = capi.ovrTrackedDevice_Object3

# tracking origin types
TRACKING_ORIGIN_EYE_LEVEL = capi.ovrTrackingOrigin_EyeLevel
TRACKING_ORIGIN_FLOOR_LEVEL = capi.ovrTrackingOrigin_FloorLevel

# trackings state status flags
STATUS_ORIENTATION_TRACKED = capi.ovrStatus_OrientationTracked
STATUS_POSITION_TRACKED = capi.ovrStatus_PositionTracked

# API version information
PRODUCT_VERSION = capi.OVR_PRODUCT_VERSION
MAJOR_VERSION = capi.OVR_MAJOR_VERSION
MINOR_VERSION = capi.OVR_MINOR_VERSION
PATCH_VERSION = capi.OVR_PATCH_VERSION
BUILD_NUMBER = capi.OVR_BUILD_NUMBER
DLL_COMPATIBLE_VERSION = capi.OVR_DLL_COMPATIBLE_VERSION
MIN_REQUESTABLE_MINOR_VERSION = capi.OVR_MIN_REQUESTABLE_MINOR_VERSION
FEATURE_VERSION = capi.OVR_FEATURE_VERSION

# API keys
KEY_USER = capi.OVR_KEY_USER
KEY_NAME = capi.OVR_KEY_NAME
KEY_GENDER = capi.OVR_KEY_GENDER
DEFAULT_GENDER = capi.OVR_DEFAULT_GENDER
KEY_PLAYER_HEIGHT = capi.OVR_KEY_PLAYER_HEIGHT
DEFAULT_PLAYER_HEIGHT = capi.OVR_DEFAULT_PLAYER_HEIGHT
KEY_EYE_HEIGHT = capi.OVR_KEY_EYE_HEIGHT
DEFAULT_EYE_HEIGHT = capi.OVR_DEFAULT_EYE_HEIGHT
KEY_NECK_TO_EYE_DISTANCE = capi.OVR_KEY_NECK_TO_EYE_DISTANCE
DEFAULT_NECK_TO_EYE_HORIZONTAL = capi.OVR_DEFAULT_NECK_TO_EYE_HORIZONTAL
DEFAULT_NECK_TO_EYE_VERTICAL = capi.OVR_DEFAULT_NECK_TO_EYE_VERTICAL
KEY_EYE_TO_NOSE_DISTANCE = capi.OVR_KEY_EYE_TO_NOSE_DISTANCE
PERF_HUD_MODE = capi.OVR_PERF_HUD_MODE
LAYER_HUD_MODE = capi.OVR_LAYER_HUD_MODE
LAYER_HUD_CURRENT_LAYER = capi.OVR_LAYER_HUD_CURRENT_LAYER
LAYER_HUD_SHOW_ALL_LAYERS = capi.OVR_LAYER_HUD_SHOW_ALL_LAYERS
DEBUG_HUD_STEREO_MODE = capi.OVR_DEBUG_HUD_STEREO_MODE
DEBUG_HUD_STEREO_GUIDE_INFO_ENABLE = capi.OVR_DEBUG_HUD_STEREO_GUIDE_INFO_ENABLE
DEBUG_HUD_STEREO_GUIDE_SIZE = capi.OVR_DEBUG_HUD_STEREO_GUIDE_SIZE
DEBUG_HUD_STEREO_GUIDE_POSITION = capi.OVR_DEBUG_HUD_STEREO_GUIDE_POSITION
DEBUG_HUD_STEREO_GUIDE_YAWPITCHROLL = capi.OVR_DEBUG_HUD_STEREO_GUIDE_YAWPITCHROLL
DEBUG_HUD_STEREO_GUIDE_COLOR = capi.OVR_DEBUG_HUD_STEREO_GUIDE_COLOR

# performance hud modes
PERF_HUD_OFF = capi.ovrPerfHud_Off
PERF_HUD_PERF_SUMMARY = capi.ovrPerfHud_PerfSummary
PERF_HUD_LATENCY_TIMING = capi.ovrPerfHud_LatencyTiming
PERF_HUD_APP_RENDER_TIMING = capi.ovrPerfHud_AppRenderTiming
PERF_HUD_COMP_RENDER_TIMING = capi.ovrPerfHud_CompRenderTiming
PERF_HUD_ASW_STATS = capi.ovrPerfHud_AswStats
PERF_HUD_VERSION_INFO = capi.ovrPerfHud_VersionInfo
PERF_HUD_COUNT = capi.ovrPerfHud_Count

# stereo debug hud
DEBUG_HUD_STEREO_MODE_OFF = capi.ovrDebugHudStereo_Off
DEBUG_HUD_STEREO_MODE_QUAD = capi.ovrDebugHudStereo_Quad
DEBUG_HUD_STEREO_MODE_QUAD_WITH_CROSSHAIR = capi.ovrDebugHudStereo_QuadWithCrosshair
DEBUG_HUD_STEREO_MODE_CROSSHAIR_AT_INFINITY = capi.ovrDebugHudStereo_CrosshairAtInfinity

# swapchain handles, more than enough for now
TEXTURE_SWAP_CHAIN0 = 0
TEXTURE_SWAP_CHAIN1 = 1
TEXTURE_SWAP_CHAIN2 = 2
TEXTURE_SWAP_CHAIN3 = 3
TEXTURE_SWAP_CHAIN4 = 4
TEXTURE_SWAP_CHAIN5 = 5
TEXTURE_SWAP_CHAIN6 = 6
TEXTURE_SWAP_CHAIN7 = 7
TEXTURE_SWAP_CHAIN8 = 8
TEXTURE_SWAP_CHAIN9 = 9
TEXTURE_SWAP_CHAIN10 = 10
TEXTURE_SWAP_CHAIN11 = 11
TEXTURE_SWAP_CHAIN12 = 12
TEXTURE_SWAP_CHAIN13 = 13
TEXTURE_SWAP_CHAIN14 = 14
TEXTURE_SWAP_CHAIN15 = 15

# boundary modes
BOUNDARY_PLAY_AREA = capi.ovrBoundary_PlayArea
BOUNDARY_OUTER = capi.ovrBoundary_Outer

# layer header flags
LAYER_FLAG_HIGH_QUALITY = capi.ovrLayerFlag_HighQuality
LAYER_FLAG_TEXTURE_ORIGIN_AT_BOTTOM_LEFT = \
    capi.ovrLayerFlag_TextureOriginAtBottomLeft
LAYER_FLAG_HEAD_LOCKED = capi.ovrLayerFlag_HeadLocked

# ------------------------------------------------------------------------------
# Wrapper factory functions
#
cdef np.npy_intp[1] VEC2_SHAPE = [2]
cdef np.npy_intp[1] VEC3_SHAPE = [3]
cdef np.npy_intp[1] FOVPORT_SHAPE = [4]
cdef np.npy_intp[1] QUAT_SHAPE = [4]
cdef np.npy_intp[2] MAT4_SHAPE = [4, 4]


cdef np.ndarray _wrap_ovrVector2f_as_ndarray(capi.ovrVector2f* prtVec):
    """Wrap an ovrVector2f object with a NumPy array."""
    return np.PyArray_SimpleNewFromData(
        1, VEC2_SHAPE, np.NPY_FLOAT32, <void*>prtVec)


cdef np.ndarray _wrap_ovrVector3f_as_ndarray(capi.ovrVector3f* prtVec):
    """Wrap an ovrVector3f object with a NumPy array."""
    return np.PyArray_SimpleNewFromData(
        1, VEC3_SHAPE, np.NPY_FLOAT32, <void*>prtVec)


cdef np.ndarray _wrap_ovrQuatf_as_ndarray(capi.ovrQuatf* prtVec):
    """Wrap an ovrQuatf object with a NumPy array."""
    return np.PyArray_SimpleNewFromData(
        1, QUAT_SHAPE, np.NPY_FLOAT32, <void*>prtVec)


#cdef np.ndarray _wrap_ovrMatrix4f_as_ndarray(capi.ovrMatrix4f* prtVec):
#    """Wrap an ovrMatrix4f object with a NumPy array."""
#    return np.PyArray_SimpleNewFromData(
#        2, MAT4_SHAPE, np.NPY_FLOAT32, <void*>prtVec.M)


cdef np.ndarray _wrap_ovrFovPort_as_ndarray(capi.ovrFovPort* prtVec):
    """Wrap an ovrFovPort object with a NumPy array."""
    return np.PyArray_SimpleNewFromData(
        1, FOVPORT_SHAPE, np.NPY_FLOAT32, <void*>prtVec)

# ------------------------------------------------------------------------------
# Classes and extension types
#

cdef class LibOVRPose(object):
    """Class for representing rigid body poses.

    This class is an abstract representation of a rigid body pose, where the
    position of the body in a scene is represented by a vector/coordinate and
    the orientation with a quaternion. LibOVR uses poses to represent the
    posture of tracked devices (e.g. HMD, touch controllers, etc.) and other
    objects in a VR scene. Poses can be manipulated and interacted with using
    class methods and attributes.

    Poses can be converted to 4x4 transformation matrices. One can use these
    matrices when rendering to transform the vertices of a model associated with
    the pose by passing them to OpenGL. Furthermore, poses can be used to
    transform other poses and vectors.

    This class is a wrapper for the ``OVR::ovrPosef`` data structure. Fields
    ``OVR::ovrPosef.Orientation`` and ``OVR::ovrPosef.Position`` are accessed
    using `ndarray` proxy objects which share the same memory. Therefore, data
    accessed and written to those objects will directly change the field values
    of the referenced ``OVR::ovrPosef`` struct. ``OVR::ovrPosef.Orientation``
    and ``OVR::ovrPosef.Position`` can be accessed and edited using the
    :py:class:`~LibOVRPose.ori` and :py:class:`~LibOVRPose.pos` class
    attributes, respectively. Methods associated with this class perform various
    operations using the functions provided by `OVR_MATH.h
    <https://developer.oculus.com/reference/libovr/1.38/o_v_r_math_8h/>`_, which
    is part of the Oculus PC SDK.

    """
    cdef capi.ovrPosef* c_data
    cdef bint ptr_owner

    cdef np.ndarray _pos
    cdef np.ndarray _ori

    def __init__(self, pos=(0., 0., 0.), ori=(0., 0., 0., 1.)):
        """
        Parameters
        ----------
        pos : tuple, list, or ndarray of float
            Position vector (x, y, z).
        ori : tuple, list, or ndarray of float
            Orientation quaternion vector (x, y, z, w).

        Attributes
        ----------
        pos : ndarray
        ori : ndarray
        posOri : tuple of ndarray
        at : ndarray
        up : ndarray

        """
        self._new_struct(pos, ori)

    def __cinit__(self, *args, **kwargs):
        self.ptr_owner = False

    @staticmethod
    cdef LibOVRPose fromPtr(capi.ovrPosef* ptr, bint owner=False):
        cdef LibOVRPose wrapper = LibOVRPose.__new__(LibOVRPose)
        wrapper.c_data = ptr
        wrapper.ptr_owner = owner

        wrapper._pos = _wrap_ovrVector3f_as_ndarray(&ptr.Position)
        wrapper._ori = _wrap_ovrQuatf_as_ndarray(&ptr.Orientation)

        return wrapper

    cdef void _new_struct(self, object pos, object ori):
        if self.c_data is not NULL:
            return

        cdef capi.ovrPosef* ptr = <capi.ovrPosef*>PyMem_Malloc(sizeof(capi.ovrPosef))

        if ptr is NULL:
            raise MemoryError

        # clear memory
        ptr.Position.x = <float>pos[0]
        ptr.Position.y = <float>pos[1]
        ptr.Position.z = <float>pos[2]
        ptr.Orientation.x = <float>ori[0]
        ptr.Orientation.y = <float>ori[1]
        ptr.Orientation.z = <float>ori[2]
        ptr.Orientation.w = <float>ori[3]

        self.c_data = ptr
        self.ptr_owner = True

        self._pos = _wrap_ovrVector3f_as_ndarray(&ptr.Position)
        self._ori = _wrap_ovrQuatf_as_ndarray(&ptr.Orientation)

    def __dealloc__(self):
        # don't do anything crazy like set c_data=NULL without deallocating!
        if self.c_data is not NULL:
            if self.ptr_owner is True:
                PyMem_Free(self.c_data)
                self.c_data = NULL

    def __mul__(LibOVRPose a, LibOVRPose b):
        """Multiplication operator (*) to combine poses.
        """
        cdef capi.ovrPosef* ptr = <capi.ovrPosef*>PyMem_Malloc(sizeof(capi.ovrPosef))

        if ptr is NULL:
            raise MemoryError

        cdef libovr_math.Posef pose_r = \
            <libovr_math.Posef>a.c_data[0] * <libovr_math.Posef>b.c_data[0]

        # copy into
        ptr[0] = <capi.ovrPosef>pose_r
        return LibOVRPose.fromPtr(ptr, True)

    def __imul__(self, LibOVRPose other):
        """Inplace multiplication operator (*=) to combine poses.
        """
        cdef libovr_math.Posef result = <libovr_math.Posef>self.c_data[0] * \
                                        <libovr_math.Posef>other.c_data[0]
        self.c_data[0] = <capi.ovrPosef>result
        return self

    def __invert__(self):
        """Invert operator (~) to invert a pose."""
        return self.inverted()

    def __eq__(self, LibOVRPose other):
        """Equality operator (==) for two poses.

        The tolerance of the comparison is defined by the Oculus SDK as 1e-5.

        """
        return (<libovr_math.Posef>self.c_data[0]).IsEqual(
            <libovr_math.Posef>other.c_data[0], <float>1e-5)

    def __ne__(self, LibOVRPose other):
        """Inequality operator (!=) for two poses.

        The tolerance of the comparison is defined by the Oculus SDK as 1e-5.

        """
        return not (<libovr_math.Posef>self.c_data[0]).IsEqual(
            <libovr_math.Posef>other.c_data[0], <float>1e-5)

    def __deepcopy__(self, memo=None):
        # create a new object with a copy of the data stored in c_data
        # allocate new struct
        cdef capi.ovrPosef* ptr = <capi.ovrPosef*>PyMem_Malloc(sizeof(capi.ovrPosef))

        if ptr is NULL:
            raise MemoryError

        cdef LibOVRPose to_return = LibOVRPose.fromPtr(ptr, owner=True)

        # copy over data
        to_return.c_data[0] = self.c_data[0]
        if memo is not None:
            memo[id(self)] = to_return

        return to_return

    def isEqual(self, LibOVRPose pose, float tolerance=1e-5):
        """Check if poses are close to equal in position and orientation.

        Same as using the equality operator (==) on poses, but you can specify
        and arbitrary value for `tolerance`.

        Parameters
        ----------
        pose : LibOVRPose
            The other pose.
        tolerance : float, optional
            Tolerance for the comparison, default is 1e-5 as defined in
            `OVR_MATH.h`.

        Returns
        -------
        bool
            True if pose components are within `tolerance` from this pose.

        """
        return (<libovr_math.Posef>self.c_data[0]).IsEqual(
            <libovr_math.Posef>pose.c_data[0], tolerance)

    def duplicate(self):
        """Create a deep copy of this object.

        Same as calling `copy.deepcopy` on an instance.

        Returns
        -------
        LibOVRPose
            An independent copy of this object.

        """
        return self.__deepcopy__()

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
        """Position vector [X, Y, Z] (`ndarray`).

        Examples
        --------

        Set the position of the pose::

            myPose.pos = [0., 0., -1.5]

        Get the x, y, and z coordinates of a pose::

            x, y, z = myPose.pos

        The `ndarray` returned by `pos` directly references the position field
        data in the pose data structure (`ovrPosef`). Updating values will
        directly edit the values in the structure. For instance, you can specify
        a component of a pose's position::

            myPose.pos[2] = -10.0  # z = -10.0

        Assigning `pos` a name will create a reference to that `ndarray` which
        can edit values in the structure::

            p = myPose.pos
            p[1] = 1.5  # sets the Y position of 'myPose' to 1.5

        """
        return self._pos

    @pos.setter
    def pos(self, object value):
        self._pos[:] = value

    def getPos(self, object out=None):
        """Position vector X, Y, Z (`ndarray` of `float`).

        The returned object is a NumPy array which contains a copy of the data
        stored in an internal structure (ovrPosef). The array is conformal with
        the internal data's type (float32) and size (length 3).

        Parameters
        ----------
        out : ndarray or None
            Option array to write values to. Must have a float32 data type.

        Returns
        -------
        ndarray
            Position coordinate of this pose.

        Raises
        ------
        ValueError
            Buffer dtype mismatch where float32 was expected.
        IndexError
            Out of bounds on buffer access.

        Examples
        --------

        Get the position coordinates::

            x, y, z = myPose.getPos()  # Python float literals
            # ... or ...
            pos = myPose.getPos()  # NumPy array shape=(3,) and dtype=float32

        Write the position to an existing array by specifying `out`::

            position = numpy.zeros((3,), dtype=numpy.float32)  # mind the dtype!
            myPose.getPos(position)  # position now contains myPose.pos

        You can also pass a view/slice to `out`::

            coords = numpy.zeros((100,3,), dtype=numpy.float32)  # big array
            myPose.getPos(coords[42,:])  # row 42

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = out

        toReturn[0] = self.c_data[0].Position.x
        toReturn[1] = self.c_data[0].Position.y
        toReturn[2] = self.c_data[0].Position.z

        return toReturn

    def setPos(self, object pos):
        """Set the position of the pose in a scene.

        Parameters
        ----------
        pos : list, tuple of float or ndarray
            Position vector [X, Y, Z].

        """
        self.c_data[0].Position.x = <float>pos[0]
        self.c_data[0].Position.y = <float>pos[1]
        self.c_data[0].Position.z = <float>pos[2]

    @property
    def ori(self):
        """Orientation quaternion [X, Y, Z, W] (`ndarray`).
        """
        return self._ori

    @ori.setter
    def ori(self, object value):
        self._ori[:] = value

    def getOri(self, object out=None):
        """Orientation quaternion X, Y, Z, W (`ndarray` of `float`).

        Components X, Y, Z are imaginary and W is real.

        The returned object is a NumPy array which references data stored in an
        internal structure (ovrPosef). The array is conformal with the internal
        data's type (float32) and size (length 4).

        Parameters
        ----------
        out : ndarray or None
            Option array to write values to. Must have a float32 data type.

        Returns
        -------
        ndarray
            Orientation quaternion of this pose.

        Raises
        ------
        ValueError
            Buffer dtype mismatch where float32 was expected.
        IndexError
            Out of bounds on buffer access.

        Notes
        -----

        * The orientation quaternion should be normalized.

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((4,), dtype=np.float32)
        else:
            toReturn = out

        toReturn[0] = self.c_data[0].Orientation.x
        toReturn[1] = self.c_data[0].Orientation.y
        toReturn[2] = self.c_data[0].Orientation.z
        toReturn[3] = self.c_data[0].Orientation.w

        return toReturn

    def setOri(self, object ori):
        """Set the orientation of the pose in a scene.

        Parameters
        ----------
        ori : list, tuple of float or ndarray
            Orientation quaternion [X, Y, Z, W].

        """
        self.c_data[0].Orientation.x = <float>ori[0]
        self.c_data[0].Orientation.y = <float>ori[1]
        self.c_data[0].Orientation.z = <float>ori[2]
        self.c_data[0].Orientation.w = <float>ori[3]

    @property
    def posOri(self):
        """Position vector and orientation quaternion."""
        return self.pos, self.ori

    @posOri.setter
    def posOri(self, object value):
        self.pos = value[0]
        self.ori = value[1]

    @property
    def at(self):
        """Forward vector of this pose (-Z is forward) (read-only)."""
        return self.getAt()

    def getAt(self, object out=None):
        """Get the `at` vector for this pose.

        Parameters
        ----------
        out : ndarray or None
            Option array to write values to. Must have shape (3,) and a float32
            data type.

        Returns
        -------
        ndarray
            The vector for `at`.

        Raises
        ------
        ValueError
            Buffer dtype mismatch where float32 was expected.
        IndexError
            Out of bounds on buffer access.

        Notes
        -----
        It's better to use the `at` property if you are not supplying an output
        array.

        Examples
        --------

        Setting the listener orientation for 3D positional audio (PyOpenAL)::

            myListener.set_orientation((*myPose.getAt(), *myPose.getUp()))

        See Also
        --------
        getUp : Get the `up` vector.

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = out

        cdef libovr_math.Vector3f at = \
            (<libovr_math.Quatf>self.c_data[0].Orientation).Rotate(
                libovr_math.Vector3f(0.0, 0.0, -1.0))

        toReturn[0] = at.x
        toReturn[1] = at.y
        toReturn[2] = at.z

        return toReturn

    @property
    def up(self):
        """Up vector of this pose (+Y is up) (read-only)."""
        return self.getUp()

    def getUp(self, object out=None):
        """Get the 'up' vector for this pose.

        Parameters
        ----------
        out : ndarray, optional
            Option array to write values to. Must have shape (3,) and a float32
            data type.

        Returns
        -------
        ndarray
            The vector for `up`.

        Raises
        ------
        ValueError
            Buffer dtype mismatch where float32 was expected.
        IndexError
            Out of bounds on buffer access.

        Notes
        -----
        It's better to use the `up` property if you are not supplying an output
        array. However, `getUp` will have the same effect as the `up` property
        if `out`=None.

        Examples
        --------

        Using the `up` vector with gluLookAt::

            up = myPose.getUp()  # myPose.up also works
            center = myPose.pos
            target = targetPose.pos  # some target pose
            gluLookAt(*(*up, *center, *target))

        See Also
        --------
        getAt : Get the `up` vector.

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = out

        cdef libovr_math.Vector3f up = \
            (<libovr_math.Quatf>self.c_data[0].Orientation).Rotate(
                libovr_math.Vector3f(0.0, 1.0, 0.0))

        toReturn[0] = up.x
        toReturn[1] = up.y
        toReturn[2] = up.z

        return toReturn

    def getYawPitchRoll(self, LibOVRPose refPose=None, object out=None):
        """Get the yaw, pitch, and roll of the orientation quaternion.

        Parameters
        ----------
        refPose : LibOVRPose, optional
            Reference pose to compute angles relative to. If None is specified,
            computed values are referenced relative to the world axes.
        out : ndarray
            Alternative place to write yaw, pitch, and roll values. Must have
            shape (3,) and a float32 data type.

        Returns
        -------
        ndarray of float or None
            Yaw, pitch, and roll of the pose in degrees.

        Notes
        -----

        * Uses ``OVR::Quatf.GetYawPitchRoll`` which is part of the Oculus PC
          SDK.

        """
        cdef float yaw, pitch, roll
        cdef libovr_math.Posef inPose = <libovr_math.Posef>self.c_data[0]
        cdef libovr_math.Posef invRef

        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = out

        if refPose is not None:
            invRef = (<libovr_math.Posef>refPose.c_data[0]).Inverted()
            inPose = invRef * inPose

        inPose.Rotation.GetYawPitchRoll(&yaw, &pitch, &roll)

        toReturn[0] = yaw
        toReturn[1] = pitch
        toReturn[2] = roll

        return toReturn

    def asMatrix(self, bint inverse=False, object out=None):
        """Convert this pose into a 4x4 transformation matrix.

        Parameters
        ----------
        inverse : bool, optional
            If True, return the inverse of the matrix.
        out : ndarray
            Alternative place to write the matrix to values. Must be a `ndarray`
            of shape (4, 4,) and have a data type of float32. Values are written
            assuming row-major order.

        Returns
        -------
        ndarray
            4x4 transformation matrix.

        """
        cdef libovr_math.Matrix4f m_pose = libovr_math.Matrix4f(
            <libovr_math.Posef>self.c_data[0])

        if inverse:
            m_pose.InvertHomogeneousTransform()  # faster than Invert() here

        cdef np.ndarray[np.float32_t, ndim=2] toReturn
        if out is None:
            toReturn =  np.zeros((4, 4), dtype=np.float32)
        else:
            toReturn = out

        # fast copy matrix to numpy array
        cdef float [:, :] mv = toReturn
        cdef Py_ssize_t i, j
        cdef Py_ssize_t N = 4
        i = j = 0
        for i in range(N):
            for j in range(N):
                mv[i, j] = m_pose.M[i][j]

        return toReturn

    def normalize(self):
        """Normalize this pose.

        Notes
        -----
        Uses ``OVR::Posef.Normalize`` which is part of the Oculus PC SDK.

        """
        (<libovr_math.Posef>self.c_data[0]).Normalize()

    def inverted(self):
        """Get the inverse of the pose.

        Returns
        -------
        :py:class:`LibOVRPose`
            Inverted pose.

        Notes
        -----
        * Uses ``OVR::Posef.Inverted`` which is part of the Oculus PC SDK.

        """
        cdef capi.ovrPosef* ptr = <capi.ovrPosef*>PyMem_Malloc(
            sizeof(capi.ovrPosef))

        if ptr is NULL:
            raise MemoryError

        cdef libovr_math.Quatf inv_ori = \
            (<libovr_math.Quatf>self.c_data[0].Orientation).Inverted()
        cdef libovr_math.Vector3f inv_pos = \
            (<libovr_math.Quatf>inv_ori).Rotate(
                -(<libovr_math.Vector3f>self.c_data[0].Position))

        ptr[0].Orientation.x = inv_ori.x
        ptr[0].Orientation.y = inv_ori.y
        ptr[0].Orientation.z = inv_ori.z
        ptr[0].Orientation.w = inv_ori.w
        ptr[0].Position.x = inv_pos.x
        ptr[0].Position.y = inv_pos.y
        ptr[0].Position.z = inv_pos.z

        return LibOVRPose.fromPtr(ptr, True)

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

        Notes
        -----
        * Uses ``OVR::Posef.Rotate`` which is part of the Oculus PC SDK.

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

        Notes
        -----
        * Uses ``OVR::Vector3f.InverseRotate`` which is part of the Oculus PC SDK.

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
            Vector to translate [x, y, z].

        Returns
        -------
        ndarray
            Vector translated by the pose's position.

        Notes
        -----
        * Uses ``OVR::Vector3f.Translate`` which is part of the Oculus PC SDK.

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
            Vector to transform [x, y, z].

        Returns
        -------
        ndarray
            Vector transformed by the pose's position and orientation.

        Notes
        -----
        * Uses ``OVR::Vector3f.Transform`` which is part of the Oculus PC SDK.

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

        Notes
        -----
        * Uses ``OVR::Vector3f.InverseTransform`` which is part of the Oculus PC
          SDK.

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

        Notes
        -----
        * Uses ``OVR::Vector3f.TransformNormal`` which is part of the Oculus PC
          SDK.

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

        Notes
        -----
        * Uses ``OVR::Vector3f.InverseTransformNormal`` which is part of the
          Oculus PC SDK.

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

        Examples
        --------

        Get the distance between poses::

            distance = thisPose.distanceTo(otherPose)

        Get the distance to a point coordinate::

            distance = thisPose.distanceTo([0.0, 0.0, 5.0])

        Do something if the tracked right hand pose is within 0.5 meters of some
        object::

            # use 'getTrackingState' instead for hand poses, just an example
            handPose = getDevicePoses(TRACKED_DEVICE_TYPE_RTOUCH,
                                      absTime, latencyMarker=False)
            # object pose
            objPose = LibOVRPose((0.0, 1.0, -0.5))

            if handPose.distanceTo(objPose) < 0.5:
                # do something here ...

        Vary the touch controller's vibration amplitude as a function of
        distance to some pose. As the hand gets closer to the point, the
        amplitude of the vibration increases::

            dist = handPose.distanceTo(objPose)
            vibrationRadius = 0.5

            if dist < vibrationRadius:  # inside vibration radius
                amplitude = 1.0 - dist / vibrationRadius
                setControllerVibration(CONTROLLER_TYPE_RTOUCH,
                    'low', amplitude)
            else:
                # turn off vibration
                setControllerVibration(CONTROLLER_TYPE_RTOUCH, 'off')

        """
        cdef libovr_math.Vector3f pos_in

        if isinstance(v, LibOVRPose):
            pos_in = <libovr_math.Vector3f>((<LibOVRPose>v).c_data[0]).Position
        else:
            pos_in = libovr_math.Vector3f(<float>v[0], <float>v[1], <float>v[2])

        cdef float to_return = \
            (<libovr_math.Posef>self.c_data[0]).Translation.Distance(pos_in)

        return to_return

    def raycastSphere(self, object targetPose, float radius=0.5, object rayDir=(0., 0., -1.), float maxRange=0.0):
        """Raycast to a sphere.

        Project an invisible ray of finite or infinite length from this pose in
        `rayDir` and check if it intersects with the targetPose bounding sphere.

        This method allows for very basic interaction between objects
        represented by poses in a scene, including tracked devices.

        Specifying `maxRange` as >0.0 casts a ray of finite length in world
        units. The distance between the target and ray origin position are
        checked prior to casting the ray; automatically failing if the ray can
        never reach the edge of the bounding sphere centered about `targetPose`.
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
            the target is out of range. Ray is infinite if `maxRange`=0.0.

        Returns
        -------
        bool
            True if the ray intersects anywhere on the bounding sphere, False in
            every other condition.

        Examples
        --------

        Basic example to check if the HMD is aligned to some target::

            targetPose = LibOVRPose((0.0, 1.5, -5.0))
            targetRadius = 0.5  # 2.5 cm
            isTouching = hmdPose.raycastSphere(targetPose,
                                               radius=targetRadius)

        Check if someone is touching a target with their finger when making a
        pointing gesture while providing haptic feedback::

            targetPose = LibOVRPose((0.0, 1.5, -0.25))
            targetRadius = 0.025  # 2.5 cm
            fingerLength = 0.1  # 10 cm

            # check if making a pointing gesture with their right hand
            isPointing = getTouch(CONTROLLER_TYPE_RTOUCH,
                TOUCH_RINDEXPOINTING)

            if isPointing:
                # do raycasting operation
                isTouching = handPose.raycastSphere(
                    targetPose, radius=targetRadius, maxRange=fingerLength)
                if isTouching:
                    # do something here, like make the controller vibrate
                    setControllerVibration(
                        CONTROLLER_TYPE_RTOUCH, 'low', 0.5)
                else:
                    # stop vibration if no longer touching
                    setControllerVibration(CONTROLLER_TYPE_RTOUCH, 'off')

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

    def interp(self, LibOVRPose end, float s, bint fast=False):
        """Interpolate between poses.

        Linear interpolation is used on position (Lerp) while the orientation
        has spherical linear interpolation (Slerp) applied.

        Parameters
        ----------
        end : LibOVRPose
            End pose.
        s : float
            Interpolation factor between interval 0.0 and 1.0.
        fast : bool, optional
            If True, use fast interpolation which is quicker but less accurate
            over larger distances.

        Returns
        -------
        LibOVRPose
            Interpolated pose at `s`.

        Notes
        -----
        * Uses ``OVR::Posef.Lerp`` and ``OVR::Posef.FastLerp`` which is part of
          the Oculus PC SDK.

        """
        if 0.0 > s > 1.0:
            raise ValueError("Interpolation factor must be between 0.0 and 1.0.")

        cdef libovr_math.Posef toPose = <libovr_math.Posef>end.c_data[0]
        cdef capi.ovrPosef* ptr = <capi.ovrPosef*>PyMem_Malloc(sizeof(capi.ovrPosef))

        if ptr is NULL:
            raise MemoryError

        if not fast:
            ptr[0] = <capi.ovrPosef>(
                (<libovr_math.Posef>self.c_data[0]).Lerp(toPose, s))
        else:
            ptr[0] = <capi.ovrPosef>(
                (<libovr_math.Posef>self.c_data[0]).FastLerp(toPose, s))

        return LibOVRPose.fromPtr(ptr, True)


cdef class LibOVRPoseState(object):
    """Class for representing rigid body poses with additional state
    information.

    Pose, angular and linear motion derivatives of a tracked rigid body reported
    by LibOVR. Functions :func:`getTrackingState` and :func:`getDevicePoses`
    returns an instance of this class. Velocity and acceleration for linear and
    angular motion can be used to compute forces applied to rigid bodies and
    predict the future positions of objects (see
    :py:method:`~psychxr.libovr.LibOVRPoseState.timeIntegrate`).

    """
    cdef capi.ovrPoseStatef* c_data
    cdef bint ptr_owner  # owns the data

    # these will hold references until this object is de-allocated
    cdef LibOVRPose _pose
    cdef np.ndarray _linearVelocity
    cdef np.ndarray _angularVelocity
    cdef np.ndarray _linearAcceleration
    cdef np.ndarray _angularAcceleration

    def __init__(self):
        """
        Attributes
        ----------
        pose : :obj:`LibOVRPose`
        angularVelocity : `ndarray`
        linearVelocity : `ndarray`
        angularAcceleration : `ndarray`
        linearAcceleration : `ndarray`
        timeInSeconds : `float`

        """
        self._new_struct()

    def __cinit__(self):
        self.ptr_owner = False

    @staticmethod
    cdef LibOVRPoseState fromPtr(capi.ovrPoseStatef* ptr, bint owner=False):
        # bypass __init__ if wrapping a pointer
        cdef LibOVRPoseState wrapper = LibOVRPoseState.__new__(LibOVRPoseState)
        wrapper.c_data = ptr
        wrapper.ptr_owner = owner

        wrapper._pose = LibOVRPose.fromPtr(&wrapper.c_data.ThePose)
        wrapper._linearVelocity = _wrap_ovrVector3f_as_ndarray(
                &wrapper.c_data.LinearVelocity)
        wrapper._linearAcceleration = _wrap_ovrVector3f_as_ndarray(
                &wrapper.c_data.LinearAcceleration)
        wrapper._angularVelocity = _wrap_ovrVector3f_as_ndarray(
                &wrapper.c_data.AngularVelocity)
        wrapper._angularAcceleration = _wrap_ovrVector3f_as_ndarray(
                &wrapper.c_data.AngularAcceleration)

        return wrapper

    cdef void _new_struct(self):
        if self.c_data is not NULL:  # already allocated, __init__ called twice?
            return

        cdef capi.ovrPoseStatef* _ptr = \
            <capi.ovrPoseStatef*>PyMem_Malloc(
                sizeof(capi.ovrPoseStatef))

        if _ptr is NULL:
            raise MemoryError

        # clear memory to defaults
        _ptr.ThePose.Position = [0., 0., 0.]
        _ptr.ThePose.Orientation = [0., 0., 0., 1.]
        _ptr.AngularVelocity = [0., 0., 0.]
        _ptr.LinearVelocity = [0., 0., 0.]
        _ptr.AngularAcceleration = [0., 0., 0.]
        _ptr.LinearAcceleration = [0., 0., 0.]
        _ptr.TimeInSeconds = 0.0

        self.c_data = _ptr
        self.ptr_owner = True

        # setup property wrappers
        self._pose = LibOVRPose.fromPtr(&self.c_data.ThePose)
        self._linearVelocity = _wrap_ovrVector3f_as_ndarray(
            &self.c_data.LinearVelocity)
        self._linearAcceleration = _wrap_ovrVector3f_as_ndarray(
            &self.c_data.LinearAcceleration)
        self._angularVelocity = _wrap_ovrVector3f_as_ndarray(
            &self.c_data.AngularVelocity)
        self._angularAcceleration = _wrap_ovrVector3f_as_ndarray(
            &self.c_data.AngularAcceleration)

    def __dealloc__(self):
        # don't do anything crazy like set c_data=NULL without deallocating!
        if self.c_data is not NULL:
            if self.ptr_owner is True:
                PyMem_Free(self.c_data)

    def __deepcopy__(self, memo=None):
        """Deep copy returned by :py:func:`copy.deepcopy`.

        New `LibOVRPoseState` instance with a copy of the data in a separate
        memory location. Does not increase the reference count of the object
        being copied.

        Examples
        --------

        Deep copy::

            import copy
            a = LibOVRPoseState()
            b = copy.deepcopy(a)  # create independent copy of 'a'

        """
        cdef capi.ovrPoseStatef* ptr = \
            <capi.ovrPoseStatef*>PyMem_Malloc(sizeof(capi.ovrPoseStatef))

        if ptr is NULL:
            raise MemoryError

        cdef LibOVRPoseState to_return = LibOVRPoseState.fromPtr(ptr, True)

        # copy over data
        to_return.c_data[0] = self.c_data[0]

        if memo is not None:
            memo[id(self)] = to_return

        return to_return

    def duplicate(self):
        """Create a deep copy of this object.

        Same as calling `copy.deepcopy` on an instance.

        Returns
        -------
        LibOVRPoseState
            An independent copy of this object.

        """
        return self.__deepcopy__()

    @property
    def pose(self):
        """Rigid body pose."""
        return self._pose

    @pose.setter
    def pose(self, LibOVRPose value):
        self.c_data.ThePose = value.c_data[0]  # copy into

    @property
    def angularVelocity(self):
        """Angular velocity vector in radians/sec."""
        return self._angularVelocity

    @angularVelocity.setter
    def angularVelocity(self, object value):
        self._angularVelocity[:] = value

    @property
    def linearVelocity(self):
        """Linear velocity vector in meters/sec."""
        return self._linearVelocity

    @linearVelocity.setter
    def linearVelocity(self, object value):
        self._linearVelocity[:] = value

    @property
    def angularAcceleration(self):
        """Angular acceleration vector in radians/s^2."""
        return self._angularAcceleration

    @angularAcceleration.setter
    def angularAcceleration(self, object value):
        self._angularAcceleration[:] = value

    @property
    def linearAcceleration(self):
        """Linear acceleration vector in meters/s^2."""
        return self._linearAcceleration

    @linearAcceleration.setter
    def linearAcceleration(self, object value):
        self._linearAcceleration[:] = value

    @property
    def timeInSeconds(self):
        """Absolute time this data refers to in seconds."""
        return <double>self.c_data[0].TimeInSeconds

    @timeInSeconds.setter
    def timeInSeconds(self, double value):
        self.c_data[0].TimeInSeconds = value

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
            Pose at `dt`.

        Examples
        --------

        Time integrate a pose for a 1/4 second (note the returned object is a
        :py:mod:`LibOVRPose`, not a :py:class:`LibOVRPoseState`)::

            newPose = oldPose.timeIntegrate(0.25)
            pos, ori = newPose.posOri  # extract components

        """
        cdef libovr_math.Posef res = \
            (<libovr_math.Posef>self.c_data[0].ThePose).TimeIntegrate(
                <libovr_math.Vector3f>self.c_data[0].LinearVelocity,
                <libovr_math.Vector3f>self.c_data[0].AngularVelocity,
                <libovr_math.Vector3f>self.c_data[0].LinearAcceleration,
                <libovr_math.Vector3f>self.c_data[0].AngularAcceleration,
                dt)

        cdef capi.ovrPosef* ptr = \
            <capi.ovrPosef*>PyMem_Malloc(sizeof(capi.ovrPosef))

        if ptr is NULL:
            raise MemoryError(
                "Failed to allocate 'ovrPosef' in 'timeIntegrate'.")

        cdef LibOVRPose to_return = LibOVRPose.fromPtr(ptr, True)

        # copy over data
        to_return.c_data[0] = <capi.ovrPosef>res

        return to_return


cdef class LibOVRTrackerInfo(object):
    """Class for storing tracker (sensor) information such as pose, status, and
    camera frustum information. This object is returned by calling
    :func:`~psychxr.libovr.getTrackerInfo`. Attributes of this class are
    read-only.

    """
    cdef capi.ovrTrackerPose c_ovrTrackerPose
    cdef capi.ovrTrackerDesc c_ovrTrackerDesc

    cdef LibOVRPose _pose
    cdef LibOVRPose _leveledPose

    cdef unsigned int _trackerIndex

    def __init__(self):
        """
        Attributes
        ----------
        trackerIndex : int
            Tracker index this objects refers to (read-only).
        pose : LibOVRPose
            The pose of the sensor (read-only).
        leveledPose : LibOVRPose
            Gravity aligned pose of the sensor (read-only).
        isConnected : bool
            True if the sensor is connected and available (read-only).
        isPoseTracked : bool
            True if the sensor has a valid pose (read-only).
        horizontalFov : float
            Horizontal FOV of the sensor in radians (read-only).
        verticalFov : float
            Vertical FOV of the sensor in radians (read-only).
        nearZ : float
            Near clipping plane of the sensor frustum in meters (read-only).
        farZ : float
            Far clipping plane of the sensor frustum in meters (read-only).

        """
        pass

    def __cinit__(self):
        self._pose = LibOVRPose.fromPtr(&self.c_ovrTrackerPose.Pose)
        self._leveledPose = LibOVRPose.fromPtr(&self.c_ovrTrackerPose.LeveledPose)

    @property
    def trackerIndex(self):
        """Tracker index this objects refers to (read-only)."""
        return self._trackerIndex

    @property
    def pose(self):
        """The pose of the sensor (read-only)."""
        return self._pose

    @property
    def leveledPose(self):
        """Gravity aligned pose of the sensor (read-only)."""
        return self._leveledPose

    @property
    def isConnected(self):
        """True if the sensor is connected and available (read-only)."""
        return <bint>((capi.ovrTracker_Connected &
             self.c_ovrTrackerPose.TrackerFlags) ==
                      capi.ovrTracker_Connected)

    @property
    def isPoseTracked(self):
        """True if the sensor has a valid pose (read-only)."""
        return <bint>((capi.ovrTracker_PoseTracked &
             self.c_ovrTrackerPose.TrackerFlags) ==
                      capi.ovrTracker_PoseTracked)

    @property
    def horizontalFov(self):
        """Horizontal FOV of the sensor in radians (read-only)."""
        return self.c_ovrTrackerDesc.FrustumHFovInRadians

    @property
    def verticalFov(self):
        """Vertical FOV of the sensor in radians (read-only)."""
        return self.c_ovrTrackerDesc.FrustumVFovInRadians

    @property
    def nearZ(self):
        """Near clipping plane of the sensor frustum in meters (read-only)."""
        return self.c_ovrTrackerDesc.FrustumNearZInMeters

    @property
    def farZ(self):
        """Far clipping plane of the sensor frustum in meters (read-only)."""
        return self.c_ovrTrackerDesc.FrustumFarZInMeters


cdef class LibOVRHmdInfo(object):
    """Class for general HMD information and capabilities. An instance of this
    class is returned by calling :func:`~psychxr.libovr.getHmdInfo`.

    """
    cdef capi.ovrHmdDesc* c_data
    cdef bint ptr_owner

    def __init__(self):
        self.newStruct()

    def __cinit__(self):
        self.ptr_owner = False

    @staticmethod
    cdef LibOVRHmdInfo fromPtr(capi.ovrHmdDesc* ptr, bint owner=False):
        # bypass __init__ if wrapping a pointer
        cdef LibOVRHmdInfo wrapper = LibOVRHmdInfo.__new__(LibOVRHmdInfo)
        wrapper.c_data = ptr
        wrapper.ptr_owner = owner

        return wrapper

    cdef void newStruct(self):
        if self.c_data is not NULL:  # already allocated, __init__ called twice?
            return

        cdef capi.ovrHmdDesc* _ptr = <capi.ovrHmdDesc*>PyMem_Malloc(
            sizeof(capi.ovrHmdDesc))

        if _ptr is NULL:
            raise MemoryError

        self.c_data = _ptr
        self.ptr_owner = True

    def __dealloc__(self):
        if self.c_data is not NULL and self.ptr_owner is True:
            PyMem_Free(self.c_data)
            self.c_data = NULL

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
        cdef libovr_math.FovPort fov_left = \
            <libovr_math.FovPort>self.c_data[0].DefaultEyeFov[0]
        cdef libovr_math.FovPort fov_right = \
            <libovr_math.FovPort>self.c_data[0].DefaultEyeFov[1]

        cdef libovr_math.FovPort fov_max = libovr_math.FovPort.Max(
            <libovr_math.FovPort>fov_left, <libovr_math.FovPort>fov_right)

        cdef float tan_half_fov_horz = maxf(fov_max.LeftTan, fov_max.RightTan)
        cdef float tan_half_fov_vert = maxf(fov_max.DownTan, fov_max.UpTan)

        cdef capi.ovrFovPort fov_both
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


# cdef class LibOVRHapticsBuffer(object):
#     """Class for storing haptics buffer data."""
#     cdef capi.ovrHapticsBuffer c_data
#
#     def __init__(self, *args, **kwargs):
#         pass
#
#     def __cinit__(self, np.ndarray[np.float32, ndim=1] samples):
#
#         cdef int numSamples = <int>samples.shape[0]
#         if numSamples > capi.OVR_HAPTICS_BUFFER_SAMPLES_MAX:
#             raise ValueError(
#                 "Too many buffer samples, must be <{}.".format(
#                     capi.OVR_HAPTICS_BUFFER_SAMPLES_MAX))
#
#         self.c_data.Samples = <void>&samples[0]
#         self.c_data.SamplesCount = numSamples
#
#     @property
#     def samplesCount(self):
#         return self.c_data.SamplesCount


cdef class LibOVRSessionStatus(object):
    """Class for storing session status information. An instance of this class
    is returned when :func:`getSessionStatus` is called.

    One can check if there was a status change between calls of
    :func:`getSessionStatus` by using the ``==`` and ``!=`` operators on the
    returned :py:class:`LibOVRSessionStatus` instances.

    """
    cdef capi.ovrSessionStatus c_data

    def __init__(self):
        """
        Attributes
        ----------
        isVisible : bool
        hmdPresent : bool
        hmdMounted : bool
        displayLost : bool
        shouldQuit : bool
        shouldRecenter : bool
        hasInputFocus : bool
        overlayPresent : bool
        depthRequested : bool

        """
        pass

    def __cinit__(self):
        pass

    def __eq__(self, LibOVRSessionStatus other):
        """Equality test between status objects. Use this to check if the status
        is unchanged between two :func:`getSessionStatus` calls.

        """
        return (self.c_data.IsVisible == other.c_data.IsVisible and
               self.c_data.HmdPresent == other.c_data.HmdPresent and
               self.c_data.hmdMounted == other.c_data.hmdMounted and
               self.c_data.displayLost == other.c_data.displayLost and
               self.c_data.shouldQuit == other.c_data.shouldQuit and
               self.c_data.shouldRecenter == other.c_data.shouldRecenter and
               self.c_data.hasInputFocus == other.c_data.hasInputFocus and
               self.c_data.overlayPresent == other.c_data.overlayPresent and
               self.c_data.depthRequested == other.c_data.depthRequested)

    def __ne__(self, LibOVRSessionStatus other):
        """Equality test between status objects. Use this to check if the status
        differs between two :func:`getSessionStatus` calls.

        """
        return not (self.c_data.IsVisible == other.c_data.IsVisible and
               self.c_data.HmdPresent == other.c_data.HmdPresent and
               self.c_data.hmdMounted == other.c_data.hmdMounted and
               self.c_data.displayLost == other.c_data.displayLost and
               self.c_data.shouldQuit == other.c_data.shouldQuit and
               self.c_data.shouldRecenter == other.c_data.shouldRecenter and
               self.c_data.hasInputFocus == other.c_data.hasInputFocus and
               self.c_data.overlayPresent == other.c_data.overlayPresent and
               self.c_data.depthRequested == other.c_data.depthRequested)

    @property
    def isVisible(self):
        """``True`` the application has focus and visible in the HMD."""
        return self.c_data.IsVisible == capi.ovrTrue

    @property
    def hmdPresent(self):
        """``True`` if the HMD is present."""
        return self.c_data.HmdPresent == capi.ovrTrue

    @property
    def hmdMounted(self):
        """``True`` if the HMD is being worn on the user's head."""
        return self.c_data.HmdMounted == capi.ovrTrue

    @property
    def displayLost(self):
        """``True`` the the display was lost.

        If occurs, the HMD was disconnected and the current session is invalid.
        You need to destroy all resources associated with current session and
        call :func:`create` again. Alternatively, you can raise an error and
        shutdown the application.
        """
        return self.c_data.DisplayLost == capi.ovrTrue

    @property
    def shouldQuit(self):
        """``True`` the application was signaled to quit.

        This can occur if the user requests the application exit through the
        system UI menu. You can ignore this flag if needed.
        """
        return self.c_data.ShouldQuit == capi.ovrTrue

    @property
    def shouldRecenter(self):
        """``True`` if the application was signaled to recenter.

        This happens when the user requests the application recenter the VR
        scene on their current physical location through the system UI. You can
        ignore this request or clear it by calling
        :func:`clearShouldRecenterFlag`.

        """
        return self.c_data.ShouldRecenter == capi.ovrTrue

    @property
    def hasInputFocus(self):
        """``True`` if the application has input focus.

        If the application has focus, the statistics presented by the
        performance HUD will reflect the current application's frame statistics.

        """
        return self.c_data.HasInputFocus == capi.ovrTrue

    @property
    def overlayPresent(self):
        """``True`` if the system UI is visible."""
        return self.c_data.OverlayPresent == capi.ovrTrue

    @property
    def depthRequested(self):
        """``True`` if the a depth texture is requested.

        Notes
        -----
        * This feature is currently unused by PsychXR.

        """
        return self.c_data.DepthRequested == capi.ovrTrue


# ------------------------------------------------------------------------------
# Functions
#

def success(int result):
    """Check if an API return indicates success.

    Returns
    -------
    bool
        ``True`` if API call was an successful (`result` > 0).

    """
    return <bint>capi.OVR_SUCCESS(result)


def unqualifiedSuccess(int result):
    """Check if an API return indicates unqualified success.

    Returns
    -------
    bool
        ``True`` if API call was an unqualified success (`result` == 0).

    """
    return <bint>capi.OVR_UNQUALIFIED_SUCCESS(result)


def failure(int result):
    """Check if an API return indicates failure (error).

    Returns
    -------
    bool
        ``True`` if API call returned an error (`result` < 0).

    """
    return <bint>capi.OVR_FAILURE(result)


def getBool(bytes propertyName, bint defaultVal=False):
    """Read a LibOVR boolean property.

    Parameters
    ----------
    propertyName : bytes
        Name of the property to get.
    defaultVal : bool, optional
        Return value if the property could not be set. The default value is
        ``False``.

    Returns
    -------
    bool
        Value of the property. Returns `defaultVal` if the property does not
        exist.

    """
    global _ptrSession
    cdef capi.ovrBool val = capi.ovrTrue if defaultVal else capi.ovrFalse

    cdef capi.ovrBool to_return = capi.ovr_GetBool(
        _ptrSession,
        propertyName,
        defaultVal
    )

    return to_return == capi.ovrTrue


def setBool(bytes propertyName, bint value=True):
    """Write a LibOVR boolean property.

    Parameters
    ----------
    propertyName : bytes
        Name of the property to set.
    value : bool
        Value to write.

    Returns
    -------
    bool
        ``True`` if the property was set successfully, ``False`` if the property
        was read-only or does not exist.

    """
    global _ptrSession
    cdef capi.ovrBool val = capi.ovrTrue if value else capi.ovrFalse

    cdef capi.ovrBool to_return = capi.ovr_SetBool(
        _ptrSession,
        propertyName,
        val
    )

    return to_return == capi.ovrTrue


def getInt(bytes propertyName, int defaultVal=0):
    """Read a LibOVR integer property.

    Parameters
    ----------
    propertyName : bytes
        Name of the property to get.
    defaultVal : int, optional
        Return value if the property could not be set.

    Returns
    -------
    int
        Value of the property. Returns `defaultVal` if the property does not
        exist.

    """
    global _ptrSession

    cdef int to_return = capi.ovr_GetInt(
        _ptrSession,
        propertyName,
        defaultVal
    )

    return to_return


def setInt(bytes propertyName, int value):
    """Write a LibOVR integer property.

    Parameters
    ----------
    propertyName : bytes
        Name of the property to set.
    value : int
        Value to write.

    Returns
    -------
    bool
        ``True`` if the property was set successfully, ``False`` if the property
        was read-only or does not exist.

    Examples
    --------

    Set the performance HUD mode to show summary information::

        setInt(PERF_HUD_MODE, PERF_HUD_PERF_SUMMARY)

    Switch off the performance HUD::

        setInt(PERF_HUD_MODE, PERF_OFF)

    """
    global _ptrSession

    cdef capi.ovrBool to_return = capi.ovr_SetInt(
        _ptrSession,
        propertyName,
        value
    )

    return to_return == capi.ovrTrue


def getFloat(bytes propertyName, float defaultVal=0.0):
    """Read a LibOVR floating point number property.

    Parameters
    ----------
    propertyName : bytes
        Name of the property to get.
    defaultVal : float, optional
        Return value if the property could not be set. Returns 0.0 if not
        specified.

    Returns
    -------
    float
        Value of the property. Returns `defaultVal` if the property does not
        exist.

    """
    global _ptrSession

    cdef float to_return = capi.ovr_GetFloat(
        _ptrSession,
        propertyName,
        defaultVal)

    return to_return


def setFloat(bytes propertyName, float value):
    """Write a LibOVR floating point number property.

    Parameters
    ----------
    propertyName : bytes
        Name of the property to set.
    value : float
        Value to write.

    Returns
    -------
    bool
        ``True`` if the property was set successfully, ``False`` if the property
        was read-only or does not exist.

    """
    global _ptrSession

    cdef capi.ovrBool to_return = capi.ovr_SetFloat(
        _ptrSession,
        propertyName,
        value)

    return to_return == capi.ovrTrue


def setFloatArray(bytes propertyName, np.ndarray[np.float32_t, ndim=1] values):
    """Write a LibOVR floating point number property.

    Parameters
    ----------
    propertyName : bytes
        Name of the property to set.
    values : ndarray
        Value to write, must be 1-D and have dtype=float32.

    Returns
    -------
    bool
        ``True`` if the property was set successfully, ``False`` if the property
        was read-only or does not exist.

    Examples
    --------

    Set the position of the stereo debug guide::

        guidePos = numpy.asarray([0., 0., -10.0], dtype=np.float32)
        setFloatArray(DEBUG_HUD_STEREO_GUIDE_POSITION, guidePos)

    """
    global _ptrSession

    cdef Py_ssize_t valuesCapacity = len(values)
    cdef capi.ovrBool to_return = capi.ovr_SetFloatArray(
        _ptrSession,
        propertyName,
        &values[0],
        <unsigned int>valuesCapacity)

    return to_return == capi.ovrTrue


def getFloatArray(bytes propertyName, np.ndarray[np.float32_t, ndim=1] values not None):
    """Read a LibOVR float array property.

    Parameters
    ----------
    propertyName : bytes
        Name of the property to get.
    values : ndarray
        Output array array for values, must be 1-D and have dtype=float32.

    Returns
    -------
    int
        Number of values successfully read from the property.

    Examples
    --------

    Get the position of the stereo debug guide::

        guidePos = numpy.zeros((3,), dtype=np.float32)  # array to write to
        result = getFloatArray(DEBUG_HUD_STEREO_GUIDE_POSITION, guidePos)

        # check if the array we specified was long enough to store the values
        if result <= len(guidePos):
            # success

    """
    global _ptrSession

    cdef Py_ssize_t valuesCapacity = len(values)
    cdef unsigned int to_return = capi.ovr_GetFloatArray(
        _ptrSession,
        propertyName,
        &values[0],
        <unsigned int>valuesCapacity)

    return to_return


def setString(bytes propertyName, object value):
    """Write a LibOVR floating point number property.

    Parameters
    ----------
    propertyName : bytes
        Name of the property to set.
    value : str or bytes
        Value to write.

    Returns
    -------
    bool
        ``True`` if the property was set successfully, ``False`` if the property
        was read-only or does not exist.

    """
    global _ptrSession

    cdef object value_in = None
    if isinstance(value, str):
        value_in = value.encode('UTF-8')
    elif isinstance(value, bytes):
        value_in = value
    else:
        raise TypeError("Default value must be type 'str' or 'bytes'.")

    cdef capi.ovrBool to_return = capi.ovr_SetString(
        _ptrSession,
        propertyName,
        value_in)

    return to_return == capi.ovrTrue


def getString(bytes propertyName, object defaultVal=''):
    """Read a LibOVR string property.

    Parameters
    ----------
    propertyName : bytes
        Name of the property to get.
    defaultVal : str, bytes, optional
        Return value if the property could not be set. Returns 0.0 if not
        specified.

    Returns
    -------
    str
        Value of the property. Returns `defaultVal` if the property does not
        exist.

    Notes
    -----

    * Strings passed to this function are converted to bytes before being passed
      to ``OVR::ovr_GetString``.

    """
    global _ptrSession

    cdef object value_in = None
    if isinstance(defaultVal, str):
        value_in = defaultVal.encode('UTF-8')
    elif isinstance(defaultVal, bytes):
        value_in = defaultVal
    else:
        raise TypeError("Default value must be type 'str' or 'bytes'.")

    cdef const char* to_return = capi.ovr_GetString(
        _ptrSession,
        propertyName,
        value_in)

    return to_return.decode('UTF-8')


def isOculusServiceRunning(int timeoutMs=100):
    """Check if the Oculus Runtime is loaded and running.

    Parameters
    ----------
    timeoutMS : int
        Timeout in milliseconds.

    Returns
    -------
    bool
        True if the Oculus background service is running.

    """
    cdef capi.ovrDetectResult result = capi.ovr_Detect(
        timeoutMs)

    return <bint>result.IsOculusServiceRunning


def isHmdConnected(int timeoutMs=100):
    """Check if an HMD is connected.

    Parameters
    ----------
    timeoutMs : int
        Timeout in milliseconds.

    Returns
    -------
    bool
        True if a LibOVR compatible HMD is connected.

    """
    cdef capi.ovrDetectResult result = capi.ovr_Detect(
        timeoutMs)

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
        Return code of the LibOVR API call ``OVR::ovr_Initialize``. Returns
        ``SUCCESS`` if completed without errors. In the event of an
        error, possible return values are:

        * ``ERROR_INITIALIZE``: Initialization error.
        * ``ERROR_LIB_LOAD``:  Failed to load LibOVRRT.
        * ``ERROR_LIB_VERSION``:  LibOVRRT version incompatible.
        * ``ERROR_SERVICE_CONNECTION``:  Cannot connect to OVR service.
        * ``ERROR_SERVICE_VERSION``: OVR service version is incompatible.
        * ``ERROR_INCOMPATIBLE_OS``: Operating system version is incompatible.
        * ``ERROR_DISPLAY_INIT``: Unable to initialize the HMD.
        * ``ERROR_SERVER_START``:  Cannot start a server.
        * ``ERROR_REINITIALIZATION``: Reinitialized with a different version.

    """
    cdef int32_t flags = capi.ovrInit_RequestVersion
    if focusAware is True:
        flags |= capi.ovrInit_FocusAware

    #if debug is True:
    #    flags |= capi.ovrInit_Debug
    global _initParams
    _initParams.Flags = flags
    _initParams.RequestedMinorVersion = capi.OVR_MINOR_VERSION
    _initParams.LogCallback = NULL  # not used yet
    _initParams.ConnectionTimeoutMS = <uint32_t>connectionTimeout
    cdef capi.ovrResult result = capi.ovr_Initialize(
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
        Result of the ``OVR::ovr_Create`` API call. A session was successfully
        created if the result is ``SUCCESS``.

    """
    global _ptrSession
    global _gfxLuid
    global _eyeLayer
    global _hmdDesc
    global _eyeRenderDesc

    result = capi.ovr_Create(&_ptrSession, &_gfxLuid)
    check_result(result)
    if capi.OVR_FAILURE(result):
        return result  # failed to create session, return error code

    # if we got to this point, everything should be fine
    # get HMD descriptor
    _hmdDesc = capi.ovr_GetHmdDesc(_ptrSession)

    # configure the eye render descriptor to use the recommended FOV, this
    # can be changed later
    cdef Py_ssize_t i = 0
    for i in range(capi.ovrEye_Count):
        _eyeRenderDesc[i] = capi.ovr_GetRenderDesc(
            _ptrSession,
            <capi.ovrEyeType>i,
            _hmdDesc.DefaultEyeFov[i])

        _eyeLayer.Fov[i] = _eyeRenderDesc[i].Fov

    return result


def destroyTextureSwapChain(int swapChain):
    """Destroy a texture swap chain.

    Once destroyed, the swap chain's resources will be freed.

    Parameters
    ----------
    swapChain : int
        Swap chain identifier/index.

    """
    global _ptrSession
    global _swapChains
    capi.ovr_DestroyTextureSwapChain(_ptrSession, _swapChains[swapChain])
    _swapChains[swapChain] = NULL


def destroyMirrorTexture():
    """Destroy the mirror texture.
    """
    global _ptrSession
    global _mirrorTexture
    if _mirrorTexture != NULL:
        capi.ovr_DestroyMirrorTexture(_ptrSession, _mirrorTexture)


def destroy():
    """Destroy a session.

    Must be called after every successful :func:`create` call. Calling destroy
    will invalidate the current session and all resources must be freed and
    re-created.

    """
    global _ptrSession
    global _eyeLayer
    # null eye textures in eye layer
    _eyeLayer.ColorTexture[0] = _eyeLayer.ColorTexture[1] = NULL

    # destroy the current session and shutdown
    capi.ovr_Destroy(_ptrSession)


def shutdown():
    """End the current session.

    Clean-up routines are executed that destroy all swap chains and mirror
    texture buffers, afterwards control is returned to Oculus Home. This
    must be called after every successful :func:`initialize` call.

    """
    capi.ovr_Shutdown()


def getGraphicsLUID():
    """The graphics device LUID.

    Returns
    -------
    str
        Reserved graphics LUID.

    """
    global _gfxLuid
    return _gfxLuid.Reserved.decode('utf-8')


def setHighQuality(bint enable):
    """Enable high quality mode.

    This enables 4x anisotropic sampling by the compositor to reduce the
    appearance of high-frequency artifacts in the visual periphery due to
    distortion.

    Parameters
    ----------
    enable : bool
        Enable high-quality mode.

    """
    global _eyeLayer
    if enable:
        _eyeLayer.Header.Flags |= capi.ovrLayerFlag_HighQuality
    else:
        _eyeLayer.Header.Flags &= ~capi.ovrLayerFlag_HighQuality


def setHeadLocked(bint enable):
    """True when head-locked mode is enabled.

    This is disabled by default when a session is started. Enable this if you
    are considering to use custom head poses.

    Parameters
    ----------
    enable : bool
        Enable head-locking when rendering to the eye render layer.

    """
    global _eyeLayer
    if enable:
        _eyeLayer.Header.Flags |= capi.ovrLayerFlag_HeadLocked
    else:
        _eyeLayer.Header.Flags &= ~capi.ovrLayerFlag_HeadLocked


def getPixelsPerTanAngleAtCenter(int eye):
    """Get pixels per tan angle (=1) at the center of the display.

    Values reflect the FOVs set by the last call to :func:`setEyeRenderFov` (or
    else the default FOVs will be used.)

    Parameters
    ----------
    eye : int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.

    Returns
    -------
    tuple of floats
        Pixels per tan angle at the center of the screen.

    """
    global _eyeRenderDesc

    cdef capi.ovrVector2f toReturn = \
        _eyeRenderDesc[eye].PixelsPerTanAngleAtCenter

    return toReturn.x, toReturn.y


def getTanAngleToRenderTargetNDC(int eye, object tanAngle):
    """Convert FOV tan angle to normalized device coordinates (NDC).

    Parameters
    ----------
    eye : int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.
    tanAngle : tuple, list of float or ndarray
        Horizontal and vertical tan angles [X, Y] from display center.

    Returns
    -------
    tuple
        NDC coordinates X, Y [-1, 1].

    """
    global _eyeRenderDesc

    cdef libovr_math.Vector2f vecIn
    vecIn.x = tanAngle[0]
    vecIn.y = tanAngle[1]

    cdef libovr_math.Vector2f toReturn = \
        (<libovr_math.FovPort>_eyeRenderDesc[eye].Fov).TanAngleToRendertargetNDC(
            vecIn)

    return toReturn.x, toReturn.y


# def getPixelsPerDegree(int eye):
#     """Get pixels per degree at the center of the display.
#
#     Values reflect the FOVs set by the last call to :func:`setEyeRenderFov` (or
#     else the default FOVs will be used.)
#
#     Parameters
#     ----------
#     eye: int
#         Eye index. Use either :data:`EYE_LEFT` or
#         :data:`EYE_RIGHT`.
#
#     Returns
#     -------
#     tuple of floats
#         Pixels per degree at the center of the screen.
#
#     """
#     global _eyeRenderDesc
#
#     cdef capi.ovrVector2f pixelsPerTanAngle = \
#         _eyeRenderDesc[eye].PixelsPerTanAngleAtCenter
#
#     # tan(angle)=1 -> 45 deg
#     cdef float horzPixelPerDeg = pixelsPerTanAngle.x / 45.0
#     cdef float vertPixelPerDeg = pixelsPerTanAngle.y / 45.0
#
#     return horzPixelPerDeg, vertPixelPerDeg


def getDistortedViewport(int eye):
    """Get the distorted viewport.

    You must call :func:`setEyeRenderFov` first for values to be valid.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.

    """
    cdef capi.ovrRecti distVp = _eyeRenderDesc[eye].DistortedViewport

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
    (recommended) FOVs after :func:`create` is called.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.

    Returns
    -------
    ndarray of float
        Eye FOV tangent angles [UpTan, DownTan, LeftTan, RightTan].

    Examples
    --------
    Getting the tangent angles::

        leftFov = getEyeRenderFOV(EYE_LEFT)
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

    By default, this function will return the default FOVs after :func:`create`
    is called (see :py:attr:`LibOVRHmdInfo.defaultEyeFov`). You can override
    these values using :py:attr:`LibOVRHmdInfo.maxEyeFov` and
    :py:attr:`LibOVRHmdInfo.symmetricEyeFov`, or with custom values (see
    Examples below).

    Parameters
    ----------
    eye : int
        Eye index. Values are ``EYE_LEFT`` and ``EYE_RIGHT``.
    fov : tuple, list or ndarray of floats
        Eye FOV tangent angles [UpTan, DownTan, LeftTan, RightTan].

    Examples
    --------

    Setting eye render FOVs to symmetric (needed for mono rendering)::

        leftFov, rightFov = getSymmetricEyeFOVs()
        setEyeRenderFOV(EYE_LEFT, leftFov)
        setEyeRenderFOV(EYE_RIGHT, rightFov)

    Using custom values::

        # Up, Down, Left, Right tan angles
        setEyeRenderFOV(EYE_LEFT, [1.0, -1.0, -1.0, 1.0])

    """
    global _ptrSession
    global _eyeRenderDesc
    global _eyeLayer

    cdef capi.ovrFovPort fov_in
    fov_in.UpTan = <float>fov[0]
    fov_in.DownTan = <float>fov[1]
    fov_in.LeftTan = <float>fov[2]
    fov_in.RightTan = <float>fov[3]

    _eyeRenderDesc[<int>eye] = capi.ovr_GetRenderDesc(
        _ptrSession,
        <capi.ovrEyeType>eye,
        fov_in)

    # set in eye layer too
    _eyeLayer.Fov[eye] = _eyeRenderDesc[eye].Fov


def getEyeAspectRatio(int eye):
    """Get the aspect ratio of an eye.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.

    Returns
    -------
    float
        Aspect ratio of the eye's FOV (width / height).

    """
    cdef libovr_math.FovPort fovPort = \
        <libovr_math.FovPort>_eyeRenderDesc[eye].Fov

    return (fovPort.LeftTan + fovPort.RightTan) / \
           (fovPort.UpTan + fovPort.DownTan)


def getEyeHorizontalFovRadians(int eye):
    """Get the angle of the horizontal field-of-view (FOV) for a given eye.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.

    Returns
    -------
    float
        Horizontal FOV of a given eye in radians.

    """
    cdef libovr_math.FovPort fovPort = \
        <libovr_math.FovPort>_eyeRenderDesc[eye].Fov

    return fovPort.GetHorizontalFovRadians()


def getEyeVerticalFovRadians(int eye):
    """Get the angle of the vertical field-of-view (FOV) for a given eye.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.

    Returns
    -------
    float
        Vertical FOV of a given eye in radians.

    """
    cdef libovr_math.FovPort fovPort = \
        <libovr_math.FovPort>_eyeRenderDesc[eye].Fov

    return fovPort.GetVerticalFovRadians()


def getEyeFocalLength(int eye):
    """Get the focal length of the eye's frustum.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.

    Returns
    -------
    float
        Focal length in meters.

    """
    return 1.0 / tan(getEyeHorizontalFovRadians(eye) / 2.0)


def calcEyeBufferSize(int eye, float texelsPerPixel=1.0):
    """Get the recommended buffer (texture) sizes for eye buffers.

    Should be called after :func:`setEyeRenderFov`. Returns buffer resolutions in
    pixels (w, h). The values can be used when configuring a framebuffer or swap
    chain for rendering.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.
    texelsPerPixel : float, optional
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

    Getting the buffer size for the swap chain::

        # get HMD info
        hmdInfo = getHmdInfo()

        # eye FOVs must be set first!
        leftFov, rightFov = hmdInfo.defaultEyeFov
        setEyeRenderFov(EYE_LEFT, leftFov)
        setEyeRenderFov(EYE_RIGHT, rightFov)

        leftBufferSize, rightBufferSize = calcEyeBufferSize()
        leftW leftH = leftBufferSize
        rightW, rightH = rightBufferSize
        # combined size if using a single texture buffer for both eyes
        bufferW, bufferH = leftW + rightW, max(leftH, rightH)

        # create a swap chain
        createTextureSwapChainGL(TEXTURE_SWAP_CHAIN0, bufferW, bufferH)

    Notes
    -----
    * This function returns the recommended texture resolution for a specified
      eye. If you are using a single buffer for both eyes, that buffer should be
      as wide as the combined width of both eye's values.

    """
    global _ptrSession
    global _eyeRenderDesc

    cdef capi.ovrSizei bufferSize = capi.ovr_GetFovTextureSize(
        _ptrSession,
        <capi.ovrEyeType>0,
        _eyeRenderDesc[0].Fov,
        <float>texelsPerPixel)

    return bufferSize.w, bufferSize.h


def getLayerEyeFovFlags():
    """Get header flags for the render layer.

    Returns
    -------
    unsigned int
        Flags from ``OVR::ovrLayerEyeFov.Header.Flags``.

    """
    global _eyeLayer
    return <unsigned int>_eyeLayer.Header.Flags


def setLayerEyeFovFlags(unsigned int flags):
    """Set header flags for the render layer.

    Parameters
    ----------
    flags : int
        Flags to set. Flags can be ORed together to apply multiple settings.
        Valid values for flags are:

        * ``LAYER_FLAG_HIGH_QUALITY`` : Enable high quality mode which tells the
          compositor to use 4x anisotropic filtering when sampling.
        * ``LAYER_FLAG_TEXTURE_ORIGIN_AT_BOTTOM_LEFT`` : Tell the compositor the
          texture origin is at the bottom left, required for using OpenGL
          textures.
        * ``LAYER_FLAG_HEAD_LOCKED`` : Enable head locking, which forces the
          render layer transformations to remain head referenced.

    Notes
    -----
    * ``LAYER_FLAG_HIGH_QUALITY`` and
      ``LAYER_FLAG_TEXTURE_ORIGIN_AT_BOTTOM_LEFT`` are recommended settings and
      are enabled by default.

    """
    global _eyeLayer
    _eyeLayer.Header.Flags = <capi.ovrLayerFlags>flags


def getTextureSwapChainLengthGL(int swapChain):
    """Get the length of a specified swap chain.

    Parameters
    ----------
    swapChain : int
        Swap chain handle to query. Must be a swap chain initialized by a
        previous call to :func:`createTextureSwapChainGL`.

    Returns
    -------
    tuple of int
        Result of the ``ovr_GetTextureSwapChainLength`` API call and the
        length of that swap chain.

    See Also
    --------
    getTextureSwapChainCurrentIndex : Get the current swap chain index.
    getTextureSwapChainBufferGL : Get the current OpenGL swap chain buffer.

    Examples
    --------

    Get the swap chain length for the previously created
    ``TEXTURE_SWAP_CHAIN0``::

        result, length = getTextureSwapChainLengthGL(TEXTURE_SWAP_CHAIN0)

    """
    cdef int outLength
    cdef capi.ovrResult result = 0
    global _swapChains
    global _ptrSession
    global _eyeLayer

    # check if there is a swap chain in the slot
    if _eyeLayer.ColorTexture[swapChain] == NULL:
        raise RuntimeError(
            "Cannot get swap chain length, NULL eye buffer texture.")

    # get the current texture index within the swap chain
    result = capi.ovr_GetTextureSwapChainLength(
        _ptrSession, _swapChains[swapChain], &outLength)

    return result, outLength


def getTextureSwapChainCurrentIndex(int swapChain):
    """Get the current buffer index within the swap chain.

    Parameters
    ----------
    swapChain : int
        Swap chain handle to query. Must be a swap chain initialized by a
        previous call to :func:`createTextureSwapChainGL`.

    Returns
    -------
    tuple of int
        Result of the ``OVR::ovr_GetTextureSwapChainCurrentIndex`` API call and
        the index of the buffer.

    See Also
    --------
    getTextureSwapChainLengthGL : Get the length of a swap chain.
    getTextureSwapChainBufferGL : Get the current OpenGL swap chain buffer.

    """
    cdef int current_idx = 0
    global _swapChains
    global _eyeLayer
    global _ptrSession

    # check if there is a swap chain in the slot
    if _eyeLayer.ColorTexture[swapChain] == NULL:
        raise RuntimeError(
            "Cannot get buffer ID, NULL eye buffer texture.")

    # get the current texture index within the swap chain
    cdef capi.ovrResult result = capi.ovr_GetTextureSwapChainCurrentIndex(
        _ptrSession, _swapChains[swapChain], &current_idx)

    return result, current_idx


def getTextureSwapChainBufferGL(int swapChain, int index):
    """Get the texture buffer as an OpenGL name at a specific index in the
    swap chain for a given swapChain.

    Parameters
    ----------
    swapChain : int
        Swap chain handle to query. Must be a swap chain initialized by a
        previous call to :func:`createTextureSwapChainGL`.
    index : int
        Index within the swap chain to retrieve its OpenGL texture name.

    Returns
    -------
    tuple of ints
        Result of the ``OVR::ovr_GetTextureSwapChainBufferGL`` API call and the
        OpenGL texture buffer name. A OpenGL buffer name is invalid when 0,
        check the returned API call result for an error condition.

    Examples
    --------

    Get the OpenGL texture buffer name associated with the swap chain index::

        # get the current available index
        swapChain = TEXTURE_SWAP_CHAIN0
        result, currentIdx = getSwapChainCurrentIndex(swapChain)

        # get the OpenGL buffer name
        result, texId = getTextureSwapChainBufferGL(swapChain, currentIdx)

        # bind the texture
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D, texId, 0)

    """
    cdef unsigned int tex_id = 0  # OpenGL texture handle
    global _swapChains
    global _eyeLayer
    global _ptrSession

    # get the next available texture ID from the swap chain
    cdef capi.ovrResult result = capi.ovr_GetTextureSwapChainBufferGL(
        _ptrSession, _swapChains[swapChain], index, &tex_id)

    return result, tex_id


def createTextureSwapChainGL(int swapChain, int width, int height, int textureFormat=FORMAT_R8G8B8A8_UNORM_SRGB, int levels=1):
    """Create a texture swap chain for eye image buffers.

    Parameters
    ----------
    swapChain : int
        Swap chain handle to initialize, usually :data:`SWAP_CHAIN*`.
    width : int
        Width of texture in pixels.
    height : int
        Height of texture in pixels.
    textureFormat : int
        Texture format to use. Valid color texture formats are:

        * ``FORMAT_R8G8B8A8_UNORM``
        * ``FORMAT_R8G8B8A8_UNORM_SRGB``
        * ``FORMAT_R16G16B16A16_FLOAT``
        * ``FORMAT_R11G11B10_FLOAT``

        Depth texture formats:

        * ``FORMAT_D16_UNORM``
        * ``FORMAT_D24_UNORM_S8_UINT``
        * ``FORMAT_D32_FLOAT``

    Other Parameters
    ----------------
    levels : int
        Mip levels to use, default is 1.

    Returns
    -------
    int
        The result of the ``OVR::ovr_CreateTextureSwapChainGL`` API call.

    Examples
    --------

    Create a texture swap chain::

        result = createTextureSwapChainGL(TEXTURE_SWAP_CHAIN0,
            texWidth, texHeight, FORMAT_R8G8B8A8_UNORM)
        # set the swap chain for each eye buffer
        for eye in range(EYE_COUNT):
            setEyeColorTextureSwapChain(eye, TEXTURE_SWAP_CHAIN0)

    """
    global _swapChains
    global _ptrSession

    if _swapChains[swapChain] != NULL:
        raise ValueError("Swap chain TEXTURE_SWAP_CHAIN{} already "
                         "initialized!".format(swapChain))

    # configure the texture
    cdef capi.ovrTextureSwapChainDesc swapConfig
    swapConfig.Type = capi.ovrTexture_2D
    swapConfig.Format = <capi.ovrTextureFormat>textureFormat
    swapConfig.ArraySize = 1
    swapConfig.Width = <int>width
    swapConfig.Height = <int>height
    swapConfig.MipLevels = <int>levels
    swapConfig.SampleCount = 1
    swapConfig.StaticImage = capi.ovrFalse  # always buffered
    swapConfig.MiscFlags = capi.ovrTextureMisc_None
    swapConfig.BindFlags = capi.ovrTextureBind_None

    # create the swap chain
    cdef capi.ovrResult result = \
        capi.ovr_CreateTextureSwapChainGL(
            _ptrSession,
            &swapConfig,
            &_swapChains[swapChain])

    #_eyeLayer.ColorTexture[swapChain] = _swapChains[swapChain]

    return result


def setEyeColorTextureSwapChain(int eye, int swapChain):
    """Set the color texture swap chain for a given eye.

    Should be called after a successful :func:`createTextureSwapChainGL` call
    but before any rendering is done.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.
    swapChain : int
        Swap chain handle to query. Must be a swap chain initialized by a
        previous call to :func:`createTextureSwapChainGL`.

    See Also
    --------
    createTextureSwapChainGL : Create a OpenGL buffer swap chain.

    Examples
    --------

    Associate the swap chain with both eyes (single buffer for stereo views)::

        setEyeColorTextureSwapChain(EYE_LEFT, TEXTURE_SWAP_CHAIN0)
        setEyeColorTextureSwapChain(EYE_RIGHT, TEXTURE_SWAP_CHAIN0)

        # same as above but with a loop
        for eye in range(EYE_COUNT):
            setEyeColorTextureSwapChain(eye, TEXTURE_SWAP_CHAIN0)

    Associate a swap chain with each eye (separate buffer for stereo views)::

        setEyeColorTextureSwapChain(EYE_LEFT, TEXTURE_SWAP_CHAIN0)
        setEyeColorTextureSwapChain(EYE_RIGHT, TEXTURE_SWAP_CHAIN1)

        # with a loop ...
        for eye in range(EYE_COUNT):
            setEyeColorTextureSwapChain(eye, TEXTURE_SWAP_CHAIN0 + eye)

    """
    global _swapChains
    global _eyeLayer

    _eyeLayer.ColorTexture[eye] = _swapChains[swapChain]


def createMirrorTexture(int width, int height, int textureFormat=FORMAT_R8G8B8A8_UNORM_SRGB):
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

        * ``FORMAT_R8G8B8A8_UNORM``
        * ``FORMAT_R8G8B8A8_UNORM_SRGB``
        * ``FORMAT_R16G16B16A16_FLOAT``
        * ``FORMAT_R11G11B10_FLOAT``

    Returns
    -------
    int
        Result of API call ``OVR::ovr_CreateMirrorTextureGL``.

    """
    # additional options
    #cdef unsigned int mirror_options = capi.ovrMirrorOption_Default
    # set the mirror texture mode
    #if mirrorMode == 'Default':
    #    mirror_options = <capi.ovrMirrorOptions>capi.ovrMirrorOption_Default
    #elif mirrorMode == 'PostDistortion':
    #    mirror_options = <capi.ovrMirrorOptions>capi.ovrMirrorOption_PostDistortion
    #elif mirrorMode == 'LeftEyeOnly':
    #    mirror_options = <capi.ovrMirrorOptions>capi.ovrMirrorOption_LeftEyeOnly
    #elif mirrorMode == 'RightEyeOnly':
    #    mirror_options = <capi.ovrMirrorOptions>capi.ovrMirrorOption_RightEyeOnly
    #else:
    #    raise RuntimeError("Invalid 'mirrorMode' mode specified.")

    #if include_guardian:
    #    mirror_options |= capi.ovrMirrorOption_IncludeGuardian
    #if include_notifications:
    #    mirror_options |= capi.ovrMirrorOption_IncludeNotifications
    #if include_system_gui:
    #    mirror_options |= capi.ovrMirrorOption_IncludeSystemGui

    # create the descriptor
    cdef capi.ovrMirrorTextureDesc mirrorDesc
    global _ptrSession
    global _mirrorTexture

    mirrorDesc.Format = <capi.ovrTextureFormat>textureFormat
    mirrorDesc.Width = <int>width
    mirrorDesc.Height = <int>height
    mirrorDesc.MiscFlags = capi.ovrTextureMisc_None
    mirrorDesc.MirrorOptions = capi.ovrMirrorOption_Default

    cdef capi.ovrResult result = capi.ovr_CreateMirrorTextureGL(
        _ptrSession, &mirrorDesc, &_mirrorTexture)

    return <int>result


def getMirrorTexture():
    """Mirror texture ID.

    Returns
    -------
    tuple of int
        Result of API call ``OVR::ovr_GetMirrorTextureBufferGL`` and the mirror
        texture ID. A mirror texture ID == 0 is invalid.

    Examples
    --------

    Getting the mirror texture for use::

        # get the mirror texture
        result, mirrorTexId = getMirrorTexture()
        if failure(result):
            # raise error ...

        # bind the mirror texture texture to the framebuffer
        glFramebufferTexture2D(
            GL_READ_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D, mirrorTexId, 0)

    """
    cdef unsigned int mirror_id

    global _ptrSession
    global _mirrorTexture

    if _mirrorTexture == NULL:  # no texture created
        return None

    cdef capi.ovrResult result = \
        capi.ovr_GetMirrorTextureBufferGL(
            _ptrSession,
            _mirrorTexture,
            &mirror_id)

    return <int>result, <unsigned int>mirror_id


def getTrackingState(double absTime, bint latencyMarker=True):
    """Get the current poses of the head and hands.

    Parameters
    ----------
    absTime : `float`
        Absolute time in seconds which the tracking state refers to.
    latencyMarker : `bool`
        Insert a latency marker for motion-to-photon calculation.

    Returns
    -------
    tuple of dict, :class:`LibOVRPose`
        Dictionary of tracking states where keys are
        ``TRACKED_DEVICE_TYPE_HMD``,
        ``TRACKED_DEVICE_TYPE_LTOUCH``, and
        ``TRACKED_DEVICE_TYPE_RTOUCH``. The value referenced by each
        key is a tuple containing a :class:`LibOVRPoseState` and `int` for
        status flags. The second value is :class:`LibOVRPose` with the
        calibrated origin used for tracking.

    Examples
    --------
    Getting the head pose and calculating eye render poses::

        t = hmd.getPredictedDisplayTime()
        trackedPoses, calibratedOrigin = hmd.getTrackingState(t)
        headPoseState, status = trackedPoses[TRACKED_DEVICE_TYPE_HMD]

        # tracking state flags
        flags = STATUS_ORIENTATION_TRACKED | STATUS_ORIENTATION_TRACKED

        # check if tracking
        if (flags & status) == flags:
            hmd.calcEyePose(headPoseState.thePose)  # calculate eye poses

    """
    global _ptrSession
    global _eyeLayer
    global _trackingState

    cdef capi.ovrBool use_marker = \
        capi.ovrTrue if latencyMarker else capi.ovrFalse

    # tracking state object that is actually returned to Python land
    _trackingState = capi.ovr_GetTrackingState(_ptrSession, absTime, use_marker)

    # init objects pointing to tracking state fields
    cdef LibOVRPoseState headPoseState = \
        LibOVRPoseState.fromPtr(&_trackingState.HeadPose)
    cdef LibOVRPoseState leftHandPoseState = \
        LibOVRPoseState.fromPtr(&_trackingState.HandPoses[0])
    cdef LibOVRPoseState rightHandPoseState = \
        LibOVRPoseState.fromPtr(&_trackingState.HandPoses[1])
    cdef LibOVRPose calibratedOrigin = LibOVRPose.fromPtr(
        &_trackingState.CalibratedOrigin)

    # build dictionary of returned values
    cdef dict poseStates = {
        TRACKED_DEVICE_TYPE_HMD:
            (headPoseState, _trackingState.StatusFlags),
        TRACKED_DEVICE_TYPE_LTOUCH:
            (leftHandPoseState, _trackingState.HandStatusFlags[0]),
        TRACKED_DEVICE_TYPE_RTOUCH:
            (rightHandPoseState, _trackingState.HandStatusFlags[1])
    }

    # for computing app photon-to-motion latency
    #_eyeLayer.SensorSampleTime = toReturn.c_data[0].HeadPose.TimeInSeconds

    return poseStates, calibratedOrigin


def getDevicePoses(object deviceTypes, double absTime, bint latencyMarker=True):
    """Get tracked device poses.

    Each pose in the returned array matches the device type at each index
    specified in `deviceTypes`. You need to call this function to get the poses
    for 'objects', which are additional Touch controllers that can be paired and
    tracked in the scene.

    It is recommended that :func:`getTrackingState` is used for obtaining the
    head and hand poses.

    Parameters
    ----------
    deviceTypes : `list` or `tuple` of `int`
        List of device types. Valid device types identifiers are:

        * ``TRACKED_DEVICE_TYPE_HMD`` : The head or HMD.
        * ``TRACKED_DEVICE_TYPE_LTOUCH`` : Left touch controller or hand.
        * ``TRACKED_DEVICE_TYPE_RTOUCH`` : Right touch controller or hand.
        * ``TRACKED_DEVICE_TYPE_TOUCH`` : Both touch controllers.

        Up to four additional touch controllers can be paired and tracked, they
        are assigned as:

        * ``TRACKED_DEVICE_TYPE_OBJECT0``
        * ``TRACKED_DEVICE_TYPE_OBJECT1``
        * ``TRACKED_DEVICE_TYPE_OBJECT2``
        * ``TRACKED_DEVICE_TYPE_OBJECT3``

    absTime : `float`
        Absolute time in seconds poses refer to.
    latencyMarker: `bool`, optional
        Insert a marker for motion-to-photon latency calculation. Set this to
        False if :func:`getTrackingState` was previously called and a latency
        marker was set there.

    Returns
    -------
    tuple
        Return code (`int`) of the ``OVR::ovr_GetDevicePoses`` API call and list 
        of tracked device poses (`list` of `LibOVRPoseState`). If a device 
        cannot be tracked, the return code will be ``ERROR_LOST_TRACKING``.

    Warning
    -------
    If multiple devices were specified with `deviceTypes`, the return code will
    be ``ERROR_LOST_TRACKING`` if ANY of the devices lost tracking.

    Examples
    --------

    Get HMD and touch controller poses::

        deviceTypes = (TRACKED_DEVICE_TYPE_HMD,
                       TRACKED_DEVICE_TYPE_LTOUCH,
                       TRACKED_DEVICE_TYPE_RTOUCH)
        headPose, leftHandPose, rightHandPose = getDevicePoses(
            deviceTypes, absTime)

    """
    # give a success code and empty pose list if an empty list was specified
    if not deviceTypes:
        return capi.ovrSuccess, []

    global _ptrSession
    global _eyeLayer
    #global _devicePoses

    # for computing app photon-to-motion latency
    if latencyMarker:
        _eyeLayer.SensorSampleTime = absTime

    # allocate arrays to store pose types and poses
    cdef int count = <int>len(deviceTypes)
    cdef capi.ovrTrackedDeviceType* devices = \
        <capi.ovrTrackedDeviceType*>PyMem_Malloc(
            count * sizeof(capi.ovrTrackedDeviceType))
    if not devices:
        raise MemoryError("Failed to allocate array 'devices'.")

    cdef int i = 0
    for i in range(count):
        devices[i] = <capi.ovrTrackedDeviceType>deviceTypes[i]

    cdef capi.ovrPoseStatef* devicePoses = \
        <capi.ovrPoseStatef*>PyMem_Malloc(
            count * sizeof(capi.ovrPoseStatef))
    if not devicePoses:
        raise MemoryError("Failed to allocate array 'devicePoses'.")

    # get the device poses
    cdef capi.ovrResult result = capi.ovr_GetDevicePoses(
        _ptrSession,
        devices,
        count,
        absTime,
        devicePoses)

    # build list of device poses
    cdef list outPoses = list()
    cdef LibOVRPoseState thisPose
    for i in range(count):
        thisPose = LibOVRPoseState()  # new
        thisPose.c_data[0] = devicePoses[i]
        outPoses.append(thisPose)

    # free the allocated arrays
    PyMem_Free(devices)
    PyMem_Free(devicePoses)

    return result, outPoses


def calcEyePoses(LibOVRPose headPose):
    """Calculate eye poses using a given pose state.

    Eye poses are derived from the head pose stored in the pose state and
    the HMD to eye poses reported by LibOVR. Calculated eye poses are stored
    and passed to the compositor when :func:`endFrame` is called for additional
    rendering.

    You can access the computed poses via the :func:`getEyeRenderPose` function.

    Parameters
    ----------
    headPose : :py:class:`LibOVRPose`
        Head pose.

    Notes
    -----

    * :func:`getTrackingState` must still be called every frame, even if you
      are specifying your own `headPose`.

    * When specifying a head pose defined by any other means than returned by
      :func:`getTrackingState`. The `headPose` will be incongruent to that
      computed by the LibOVR runtime, causing the render layer to 'slip' during
      composting. To prevent this, you must enable head-locking of the render
      layer by calling :func:`setHeadLocked` if using custom head poses.

    Examples
    --------

    Compute the eye poses from tracker data::

        t = getPredictedDisplayTime()
        trackingState = getTrackingState(t)

        # check if tracking
        if head.orientationTracked and head.positionTracked:
            calcEyePoses(trackingState.headPose.thePose)  # calculate eye poses
        else:
            # do something ...

        # computed render poses appear here
        renderPoseLeft, renderPoseRight = hmd.getEyeRenderPoses()

    Use a custom head pose::

        # note headLocked(True) should be called prior
        headPose = LibOVRPose((0., 1.5, 0.))  # eyes 1.5 meters off the ground
        hmd.calcEyePoses(headPose)  # calculate eye poses

    """
    global _ptrSession
    global _eyeLayer
    global _eyeRenderDesc
    global _eyeViewMatrix
    global _eyeProjectionMatrix
    global _eyeViewProjectionMatrix

    cdef capi.ovrPosef[2] hmdToEyePoses
    hmdToEyePoses[0] = _eyeRenderDesc[0].HmdToEyePose
    hmdToEyePoses[1] = _eyeRenderDesc[1].HmdToEyePose

     # calculate the eye poses
    capi.ovr_CalcEyePoses2(
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
    for eye in range(capi.ovrEye_Count):
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


def getHmdToEyePose(int eye):
    """HMD to eye pose.

    These are the prototype eye poses specified by LibOVR, defined only
    after :func:`create` is called. These poses are referenced to the HMD
    origin. Poses are transformed by calling :func:`calcEyePoses`, updating the
    values returned by :func:`getEyeRenderPose`.

    The horizontal (x-axis) separation of the eye poses are determined by the
    configured lens spacing (slider adjustment). This spacing is supposed to
    correspond to the actual inter-ocular distance (IOD) of the user. You can
    get the IOD used for rendering by adding up the absolute values of the
    x-components of the eye poses, or by multiplying the value of
    :func:`getEyeToNoseDist` by two. Furthermore, the IOD values can be altered,
    prior to calling :func`calcEyePoses`, to override the values specified by
    LibOVR.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.

    Returns
    -------
    `tuple` of :py:class:`LibOVRPose`
        Copy of the HMD to eye pose.

    See Also
    --------
    setHmdToEyePose : Set the HMD to eye pose.

    Examples
    --------
    Get the HMD to eye poses::

        leftPose = getHmdToEyePose(EYE_LEFT)
        rightPose = getHmdToEyePose(EYE_RIGHT)

    """
    global _eyeRenderDesc
    return LibOVRPose.fromPtr(&_eyeRenderDesc[eye].HmdToEyePose)


def setHmdToEyePose(int eye, LibOVRPose eyePose):
    """Set the HMD eye poses.

    This overwrites the values returned by LibOVR and will be used in successive
    calls of :func:`calcEyePoses` to compute eye render poses. Note that the
    poses store the view space translations, not the relative position in the
    scene.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.

    See Also
    --------
    getHmdToEyePose : Get the current HMD to eye pose.

    Examples
    --------
    Set both HMD to eye poses::

        eyePoses = [LibOVRPose((0.035, 0.0, 0.0)), LibOVRPose((-0.035, 0.0, 0.0))]
        for eye in enumerate(eyePoses):
            setHmdToEyePose(eye, eyePoses[eye])

    """
    global _eyeRenderDesc
    _eyeRenderDesc[0].HmdToEyePose = eyePose.c_data[0]


def getEyeRenderPose(int eye):
    """Get eye render poses.

    Pose are those computed by the last :func:`calcEyePoses` call. Returned
    objects are copies of the data stored internally by the session
    instance. These poses are used to derive the view matrix when rendering
    for each eye.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.

    Returns
    -------
    `tuple` of `LibOVRPose`
        Copies of the HMD to eye poses for the left and right eye.

    See Also
    --------
    setEyeRenderPose : Set an eye's render pose.

    Examples
    --------

    Get the eye render poses::

        leftPose = getHmdToEyePose(EYE_LEFT)
        rightPose = getHmdToEyePose(EYE_RIGHT)

    Get the left and right view matrices::

        eyeViewMatrices = []
        for eye in enumerate(EYE_COUNT):
            eyeViewMatrices.append(getHmdToEyePose(eye).asMatrix())

    Same as above, but overwrites existing view matrices::

        # identity 4x4 matrices
        eyeViewMatrices = [
            numpy.identity(4, dtype=numpy.float32),
            numpy.identity(4, dtype=numpy.float32)]
        for eye in range(EYE_COUNT):
            getHmdToEyePose(eye).asMatrix(eyeViewMatrices[eye])

    """
    global _eyeLayer
    return LibOVRPose.fromPtr(&_eyeLayer.RenderPose[eye])


def setEyeRenderPose(int eye, LibOVRPose eyePose):
    """Set eye render pose.

    Setting the eye render pose will update the values returned by
    :func:`getEyeRenderPose`.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.

    See Also
    --------
    getEyeRenderPose : Get an eye's render pose.

    """
    global _eyeLayer
    global _eyeViewMatrix
    global _eyeViewProjectionMatrix

    _eyeLayer.RenderPose[eye] = eyePose.c_data[0]

    # re-compute the eye transformation matrices from poses
    cdef libovr_math.Vector3f pos
    cdef libovr_math.Quatf ori
    cdef libovr_math.Vector3f up
    cdef libovr_math.Vector3f forward
    cdef libovr_math.Matrix4f rm

    pos = <libovr_math.Vector3f>_eyeLayer.RenderPose[eye].Position
    ori = <libovr_math.Quatf>_eyeLayer.RenderPose[eye].Orientation

    if not ori.IsNormalized():  # make sure orientation is normalized
        ori.Normalize()

    rm = libovr_math.Matrix4f(ori)
    up = rm.Transform(libovr_math.Vector3f(0., 1., 0.))
    forward = rm.Transform(libovr_math.Vector3f(0., 0., -1.))
    _eyeViewMatrix[eye] = \
        libovr_math.Matrix4f.LookAtRH(pos, pos + forward, up)
    # VP matrix
    _eyeViewProjectionMatrix[eye] = \
        _eyeProjectionMatrix[eye] * _eyeViewMatrix[eye]


def getEyeProjectionMatrix(int eye, float nearClip=0.01, float farClip=1000.0, object out=None):
    """Compute the projection matrix.

    The projection matrix is computed by the runtime using the eye FOV
    parameters set with :py:attr:`libovr.LibOVRSession.setEyeRenderFov` calls.

    Parameters
    ----------
    eye: int
        Eye index. Use either `EYE_LEFT`` or ``EYE_RIGHT``.
    nearClip : `float`, optional
        Near clipping plane in meters.
    farClip : `float`, optional
        Far clipping plane in meters.
    out : `ndarray` or `None`, optional
        Alternative matrix to write values to instead of returning a new one.

    Returns
    -------
    `ndarray`
        4x4 projection matrix.

    Raises
    ------
    AssertionError
        Dimensions, shape, and data type for `out` is incorrect.

    Examples
    --------

    Get the left and right projection matrices as a list::

        eyeProjectionMatrices = []
        for eye in range(EYE_COUNT):
            eyeProjectionMatrices.append(getEyeProjectionMatrix(eye))

    Same as above, but overwrites existing view matrices::

        # identity 4x4 matrices
        eyeProjectionMatrices = [
            numpy.identity(4, dtype=numpy.float32),
            numpy.identity(4, dtype=numpy.float32)]

        # for eye in range(EYE_COUNT) also works
        for eye in enumerate(eyeProjectionMatrices):
            getEyeProjectionMatrix(eye, outMatrix=eyeProjectionMatrices[eye])

    """
    global _eyeProjectionMatrix
    global _eyeRenderDesc

    cdef np.ndarray[np.float32_t, ndim=2] to_return

    if out is None:
        to_return = np.zeros((4, 4), dtype=np.float32)
    else:
        try:
            assert out.ndim == 2 and \
                   out.shape == (4,4,) and \
                   out.dtype == np.float32
        except AssertionError:
            raise AssertionError("'outMatrix' has wrong type or dimensions.")

        to_return = out

    _eyeProjectionMatrix[eye] = \
        <libovr_math.Matrix4f>capi.ovrMatrix4f_Projection(
            _eyeRenderDesc[eye].Fov,
            nearClip,
            farClip,
            capi.ovrProjection_ClipRangeOpenGL)

    # fast copy matrix to numpy array
    cdef float [:, :] mv = to_return
    cdef Py_ssize_t i, j
    cdef Py_ssize_t N = 4
    i = j = 0
    for i in range(N):
        for j in range(N):
            mv[i, j] = _eyeProjectionMatrix[eye].M[i][j]

    return to_return


def getEyeRenderViewport(int eye, object out=None):
    """Get the eye render viewport.

    The viewport defines the region on the swap texture a given eye's image is
    drawn to.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.
    out : `ndarray`, optional
        Optional NumPy array to place values. If None, this function will return
        a new array. Must be dtype=int and length 4.

    Returns
    -------
    ndarray of int or None
        Viewport rectangle [x, y, w, h]. None if 'outRect' was specified.

    """
    global _eyeLayer
    cdef np.ndarray[np.int_t, ndim=1] to_return

    if out is None:
        to_return = np.zeros((4,), dtype=np.int)
    else:
        to_return = out

    to_return[0] = _eyeLayer.Viewport[eye].Pos.x
    to_return[1] = _eyeLayer.Viewport[eye].Pos.y
    to_return[2] = _eyeLayer.Viewport[eye].Size.w
    to_return[3] = _eyeLayer.Viewport[eye].Size.h

    return to_return


def setEyeRenderViewport(int eye, object values):
    """Set the eye render viewport.

    The viewport defines the region on the swap texture a given eye's image is
    drawn to.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.
    `ndarray`, `list`, or `tuple` of `ints`
        Viewport rectangle [x, y, w, h].

    Examples
    --------

    Setting the viewports for both eyes on a single swap chain buffer::

        # Calculate the optimal eye buffer sizes for the FOVs, these will define
        # the dimensions of the render target.
        leftBufferSize, rightBufferSize = calcEyeBufferSizes()

        # Define the viewports, which specifies the region on the render target a
        # eye's image will be drawn to and accessed from. Viewports are rectangles
        # defined like [x, y, w, h]. The x-position of the rightViewport is offset
        # by the width of the left viewport.
        leftViewport = [0, 0, leftBufferSize[0], leftBufferSize[1]]
        rightViewport = [leftBufferSize[0], 0,
                         rightBufferSize[0], rightBufferSize[1]]

        # set both viewports
        setEyeRenderViewport(EYE_LEFT, leftViewport)
        setEyeRenderViewport(EYE_RIGHT, rightViewport)

    """
    global _eyeLayer
    _eyeLayer.Viewport[eye].Pos.x = <int>values[0]
    _eyeLayer.Viewport[eye].Pos.y = <int>values[1]
    _eyeLayer.Viewport[eye].Size.w = <int>values[2]
    _eyeLayer.Viewport[eye].Size.h = <int>values[3]


def getEyeViewMatrix(int eye, object out=None):
    """Compute a view matrix for a specified eye.

    View matrices are derived from the eye render poses calculated by the
    last :func:`calcEyePoses` call or update by :func:`setEyeRenderPose`.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.
    out : `ndarray` or `None`, optional
        Optional array to write to. Must have ndim=2, dtype=np.float32, and
        shape == (4,4).

    Returns
    -------
    ndarray
        4x4 view matrix. Object `out` will be returned if specified.

    """
    global _eyeViewMatrix
    cdef np.ndarray[np.float32_t, ndim=2] to_return

    if out is None:
        to_return = np.zeros((4, 4), dtype=np.float32)
    else:
        to_return = out

    cdef Py_ssize_t i, j, N
    i = j = 0
    N = 4
    for i in range(N):
        for j in range(N):
            to_return[i, j] = _eyeViewMatrix[eye].M[i][j]

    return to_return


def getPredictedDisplayTime(unsigned int frameIndex=0):
    """Get the predicted time a frame will be displayed.

    Parameters
    ----------
    frameIndex : `int`
        Frame index.

    Returns
    -------
    `float`
        Absolute frame mid-point time for the given frame index in seconds.

    """
    global _ptrSession
    cdef double t_sec = capi.ovr_GetPredictedDisplayTime(
        _ptrSession,
        frameIndex)

    return t_sec


def timeInSeconds():
    """Absolute time in seconds.

    Returns
    -------
    `float`
        Time in seconds.

    """
    cdef double t_sec = capi.ovr_GetTimeInSeconds()

    return t_sec


def waitToBeginFrame(unsigned int frameIndex=0):
    """Wait until a buffer is available so frame rendering can begin. Must be
    called before :func:`beginFrame`.

    Parameters
    ----------
    frameIndex : `int`
        The target frame index.

    Returns
    -------
    int
        Return code of the LibOVR API call ``OVR::ovr_WaitToBeginFrame``. 
        Returns ``SUCCESS`` if completed without errors. May return
        ``ERROR_DISPLAY_LOST`` if the device was removed, rendering the current 
        session invalid.

    """
    global _ptrSession
    cdef capi.ovrResult result = \
        capi.ovr_WaitToBeginFrame(_ptrSession, frameIndex)

    return <int>result


def beginFrame(unsigned int frameIndex=0):
    """Begin rendering the frame. Must be called prior to drawing and
    :func:`endFrame`.

    Parameters
    ----------
    frameIndex : `int`
        The target frame index.

    Returns
    -------
    int
        Error code returned by ``OVR::ovr_BeginFrame``.

    """
    global _ptrSession
    cdef capi.ovrResult result = \
        capi.ovr_BeginFrame(_ptrSession, frameIndex)

    return <int>result


def commitTextureSwapChain(int eye):
    """Commit changes to a given eye's texture swap chain. When called, the
    runtime is notified that the texture is ready for use, and the swap
    chain index is incremented.

    Parameters
    ----------
    eye : `int`
        Eye buffer index.

    Returns
    -------
    int
        Error code returned by API call ``OVR::ovr_CommitTextureSwapChain``. 
        Will return ``SUCCESS`` if successful. Returns error code
        ``ERROR_TEXTURE_SWAP_CHAIN_FULL`` if called too many times without 
        calling :func:`endFrame`.

    Warning
    -------

    No additional drawing operations are permitted once the texture is committed
    until the SDK dereferences it, making it available again.

    """
    global _swapChains
    global _ptrSession
    cdef capi.ovrResult result = capi.ovr_CommitTextureSwapChain(
        _ptrSession,
        _swapChains[eye])

    return <int>result


def endFrame(unsigned int frameIndex=0):
    """Call when rendering a frame has completed. Buffers which have been
    committed are passed to the compositor for distortion.

    Successful :func:`waitToBeginFrame` and :func:`beginFrame` call must precede
    calling :func:`endFrame`.

    Parameters
    ----------
    frameIndex : `int`
        The target frame index.

    Returns
    -------
    `int`
        Error code returned by API call `ovr_EndFrame`. Check against
        ``SUCCESS`, ``SUCCESS_NOT_VISIBLE``, ``SUCCESS_BOUNDARY_INVALID``,
        and ``SUCCESS_DEVICE_UNAVAILABLE``.

    """
    global _ptrSession
    global _eyeLayer

    cdef capi.ovrLayerHeader* layers = &_eyeLayer.Header
    cdef capi.ovrResult result = capi.ovr_EndFrame(
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
        Error code returned by `OVR::ovr_ResetPerfStats`.

    """
    global _ptrSession
    cdef capi.ovrResult result = capi.ovr_ResetPerfStats(_ptrSession)

    return result


def getTrackingOriginType():
    """Get the current tracking origin type.

    The tracking origin type specifies where the origin is placed when computing
    the pose of tracked objects (i.e. the head and touch controllers.) Valid
    values are ``TRACKING_ORIGIN_EYE_LEVEL`` and 
    ``TRACKING_ORIGIN_FLOOR_LEVEL``.

    See Also
    --------
    setTrackingOriginType : Set the tracking origin type.

    """
    global _ptrSession
    cdef capi.ovrTrackingOrigin originType = \
        capi.ovr_GetTrackingOriginType(_ptrSession)

    if originType == capi.ovrTrackingOrigin_FloorLevel:
        return TRACKING_ORIGIN_FLOOR_LEVEL
    elif originType == capi.ovrTrackingOrigin_EyeLevel:
        return TRACKING_ORIGIN_EYE_LEVEL


def setTrackingOriginType(int value):
    """Set the tracking origin type.

    Specify the tracking origin to use when computing eye poses. Subsequent
    calls of :func:`calcEyePoses` will use the set tracking origin.

    Parameters
    ----------
    value : int
        Tracking origin type, must be either
        ``TRACKING_ORIGIN_FLOOR_LEVEL`` or
        ``TRACKING_ORIGIN_EYE_LEVEL``.

    Returns
    -------
    int
        Result of the ``OVR::ovr_SetTrackingOriginType`` LibOVR API call.

    See Also
    --------
    getTrackingOriginType : Get the current tracking origin type.

    """
    cdef capi.ovrResult result
    global _ptrSession
    if value == TRACKING_ORIGIN_FLOOR_LEVEL:
        result = capi.ovr_SetTrackingOriginType(
            _ptrSession, capi.ovrTrackingOrigin_FloorLevel)
    elif value == TRACKING_ORIGIN_EYE_LEVEL:
        result = capi.ovr_SetTrackingOriginType(
            _ptrSession, capi.ovrTrackingOrigin_EyeLevel)
    else:
        raise ValueError("Invalid tracking origin type specified "
                         "must be 'TRACKING_ORIGIN_FLOOR_LEVEL' or "
                         "'TRACKING_ORIGIN_EYE_LEVEL'.")

    return result


def recenterTrackingOrigin():
    """Recenter the tracking origin.

    Returns
    -------
    int
        The result of the LibOVR API call ``OVR::ovr_RecenterTrackingOrigin``.

    Examples
    --------

    Recenter the tracking origin if requested by the session status::

        sessionStatus = getSessionStatus()
        if sessionStatus.shouldRecenter:
            recenterTrackingOrigin()

    """
    global _ptrSession
    cdef capi.ovrResult result = capi.ovr_RecenterTrackingOrigin(
        _ptrSession)

    return result


def specifyTrackingOrigin(LibOVRPose newOrigin):
    """Specify a new tracking origin.

    Parameters
    ----------
    newOrigin : :class:`LibOVRPose`
        New origin to use.

    """
    global _ptrSession
    cdef capi.ovrResult result = capi.ovr_SpecifyTrackingOrigin(
        _ptrSession,
        newOrigin.c_data[0])

    return result


def clearShouldRecenterFlag():
    """Clear the :py:attr:`LibOVRSessionStatus.shouldRecenter` flag.

    """
    global _ptrSession
    capi.ovr_ClearShouldRecenterFlag(_ptrSession)


def getTrackerCount():
    """Get the number of attached trackers.

    Returns
    -------
    int
        Number of trackers reported by LibOVR.

    """
    global _ptrSession
    cdef unsigned int trackerCount = capi.ovr_GetTrackerCount(
        _ptrSession)

    return <int>trackerCount


def getTrackerInfo(int trackerIndex):
    """Get information about a given tracker.

    Parameters
    ----------
    trackerIndex : int
        The index of the sensor to query. Valid values are between 0 and
        :func:`getTrackerCount` - 1.

    """
    cdef LibOVRTrackerInfo to_return = LibOVRTrackerInfo()
    global _ptrSession

    # set the tracker index
    to_return._trackerIndex = <unsigned int>trackerIndex

    # set the descriptor data
    to_return.c_ovrTrackerDesc = capi.ovr_GetTrackerDesc(
        _ptrSession, <unsigned int>trackerIndex)
    # get the tracker pose
    to_return.c_ovrTrackerPose = capi.ovr_GetTrackerPose(
        _ptrSession, <unsigned int>trackerIndex)

    return to_return


def updatePerfStats():
    """Update performance stats.

    Should be called once per frame.

    Returns
    -------
    int
        Result of the ``OVR::ovr_GetPerfStats`` LibOVR API call.

    """
    global _ptrSession
    global _frameStats
    global _lastFrameStats

    if _frameStats.FrameStatsCount > 0:
        if _frameStats.FrameStats[0].HmdVsyncIndex > 0:
            # copy last frame stats
            _lastFrameStats = _frameStats.FrameStats[0]

    cdef capi.ovrResult result = capi.ovr_GetPerfStats(
        _ptrSession, &_frameStats)

    return result


def getAdaptiveGpuPerformanceScale():
    """Get the adaptive GPU performance scale.

    Returns
    -------
    float
        GPU performance scaling factor.

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

    This occurs when :func:`updatePerfStats` is called fewer than once every 5
    frames.

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


LibOVRFramePerfStats = collections.namedtuple('LibOVRFramePerfStats',
    ['hmdVsyncIndex', 'appFrameIndex', 'appDroppedFrameCount',
     'appMotionToPhotonLatency', 'appQueueAheadTime', 'appCpuElapsedTime',
     'appGpuElapsedTime', 'compositorFrameIndex',
     'compositorLatency', 'compositorCpuElapsedTime',
     'compositorGpuElapsedTime', 'compositorCpuStartToGpuEndElapsedTime',
     'compositorGpuEndToVsyncElapsedTime'])


def getFrameStats(int frameStatIndex=0):
    """Get detailed compositor frame statistics.

    Returned frame statistics reflect the values contemporaneous with the last
    :func:`updatePerfStats` call.

    Parameters
    ----------
    frameStatIndex : int (default 0)
        Frame statistics index to retrieve. The most recent frame statistic is
        at index 0.

    Returns
    -------
    namedtuple
        Frame statistics from the compositor. Available stats are accessible
        using the following attributes of the returned `LibOVRFramePerfStats`
        namedtuple:

        * `hmdVsyncIndex` - Increments every HMD vertical sync signal.
        * `appFrameIndex` - Index increments after each call to :func:`endFrame`.
        * `appDroppedFrameCount` - If :func:`endFrame` is not called on-time,
          this will increment (i.e. missed HMD vertical sync deadline).
        * `appMotionToPhotonLatency` - Motion-to-photon latency in seconds
          computed using the marker set by :func:`getTrackingState`.
        * `appQueueAheadTime` - Queue-ahead time in seconds. If >11 ms, the CPU
          is outpacing the GPU workload by 1 frame.
        * `appCpuElapsedTime` - Time in seconds the CPU spent between calls of
          :func:`endFrame`. Form the point when :func:`endFrame` releases
          control back to the application, to the next time it is called.
        * `appGpuElapsedTime` - Time in seconds the GPU spent between calls of
          :func:`endFrame`.
        * `compositorFrameIndex` - Increments when the compositor completes a
          distortion pass, happens regardless if :func:`endFrame` was called
          late.
        * `compositorLatency` - Motion-to-photon latency of the compositor, which
          include the latency of 'timewarp' needed to correct for application
          latency and dropped application frames.
        * `compositorCpuElapsedTime` - Time in seconds the compositor spends on
          the CPU.
        * `compositorGpuElapsedTime` - Time in seconds the compositor spends on
          the GPU.
        * `compositorCpuStartToGpuEndElapsedTime` - Time in seconds between the
          point the compositor executes and completes distortion/timewarp.
          Value is -1.0 if GPU time is not available.
        * `compositorGpuEndToVsyncElapsedTime` - Time in seconds left between the
          compositor is complete and the target vertical synchronization on the
          HMD.

    See Also
    --------
    updatePerfStats : Update performance and frame statistics.

    Notes
    -----

    * If :func:`updatePerfStats` was called less than once per frame, more than
      one frame statistic will be available. Check :func:`getFrameStatsCount`
      for the number of queued stats and use an index >0 to access them. Stats
      are dropped if the queue is larger than 5 items.

    Examples
    --------

    Get the time spent by the application between :func:`endFrame` calls::

        result = updatePerfStats()

        if getFrameStatsCount() > 0:
            frameStats = getFrameStats(0)  # only the most recent
            appTime = frameStats.appCpuElapsedTime

    Get all available frame statistics::

        stats = []
        for i in range(getFrameStatsCount()):
            stats.append(getFrameStats(i))

    """
    global _frameStats

    if 0 > frameStatIndex >= _frameStats.FrameStatsCount:
        raise IndexError("Frame stats index out of range.")

    cdef capi.ovrPerfStatsPerCompositorFrame stat = \
        _frameStats.FrameStats[frameStatIndex]

    return LibOVRFramePerfStats(
        stat.HmdVsyncIndex,
        stat.AppFrameIndex,
        stat.AppDroppedFrameCount,
        stat.AppMotionToPhotonLatency,
        stat.AppQueueAheadTime,
        stat.AppCpuElapsedTime,
        stat.AppGpuElapsedTime,
        stat.CompositorFrameIndex,
        stat.CompositorLatency,
        stat.CompositorCpuElapsedTime,
        stat.CompositorGpuElapsedTime,
        stat.CompositorCpuStartToGpuEndElapsedTime,
        stat.CompositorGpuEndToVsyncElapsedTime)


def getLastErrorInfo():
    """Get the last error code and information string reported by the API.

    This function can be used when implementing custom error handlers for
    dealing with exceptions raised by LibOVR. You must call
    :func:`getLastErrorInfo` every time after any function which makes an LibOVR
    API call if you wish to catch all errors, since only the most recent is
    returned.

    Returns
    -------
    `tuple` of `int`, `str`
        Tuple of the API call result and error string. If there was no API
        error, the function will return tuple (0, '<unknown>').

    Examples
    --------

    Raise a Python exception if LibOVR reports an error::

        result = create()
        if failure(result):
            errorVal, errorMsg = getLastErrorInfo()
            raise RuntimeError(errorMsg)  # Python generic runtime error!

    """
    cdef capi.ovrErrorInfo lastErrorInfo  # store our last error here
    capi.ovr_GetLastErrorInfo(&lastErrorInfo)

    cdef capi.ovrResult result = lastErrorInfo.Result
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

    Returns
    -------
    int
        Result of the LibOVR API call ``OVR::ovr_SetBoundaryLookAndFeel``.

    """
    global _boundryStyle
    global _ptrSession

    cdef capi.ovrColorf color
    color.r = <float>red
    color.g = <float>green
    color.b = <float>blue

    _boundryStyle.Color = color

    cdef capi.ovrResult result = capi.ovr_SetBoundaryLookAndFeel(
        _ptrSession,
        &_boundryStyle)

    return result


def resetBoundaryColor():
    """Reset the boundary color to system default.

    Returns
    -------
    int
        Result of the LibOVR API call ``OVR::ovr_ResetBoundaryLookAndFeel``.

    """
    global _ptrSession
    cdef capi.ovrResult result = capi.ovr_ResetBoundaryLookAndFeel(
        _ptrSession)

    return result


def getBoundaryVisible():
    """Check if the Guardian boundary is visible.

    The boundary is drawn by the compositor which overlays the extents of
    the physical space where the user can safely move.

    Returns
    -------
    tuple of int, bool
        Result of the LibOVR API call ``OVR::ovr_GetBoundaryVisible`` and the 
        boundary state.

    Notes
    -----
    * Since the boundary has a fade-in effect, the boundary might be reported as
      visible but difficult to actually see.

    """
    global _ptrSession
    cdef capi.ovrBool isVisible
    cdef capi.ovrResult result = capi.ovr_GetBoundaryVisible(
        _ptrSession, &isVisible)

    return result, isVisible


def showBoundary():
    """Show the boundary.

    The boundary is drawn by the compositor which overlays the extents of
    the physical space where the user can safely move.

    Returns
    -------
    int
        Result of LibOVR API call ``OVR::ovr_RequestBoundaryVisible``.

    """
    global _ptrSession
    cdef capi.ovrResult result = capi.ovr_RequestBoundaryVisible(
        _ptrSession, capi.ovrTrue)

    return result


def hideBoundary():
    """Hide the boundry.

    Returns
    -------
    int
        Result of LibOVR API call ``OVR::ovr_RequestBoundaryVisible``.

    """
    global _ptrSession
    cdef capi.ovrResult result = capi.ovr_RequestBoundaryVisible(
        _ptrSession, capi.ovrFalse)

    return result


def getBoundaryDimensions(int boundaryType):
    """Get the dimensions of the boundary.

    Parameters
    ----------
    boundaryType : int
        Boundary type, can be ``BOUNDARY_OUTER`` or ``BOUNDARY_PLAY_AREA``.

    Returns
    -------
    tuple of int, ndarray
        Result of the LibOVR APi call ``OVR::ovr_GetBoundaryDimensions`` and the
        dimensions of the boundary in meters [x, y, z].

    """
    global _ptrSession
    cdef capi.ovrBoundaryType btype = <capi.ovrBoundaryType>boundaryType
    if not (boundaryType == capi.ovrBoundary_PlayArea or
            boundaryType == capi.ovrBoundary_Outer):
        raise ValueError("Invalid boundary type specified.")

    cdef capi.ovrVector3f vec_out
    cdef capi.ovrResult result = capi.ovr_GetBoundaryDimensions(
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
    list of int
        IDs of connected controller types. Possible values returned are:

        * ``CONTROLLER_TYPE_XBOX`` : XBox gamepad.
        * ``CONTROLLER_TYPE_REMOTE`` : Oculus Remote.
        * ``CONTROLLER_TYPE_TOUCH`` : Combined Touch controllers.
        * ``CONTROLLER_TYPE_LTOUCH`` : Left Touch controller.
        * ``CONTROLLER_TYPE_RTOUCH`` : Right Touch controller.
        * ``CONTROLLER_TYPE_OBJECT0`` : Object 0 controller.
        * ``CONTROLLER_TYPE_OBJECT1`` : Object 1 controller.
        * ``CONTROLLER_TYPE_OBJECT2`` : Object 2 controller.
        * ``CONTROLLER_TYPE_OBJECT3`` : Object 3 controller.

    See Also
    --------
    updateInputState : Poll a controller's current state.

    Examples
    --------

    Check if the left touch controller is paired::

        controllers = getConnectedControllerTypes()
        hasLeftTouch = CONTROLLER_TYPE_LTOUCH in controllers

    Update all connected controller states::

        for controller in getConnectedControllerTypes():
            result, time = updateInputState(controller)

    """
    global _ptrSession
    cdef unsigned int result = capi.ovr_GetConnectedControllerTypes(
        _ptrSession)

    cdef list toReturn = list()
    if (capi.ovrControllerType_XBox & result) == capi.ovrControllerType_XBox:
        toReturn.append(CONTROLLER_TYPE_XBOX)
    if (capi.ovrControllerType_Remote & result) == capi.ovrControllerType_Remote:
        toReturn.append(CONTROLLER_TYPE_REMOTE)
    if (capi.ovrControllerType_Touch & result) == capi.ovrControllerType_Touch:
        toReturn.append(CONTROLLER_TYPE_TOUCH)
    if (capi.ovrControllerType_LTouch & result) == capi.ovrControllerType_LTouch:
        toReturn.append(CONTROLLER_TYPE_LTOUCH)
    if (capi.ovrControllerType_RTouch & result) == capi.ovrControllerType_RTouch:
        toReturn.append(CONTROLLER_TYPE_RTOUCH)
    if (capi.ovrControllerType_Object0 & result) == capi.ovrControllerType_Object0:
        toReturn.append(CONTROLLER_TYPE_OBJECT0)
    if (capi.ovrControllerType_Object1 & result) == capi.ovrControllerType_Object1:
        toReturn.append(CONTROLLER_TYPE_OBJECT1)
    if (capi.ovrControllerType_Object2 & result) == capi.ovrControllerType_Object2:
        toReturn.append(CONTROLLER_TYPE_OBJECT2)
    if (capi.ovrControllerType_Object3 & result) == capi.ovrControllerType_Object3:
        toReturn.append(CONTROLLER_TYPE_OBJECT3)

    return toReturn


def updateInputState(int controller):
    """Refresh the input state of a controller.

    Subsequent :func:`getButton`, :func:`getTouch`, :func:`getThumbstickValues`,
    :func:`getIndexTriggerValues`, and :func:`getHandTriggerValues` calls using
    the same `controller` value will reflect the new state.

    Parameters
    ----------
    controller : int
        Controller name. Valid values are:

        * ``CONTROLLER_TYPE_XBOX`` : XBox gamepad.
        * ``CONTROLLER_TYPE_REMOTE`` : Oculus Remote.
        * ``CONTROLLER_TYPE_TOUCH`` : Combined Touch controllers.
        * ``CONTROLLER_TYPE_LTOUCH`` : Left Touch controller.
        * ``CONTROLLER_TYPE_RTOUCH`` : Right Touch controller.
        * ``CONTROLLER_TYPE_OBJECT0`` : Object 0 controller.
        * ``CONTROLLER_TYPE_OBJECT1`` : Object 1 controller.
        * ``CONTROLLER_TYPE_OBJECT2`` : Object 2 controller.
        * ``CONTROLLER_TYPE_OBJECT3`` : Object 3 controller.

    Returns
    -------
    tuple of int, float
        Result of the ``OVR::ovr_GetInputState`` LibOVR API call and polling 
        time in seconds.

    See Also
    --------
    getConnectedControllerTypes : Get a list of connected controllers.
    getButton: Get button states.
    getTouch: Get touches.

    """
    global _prevInputState
    global _inputStates
    global _ptrSession

    # get the controller index in the states array
    cdef int idx
    cdef capi.ovrInputState* previousInputState
    cdef capi.ovrInputState* currentInputState
    cdef capi.ovrResult result

    if controller == CONTROLLER_TYPE_XBOX:
        idx = 0
    elif controller == CONTROLLER_TYPE_REMOTE:
        idx = 1
    elif controller == CONTROLLER_TYPE_TOUCH:
        idx = 2
    elif controller == CONTROLLER_TYPE_LTOUCH:
        idx = 3
    elif controller == CONTROLLER_TYPE_RTOUCH:
        idx = 4
    elif controller == CONTROLLER_TYPE_OBJECT0:
        idx = 5
    elif controller == CONTROLLER_TYPE_OBJECT1:
        idx = 6
    elif controller == CONTROLLER_TYPE_OBJECT2:
        idx = 7
    elif controller == CONTROLLER_TYPE_OBJECT3:
        idx = 8
    else:
        raise ValueError("Invalid controller type specified.")

    # pointer to the current and previous input state
    previousInputState = &_prevInputState[idx]
    currentInputState = &_inputStates[idx]

    # copy the current input state into the previous before updating
    previousInputState[0] = currentInputState[0]

    # get the current input state
    result = capi.ovr_GetInputState(
        _ptrSession,
        <capi.ovrControllerType>controller,
        currentInputState)

    return result, currentInputState.TimeInSeconds


def getButton(int controller, int button, str testState='continuous'):
    """Get a button state.

    The `controller` to test is specified by its ID, defined as constants
    starting with :data:`CONTROLLER_TYPE_*`. Buttons to test are
    specified using their ID, defined as constants starting with
    :data:`BUTTON_*`. Button IDs can be ORed together for testing
    multiple button states. The returned value represents the button state
    during the last :func:`updateInputState` call for the specified
    `controller`.

    An optional trigger mode may be specified which defines the button's
    activation criteria. By default, `testState` is 'continuous' will return the
    immediate state of the button. Using 'rising' (or 'pressed') will
    return True once when the button transitions to being pressed between
    subsequent :func:`updateInputState` calls, whereas 'falling' (and
    'released') will return True once the button is released. If
    :func:`updateInputState` was called only once, 'rising' and 'falling' will
    return False.

    Parameters
    ----------
    controller : int
        Controller name. Valid values are:

        * ``CONTROLLER_TYPE_XBOX`` : XBox gamepad.
        * ``CONTROLLER_TYPE_REMOTE`` : Oculus Remote.
        * ``CONTROLLER_TYPE_TOUCH`` : Combined Touch controllers.
        * ``CONTROLLER_TYPE_LTOUCH`` : Left Touch controller.
        * ``CONTROLLER_TYPE_RTOUCH`` : Right Touch controller.
        * ``CONTROLLER_TYPE_OBJECT0`` : Object 0 controller.
        * ``CONTROLLER_TYPE_OBJECT1`` : Object 1 controller.
        * ``CONTROLLER_TYPE_OBJECT2`` : Object 2 controller.
        * ``CONTROLLER_TYPE_OBJECT3`` : Object 3 controller.

    button : int
        Button to check. Values can be ORed together to test for multiple button
        presses. If a given controller does not have a particular button, False
        will always be returned. Valid button values are:

        * ``BUTTON_A``
        * ``BUTTON_B``
        * ``BUTTON_RTHUMB``
        * ``BUTTON_RSHOULDER``
        * ``BUTTON_X``
        * ``BUTTON_Y``
        * ``BUTTON_LTHUMB``
        * ``BUTTON_LSHOULDER``
        * ``BUTTON_UP``
        * ``BUTTON_DOWN``
        * ``BUTTON_LEFT``
        * ``BUTTON_RIGHT``
        * ``BUTTON_ENTER``
        * ``BUTTON_BACK``
        * ``BUTTON_VOLUP``
        * ``BUTTON_VOLDOWN``
        * ``BUTTON_HOME``
        * ``BUTTON_PRIVATE``
        * ``BUTTON_RMASK``
        * ``BUTTON_LMASK``

    testState : str
        State to test buttons for. Valid states are 'rising', 'falling',
        'continuous', 'pressed', and 'released'.

    Returns
    -------
    tuple of bool and float
        Result of the button press and the time in seconds it was polled.

    Raises
    ------
    ValueError
        When an invalid controller, button, or state identifier is passed.

    See Also
    --------
    getTouch : Get touches.

    Examples
    --------
    Check if the 'X' button on the touch controllers was pressed::

        isPressed = getButtons(CONTROLLER_TYPE_TOUCH,
            BUTTON_X, 'pressed')

    Test for multiple buttons (e.g. 'X' and 'Y') being released::

        buttons = BUTTON_X | BUTTON_Y
        controller = CONTROLLER_TYPE_TOUCH
        isReleased = getButtons(controller, buttons, 'released')

    """
    global _prevInputState
    global _inputStates

    # get the controller index in the states array
    cdef int idx
    if controller == CONTROLLER_TYPE_XBOX:
        idx = 0
    elif controller == CONTROLLER_TYPE_REMOTE:
        idx = 1
    elif controller == CONTROLLER_TYPE_TOUCH:
        idx = 2
    elif controller == CONTROLLER_TYPE_LTOUCH:
        idx = 3
    elif controller == CONTROLLER_TYPE_RTOUCH:
        idx = 4
    elif controller == CONTROLLER_TYPE_OBJECT0:
        idx = 5
    elif controller == CONTROLLER_TYPE_OBJECT1:
        idx = 6
    elif controller == CONTROLLER_TYPE_OBJECT2:
        idx = 7
    elif controller == CONTROLLER_TYPE_OBJECT3:
        idx = 8
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


def getTouch(int controller, int touch, str testState='continuous'):
    """Get a touch state.

    The `controller` to test is specified by its ID, defined as constants
    starting with :data:`CONTROLLER_TYPE_*`. Touches to test are
    specified using their ID, defined as constants starting with
    :data:`TOUCH_*`. Touch IDs can be ORed together for testing multiple
    touch states. The returned value represents the touch state during the last
    :func:`updateInputState` call for the specified `controller`.

    An optional trigger mode may be specified which defines the button's
    activation criteria. By default, `testState` is 'continuous' will return the
    immediate state of the button. Using 'rising' (or 'pressed') will
    return True once when something is touched between subsequent
    :func:`updateInputState` calls, whereas 'falling' (and 'released') will
    return True once the touch is discontinued. If :func:`updateInputState` was
    called only once, 'rising' and 'falling' will return False.

    Parameters
    ----------
    controller : `int`
        Controller name. Valid values are:

        * ``CONTROLLER_TYPE_XBOX`` : XBox gamepad.
        * ``CONTROLLER_TYPE_REMOTE`` : Oculus Remote.
        * ``CONTROLLER_TYPE_TOUCH`` : Combined Touch controllers.
        * ``CONTROLLER_TYPE_LTOUCH`` : Left Touch controller.
        * ``CONTROLLER_TYPE_RTOUCH`` : Right Touch controller.
        * ``CONTROLLER_TYPE_OBJECT0`` : Object 0 controller.
        * ``CONTROLLER_TYPE_OBJECT1`` : Object 1 controller.
        * ``CONTROLLER_TYPE_OBJECT2`` : Object 2 controller.
        * ``CONTROLLER_TYPE_OBJECT3`` : Object 3 controller.

    touch : `int`
        Touch to check. Values can be ORed together to test for multiple
        touches. If a given controller does not have a particular touch, False
        will always be returned. Valid button values are:

        * ``TOUCH_A``
        * ``TOUCH_B``
        * ``TOUCH_RTHUMB``
        * ``TOUCH_RSHOULDER``
        * ``TOUCH_X``
        * ``TOUCH_Y``
        * ``TOUCH_LTHUMB``
        * ``TOUCH_LSHOULDER``
        * ``TOUCH_LINDEXTRIGGER``
        * ``TOUCH_LINDEXTRIGGER``
        * ``TOUCH_LTHUMBREST``
        * ``TOUCH_RTHUMBREST``
        * ``TOUCH_RINDEXPOINTING``
        * ``TOUCH_RTHUMBUP``
        * ``TOUCH_LINDEXPOINTING``
        * ``TOUCH_LTHUMBUP``

    testState : `str`
        State to test touches for. Valid states are 'rising', 'falling',
        'continuous', 'pressed', and 'released'.

    Returns
    -------
    tuple of bool and float
        Result of the touches and the time in seconds it was polled.

    See Also
    --------
    getButton : Get a button state.

    Notes
    -----

    * Not every controller type supports touch. Unsupported controllers will
      always return False.
    * Special 'touches' ``TOUCH_RINDEXPOINTING``, ``TOUCH_RTHUMBUP``, 
      ``TOUCH_RTHUMBREST``, ``TOUCH_LINDEXPOINTING``, ``TOUCH_LINDEXPOINTING``,
      and ``TOUCH_LINDEXPOINTING``, can be used to recognise hand pose/gestures.

    Examples
    --------

    Check if the user is making a pointing gesture with their right index
    finger::

        isPointing = getTouch(
            controller=CONTROLLER_TYPE_LTOUCH,
            touch=TOUCH_LINDEXPOINTING)

    """
    global _prevInputState
    global _inputStates

    # get the controller index in the states array
    cdef int idx
    if controller == CONTROLLER_TYPE_XBOX:
        idx = 0
    elif controller == CONTROLLER_TYPE_REMOTE:
        idx = 1
    elif controller == CONTROLLER_TYPE_TOUCH:
        idx = 2
    elif controller == CONTROLLER_TYPE_LTOUCH:
        idx = 3
    elif controller == CONTROLLER_TYPE_RTOUCH:
        idx = 4
    elif controller == CONTROLLER_TYPE_OBJECT0:
        idx = 5
    elif controller == CONTROLLER_TYPE_OBJECT1:
        idx = 6
    elif controller == CONTROLLER_TYPE_OBJECT2:
        idx = 7
    elif controller == CONTROLLER_TYPE_OBJECT3:
        idx = 8
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
    controller : `int`
        Controller name. Valid values are:

        * ``CONTROLLER_TYPE_XBOX`` : XBox gamepad.
        * ``CONTROLLER_TYPE_REMOTE`` : Oculus Remote.
        * ``CONTROLLER_TYPE_TOUCH`` : Combined Touch controllers.
        * ``CONTROLLER_TYPE_LTOUCH`` : Left Touch controller.
        * ``CONTROLLER_TYPE_RTOUCH`` : Right Touch controller.
        * ``CONTROLLER_TYPE_OBJECT0`` : Object 0 controller.
        * ``CONTROLLER_TYPE_OBJECT1`` : Object 1 controller.
        * ``CONTROLLER_TYPE_OBJECT2`` : Object 2 controller.
        * ``CONTROLLER_TYPE_OBJECT3`` : Object 3 controller.

    deadzone : `bool`
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
            ovr.CONTROLLER_TYPE_TOUCH, deadzone=True)
        x, y = rightThumbStick  # left-right, up-down values for right stick

    """
    global _inputStates

    # get the controller index in the states array
    cdef int idx
    if controller == CONTROLLER_TYPE_XBOX:
        idx = 0
    elif controller == CONTROLLER_TYPE_REMOTE:
        idx = 1
    elif controller == CONTROLLER_TYPE_TOUCH:
        idx = 2
    elif controller == CONTROLLER_TYPE_LTOUCH:
        idx = 3
    elif controller == CONTROLLER_TYPE_RTOUCH:
        idx = 4
    elif controller == CONTROLLER_TYPE_OBJECT0:
        idx = 5
    elif controller == CONTROLLER_TYPE_OBJECT1:
        idx = 6
    elif controller == CONTROLLER_TYPE_OBJECT2:
        idx = 7
    elif controller == CONTROLLER_TYPE_OBJECT3:
        idx = 8
    else:
        raise ValueError("Invalid controller type specified.")

    # pointer to the current and previous input state
    cdef capi.ovrInputState* currentInputState = &_inputStates[idx]

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

    return np.array((thumbstick_x0, thumbstick_y0), dtype=np.float32), \
           np.array((thumbstick_x1, thumbstick_y1), dtype=np.float32)


def getIndexTriggerValues(int controller, bint deadzone=False):
    """Get analog index trigger values.

    Get the values indicating the displacement of the controller's analog
    index triggers. Returns values for the left an right sticks. Values range
    from -1 to 1.

    Parameters
    ----------
    controller : `int`
        Controller name. Valid values are:

        * ``CONTROLLER_TYPE_XBOX`` : XBox gamepad.
        * ``CONTROLLER_TYPE_REMOTE`` : Oculus Remote.
        * ``CONTROLLER_TYPE_TOUCH`` : Combined Touch controllers.
        * ``CONTROLLER_TYPE_LTOUCH`` : Left Touch controller.
        * ``CONTROLLER_TYPE_RTOUCH`` : Right Touch controller.
        * ``CONTROLLER_TYPE_OBJECT0`` : Object 0 controller.
        * ``CONTROLLER_TYPE_OBJECT1`` : Object 1 controller.
        * ``CONTROLLER_TYPE_OBJECT2`` : Object 2 controller.
        * ``CONTROLLER_TYPE_OBJECT3`` : Object 3 controller.

    Returns
    -------
    `tuple`
        Trigger values (left, right).

    See Also
    --------
    getThumbstickValues : Get thumbstick displacements.
    getHandTriggerValues : Get hand trigger values.

    Examples
    --------

    Get the index trigger values for touch controllers (with deadzone)::

        leftVal, rightVal = getIndexTriggerValues(CONTROLLER_TYPE_TOUCH,
            deadzone=True)

    Cast a ray from the controller when a trigger is pulled::

        _, rightVal = getIndexTriggerValues(CONTROLLER_TYPE_TOUCH,
            deadzone=True)

        # handPose of right hand from the last tracking state
        if rightVal > 0.75:  # 75% thresholds
            if handPose.raycastSphere(target):  # target is LibOVRPose
                print('Target hit!')
            else:
                print('Missed!')

    """
    # convert the string to an index
    global _inputStates

    # get the controller index in the states array
    cdef int idx
    if controller == CONTROLLER_TYPE_XBOX:
        idx = 0
    elif controller == CONTROLLER_TYPE_REMOTE:
        idx = 1
    elif controller == CONTROLLER_TYPE_TOUCH:
        idx = 2
    elif controller == CONTROLLER_TYPE_LTOUCH:
        idx = 3
    elif controller == CONTROLLER_TYPE_RTOUCH:
        idx = 4
    elif controller == CONTROLLER_TYPE_OBJECT0:
        idx = 5
    elif controller == CONTROLLER_TYPE_OBJECT1:
        idx = 6
    elif controller == CONTROLLER_TYPE_OBJECT2:
        idx = 7
    elif controller == CONTROLLER_TYPE_OBJECT3:
        idx = 8
    else:
        raise ValueError("Invalid controller type specified.")

    # pointer to the current and previous input state
    cdef capi.ovrInputState* currentInputState = &_inputStates[idx]

    cdef float triggerLeft = 0.0
    cdef float triggerRight = 0.0

    if deadzone:
        triggerLeft = currentInputState[0].IndexTrigger[0]
        triggerRight = currentInputState[0].IndexTrigger[1]
    else:
        triggerLeft = currentInputState[0].IndexTriggerNoDeadzone[0]
        triggerRight = currentInputState[0].IndexTriggerNoDeadzone[1]

    return np.array((triggerLeft, triggerRight), dtype=np.float32)


def getHandTriggerValues(int controller, bint deadzone=False):
    """Get analog hand trigger values.

    Get the values indicating the displacement of the controller's analog
    hand triggers. Returns two values for the left and right sticks. Values
    range from -1 to 1.

    Parameters
    ----------
    controller : `int`
        Controller name. Valid values are:

        * ``CONTROLLER_TYPE_XBOX`` : XBox gamepad.
        * ``CONTROLLER_TYPE_REMOTE`` : Oculus Remote.
        * ``CONTROLLER_TYPE_TOUCH`` : Combined Touch controllers.
        * ``CONTROLLER_TYPE_LTOUCH`` : Left Touch controller.
        * ``CONTROLLER_TYPE_RTOUCH`` : Right Touch controller.
        * ``CONTROLLER_TYPE_OBJECT0`` : Object 0 controller.
        * ``CONTROLLER_TYPE_OBJECT1`` : Object 1 controller.
        * ``CONTROLLER_TYPE_OBJECT2`` : Object 2 controller.
        * ``CONTROLLER_TYPE_OBJECT3`` : Object 3 controller.

    Returns
    -------
    tuple
        Trigger values (left, right).

    See Also
    --------
    getThumbstickValues : Get thumbstick displacements.
    getIndexTriggerValues : Get index trigger values.

    Examples
    --------

    Get the hand trigger values for touch controllers (with deadzone)::

        leftVal, rightVal = getHandTriggerValues(CONTROLLER_TYPE_TOUCH,
            deadzone=True)

    Grip an object if near a hand. Simply set the pose of the object to match
    that of the hand when gripping within some distance of the object's
    origin. When the grip is released, the object will assume the last pose
    before being released. Here is a very basic example of object gripping::

        _, rightVal = getHandTriggerValues(CONTROLLER_TYPE_TOUCH,
            deadzone=True)

        # thing and handPose are LibOVRPoses, handPose is from tracking state
        distanceToHand = abs(handPose.distanceTo(thing.pos))
        if rightVal > 0.75 and distanceToHand < 0.01:
            thing.posOri = handPose.posOri

    """
    global _inputStates

    # get the controller index in the states array
    cdef int idx
    if controller == CONTROLLER_TYPE_XBOX:
        idx = 0
    elif controller == CONTROLLER_TYPE_REMOTE:
        idx = 1
    elif controller == CONTROLLER_TYPE_TOUCH:
        idx = 2
    elif controller == CONTROLLER_TYPE_LTOUCH:
        idx = 3
    elif controller == CONTROLLER_TYPE_RTOUCH:
        idx = 4
    elif controller == CONTROLLER_TYPE_OBJECT0:
        idx = 5
    elif controller == CONTROLLER_TYPE_OBJECT1:
        idx = 6
    elif controller == CONTROLLER_TYPE_OBJECT2:
        idx = 7
    elif controller == CONTROLLER_TYPE_OBJECT3:
        idx = 8
    else:
        raise ValueError("Invalid controller type specified.")

    # pointer to the current and previous input state
    cdef capi.ovrInputState* currentInputState = &_inputStates[idx]

    cdef float triggerLeft = 0.0
    cdef float triggerRight = 0.0

    if deadzone:
        triggerLeft = currentInputState[0].HandTrigger[0]
        triggerRight = currentInputState[0].HandTrigger[1]
    else:
        triggerLeft = currentInputState[0].HandTriggerNoDeadzone[0]
        triggerRight = currentInputState[0].HandTriggerNoDeadzone[1]

    return np.array((triggerLeft, triggerRight), dtype=np.float32)


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

        * ``CONTROLLER_TYPE_XBOX`` : XBox gamepad.
        * ``CONTROLLER_TYPE_REMOTE`` : Oculus Remote.
        * ``CONTROLLER_TYPE_TOUCH`` : Combined Touch controllers.
        * ``CONTROLLER_TYPE_LTOUCH`` : Left Touch controller.
        * ``CONTROLLER_TYPE_RTOUCH`` : Right Touch controller.
        * ``CONTROLLER_TYPE_OBJECT0`` : Object 0 controller.
        * ``CONTROLLER_TYPE_OBJECT1`` : Object 1 controller.
        * ``CONTROLLER_TYPE_OBJECT2`` : Object 2 controller.
        * ``CONTROLLER_TYPE_OBJECT3`` : Object 3 controller.

    frequency : str
        Vibration frequency. Valid values are: 'off', 'low', or 'high'.
    amplitude : float
        Vibration amplitude in the range of [0.0 and 1.0]. Values outside
        this range are clamped.

    Returns
    -------
    int
        Return value of API call ``OVR::ovr_SetControllerVibration``. Can return
        ``SUCCESS_DEVICE_UNAVAILABLE`` if no device is present.

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

    cdef capi.ovrResult result = capi.ovr_SetControllerVibration(
        _ptrSession,
        <capi.ovrControllerType>controller,
        freq,
        amplitude)

    return result


def getSessionStatus():
    """Get the current session status.

    Returns
    -------
    tuple of int, tuple of bool
        Result of LibOVR API call ``OVR::ovr_GetSessionStatus`` and a
        :py:class:`LibOVRSessionStatus`.

    Examples
    --------

    Check if the display is visible to the user::

        result, sessionStatus = getSessionStatus()
        if sessionStatus.isVisible:
            # begin frame rendering ...

    Quit if the user requests to through the Oculus overlay::

        result, sessionStatus = getSessionStatus()
        if sessionStatus.shouldQuit:
            # destroy any swap chains ...
            destroy()
            shutdown()

    """
    global _ptrSession
    global _sessionStatus

    cdef capi.ovrResult result = capi.ovr_GetSessionStatus(_ptrSession,
                                                           &_sessionStatus)

    cdef LibOVRSessionStatus to_return = LibOVRSessionStatus()
    to_return.c_data = _sessionStatus

    return result, to_return

# def testBBoxVisible(int eye, object boundingBox):
#     """Test if an object's bounding box falls outside of the current view
#     volume.
#
#     Parameters
#     ----------
#     eye : int
#         Eye index.
#     boundingBox : ndarray
#         Bounding box as an 8x3 array of floats, where each row contains the
#         [X, Y, Z] coordinate of a bounding box vertex.
#
#     Returns
#     -------
#     bool
#         True if the object is not visible.
#
#     """
#     global _eyeViewProjectionMatrix
#
#     cdef float m[16]
#     cdef float planeEq[6][4]
#
#     # compute plane equations, in VR these change constantly so they need to be
#     # recalculated every frame
#
#     cdef int i, j, k
#     k = 0
#     for i in range(4):
#         for j in range(4):
#             m[k] = _eyeViewProjectionMatrix[eye].M[j][i]  # column-wise
#             k += 1
#
#     planeEq[0][0] = m[3] - m[0]
#     planeEq[0][1] = m[7] - m[4]
#     planeEq[0][2] = m[11] - m[8]
#     planeEq[0][3] = m[15] - m[12]
#
#     planeEq[1][0] = m[3] + m[0]
#     planeEq[1][1] = m[7] + m[4]
#     planeEq[1][2] = m[11] + m[8]
#     planeEq[1][3] = m[15] + m[12]
#
#     planeEq[2][0] = m[3] + m[1]
#     planeEq[2][1] = m[7] + m[5]
#     planeEq[2][2] = m[11] + m[9]
#     planeEq[2][3] = m[15] + m[13]
#
#     planeEq[3][0] = m[3] - m[1]
#     planeEq[3][1] = m[7] - m[5]
#     planeEq[3][2] = m[11] - m[9]
#     planeEq[3][3] = m[15] - m[13]
#
#     planeEq[4][0] = m[3] + m[2]
#     planeEq[4][1] = m[7] + m[6]
#     planeEq[4][2] = m[11] + m[10]
#     planeEq[4][3] = m[15] + m[14]
#
#     planeEq[5][0] = m[3] - m[2]
#     planeEq[5][1] = m[7] - m[6]
#     planeEq[5][2] = m[11] - m[10]
#     planeEq[5][3] = m[15] - m[14]
#
#     # get the bounding box
#     cdef float[:,:] bbox = np.asarray(boundingBox, dtype=np.float32)
#
#     # test if the vertices of the bounding box are all outside of the view
#     cdef int culled
#     cdef float distanceFromPlane = 0.0
#
#     for i in range(6):
#         culled = 0
#         for j in range(8):
#             distanceFromPlane = planeEq[i][0] * bbox[j, 0] + \
#                                 planeEq[i][1] * bbox[j, 1] + \
#                                 planeEq[i][2] * bbox[j, 2] + \
#                                 planeEq[i][3]
#
#             if distanceFromPlane < 0.:
#                 culled |= 1 << j
#
#         if culled == 0xff:
#             return 1
#
#     return 0

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
#     #global _nearClip
#     #global _farClip
#
#     cdef np.ndarray[np.float32_t, ndim=2] pointsIn = \
#         np.array(points, dtype=np.float32, ndmin=2, copy=False)
#
#     cdef np.ndarray[np.uint8_t, ndim=2] testOut
#     if out is not None:
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
#         for eye in range(capi.ovrEye_Count):
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
#         for eye in range(capi.ovrEye_Count):
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
