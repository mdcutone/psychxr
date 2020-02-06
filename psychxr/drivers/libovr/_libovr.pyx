# distutils: language=c++
#  =============================================================================
#  _libovr.pyx - Python Interface Module for LibOVR
#  =============================================================================
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
__version__ = "0.2.4"
__status__ = "Stable"
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
    'TOUCH_A',
    'TOUCH_B',
    'TOUCH_RTHUMB',
    'TOUCH_RTHUMBREST',
    'TOUCH_X',
    'TOUCH_Y',
    'TOUCH_LTHUMB',
    'TOUCH_LTHUMBREST',
    'TOUCH_LINDEXTRIGGER',
    'TOUCH_RINDEXPOINTING',
    'TOUCH_RTHUMBUP',
    'TOUCH_LINDEXPOINTING',
    'TOUCH_LTHUMBUP',
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
    'TEXTURE_SWAP_CHAIN_COUNT',
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
    'STATUS_ORIENTATION_VALID',
    'STATUS_POSITION_VALID',
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
    'HMD_NONE',
    'HMD_DK1',
    'HMD_DKHD',
    'HMD_DK2',
    'HMD_CB',
    'HMD_OTHER',
    'HMD_E3_2015',
    'HMD_ES06',
    'HMD_ES09',
    'HMD_ES11',
    'HMD_CV1',
    'HMD_RIFTS',
    'HAPTICS_BUFFER_SAMPLES_MAX',
    'MIRROR_OPTION_DEFAULT',
    'MIRROR_OPTION_POST_DISTORTION',
    'MIRROR_OPTION_LEFT_EYE_ONLY',
    'MIRROR_OPTION_RIGHT_EYE_ONLY',
    'MIRROR_OPTION_INCLUDE_GUARDIAN',
    'MIRROR_OPTION_INCLUDE_NOTIFICATIONS',
    'MIRROR_OPTION_INCLUDE_SYSTEM_GUI',
    'MIRROR_OPTION_FORCE_SYMMETRIC_FOV',
    'LOG_LEVEL_DEBUG',
    'LOG_LEVEL_INFO',
    'LOG_LEVEL_ERROR',
    'HMD_RIFTS',
    'LibOVRPose',
    'LibOVRPoseState',
    'LibOVRTrackingState',
    'LibOVRBounds',
    'LibOVRTrackerInfo',
    'LibOVRHmdInfo',
    'LibOVRSessionStatus',
    'LibOVRBoundaryTestResult',
    'LibOVRPerfStatsPerCompositorFrame',
    'LibOVRPerfStats',
    'LibOVRHapticsInfo',
    'LibOVRHapticsBuffer',
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
    'isHeadLocked',
    'getPixelsPerTanAngleAtCenter',
    'getTanAngleToRenderTargetNDC',
    'getPixelsPerDegree',
    'getDistortedViewport',
    'getEyeRenderFov',
    'setEyeRenderFov',
    'getEyeAspectRatio',
    'getEyeHorizontalFovRadians',
    'getEyeVerticalFovRadians',
    'getEyeFocalLength',
    'calcEyeBufferSize',
    'getLayerEyeFovFlags',
    'setLayerEyeFovFlags',
    'createTextureSwapChainGL',
    'getTextureSwapChainLengthGL',
    'getTextureSwapChainCurrentIndex',
    'getTextureSwapChainBufferGL',
    'setEyeColorTextureSwapChain',
    'createMirrorTexture',
    'getMirrorTexture',
    'getSensorSampleTime',
    'setSensorSampleTime',
    'getTrackingState',
    'getDevicePoses',
    'calcEyePoses',
    'getHmdToEyePose',
    'setHmdToEyePose',
    'getEyeRenderPose',
    'setEyeRenderPose',
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
    'getTrackingOriginType',
    'setTrackingOriginType',
    'recenterTrackingOrigin',
    'specifyTrackingOrigin',
    'clearShouldRecenterFlag',
    'getTrackerCount',
    'getTrackerInfo',
    'getSessionStatus',
    'getPerfStats',
    'resetPerfStats',
    'getLastErrorInfo',
    'setBoundaryColor',
    'resetBoundaryColor',
    'getBoundaryVisible',
    'showBoundary',
    'hideBoundary',
    'getBoundaryDimensions',
    'testBoundary',
    'getConnectedControllerTypes',
    'updateInputState',
    'getButton',
    'getTouch',
    'getThumbstickValues',
    'getIndexTriggerValues',
    'getHandTriggerValues',
    'setControllerVibration',
    'getHapticsInfo',
    'submitControllerVibration',
    'getControllerPlaybackState',
    'cullPose',
    'checkSessionStarted'
]

from .cimport libovr_capi as capi
from .cimport libovr_math

from libc.stdint cimport int32_t, uint32_t, uintptr_t
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.math cimport pow, tan, M_PI, atan2

cimport numpy as np
import numpy as np
np.import_array()
import warnings


# helper functions and data
RAD_TO_DEGF = <float>180.0 / M_PI
DEG_TO_RADF = M_PI / <float>180.0


cdef float maxf(float a, float b):
    return a if a >= b else b


cdef char* str2bytes(str strIn):
    """Convert UTF-8 encoded strings to bytes."""
    py_bytes = strIn.encode('UTF-8')
    cdef char* to_return = py_bytes

    return to_return


cdef str bytes2str(char* bytesIn):
    """Convert UTF-8 encoded strings to bytes."""
    return bytesIn.decode('UTF-8')

# ------------------------------------------------------------------------------
# Version checker

if capi.OVR_MAJOR_VERSION != 1 or capi.OVR_MINOR_VERSION != 43:
    # raise a warning if the version of the Oculus SDK may be incompatible
    warnings.warn(
        "PsychXR was built using version {major}.{minor} of the Oculus PC SDK "
        "however 1.43 is recommended. This might be perfectly fine if there "
        "aren't any API breaking changes between versions.".format(
            major=capi.OVR_MAJOR_VERSION, minor=capi.OVR_MINOR_VERSION),
        RuntimeWarning)

# ------------------------------------------------------------------------------
# Constants
#
EYE_LEFT = capi.ovrEye_Left
EYE_RIGHT = capi.ovrEye_Right
EYE_COUNT = capi.ovrEye_Count
HAND_LEFT = capi.ovrHand_Left
HAND_RIGHT = capi.ovrHand_Right
HAND_COUNT = capi.ovrHand_Count

# ------------------------------------------------------------------------------
# Includes
#
include "include/libovr_logging.pxi"
include "include/libovr_errors.pxi"
include "include/libovr_wrappers.pxi"
include "include/libovr_session.pxi"
include "include/libovr_api.pxi"
include "include/libovr_timing.pxi"
include "include/libovr_pose.pxi"
include "include/libovr_tracking.pxi"
include "include/libovr_hmdinfo.pxi"
include "include/libovr_view.pxi"
include "include/libovr_frame.pxi"
include "include/libovr_bounds.pxi"
include "include/libovr_perfstats.pxi"
include "include/libovr_input.pxi"
include "include/libovr_haptics.pxi"
include "include/libovr_extras.pxi"
