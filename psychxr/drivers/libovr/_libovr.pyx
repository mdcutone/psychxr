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


# ------------------------------------------------------------------------------
# Initialize module
#

# VR related data persistent across frames
cdef capi.ovrLayerEyeFov _eyeLayer
cdef capi.ovrPosef[2] _eyeRenderPoses
cdef capi.ovrEyeRenderDesc[2] _eyeRenderDesc
# cdef capi.ovrViewScaleDesc _viewScale

# near and far clipping planes
cdef float[2] _nearClip
cdef float[2] _farClip

# prepare the render layer
_eyeLayer.Header.Type = capi.ovrLayerType_EyeFov
_eyeLayer.Header.Flags = \
    capi.ovrLayerFlag_TextureOriginAtBottomLeft | \
    capi.ovrLayerFlag_HighQuality
_eyeLayer.ColorTexture[0] = _eyeLayer.ColorTexture[1] = NULL

# geometric data
cdef libovr_math.Matrix4f[2] _eyeProjectionMatrix
cdef libovr_math.Matrix4f[2] _eyeViewMatrix
cdef libovr_math.Matrix4f[2] _eyeViewProjectionMatrix

# clock offset in seconds
cdef double t_offset = 0.0

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
# Logging Callback
#

cdef void LibOVRLogCallback(uintptr_t userData, int level, char* message) with gil:
    """Callback function for LibOVR logging messages.
    """
    (<object>(<void*>userData))(level, bytes2str(message))


# ------------------------------------------------------------------------------
# Includes
#
include "libovr_const.pxi"
include "libovr_wrappers.pxi"
include "libovr_errors.pxi"
include "libovr_session.pxi"
include "libovr_api.pxi"
include "libovr_pose.pxi"
include "libovr_posestate.pxi"
include "libovr_tracking.pxi"
include "libovr_bounds.pxi"
include "libovr_hmdinfo.pxi"

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
    """Set the render layer state for head locking.

    Head-locking prevents the compositor from applying asynchronous time warp
    (ASW) which compensates for rendering latency. Under normal circumstances
    where head pose data is retrieved from `LibOVR` using
    :func:`getTrackingState` or :func:`getDevicePoses` calls, it
    should be enabled to prevent juddering and improve visual stability.

    However, when using custom head poses (eg. fixed, or from a motion tracker)
    this system may cause the render layer to slip around, as internal IMU data
    will be incongruous with externally supplied head posture data. If you plan
    on passing custom poses to :func:`calcEyePoses`, ensure that head locking is
    enabled.

    Head locking is disabled by default when a session is started.

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


def isHeadLocked():
    """Check if head locking is enabled.

    Returns
    -------
    bool
        ``True`` if head-locking is enabled.

    See Also
    --------
    setHeadLocked

    """
    return (_eyeLayer.Header.Flags & capi.ovrLayerFlag_HeadLocked) == \
           capi.ovrLayerFlag_HeadLocked


def setReferenceTime(double refTime):
    """Set a reference time to synchronize the time source used by the LibOVR
    driver with an external clock.

    This function computes a time offset between the external clock and the one
    used by the LibOVR driver. The offset is then applied when calling any
    function which requires or retrieves absolute time information (eg.
    :func:`getPredictedDisplayTime`). This is useful for cases where the
    application interfacing with the HMD is using its own time source.

    Parameters
    ----------
    refTime : float
        Current time of the external clock in seconds (must be >=0.0).

    Returns
    -------
    float
        The difference between the external and LibOVR time sources in seconds.

    Notes
    -----
    * If the reference time is changed, any previously reported time will be
      invalid.
    * Allows for some error on the order of a few microseconds when the time
      offset is computed.
    * It is assumed that the an external time source operating on the exact same
      frequency as the time source used by LibOVR.

    """
    global t_offset

    if refTime < 0:
        raise ValueError("Value for `refTime` must be >=0.")

    t_offset = refTime - capi.ovr_GetTimeInSeconds()  # compute the offset

    return t_offset


def getFrameOnsetTime(int frameIndex):
    """Get the estimated frame onset time.

    This function **estimates** the onset time of `frameIndex` by subtracting
    half the display's frequency from the predicted mid-frame display time
    reported by LibOVR.

    Returns
    -------
    float
        Estimated onset time of the next frame in seconds.

    Notes
    -----
    * Onset times are estimated and one should use caution when using the
      value reported by this function.

    """
    global _hmdDesc
    global _ptrSession
    cdef double halfRefresh = (1.0 / <double>_hmdDesc.DisplayRefreshRate) / 2.0

    return capi.ovr_GetPredictedDisplayTime(_ptrSession, frameIndex) - \
           halfRefresh


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
    tuple
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


def getPixelsPerDegree(int eye):
    """Get pixels per degree at the center of the display.

    Values reflect the FOVs set by the last call to :func:`setEyeRenderFov` (or
    else the default FOVs will be used.)

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.

    Returns
    -------
    tuple
        Pixels per degree at the center of the screen (h, v).

    """
    global _eyeRenderDesc

    cdef capi.ovrVector2f pixelsPerTanAngle = \
        _eyeRenderDesc[eye].PixelsPerTanAngleAtCenter

    # tan(angle)=1 -> 45 deg
    cdef float horzPixelPerDeg = <float>pixelsPerTanAngle.x / <float>45.0
    cdef float vertPixelPerDeg = <float>pixelsPerTanAngle.y / <float>45.0

    return horzPixelPerDeg, vertPixelPerDeg


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
    tuple
        Eye FOV tangent angles [UpTan, DownTan, LeftTan, RightTan], distance to
        near and far clipping planes in meters.

    Examples
    --------
    Getting the tangent angles::

        leftFov, nearClip, farClip = getEyeRenderFOV(EYE_LEFT)
        # left FOV tangent angles, do the same for the right
        upTan, downTan, leftTan, rightTan =  leftFov

    """
    global _eyeRenderDesc
    global _nearClip
    global _farClip

    cdef np.ndarray to_return = np.asarray([
        _eyeRenderDesc[eye].Fov.UpTan,
        _eyeRenderDesc[eye].Fov.DownTan,
        _eyeRenderDesc[eye].Fov.LeftTan,
        _eyeRenderDesc[eye].Fov.RightTan],
        dtype=np.float32)

    return to_return, _nearClip[eye], _farClip[eye]


def setEyeRenderFov(int eye, object fov, float nearClip=0.01, float farClip=1000.):
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
    fov : array_like
        Eye FOV tangent angles [UpTan, DownTan, LeftTan, RightTan].
    nearClip, farClip : float
        Near and far clipping planes in meters. Used when computing the
        projection matrix.

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
    global _nearClip
    global _farClip
    global _eyeProjectionMatrix

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

    # set clipping planes
    _nearClip[<int>eye] = nearClip
    _farClip[<int>eye] = farClip

    # compute the projection matrix
    _eyeProjectionMatrix[eye] = \
        <libovr_math.Matrix4f>capi.ovrMatrix4f_Projection(
            _eyeRenderDesc[eye].Fov,
            _nearClip[eye],
            _farClip[eye],
            capi.ovrProjection_ClipRangeOpenGL)


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

    Notes
    -----
    * This does not reflect the optical focal length of the HMD.

    """
    return 1.0 / tan(getEyeHorizontalFovRadians(eye) / 2.0)


def getLayerEyeFovFlags():
    """Get header flags for the render layer.

    Returns
    -------
    unsigned int
        Flags from ``OVR::ovrLayerEyeFov.Header.Flags``.

    See Also
    --------
    setLayerEyeFovFlags : Set layer flags.

    Examples
    --------
    Check if a flag is set::

        layerFlags = getLayerEyeFovFlags()
        if (layerFlags & LAYER_FLAG_HIGH_QUALITY) == LAYER_FLAG_HIGH_QUALITY:
            print('high quality enabled!')

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
          render layer transformations to be head referenced.

    See Also
    --------
    getLayerEyeFovFlags : Get layer flags.

    Notes
    -----
    * ``LAYER_FLAG_HIGH_QUALITY`` and
      ``LAYER_FLAG_TEXTURE_ORIGIN_AT_BOTTOM_LEFT`` are recommended settings and
      are enabled by default.

    Examples
    --------
    Enable head-locked mode::

        layerFlags = getLayerEyeFovFlags()  # get current flags
        layerFlags |= LAYER_FLAG_HEAD_LOCKED  # set head-locking
        setLayerEyeFovFlags(layerFlags)  # set the flags again

    """
    global _eyeLayer
    _eyeLayer.Header.Flags = <capi.ovrLayerFlags>flags


include "libovr_gl.pxi"


def calcEyePoses(LibOVRPose headPose, object originPose=None):
    """Calculate eye poses using a given head pose.

    Eye poses are derived from the specified head pose, relative eye poses, and
    the scene tracking origin.

    Calculated eye poses are stored and passed to the compositor when
    :func:`endFrame` is called unless ``LAYER_FLAG_HEAD_LOCKED`` is set. You can
    access the computed poses via the :func:`getEyeRenderPose` function. If
    using custom head poses, ensure :func:`setHeadLocked` is ``True`` or the
    ``LAYER_FLAG_HEAD_LOCKED`` render layer flag is set.

    Parameters
    ----------
    headPose : :py:class:`LibOVRPose`
        Head pose.
    originPose : :py:class:`LibOVRPose`, optional
        Optional world origin pose to transform head pose. You can apply
        transformations to this pose to simulate movement through a scene.

    Examples
    --------

    Compute the eye poses from tracker data::

        abs_time = getPredictedDisplayTime()
        tracking_state, calibrated_origin = getTrackingState(abs_time, True)
        headPoseState, status = tracking_state[TRACKED_DEVICE_TYPE_HMD]

        # calculate head pose
        hmd.calcEyePoses(headPoseState.pose)

        # computed render poses appear here
        renderPoseLeft, renderPoseRight = hmd.getEyeRenderPoses()

    Using external data to set the head pose from a motion capture system::

        # rigid body in the scene defining the scene origin
        rbHead = LibOVRPose(*headRb.posOri)
        calcEyePoses(rbHead)

    Note that the external tracker latency might be larger than builtin
    tracking. To get around this, enable forward prediction in your mocap
    software to equal roughly to average `getPredictedDisplayTime() -
    mocapMidExposureTime`, or time integrate poses to mid-frame time.

    """
    global _ptrSession
    global _eyeLayer
    global _eyeRenderPoses
    global _eyeRenderDesc
    global _eyeViewMatrix
    global _eyeProjectionMatrix
    global _eyeViewProjectionMatrix

    cdef capi.ovrPosef[2] hmdToEyePoses
    hmdToEyePoses[0] = _eyeRenderDesc[0].HmdToEyePose
    hmdToEyePoses[1] = _eyeRenderDesc[1].HmdToEyePose

    # calculate the eye poses
    capi.ovr_CalcEyePoses2(headPose.c_data[0], hmdToEyePoses, _eyeRenderPoses)

    # compute the eye transformation matrices from poses
    cdef libovr_math.Vector3f pos, originPos
    cdef libovr_math.Quatf ori, originOri
    cdef libovr_math.Vector3f up
    cdef libovr_math.Vector3f forward
    cdef libovr_math.Matrix4f rm

    # get origin pose components
    if originPose is not None:
        originPos = <libovr_math.Vector3f>(<LibOVRPose>originPose).c_data.Position
        originOri = <libovr_math.Quatf>(<LibOVRPose>originPose).c_data.Orientation
        if not originOri.IsNormalized():  # make sure orientation is normalized
            originOri.Normalize()

    cdef int eye = 0
    for eye in range(capi.ovrEye_Count):
        if originPose is not None:
            pos = originPos + <libovr_math.Vector3f>_eyeRenderPoses[eye].Position
            ori = originOri * <libovr_math.Quatf>_eyeRenderPoses[eye].Orientation
        else:
            pos = <libovr_math.Vector3f>_eyeRenderPoses[eye].Position
            ori = <libovr_math.Quatf>_eyeRenderPoses[eye].Orientation

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
    tuple (LibOVRPose, LibOVRPose)
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
    for each eye, and used for visibility culling.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.

    Returns
    -------
    tuple (LibOVRPose, LibOVRPose)
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
    global _eyeRenderPoses
    return LibOVRPose.fromPtr(&_eyeRenderPoses[eye])


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
    global _eyeRenderPoses
    global _eyeViewMatrix
    global _eyeProjectionMatrix
    global _eyeViewProjectionMatrix

    _eyeRenderPoses[eye] = eyePose.c_data[0]

    # re-compute the eye transformation matrices from poses
    cdef libovr_math.Vector3f pos
    cdef libovr_math.Quatf ori
    cdef libovr_math.Vector3f up
    cdef libovr_math.Vector3f forward
    cdef libovr_math.Matrix4f rm

    pos = <libovr_math.Vector3f>_eyeRenderPoses[eye].Position
    ori = <libovr_math.Quatf>_eyeRenderPoses[eye].Orientation

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


def getEyeProjectionMatrix(int eye, np.ndarray[np.float32_t, ndim=2] out=None):
    """Compute the projection matrix.

    The projection matrix is computed by the runtime using the eye FOV
    parameters set with :py:attr:`libovr.LibOVRSession.setEyeRenderFov` calls.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.
    out : `ndarray` or `None`, optional
        Alternative matrix to write values to instead of returning a new one.

    Returns
    -------
    ndarray
        4x4 projection matrix.

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
            getEyeProjectionMatrix(eye, out=eyeProjectionMatrices[eye])

    Using eye projection matrices with PyOpenGL (fixed-function)::

        P = getEyeProjectionMatrix(eye)
        glMatrixMode(GL.GL_PROJECTION)
        glLoadTransposeMatrixf(P)

    For `Pyglet` (which is the stardard GL interface for `PsychoPy`), you need
    to convert the matrix to a C-types pointer before passing it to
    `glLoadTransposeMatrixf`::

        P = getEyeProjectionMatrix(eye)
        P = P.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        glMatrixMode(GL.GL_PROJECTION)
        glLoadTransposeMatrixf(P)

    If using fragment shaders, the matrix can be passed on to them as such::

        P = getEyeProjectionMatrix(eye)
        P = P.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # after the program was installed in the current rendering state via
        # `glUseProgram` ...
        loc = glGetUniformLocation(program, b"m_Projection")
        glUniformMatrix4fv(loc, 1, GL_TRUE, P)  # `transpose` must be `True`

    """
    global _eyeProjectionMatrix

    cdef np.ndarray[np.float32_t, ndim=2] to_return

    if out is None:
        to_return = np.zeros((4, 4), dtype=np.float32)
    else:
        to_return = out

    # fast copy matrix to numpy array
    cdef float [:, :] mv = to_return
    cdef Py_ssize_t i, j
    cdef Py_ssize_t N = 4
    i = j = 0
    for i in range(N):
        for j in range(N):
            mv[i, j] = _eyeProjectionMatrix[eye].M[i][j]

    return to_return


def getEyeRenderViewport(int eye, np.ndarray[np.int_t, ndim=1] out=None):
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
    ndarray
        Viewport rectangle [x, y, w, h].

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
    values : array_like
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


def getEyeViewMatrix(int eye, np.ndarray[np.float32_t, ndim=2] out=None):
    """Compute a view matrix for a specified eye.

    View matrices are derived from the eye render poses calculated by the
    last :func:`calcEyePoses` call or update by :func:`setEyeRenderPose`.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.
    out : ndarray or None, optional
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
    frameIndex : int
        Frame index.

    Returns
    -------
    float
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
    float
        Time in seconds.

    """
    cdef double t_sec = capi.ovr_GetTimeInSeconds()

    return t_sec


def waitToBeginFrame(unsigned int frameIndex=0):
    """Wait until a buffer is available so frame rendering can begin. Must be
    called before :func:`beginFrame`.

    Parameters
    ----------
    frameIndex : int
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
    frameIndex : int
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
    eye : int
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
    frameIndex : int
        The target frame index.

    Returns
    -------
    tuple (int, float)
        Error code returned by API call `OVR::ovr_EndFrame` and the absolute
        time in seconds `OVR::ovr_EndFrame` returned.

    """
    global _ptrSession
    global _eyeLayer
    global _eyeRenderPoses

    # if head locking is enabled, make sure the render poses are fixed to
    # HmdToEyePose
    if (_eyeLayer.Header.Flags & capi.ovrLayerFlag_HeadLocked) != \
            capi.ovrLayerFlag_HeadLocked:
        _eyeLayer.RenderPose = _eyeRenderPoses
    else:
        _eyeLayer.RenderPose[0] = _eyeRenderDesc[0].HmdToEyePose
        _eyeLayer.RenderPose[1] = _eyeRenderDesc[1].HmdToEyePose

    cdef capi.ovrLayerHeader* layers = &_eyeLayer.Header
    cdef capi.ovrResult result = capi.ovr_EndFrame(
        _ptrSession,
        frameIndex,
        NULL,
        &layers,
        <unsigned int>1)

    cdef double absTime = capi.ovr_GetTimeInSeconds()

    return result, absTime

include "libovr_perfstats.pxi"
include "libovr_input.pxi"
include "libovr_haptics.pxi"


def cullPose(int eye, LibOVRPose pose):
    """Test if a pose's bounding box or position falls outside of an eye's view
    frustum.

    Poses can be assigned bounding boxes which enclose any 3D models associated
    with them. A model is not visible if all the corners of the bounding box
    fall outside the viewing frustum. Therefore any primitives (i.e. triangles)
    associated with the pose can be culled during rendering to reduce CPU/GPU
    workload.

    If `pose` does not have a valid bounding box (:py:class:`LibOVRBounds`)
    assigned to its :py:attr:`~LibOVRPose.bounds` attribute, this function will
    test is if the position of `pose` is outside the view frustum.

    Parameters
    ----------
    eye : int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.
    pose : LibOVRPose
        Pose to test.

    Returns
    -------
    bool
        ``True`` if the pose's bounding box is not visible to the given `eye`
        and should be culled during rendering.

    Examples
    --------
    Check if a pose should be culled (needs to be done for each eye)::

        cullModel = cullPose(eye, pose)
        if not cullModel:
            # ... OpenGL calls to draw the model here ...

    Notes
    -----
    * Frustums used for testing are defined by the current render FOV for the
      eye (see: :func:`getEyeRenderFov` and :func:`getEyeSetFov`).
    * This function does not test if an object is occluded by another within the
      frustum. If an object is completely occluded, it will still be fully
      rendered, and nearer object will be drawn on-top of it. A trick to
      improve performance in this case is to use ``glDepthFunc(GL_LEQUAL)`` with
      ``glEnable(GL_DEPTH_TEST)`` and render objects from nearest to farthest
      from the head pose. This will reject fragment color calculations for
      occluded locations.

    """
    # This is based on OpenXR's function `XrMatrix4x4f_CullBounds` found in
    # `xr_linear.h`
    global _eyeViewProjectionMatrix

    cdef libovr_math.Bounds3f* bbox
    cdef libovr_math.Vector4f test_point
    cdef libovr_math.Vector4f[8] corners
    cdef Py_ssize_t i

    # compute the MVP matrix to transform poses into HCS
    cdef libovr_math.Matrix4f mvp = \
        _eyeViewProjectionMatrix[eye] * \
        libovr_math.Matrix4f(<libovr_math.Posef>pose.c_data[0])

    if pose.bounds is not None:
        # has a bounding box
        bbox = pose._bbox.c_data

        # bounding box is cleared/not valid, don't cull
        if bbox.b[1].x <= bbox.b[0].x and \
                bbox.b[1].y <= bbox.b[0].y and \
                bbox.b[1].z <= bbox.b[0].z:
            return False

        # compute the corners of the bounding box
        for i in range(8):
            test_point = libovr_math.Vector4f(
                bbox.b[1].x if (i & 1) else bbox.b[0].x,
                bbox.b[1].y if (i & 2) else bbox.b[0].y,
                bbox.b[1].z if (i & 4) else bbox.b[0].z,
                1.0)
            corners[i] = mvp.Transform(test_point)

        # If any of these loops exit normally, the bounding box is completely
        # off to one side of the viewing frustum
        for i in range(8):
            if corners[i].x > -corners[i].w:
                break
        else:
            return True

        for i in range(8):
            if corners[i].x < corners[i].w:
                break
        else:
            return True

        for i in range(8):
            if corners[i].y > -corners[i].w:
                break
        else:
            return True

        for i in range(8):
            if corners[i].y < corners[i].w:
                break
        else:
            return True

        for i in range(8):
            if corners[i].z > -corners[i].w:
                break
        else:
            return True

        for i in range(8):
            if corners[i].z < corners[i].w:
                break
        else:
            return True
    else:
        # no bounding box, cull position of the pose
        test_point = mvp.Transform(
            libovr_math.Vector4f(
                pose.c_data.Position.x,
                pose.c_data.Position.y,
                pose.c_data.Position.z,
                1.0))

        if test_point.x <= -test_point.w:
            return True
        elif test_point.x >= test_point.w:
            return True
        elif test_point.y <= -test_point.w:
            return True
        elif test_point.y >= test_point.w:
            return True
        elif test_point.z <= -test_point.w:
            return True
        elif test_point.z >= test_point.w:
            return True

    return False