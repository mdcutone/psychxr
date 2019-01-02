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
from .cimport ovr_capi
from .cimport ovr_capi_gl
from .cimport ovr_errorcode
from .cimport ovr_capi_util
from .math cimport *

from libc.stdint cimport int32_t

cimport numpy as np
import numpy as np


# -----------------
# Initialize module
# -----------------
#
cdef ovr_capi.ovrInitParams _init_params_  # initialization parameters

# HMD descriptor storing information about the HMD being used.
#
cdef ovr_capi.ovrHmdDesc _hmdDesc_

# Since we are only using one session per module instance, so we are going to
# create our session pointer here and use it module-wide.
#
cdef ovr_capi.ovrSession _ptrSession_
cdef ovr_capi.ovrGraphicsLuid _ptrLuid_

# Array of texture swap chains.
#
cdef ovr_capi.ovrTextureSwapChain _swapChains_[32]

# mirror texture swap chain, we only create one here
#
cdef ovr_capi.ovrMirrorTexture _mirrorTexture_ = NULL

# Persistent VR related structures to store head pose and other data used across
# frames.
#
cdef ovr_capi.ovrEyeRenderDesc[2] _eyeRenderDesc_
cdef ovr_capi.ovrPosef[2] _hmd_to_eye_view_pose_

# Render layer
#
cdef ovr_capi.ovrLayerEyeFov _eyeLayer_

# Arrays to store device poses.
#
cdef ovr_capi.ovrTrackedDeviceType[9] _device_types_
cdef ovr_capi.ovrPoseStatef[9] _device_poses_

# Function to check for errors returned by OVRLib functions
#
cdef ovr_errorcode.ovrErrorInfo _last_error_info_  # store our last error here
def check_result(result):
    if ovr_errorcode.OVR_FAILURE(result):
        ovr_capi.ovr_GetLastErrorInfo(&_last_error_info_)
        raise RuntimeError(
            str(result) + ": " + _last_error_info_.ErrorString.decode("utf-8"))

# Enable error checking on OVRLib functions by setting 'debug_mode=True'. All
# LibOVR functions that return a 'ovrResult' type will be checked. A
# RuntimeError will be raised if the returned value indicates failure with the
# associated message passed from LibOVR.
#
debug_mode = False

# Controller indices in controller state array.
#
ctypedef enum LibOVRControllers:
    xbox = 0
    remote = 1
    touch = 2
    left_touch = 3
    right_touch = 4
    count = 5

# Store controller states.
#
cdef ovr_capi.ovrInputState _ctrl_states_[5]
cdef ovr_capi.ovrInputState _ctrl_states_prev_[5]  # previous controller states

# Controller indices look-up table.
#
cdef dict ctrl_index_lut = {
    "xbox": LibOVRControllers.xbox,
    "remote": LibOVRControllers.remote,
    "touch": LibOVRControllers.touch,
    "left_touch": LibOVRControllers.left_touch,
    "right_touch": LibOVRControllers.right_touch
}

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
button_names = list(ctrl_button_lut.keys())

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
touch_names = list(ctrl_touch_lut.keys())

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
    "AppRenderTiming" : ovr_capi.ovrPerfHud_AppRenderTiming,
    "LatencyTiming" : ovr_capi.ovrPerfHud_LatencyTiming,
    "CompRenderTiming" : ovr_capi.ovrPerfHud_CompRenderTiming,
    "AswStats" : ovr_capi.ovrPerfHud_AswStats,
    "VersionInfo" : ovr_capi.ovrPerfHud_VersionInfo
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

# -----------------
# Session Functions
# -----------------
#
cpdef bint isOculusServiceRunning(int timeout_milliseconds=100):
    cdef ovr_capi_util.ovrDetectResult result = ovr_capi_util.ovr_Detect(
        timeout_milliseconds)

    return <bint> result.IsOculusServiceRunning

cpdef bint isHmdConnected(int timeout_milliseconds=100):
    cdef ovr_capi_util.ovrDetectResult result = ovr_capi_util.ovr_Detect(
        timeout_milliseconds)

    return <bint> result.IsOculusHMDConnected


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
    cdef ovr_capi.ovrTextureSwapChain swapChains[2]
    cdef ovr_capi.ovrMirrorTexture mirrorTexture

    # status and performance information
    cdef ovr_capi.ovrSessionStatus sessionStatus
    cdef ovr_capi.ovrPerfStats perfStats

    # tracking states

    # controller states

    # error information
    cdef ovr_errorcode.ovrErrorInfo errorInfo  # store our last error here

    # debug mode
    cdef bint debugMode

    def __init__(self, debugMode=False, timeout=100, *args, **kwargs):
        pass

    def __cinit__(self, bint debugMode=False, int timeout=100, *args, **kwargs):
        self.debugMode = debugMode

        # check if the driver and service are available
        cdef ovr_capi_util.ovrDetectResult result = ovr_capi_util.ovr_Detect(
            <int>timeout)

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
        """Get the user's height in meters as reported by the LibOVR.

        Returns
        -------
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
        """Get the height of the user's eye from the floor in meters as reported
        by LibOVR. If not set, the default value is 1.675 meters.

        Returns
        -------
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
        """Distance from the neck to eyes.

        Returns
        -------
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
        """Distance between the nose and eyes.

        Returns
        -------
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

        Returns
        -------
        str
            Product name string (utf-8).

        """
        return self.hmdDesc.ProductName.decode('utf-8')

    @property
    def manufacturerName(self):
        """Get the device manufacturer name.

        Returns
        -------
        str
            Manufacturer name string (utf-8).

        """
        return self.hmdDesc.Manufacturer.decode('utf-8')

    @property
    def screenSize(self):
        """Get the horizontal and vertical resolution of the screen in pixels.

        Returns
        -------
        ndarray of int
            Resolution of the display [w, h].

        """
        return np.asarray(
            (self.hmdDesc.Resolution.w, self.hmdDesc.Resolution.h),
            dtype=int)

    @property
    def refreshRate(self):
        """Get the nominal refresh rate in Hertz of the display.

        Returns
        -------
        float
            Refresh rate in Hz.

        """
        return <float>self.hmdDesc.DisplayRefreshRate

    @property
    def hid(self):
        """Get the USB human interface device class identifiers.

        Returns
        -------
        tuple
            USB HIDs (vendor, product).

        """
        return <int>self.hmdDesc.VendorId, <int>self.hmdDesc.ProductId

    @property
    def firmwareVersion(self):
        """Get the firmware version for this device.

        Returns
        -------
        tuple
            Firmware version (major, minor).

        """
        return <int>self.hmdDesc.FirmwareMajor, <int>self.hmdDesc.FirmwareMinor

    @property
    def versionString(self):
        """LibOVRRT version as a string."""
        return self.getVersionString()

    def getVersionString(self):
        cdef const char* version = ovr_capi.ovr_GetVersionString()

        return version.decode('utf-8')  # already UTF-8?

    def start(self):
        """Start a new session. Control is handed over to the application from
        Oculus Home.

        """
        cdef ovr_capi.ovrResult result = 0
        self.initParams.RequestedMinorVersion = 25
        result = ovr_capi.ovr_Initialize(&self.initParams)

        result = ovr_capi.ovr_Create(&self.ptrSession, &self.ptrLuid)
        if ovr_errorcode.OVR_FAILURE(result):
            ovr_capi.ovr_Shutdown()

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

        # prepare the render layer
        self.eyeLayer.Header.Type = ovr_capi.ovrLayerType_EyeFov
        self.eyeLayer.Header.Flags = \
            ovr_capi.ovrLayerFlag_TextureOriginAtBottomLeft | \
            ovr_capi.ovrLayerFlag_HighQuality
        self.eyeLayer.ColorTexture[0] = self.eyeLayer.ColorTexture[1] = NULL

    def shutdown(self):
        """End the current session.

        Clean-up routines are executed that destroy all swap chains and mirror
        texture buffers, afterwards control is returned to Oculus Home. This
        must be called after every successful 'startSession' call.

        Returns
        -------
        None

        """
        # switch off the performance HUD
        cdef ovr_capi.ovrBool ret = ovr_capi.ovr_SetInt(
            self.ptrSession, b"PerfHudMode", ovr_capi.ovrPerfHud_Off)

        # destroy the mirror texture
        ovr_capi.ovr_DestroyMirrorTexture(self.ptrSession, self.mirrorTexture)

        # free all swap chains
        cdef int i = 0
        for i in range(2):
            if not self.swapChains[i] is NULL:
                ovr_capi.ovr_DestroyTextureSwapChain(
                    self.ptrSession, self.swapChains[i])
                self.swapChains[i] = NULL

        # destroy the current session and shutdown
        ovr_capi.ovr_Destroy(self.ptrSession)
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
        """Enable high-quality mode if True. The distortion compositor applies
        4x anisotropic texture filtering which reduces the visibility of
        artifacts, particularly in the periphery.

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
        """Enable head locking mode if True. Head locking places the rendered
        image as a 'billboard' in front of the viewer.

        This is disabled by default when a session is started.

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
        return self.getTrackerCount()

    def getTrackerCount(self):
        return <int>ovr_capi.ovr_GetTrackerCount(self.ptrSession)

    @property
    def defaultEyeFov(self):
        """Default or recommended eye field-of-views (FOVs) provided by the API.
        """
        return self.getDefaultEyeFov()

    def getDefaultEyeFov(self):
        """Get the default field-of-view (FOV) for the HMD for both eyes.

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
    def maxEyeFov(self):
        """Maximum eye field-of-views (FOVs) provided by the API.

        """
        return self.getMaxEyeFov()

    def getMaxEyeFov(self):
        """Get the maximum field-of-view (FOV) for the HMD.

        Returns
        -------
        tuple of ndarray
            Pair of left and right eye FOVs specified as tangent angles [Up,
            Down, Left, Right].

        """
        cdef np.ndarray fovLeft = np.asarray([
            self.hmdDesc.MaxEyeFov[0].UpTan,
            self.hmdDesc.MaxEyeFov[0].DownTan,
            self.hmdDesc.MaxEyeFov[0].LeftTan,
            self.hmdDesc.MaxEyeFov[0].RightTan],
            dtype=np.float32)

        cdef np.ndarray fovRight = np.asarray([
            self.hmdDesc.MaxEyeFov[1].UpTan,
            self.hmdDesc.MaxEyeFov[1].DownTan,
            self.hmdDesc.MaxEyeFov[1].LeftTan,
            self.hmdDesc.MaxEyeFov[1].RightTan],
            dtype=np.float32)

        return fovLeft, fovRight

    def getEyeRenderFov(self, int eye):
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

    def setEyeRenderFov(self, int eye, object fov):
        """Set the field-of-view of a given eye. This is used to compute the
        projection matrix.

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

    def getEyeBufferSize(self, int eye, object fov, float texelPerPixel=1.0):
        """Get the recommended render buffer size in pixels.

        Parameters
        ----------
        eye : int
            Eye index.
        fov : tuple, list or ndarray of floats
        texelPerPixel : float

        Returns
        -------
        ndarray of int
            Resolution of the display [w, h].

        """
        cdef ovr_capi.ovrFovPort fov_in
        fov_in.UpTan = <float>fov[0]
        fov_in.DownTan = <float>fov[1]
        fov_in.LeftTan = <float>fov[2]
        fov_in.RightTan = <float>fov[3]

        cdef ovr_capi.ovrSizei bufferSize = ovr_capi.ovr_GetFovTextureSize(
            self.ptrSession,
            <ovr_capi.ovrEyeType>eye,
            fov_in,
            texelPerPixel)

        return np.asarray([bufferSize.w, bufferSize.h], dtype=np.int)

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

        Returns
        -------
        None

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
            ovr_capi_gl.ovr_CreateTextureSwapChainGL(
                self.ptrSession,
                &swapConfig,
                &self.swapChains[eye])

        self.eyeLayer.ColorTexture[eye] = self.swapChains[eye]

        if debug_mode:
            check_result(result)

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

        if debug_mode:
            check_result(result)

        # get the next available texture ID from the swap chain
        result = ovr_capi_gl.ovr_GetTextureSwapChainBufferGL(
            self.ptrSession, self.swapChains[eye], current_idx, &tex_id)

        if debug_mode:
            check_result(result)

        return tex_id

    def getEyeProjectionMatrix(self, eye, nearClip=0.1, farClip=1000.0):
        """Create a projection matrix.

        Parameters
        ----------
        eye : int
            Eye index.
        nearClip : float
            Near clipping plane in meters.
        farClip
            Far clipping plane in meters.

        Returns
        -------
        ndarray
            4x4 projection matrix.

        """
        cdef ovr_capi.ovrMatrix4f projMat = \
            ovr_capi_util.ovrMatrix4f_Projection(
                self.eyeRenderDesc[eye].Fov,
                nearClip,
                farClip,
                ovr_capi_util.ovrProjection_ClipRangeOpenGL)

        cdef np.ndarray to_return = np.zeros((4, 4), dtype=np.float32)

        # fast copy matrix to numpy array
        cdef float [:, :] mv = to_return
        cdef Py_ssize_t i, j
        i = j = 0
        for i in range(4):
            for j in range(4):
                mv[i, j] = projMat.M[i][j]

        return to_return

    def getEyeViewMatrix(self, eyePose):
        """Create a projection matrix.

        Parameters
        ----------
        eyePose : tuple, list, or ndarray of float
            Eye pose.
        nearClip : float
            Near clipping plane in meters.
        farClip
            Far clipping plane in meters.

        Returns
        -------
        ndarray
            4x4 projection matrix.

        """

        cdef np.ndarray pos = np.zeros((4,), dtype=np.float32)
        pos[0] = eyePose.translation.x
        pos[1] = eyePose.translation.y
        pos[2] = eyePose.translation.z
        pos[3] = 1.0

        cdef float qw = eyePose.rotation.w
        cdef float qx = eyePose.rotation.x
        cdef float qy = eyePose.rotation.y
        cdef float qz = eyePose.rotation.z

        cdef float qw2 = qw * qw
        cdef float qx2 = qx * qx
        cdef float qy2 = qy * qy
        cdef float qz2 = qz * qz

        # build the rotation matrix from the pose quaternion
        cdef np.ndarray rotMat = np.zeros((4, 4), dtype=np.float32)
        cdef float[:, :] rmat_view = rotMat  # fast-access
        rmat_view[0, 0] = (qw2 + qx2 - qy2 - qz2)
        rmat_view[1, 0] = 2.0 * (qx * qy + qw * qz)
        rmat_view[2, 0] = 2.0 * (qx * qz - qw * qy)
        #rmat_view[3, 0] = 0.0
        rmat_view[0, 1] = 2.0 * (qx * qy - qw * qz)
        rmat_view[1, 1] = (qw2 - qx2 + qy2 - qz2)
        rmat_view[2, 1] = 2.0 * (qy * qz + qw * qx)
        #rmat_view[3, 1] = 0.0
        rmat_view[0, 2] = 2.0 * (qx * qz + qw * qy)
        rmat_view[1, 2] = 2.0 * (qy * qz - qw * qx)
        rmat_view[2, 2] = (qw2 - qx2 - qy2 + qz2)
        #rmat_view[3, 2] = 0.0
        #rmat_view[:3, 3] = 0.0
        rmat_view[3, 3] = 1.0

        cdef np.ndarray finalUp = np.asarray((0, 1, 0, 1), dtype=np.float32)
        cdef np.ndarray finalFwd = np.asarray((0, 0, -1, 1), dtype=np.float32)

        np.matmul(finalUp, rotMat.T, finalUp)
        np.matmul(finalFwd, rotMat.T, finalFwd)

        finalUp = finalUp[:3]
        finalFwd = finalFwd[:3]
        pos = pos[:3]

        cdef np.ndarray centerPos = pos + finalFwd
        cdef np.ndarray f = centerPos - pos
        f /= np.linalg.norm(f)
        finalUp /= np.linalg.norm(finalUp)

        cdef np.ndarray s = np.cross(f, finalUp)
        cdef np.ndarray u = np.cross(s / np.linalg.norm(s), f)

        rotMat = np.zeros((4, 4), np.float32)
        rotMat[0, :3] = s
        rotMat[1, :3] = u
        rotMat[2, :3] = -f
        rotMat[3, 3] = 1.0

        transMat = np.zeros((4, 4), np.float32)
        np.fill_diagonal(transMat, 1.0)
        transMat[:3, 3] = -pos

        return np.matmul(rotMat, transMat)

    def getPredictedDisplayTime(self, frameIndex=0):
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
        cdef double t_sec = ovr_capi.ovr_GetPredictedDisplayTime(
            self.ptrSession,
            <int>frameIndex)

        return t_sec

    def getTimeInSeconds(self):
        """Get the absolute time in seconds.

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
            Performance HUD mode to present. Valid mode strings are
            'PerfSummary', 'LatencyTiming', 'AppRenderTiming', 
            'CompRenderTiming', 'AswStats', 'VersionInfo' and 'Off'. Specifying 
            'Off' hides the performance HUD.

        Returns
        -------
        None
            
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

        Returns
        -------
        None

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

        Returns
        -------
        None

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
            Error code.

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
        """Make the eye render texture available to the compositor.

        Parameters
        ----------
        eye : int
            Eye buffer index.

        Returns
        -------
        int
            Error code returned by 'ovr_CommitTextureSwapChain'.

        """
        cdef ovr_capi.ovrResult result = ovr_capi.ovr_CommitTextureSwapChain(
            self.ptrSession,
            self.swapChains[eye])

        if debug_mode:
            check_result(result)

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
            Error code returned by 'ovr_EndFrame'.

        """
        cdef ovr_capi.ovrLayerHeader* layers = &(self.eyeLayer).Header
        cdef ovr_capi.ovrResult result = ovr_capi.ovr_EndFrame(
            self.ptrSession,
            frameIndex,
            NULL,
            &layers,
            <unsigned int> 1)

        if debug_mode:
            check_result(result)

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
        return self.getTrackingOriginType()

    @trackingOriginType.setter
    def trackingOriginType(self, str value):
        self.setTrackingOriginType(value)

    def setTrackingOriginType(self, str value):
        """Set the tracking origin type. Can either be 'floor' or 'eye'.

        """
        cdef ovr_capi.ovrResult result
        if value == 'floor':
            result = ovr_capi.ovr_SetTrackingOriginType(
                self.ptrSession, ovr_capi.ovrTrackingOrigin_FloorLevel)
        elif value == 'eye':
            result = ovr_capi.ovr_SetTrackingOriginType(
                self.ptrSession, ovr_capi.ovrTrackingOrigin_EyeLevel)

        if debug_mode:
            check_result(result)

    def getTrackingOriginType(self):
        """Get the current tracking origin type.

        """
        cdef ovr_capi.ovrTrackingOrigin origin = \
            ovr_capi.ovr_GetTrackingOriginType(self.ptrSession)

        if origin == ovr_capi.ovrTrackingOrigin_FloorLevel:
            return 'floor'
        elif origin == ovr_capi.ovrTrackingOrigin_EyeLevel:
            return 'eye'

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



# ---------------------------------
# Rendering Configuration Functions
# ---------------------------------
#
# layer header flags
ovrLayerFlag_HighQuality = 0x01
ovrLayerFlag_TextureOriginAtBottomLeft = 0x02
ovrLayerFlag_HeadLocked = 0x04

# Texture types supported by the PC version of LibOVR
#
ovrTexture_2D = ovr_capi.ovrTexture_2D
ovrTexture_Cube = ovr_capi.ovrTexture_Cube

# Texture formats supported by OpenGL
#
OVR_FORMAT_R8G8B8A8_UNORM = ovr_capi.OVR_FORMAT_R8G8B8A8_UNORM
OVR_FORMAT_R8G8B8A8_UNORM_SRGB = ovr_capi.OVR_FORMAT_R8G8B8A8_UNORM_SRGB
OVR_FORMAT_B8G8R8A8_UNORM = ovr_capi.OVR_FORMAT_B8G8R8A8_UNORM
OVR_FORMAT_B8G8R8_UNORM = ovr_capi.OVR_FORMAT_B8G8R8_UNORM
OVR_FORMAT_R16G16B16A16_FLOAT = ovr_capi.OVR_FORMAT_R16G16B16A16_FLOAT
OVR_FORMAT_R11G11B10_FLOAT = ovr_capi.OVR_FORMAT_R11G11B10_FLOAT
OVR_FORMAT_D16_UNORM = ovr_capi.OVR_FORMAT_D16_UNORM
OVR_FORMAT_D24_UNORM_S8_UINT = ovr_capi.OVR_FORMAT_D24_UNORM_S8_UINT
OVR_FORMAT_D32_FLOAT = ovr_capi.OVR_FORMAT_D32_FLOAT
OVR_FORMAT_D32_FLOAT_S8X24_UINT = ovr_capi.OVR_FORMAT_D32_FLOAT_S8X24_UINT

SWAP_CHAIN_TEXTURE0 = 0
SWAP_CHAIN_TEXTURE1 = 1

def createTextureSwapChainGL(
        LibOVRSession session,
        eye,
        textureFormat,
        width,
        height,
        levels=1):
    """Initialize a texture swap chain.

    :param swap_desc: ovrTextureSwapChainDesc
    :return: int

    """
    #global _swapChains_, _ptrSession_

    # check if the swap chain is available (if NULL)
    # if 0 > swapChainIndex >= 2:
    #     if _swapChains_[swapChainIndex] is not NULL:
    #         raise RuntimeError(
    #             "Swap chain at index '{}' already initialized!".format(
    #                 swapChainIndex))
    #     raise IndexError(
    #         "Swap chain index '{}' out-of-range, must be >0 and <32.".format(
    #             swapChainIndex))

    # configure the texture
    cdef ovr_capi.ovrTextureSwapChainDesc swapConfig
    swapConfig.Type = ovr_capi.ovrTexture_2D
    swapConfig.Format = textureFormat
    swapConfig.ArraySize = 1
    swapConfig.Width = <int>width
    swapConfig.Height = <int>height
    swapConfig.MipLevels = <int>levels
    swapConfig.SampleCount = 1
    swapConfig.StaticImage = ovr_capi.ovrFalse
    swapConfig.MiscFlags = ovr_capi.ovrTextureMisc_None
    swapConfig.BindFlags = ovr_capi.ovrTextureBind_None

    # create the swap chain
    cdef ovr_capi.ovrResult result = ovr_capi_gl.ovr_CreateTextureSwapChainGL(
        session.ptrSession,
        &swapConfig,
        &session.swapChains[eye])

    global _eyeLayer_
    session.eyeLayer.ColorTexture[eye] = session.swapChains[eye]

    if debug_mode:
        check_result(result)

# types
ovrLayerType_EyeFov = ovr_capi.ovrLayerType_EyeFov

# layer header flags
ovrLayerFlag_HighQuality = ovr_capi.ovrLayerFlag_HighQuality
ovrLayerFlag_TextureOriginAtBottomLeft = ovr_capi.ovrLayerFlag_TextureOriginAtBottomLeft
ovrLayerFlag_HeadLocked = ovr_capi.ovrLayerFlag_HeadLocked

ovrEye_Left = ovr_capi.ovrEye_Left
ovrEye_Right = ovr_capi.ovrEye_Right
ovrEye_Count = ovr_capi.ovrEye_Count

cpdef ovrSizei getFovTextureSize(
        int eye_type,
        ovrFovPort fov,
        float texels_per_pixel=1.0):
    """Compute the recommended buffer (texture) size for a specified 
    configuration.
    
    Returns a tuple with the dimensions of the required texture (w, h). The 
    values can be used when configuring a render buffer which will ultimately
    be used to draw to the HMD buffers.
    
    :return: None 
    
    """
    cdef ovrSizei to_return = ovrSizei()
    (<ovrSizei> to_return).c_data[0] = ovr_capi.ovr_GetFovTextureSize(
        _ptrSession_,
        <ovr_capi.ovrEyeType> eye_type,
        fov.c_data[0],
        texels_per_pixel)

    return to_return

cpdef void configEyeRenderDesc(LibOVRSession session, int eye_type, object fov):
    """Compute eye render descriptors for a given eye. 
    
    Each eye has an internal 'ovrEyeRenderDesc' structure which stores computed
    information which is not accessible directly from Python. You must call this
    function twice (for each eye) to fully configure the descriptors.

    :param eye_type: int
    :param fov: ovrFovPort
    :return: None
    
    """
    global _hmd_to_eye_view_pose_, _eyeLayer_
    cdef ovr_capi.ovrFovPort fov_in
    fov_in.UpTan = fov[0]
    fov_in.DownTan = fov[1]
    fov_in.LeftTan = fov[2]
    fov_in.RightTan = fov[3]

    session.eyeRenderDesc[eye_type] = ovr_capi.ovr_GetRenderDesc(
        session.ptrSession,
        <ovr_capi.ovrEyeType> eye_type,
        fov_in)

    # set the initial eye pose
    _hmd_to_eye_view_pose_[eye_type] = session.eyeRenderDesc[eye_type].HmdToEyePose

    # set the render layer FOV to what is computed
    session.eyeLayer.Fov[eye_type] = session.eyeRenderDesc[eye_type].Fov

cpdef list getHmdToEyePose(LibOVRSession session):
    """Get the HMD to eye poses from the internal eye render descriptor.
    
    :return: 
    
    """
    global _eyeRenderDesc_
    cdef ovrPosef hmdToEyePoseLeft = ovrPosef()
    cdef ovrPosef hmdToEyePoseRight = ovrPosef()

    (<ovrPosef> hmdToEyePoseLeft).c_data[0] = \
        <ovr_math.Posef> session.eyeRenderDesc[0].HmdToEyePose
    (<ovrPosef> hmdToEyePoseRight).c_data[0] = \
        <ovr_math.Posef> session.eyeRenderDesc[1].HmdToEyePose

    return [hmdToEyePoseLeft, hmdToEyePoseRight]

cpdef void setRenderSwapChain(LibOVRSession session, int eye, object swap_chain):
    """Set the swap chain for the render layer.

    :param eye: str
    :param swap_chain: int or None
    :return: None
    
    """
    # set the swap chain textures
    global _eyeLayer_
    if not swap_chain is None:
        session.eyeLayer.ColorTexture[eye] = _swapChains_[<int> swap_chain]
    else:
        session.eyeLayer.ColorTexture[eye] = NULL

cpdef ovrRecti getRenderViewport(LibOVRSession session, int eye):
    """Get the viewport rectangle for a given eye view. These will return the
    viewports set by the previous 'setRenderViewport' call.
    
    :param eye: int
    :return: None
    
    """
    global _ptrSession_, _eyeLayer_
    cdef ovrRecti to_return = ovrRecti()
    (<ovrRecti> to_return).c_data[0] = session.eyeLayer.Viewport[eye]

    return to_return

cpdef void setRenderViewport(int eye, ovrRecti viewPortRect):
    """Set the viewport rectangle for a specified eye view. This defines where
    on the swap texture the eye view is to be drawn/retrieved.
    
    :param eye: int
    :param viewPortRect: ovrRecti
    :return: None
    
    """
    global _eyeLayer_
    _eyeLayer_.Viewport[eye] = viewPortRect.c_data[0]

cpdef int getRenderLayerFlags():
    """Get the render layer's header flags.
    
    :return: int
    
    """
    global _eyeLayer_
    return <int> _eyeLayer_.Header.Flags

cpdef void setRenderLayerFlags(int layerHeaderFlags):
    """Set the render layer's header flags.
    
    :param layerHeaderFlags: 
    :return: None
    
    """
    global _eyeLayer_
    _eyeLayer_.Header.Flags = layerHeaderFlags

# ---------------------------------
# VR Tracking Classes and Functions
# ---------------------------------
#
cdef class ovrPoseStatef(object):
    """Pose state data.

    """
    cdef ovr_capi.ovrPoseStatef*c_data
    cdef ovr_capi.ovrPoseStatef  c_ovrPoseStatef

    cdef ovrPosef field_the_pose
    cdef ovrVector3f field_angular_velocity
    cdef ovrVector3f field_linear_velocity
    cdef ovrVector3f field_angular_acceleration
    cdef ovrVector3f field_linear_acceleration

    def __cinit__(self, *args, **kwargs):
        self.c_data = &self.c_ovrPoseStatef

        self.field_angular_velocity = ovrVector3f()
        self.field_linear_velocity = ovrVector3f()
        self.field_angular_acceleration = ovrVector3f()
        self.field_linear_acceleration = ovrVector3f()

    @property
    def ThePose(self):
        cdef ovrPosef to_return = ovrPosef()
        (<ovrPosef> to_return).c_data[0] = <ovr_math.Posef> self.c_data[
            0].ThePose

        return to_return

    @property
    def AngularVelocity(self):
        self.field_angular_velocity.c_data[0] = \
            (<ovr_math.Vector3f> self.c_data[0].AngularVelocity)

        return self.field_angular_velocity

    @property
    def LinearVelocity(self):
        self.field_linear_velocity.c_data[0] = \
            (<ovr_math.Vector3f> self.c_data[0].LinearVelocity)

        return self.field_linear_velocity

    @property
    def AngularAcceleration(self):
        self.field_angular_acceleration.c_data[0] = \
            (<ovr_math.Vector3f> self.c_data[0].AngularAcceleration)

        return self.field_angular_acceleration

    @property
    def LinearAcceleration(self):
        self.field_linear_acceleration.c_data[0] = \
            (<ovr_math.Vector3f> self.c_data[0].LinearAcceleration)

        return self.field_linear_acceleration

    @property
    def TimeInSeconds(self):
        return <double> self.c_data[0].TimeInSeconds

ovrStatus_OrientationTracked = ovr_capi.ovrStatus_OrientationTracked
ovrStatus_PositionTracked = ovr_capi.ovrStatus_PositionTracked

cdef class TrackingStateData(object):
    """Structure which stores tracking state information. All attributes are
    read-only, returning a copy of the data in the accessed field.

    """
    cdef ovr_capi.ovrTrackingState*c_data
    cdef ovr_capi.ovrTrackingState  c_ovrTrackingState

    def __cinit__(self, *args, **kwargs):
        self.c_data = &self.c_ovrTrackingState

    @property
    def HeadPose(self):
        cdef ovrPoseStatef to_return = ovrPoseStatef()
        (<ovrPoseStatef> to_return).c_data[0] = self.c_data[0].HeadPose

        return to_return

    @property
    def StatusFlags(self):
        return <unsigned int> self.c_data[0].StatusFlags

    @property
    def HandPoses(self):
        cdef ovrPoseStatef left_hand_pose = ovrPoseStatef()
        (<ovrPoseStatef> left_hand_pose).c_data[0] = self.c_data[0].HandPoses[0]

        cdef ovrPoseStatef right_hand_pose = ovrPoseStatef()
        (<ovrPoseStatef> right_hand_pose).c_data[0] = self.c_data[0].HandPoses[
            1]

        return left_hand_pose, right_hand_pose

    @property
    def HandStatusFlags(self):
        return <unsigned int> self.c_data[0].HandStatusFlags[0], \
               <unsigned int> self.c_data[0].HandStatusFlags[1]

cpdef TrackingStateData getTrackingState(LibOVRSession session,
        double abs_time,
        bint latency_marker=True):
    cdef ovr_capi.ovrBool use_marker = \
        ovr_capi.ovrTrue if latency_marker else ovr_capi.ovrFalse

    cdef ovr_capi.ovrTrackingState ts = ovr_capi.ovr_GetTrackingState(
        session.ptrSession, abs_time, use_marker)

    cdef TrackingStateData to_return = TrackingStateData()
    (<TrackingStateData> to_return).c_data[0] = ts

    return to_return

cpdef void setTrackingOriginType(str origin='floor'):
    """Set the tracking origin type. Can either be 'floor' or 'eye'.
    
    :param origin: str
    :return: 
    """
    global _ptrSession_
    cdef ovr_capi.ovrResult result
    if origin == 'floor':
        result = ovr_capi.ovr_SetTrackingOriginType(
            _ptrSession_, ovr_capi.ovrTrackingOrigin_FloorLevel)
    elif origin == 'eye':
        result = ovr_capi.ovr_SetTrackingOriginType(
            _ptrSession_, ovr_capi.ovrTrackingOrigin_EyeLevel)

    if debug_mode:
        check_result(result)

cpdef str getTrackingOriginType():
    """Get the current tracking origin type.
    
    :return: str
    """
    global _ptrSession_
    cdef ovr_capi.ovrTrackingOrigin origin = ovr_capi.ovr_GetTrackingOriginType(
        _ptrSession_)

    if origin == ovr_capi.ovrTrackingOrigin_FloorLevel:
        return 'floor'
    elif origin == ovr_capi.ovrTrackingOrigin_EyeLevel:
        return 'eye'

cpdef void recenterTrackingOrigin():
    """Recenter the tracking origin.
    
    :return: None
    
    """
    global _ptrSession_
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_RecenterTrackingOrigin(
        _ptrSession_)

    if debug_mode:
        check_result(result)

cpdef void specifyTrackingOrigin(ovrPosef originPose):
    """Specify a custom tracking origin.
    
    :param origin_pose: ovrVector3f
    :return: 
    """
    global _ptrSession_
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_SpecifyTrackingOrigin(
        _ptrSession_, <ovr_capi.ovrPosef> originPose.c_data[0])

    if debug_mode:
        check_result(result)

cpdef list calcEyePoses(LibOVRSession session, TrackingStateData trackingState):
    """Calculate eye poses from tracking state data.
    
    Poses are stored internally for conversion to transformation matrices by 
    calling 'get_eye_view_matrix'. Should be called at least once per frame 
    after 'wait_to_begin_frame' but before 'begin_frame' to minimize 
    motion-to-photon latency.
    
    :param tracking_state: TrackingStateData
    :return: 
    
    """
    global _hmd_to_eye_view_pose_, _eyeLayer_

    ovr_capi_util.ovr_CalcEyePoses2(
        trackingState.c_data[0].HeadPose.ThePose,
        _hmd_to_eye_view_pose_,
        session.eyeLayer.RenderPose)

    cdef ovrPosef eye_pose0 = ovrPosef()
    cdef ovrPosef eye_pose1 = ovrPosef()
    (<ovrPosef> eye_pose0).c_data[0] = <ovr_math.Posef> session.eyeLayer.RenderPose[
        0]
    (<ovrPosef> eye_pose1).c_data[0] = <ovr_math.Posef> session.eyeLayer.RenderPose[
        1]

    return [eye_pose0, eye_pose1]

cpdef ovrMatrix4f getEyeViewMatrix(ovrPosef eyePose):
    """Get the view matrix from the last calculated head pose. This should be
    called once per frame if real-time head tracking is desired.
    
    :param eye: str
    :return: 
    
    """
    cdef ovrVector3f pos = ovrVector3f()
    cdef ovrMatrix4f rot = ovrMatrix4f()
    pos.c_data[0] = <ovr_math.Vector3f> eyePose.c_data.Translation
    rot.c_data[0] = ovr_math.Matrix4f(<ovr_math.Quatf> eyePose.c_data.Rotation)

    cdef ovrVector3f final_up = \
        (<ovrVector3f> rot).transform(ovrVector3f(0, 1, 0))
    cdef ovrVector3f final_forward = \
        (<ovrVector3f> rot).transform(ovrVector3f(0, 0, -1))
    cdef ovrMatrix4f viewMatrix = \
        ovrMatrix4f.lookAt(pos, pos + final_forward, final_up)

    return viewMatrix

cpdef ovrMatrix4f getEyeProjectionMatrix(
        LibOVRSession session,
        int eye,
        float near_clip=0.2,
        float far_clip=1000.0):
    """Get the projection matrix for a specified eye. These do not need to be
    computed more than once per session unless the render layer descriptors are 
    updated, or the clipping planes have been changed.
    
    :param eye: str 
    :param near_clip: float
    :param far_clip: float
    :return: 
    
    """
    global _eyeLayer_

    cdef ovrMatrix4f projectionMatrix = ovrMatrix4f()
    (<ovrMatrix4f> projectionMatrix).c_data[0] = \
        <ovr_math.Matrix4f> ovr_capi_util.ovrMatrix4f_Projection(
            session.eyeLayer.Fov[eye],
            near_clip,
            far_clip,
            ovr_capi_util.ovrProjection_ClipRangeOpenGL)

    return projectionMatrix

# -------------------------
# Frame Rendering Functions
# -------------------------
#
cpdef double getDisplayTime(unsigned int frameIndex=0, bint predicted=True):
    """Get the current display time. If 'predicted=True', the predicted 
    mid-frame time is returned.
    
    :param frameIndex: int
    :param predicted: boolean
    :return: float
    
    """
    cdef double t_secs
    if predicted:
        t_secs = ovr_capi.ovr_GetPredictedDisplayTime(
            _ptrSession_, frameIndex)
    else:
        t_secs = ovr_capi.ovr_GetTimeInSeconds()

    return t_secs

cpdef int waitToBeginFrame(LibOVRSession session, unsigned int frameIndex=0):
    cdef ovr_capi.ovrResult result = 0
    result = ovr_capi.ovr_WaitToBeginFrame(session.ptrSession, frameIndex)

    return <int> result

cpdef int beginFrame(LibOVRSession session, unsigned int frameIndex=0):
    result = ovr_capi.ovr_BeginFrame(session.ptrSession, frameIndex)

    return <int> result

cpdef void commitSwapChain(LibOVRSession session, int eye):
    #global _ptrSession_, _swapChains_
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_CommitTextureSwapChain(
        session.ptrSession,
        session.swapChains[eye])

    if debug_mode:
        check_result(result)

cpdef void endFrame(LibOVRSession session, unsigned int frameIndex=0):
    global _eyeLayer_
    cdef ovr_capi.ovrLayerHeader* layers = &(session.eyeLayer).Header
    result = ovr_capi.ovr_EndFrame(
        session.ptrSession,
        frameIndex,
        NULL,
        &layers,
        <unsigned int> 1)

    if debug_mode:
        check_result(result)

# ------------------------
# Mirror Texture Functions
# ------------------------
#
ovrMirrorOption_Default = ovr_capi.ovrMirrorOption_Default
ovrMirrorOption_PostDistortion = ovr_capi.ovrMirrorOption_PostDistortion
ovrMirrorOption_LeftEyeOnly = ovr_capi.ovrMirrorOption_LeftEyeOnly
ovrMirrorOption_RightEyeOnly = ovr_capi.ovrMirrorOption_RightEyeOnly
ovrMirrorOption_IncludeGuardian = ovr_capi.ovrMirrorOption_IncludeGuardian
ovrMirrorOption_IncludeNotifications = ovr_capi.ovrMirrorOption_IncludeNotifications
ovrMirrorOption_IncludeSystemGui = ovr_capi.ovrMirrorOption_IncludeSystemGui

cdef class ovrMirrorTextureDesc:
    """ovrTextureSwapChainDesc
    """
    # no data pointer here
    cdef ovr_capi.ovrMirrorTextureDesc c_ovrMirrorTextureDesc

    def __cinit__(
            self,
            int _format=OVR_FORMAT_R8G8B8A8_UNORM_SRGB,
            int width=800,
            int height=600,
            int mirrorOptions=ovrMirrorOption_Default):
        self.c_ovrMirrorTextureDesc.Format = <ovr_capi.ovrTextureFormat> _format
        self.c_ovrMirrorTextureDesc.Width = width
        self.c_ovrMirrorTextureDesc.Height = height
        self.c_ovrMirrorTextureDesc.MiscFlags = ovr_capi.ovrTextureMisc_None
        self.c_ovrMirrorTextureDesc.MirrorOptions = <int32_t> mirrorOptions

    @property
    def Format(self):
        return <int> self.c_ovrMirrorTextureDesc.Format

    @Format.setter
    def Format(self, int value):
        self.c_ovrMirrorTextureDesc.Format = <ovr_capi.ovrTextureFormat> value

    @property
    def Width(self):
        return <int> self.c_ovrMirrorTextureDesc.Width

    @Width.setter
    def Width(self, int value):
        self.c_ovrMirrorTextureDesc.Width = value

    @property
    def Height(self):
        return <int> self.c_ovrMirrorTextureDesc.Height

    @Height.setter
    def Height(self, int value):
        self.c_ovrMirrorTextureDesc.Height = value

    @property
    def MirrorOptions(self):
        return <int> self.c_ovrMirrorTextureDesc.MirrorOptions

    @MirrorOptions.setter
    def MirrorOptions(self, int value):
        self.c_ovrMirrorTextureDesc.MirrorOptions = <int32_t> value

cpdef void setupMirrorTexture(LibOVRSession session, ovrMirrorTextureDesc mirrorDesc):
    """Create a mirror texture buffer.
    
    :param width: int 
    :param height: int 
    :return: None
    
    """
    #global _mirrorTexture_
    cdef ovr_capi.ovrResult result = ovr_capi_gl.ovr_CreateMirrorTextureGL(
        session.ptrSession, &mirrorDesc.c_ovrMirrorTextureDesc, &(<LibOVRSession>session).mirrorTexture)

    if debug_mode:
        check_result(result)

cpdef unsigned int getMirrorTexture(LibOVRSession session):
    """Get the mirror texture handle.
    
    :return: 
    """
    cdef unsigned int out_tex_id
    cdef ovr_capi.ovrResult result = \
        ovr_capi_gl.ovr_GetMirrorTextureBufferGL(
            session.ptrSession,
            session.mirrorTexture,
            &out_tex_id)

    return <unsigned int> out_tex_id

# types
ovrLayerType_EyeFov = ovr_capi.ovrLayerType_EyeFov

# layer header flags
ovrLayerFlag_HighQuality = ovr_capi.ovrLayerFlag_HighQuality
ovrLayerFlag_TextureOriginAtBottomLeft = ovr_capi.ovrLayerFlag_TextureOriginAtBottomLeft
ovrLayerFlag_HeadLocked = ovr_capi.ovrLayerFlag_HeadLocked

# ------------------------
# Session Status Functions
# ------------------------
#
cdef class ovrSessionStatus(object):
    cdef ovr_capi.ovrSessionStatus*c_data
    cdef ovr_capi.ovrSessionStatus  c_ovrSessionStatus

    def __cinit__(self, *args, **kwargs):
        self.c_data = &self.c_ovrSessionStatus

    @property
    def IsVisible(self):
        return <bint> self.c_data.IsVisible

    @property
    def HmdPresent(self):
        return <bint> self.c_data.HmdPresent

    @property
    def DisplayLost(self):
        return <bint> self.c_data.DisplayLost

    @property
    def ShouldQuit(self):
        return <bint> self.c_data.ShouldQuit

    @property
    def ShouldRecenter(self):
        return <bint> self.c_data.ShouldRecenter

    @property
    def HasInputFocus(self):
        return <bint> self.c_data.HasInputFocus

    @property
    def OverlayPresent(self):
        return <bint> self.c_data.OverlayPresent

    @property
    def DepthRequested(self):
        return <bint> self.c_data.DepthRequested

cpdef ovrSessionStatus getSessionStatus():
    """Get the current session status.
    
    :return: ovrSessionStatus
    
    """
    global _ptrSession_
    cdef ovrSessionStatus to_return = ovrSessionStatus()
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetSessionStatus(
        _ptrSession_, &(<ovrSessionStatus> to_return).c_data[0])

    if debug_mode:
        check_result(result)

    return to_return

# -------------------------
# HID Classes and Functions
# -------------------------
#
cdef class ovrInputState(object):
    """Class storing the state of an input device. Fields can only be updated
    by calling 'get_input_state()'.

    """
    cdef ovr_capi.ovrInputState*c_data
    cdef ovr_capi.ovrInputState c_ovrInputState

    def __cinit__(self, *args, **kwargs):
        self.c_data = &self.c_ovrInputState

    @property
    def TimeInSeconds(self):
        return <double> self.c_data.TimeInSeconds

    @property
    def Buttons(self):
        return self.c_data[0].Buttons

    @property
    def Touches(self):
        return self.c_data[0].Touches

    @property
    def IndexTrigger(self):
        cdef float index_trigger_left = self.c_data[0].IndexTrigger[0]
        cdef float index_trigger_right = self.c_data[0].IndexTrigger[1]

        return index_trigger_left, index_trigger_right

    @property
    def HandTrigger(self):
        cdef float hand_trigger_left = self.c_data[0].HandTrigger[0]
        cdef float hand_trigger_right = self.c_data[0].HandTrigger[1]

        return hand_trigger_left, hand_trigger_right

    @property
    def Thumbstick(self):
        cdef float thumbstick_x0 = self.c_data[0].Thumbstick[0].x
        cdef float thumbstick_y0 = self.c_data[0].Thumbstick[0].y
        cdef float thumbstick_x1 = self.c_data[0].Thumbstick[1].x
        cdef float thumbstick_y1 = self.c_data[0].Thumbstick[1].y

        return (thumbstick_x0, thumbstick_y0), (thumbstick_x1, thumbstick_y1)

    @property
    def ControllerType(self):
        cdef int ctrl_type = <int> self.c_data[0].ControllerType
        if ctrl_type == ovr_capi.ovrControllerType_XBox:
            return 'xbox'
        elif ctrl_type == ovr_capi.ovrControllerType_Remote:
            return 'remote'
        elif ctrl_type == ovr_capi.ovrControllerType_Touch:
            return 'touch'
        elif ctrl_type == ovr_capi.ovrControllerType_LTouch:
            return 'ltouch'
        elif ctrl_type == ovr_capi.ovrControllerType_RTouch:
            return 'rtouch'
        else:
            return None

    @property
    def IndexTriggerNoDeadzone(self):
        cdef float index_trigger_left = self.c_data[0].IndexTriggerNoDeadzone[0]
        cdef float index_trigger_right = self.c_data[0].IndexTriggerNoDeadzone[
            1]

        return index_trigger_left, index_trigger_right

    @property
    def HandTriggerNoDeadzone(self):
        cdef float hand_trigger_left = self.c_data[0].HandTriggerNoDeadzone[0]
        cdef float hand_trigger_right = self.c_data[0].HandTriggerNoDeadzone[1]

        return hand_trigger_left, hand_trigger_right

    @property
    def ThumbstickNoDeadzone(self):
        cdef float thumbstick_x0 = self.c_data[0].ThumbstickNoDeadzone[0].x
        cdef float thumbstick_y0 = self.c_data[0].ThumbstickNoDeadzone[0].y
        cdef float thumbstick_x1 = self.c_data[0].ThumbstickNoDeadzone[1].x
        cdef float thumbstick_y1 = self.c_data[0].ThumbstickNoDeadzone[1].y

        return (thumbstick_x0, thumbstick_y0), (thumbstick_x1, thumbstick_y1)

    @property
    def IndexTriggerRaw(self):
        cdef float index_trigger_left = self.c_data[0].IndexTriggerRaw[0]
        cdef float index_trigger_right = self.c_data[0].IndexTriggerRaw[1]

        return index_trigger_left, index_trigger_right

    @property
    def HandTriggerRaw(self):
        cdef float hand_trigger_left = self.c_data[0].HandTriggerRaw[0]
        cdef float hand_trigger_right = self.c_data[0].HandTriggerRaw[1]

        return hand_trigger_left, hand_trigger_right

    @property
    def ThumbstickRaw(self):
        cdef float thumbstick_x0 = self.c_data[0].ThumbstickRaw[0].x
        cdef float thumbstick_y0 = self.c_data[0].ThumbstickRaw[0].y
        cdef float thumbstick_x1 = self.c_data[0].ThumbstickRaw[1].x
        cdef float thumbstick_y1 = self.c_data[0].ThumbstickRaw[1].y

        return (thumbstick_x0, thumbstick_y0), (thumbstick_x1, thumbstick_y1)

cpdef object getInputState(str controller, object stateOut=None):
    """Get a controller state as an object. If a 'InputStateData' object is
    passed to 'state_out', that object will be updated.
    
    :param controller: str
    :param state_out: InputStateData or None
    :return: InputStateData or None
    
    """
    cdef ovr_capi.ovrControllerType ctrl_type
    if controller == 'xbox':
        ctrl_type = ovr_capi.ovrControllerType_XBox
    elif controller == 'remote':
        ctrl_type = ovr_capi.ovrControllerType_Remote
    elif controller == 'touch':
        ctrl_type = ovr_capi.ovrControllerType_Touch
    elif controller == 'left_touch':
        ctrl_type = ovr_capi.ovrControllerType_LTouch
    elif controller == 'right_touch':
        ctrl_type = ovr_capi.ovrControllerType_RTouch

    # create a controller state object and set its data
    global _ptrSession_
    cdef ovr_capi.ovrInputState*ptr_state
    cdef ovrInputState to_return = ovrInputState()

    if stateOut is None:
        ptr_state = &(<ovrInputState> to_return).c_ovrInputState
    else:
        ptr_state = &(<ovrInputState> stateOut).c_ovrInputState

    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetInputState(
        _ptrSession_,
        ctrl_type,
        ptr_state)

    if stateOut is None:
        return None

    return to_return

cpdef double pollController(str controller):
    """Poll and update specified controller's state data. The time delta in 
    seconds between the current and previous controller state is returned.
    
    :param controller: str or None
    :return: double
    
    """
    global _ptrSession_, _ctrl_states_, _ctrl_states_prev_
    cdef ovr_capi.ovrInputState*ptr_ctrl = NULL
    cdef ovr_capi.ovrInputState*ptr_ctrl_prev = NULL

    cdef ovr_capi.ovrControllerType ctrl_type
    if controller == 'xbox':
        ctrl_type = ovr_capi.ovrControllerType_XBox
        ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.xbox]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.xbox]
    elif controller == 'remote':
        ctrl_type = ovr_capi.ovrControllerType_Remote
        ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.remote]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.remote]
    elif controller == 'touch':
        ctrl_type = ovr_capi.ovrControllerType_Touch
        ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.touch]
    elif controller == 'left_touch':
        ctrl_type = ovr_capi.ovrControllerType_LTouch
        ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.left_touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.left_touch]
    elif controller == 'right_touch':
        ctrl_type = ovr_capi.ovrControllerType_RTouch
        ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.right_touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.right_touch]

    # copy the previous control state
    ptr_ctrl_prev[0] = ptr_ctrl[0]

    # update the current controller state
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetInputState(
        _ptrSession_,
        ctrl_type,
        ptr_ctrl)

    if debug_mode:
        check_result(result)

    # return the time delta between the last time the controller was polled
    return ptr_ctrl[0].TimeInSeconds - ptr_ctrl_prev[0].TimeInSeconds

cpdef double getControllerAbsTime(str controller):
    """Get the absolute time the state of the specified controller was last 
    updated.
    
    :param controller: str or None
    :return: float
    
    """
    # get pointer to control state
    global _ctrl_states_
    cdef ovr_capi.ovrInputState*ptr_ctrl_state = NULL
    if controller == 'xbox':
        ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.xbox]
    elif controller == 'remote':
        ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.remote]
    elif controller == 'touch':
        ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.touch]
    elif controller == 'left_touch':
        ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.left_touch]
    elif controller == 'right_touch':
        ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.right_touch]

    return ptr_ctrl_state[0].TimeInSeconds

cpdef tuple getIndexTriggerValues(str controller, bint deadZone=False):
    """Get index trigger values for a specified controller.
    
    :param controller: str
    :param deadZone: boolean
    :return: tuple
    
    """
    # get pointer to control state
    global _ctrl_states_
    cdef ovr_capi.ovrInputState*ptr_ctrl_state = NULL
    if controller == 'xbox':
        ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.xbox]
    elif controller == 'remote':
        ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.remote]
    elif controller == 'touch':
        ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.touch]
    elif controller == 'left_touch':
        ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.left_touch]
    elif controller == 'right_touch':
        ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.right_touch]

    cdef float index_trigger_left = 0.0
    cdef float index_trigger_right = 0.0

    # get the value with or without the deadzone
    if not deadZone:
        index_trigger_left = ptr_ctrl_state[0].IndexTriggerNoDeadzone[0]
        index_trigger_right = ptr_ctrl_state[0].IndexTriggerNoDeadzone[1]
    else:
        index_trigger_left = ptr_ctrl_state[0].IndexTrigger[0]
        index_trigger_right = ptr_ctrl_state[0].IndexTrigger[1]

    return index_trigger_left, index_trigger_right

cpdef tuple getHandTriggerValues(str controller, bint deadZone=False):
    """Get hand trigger values for a specified controller.
    
    :param controller: str
    :param deadzone: boolean
    :return: tuple
    
    """
    # get pointer to control state
    global _ctrl_states_
    cdef ovr_capi.ovrInputState*ptr_ctrl_state = NULL
    if controller == 'xbox':
        ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.xbox]
    elif controller == 'remote':
        ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.remote]
    elif controller == 'touch':
        ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.touch]
    elif controller == 'left_touch':
        ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.left_touch]
    elif controller == 'right_touch':
        ptr_ctrl_state = &_ctrl_states_[<int> LibOVRControllers.right_touch]

    cdef float hand_trigger_left = 0.0
    cdef float hand_trigger_right = 0.0

    # get the value with or without the deadzone
    if not deadZone:
        hand_trigger_left = ptr_ctrl_state[0].HandTriggerNoDeadzone[0]
        hand_trigger_right = ptr_ctrl_state[0].HandTriggerNoDeadzone[1]
    else:
        hand_trigger_left = ptr_ctrl_state[0].HandTrigger[0]
        hand_trigger_right = ptr_ctrl_state[0].HandTrigger[1]

    return hand_trigger_left, hand_trigger_right

cdef float clip_input_range(float val):
    """Constrain an analog input device's range between -1.0 and 1.0. This is 
    only accessible from module functions.
    
    :param val: float
    :return: float
    
    """
    if val > 1.0:
        val = 1.0
    elif val < 1.0:
        val = 1.0

    return val

cpdef tuple getThumbstickValues(str controller, bint deadZone=False):
    """Get thumbstick values for a specified controller.
    
    :param controller: 
    :param dead_zone: 
    :return: tuple
    
    """
    # get pointer to control state
    global _ctrl_states_
    cdef ovr_capi.ovrInputState*ptr_ctrl = NULL
    cdef ovr_capi.ovrInputState*ptr_ctrl_prev = NULL
    if controller == 'xbox':
        ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.xbox]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.xbox]
    elif controller == 'remote':
        ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.remote]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.remote]
    elif controller == 'touch':
        ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.touch]
    elif controller == 'left_touch':
        ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.left_touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.left_touch]
    elif controller == 'right_touch':
        ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.right_touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.right_touch]

    cdef float thumbstick0_x = 0.0
    cdef float thumbstick0_y = 0.0
    cdef float thumbstick1_x = 0.0
    cdef float thumbstick1_y = 0.0

    # get the value with or without the deadzone
    if not deadZone:
        thumbstick0_x = ptr_ctrl[0].Thumbstick[0].x
        thumbstick0_y = ptr_ctrl[0].Thumbstick[0].y
        thumbstick1_x = ptr_ctrl[0].Thumbstick[1].x
        thumbstick1_y = ptr_ctrl[0].Thumbstick[1].y
    else:
        thumbstick0_x = ptr_ctrl[0].ThumbstickNoDeadzone[0].x
        thumbstick0_y = ptr_ctrl[0].ThumbstickNoDeadzone[0].y
        thumbstick1_x = ptr_ctrl[0].ThumbstickNoDeadzone[1].x
        thumbstick1_y = ptr_ctrl[0].ThumbstickNoDeadzone[1].y

    # clip range
    thumbstick0_x = clip_input_range(thumbstick0_x)
    thumbstick0_y = clip_input_range(thumbstick0_y)
    thumbstick1_x = clip_input_range(thumbstick1_x)
    thumbstick1_y = clip_input_range(thumbstick1_y)

    return (thumbstick0_x, thumbstick0_y), (thumbstick1_x, thumbstick1_y)

cpdef bint getButtons(str controller, object buttonNames,
                      str trigger='continuous'):
    """Get the state of a specified button for a given controller. 
    
    Buttons to test are specified using their string names. Argument
    'button_names' accepts a single string or a list. If a list is specified,
    the returned value will reflect whether all buttons were triggered at the
    time the controller was polled last. 
    
    An optional trigger mode may be specified which defines the button's
    activation criteria. Be default, trigger='continuous' which will return the
    immediate state of the button is used. Using 'rising' will return True once 
    when the button is first pressed, whereas 'falling' will return True once 
    the button is released.
    
    :param controller: str
    :param buttonNames: str, tuple or list
    :param trigger: str
    :return: boolean
    
    """
    # get pointer to control state
    global _ctrl_states_
    cdef ovr_capi.ovrInputState*ptr_ctrl = NULL
    cdef ovr_capi.ovrInputState*ptr_ctrl_prev = NULL
    if controller == 'xbox':
        ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.xbox]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.xbox]
    elif controller == 'remote':
        ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.remote]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.remote]
    elif controller == 'touch':
        ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.touch]
    elif controller == 'left_touch':
        ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.left_touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.left_touch]
    elif controller == 'right_touch':
        ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.right_touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.right_touch]

    cdef unsigned int button_bits = 0x00000000
    cdef int i, N
    if isinstance(buttonNames, str):  # don't loop if a string is specified
        button_bits |= ctrl_button_lut[buttonNames]
    elif isinstance(buttonNames, (tuple, list)):
        # loop over all names and combine them
        N = <int> len(buttonNames)
        for i in range(N):
            button_bits |= ctrl_button_lut[buttonNames[i]]

    # test if the button was pressed
    cdef bint pressed
    if trigger == 'continuous':
        pressed = (ptr_ctrl.Buttons & button_bits) == button_bits
    elif trigger == 'rising' or trigger == 'pressed':
        # rising edge, will trigger once when pressed
        pressed = (ptr_ctrl.Buttons & button_bits) == button_bits and \
                  (ptr_ctrl_prev.Buttons & button_bits) != button_bits
    elif trigger == 'falling' or trigger == 'released':
        # falling edge, will trigger once when released
        pressed = (ptr_ctrl.Buttons & button_bits) != button_bits and \
                  (ptr_ctrl_prev.Buttons & button_bits) == button_bits
    else:
        raise ValueError("Invalid trigger mode specified.")

    return pressed

cpdef bint getTouches(str controller, object touchNames,
                      str trigger='continuous'):
    """Get touches for a specified device.
    
    Touches reveal information about the user's hand pose, for instance, whether 
    a pointing or pinching gesture is being made. Oculus Touch controllers are
    required for this functionality.

    Touch points to test are specified using their string names. Argument
    'touch_names' accepts a single string or a list. If a list is specified,
    the returned value will reflect whether all touches were triggered at the
    time the controller was polled last. 
    
    :param controller: str
    :param touchNames: str, tuple or list
    :param trigger: str
    :return: boolean
    
    """
    # get pointer to control state
    global _ctrl_states_
    cdef ovr_capi.ovrInputState*ptr_ctrl = NULL
    cdef ovr_capi.ovrInputState*ptr_ctrl_prev = NULL
    if controller == 'xbox':
        ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.xbox]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.xbox]
    elif controller == 'remote':
        ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.remote]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.remote]
    elif controller == 'touch':
        ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.touch]
    elif controller == 'left_touch':
        ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.left_touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.left_touch]
    elif controller == 'right_touch':
        ptr_ctrl = &_ctrl_states_[<int> LibOVRControllers.right_touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int> LibOVRControllers.right_touch]

    cdef unsigned int touch_bits = 0x00000000
    cdef int i, N
    if isinstance(touchNames, str):  # don't loop if a string is specified
        touch_bits |= ctrl_button_lut[touchNames]
    elif isinstance(touchNames, (tuple, list)):
        # loop over all names and combine them
        N = <int> len(touchNames)
        for i in range(N):
            touch_bits |= ctrl_button_lut[touchNames[i]]

    # test if the button was pressed
    cdef bint touched
    if trigger == 'continuous':
        touched = (ptr_ctrl.Touches & touch_bits) == touch_bits
    elif trigger == 'rising' or trigger == 'pressed':
        # rising edge, will trigger once when pressed
        touched = (ptr_ctrl.Touches & touch_bits) == touch_bits and \
                  (ptr_ctrl_prev.Touches & touch_bits) != touch_bits
    elif trigger == 'falling' or trigger == 'released':
        # falling edge, will trigger once when released
        touched = (ptr_ctrl.Touches & touch_bits) != touch_bits and \
                  (ptr_ctrl_prev.Touches & touch_bits) == touch_bits
    else:
        raise ValueError("Invalid trigger mode specified.")

    return touched

# List of controller names that are available to the user. These are handled by
# the SDK, additional joysticks, keyboards and mice must be accessed by some
# other method.
#
controller_names = ['xbox', 'remote', 'touch', 'left_touch', 'right_touch']

cpdef list getConnectedControllerTypes():
    """Get a list of currently connected controllers. You can check if a
    controller is attached by testing for its membership in the list using its
    name.
    
    :return: list  
    
    """
    cdef unsigned int result = ovr_capi.ovr_GetConnectedControllerTypes(
        _ptrSession_)

    cdef list ctrl_types = list()
    if (result & ovr_capi.ovrControllerType_XBox) == \
            ovr_capi.ovrControllerType_XBox:
        ctrl_types.append('xbox')
    elif (result & ovr_capi.ovrControllerType_Remote) == \
            ovr_capi.ovrControllerType_Remote:
        ctrl_types.append('remote')
    elif (result & ovr_capi.ovrControllerType_Touch) == \
            ovr_capi.ovrControllerType_Touch:
        ctrl_types.append('touch')
    elif (result & ovr_capi.ovrControllerType_LTouch) == \
            ovr_capi.ovrControllerType_LTouch:
        ctrl_types.append('left_touch')
    elif (result & ovr_capi.ovrControllerType_RTouch) == \
            ovr_capi.ovrControllerType_RTouch:
        ctrl_types.append('right_touch')

    return ctrl_types

# -------------------------------
# Performance/Profiling Functions
# -------------------------------
#
cdef class ovrPerfStatsPerCompositorFrame(object):
    cdef ovr_capi.ovrPerfStatsPerCompositorFrame*c_data
    cdef ovr_capi.ovrPerfStatsPerCompositorFrame  c_ovrPerfStatsPerCompositorFrame

    def __cinit__(self, *args, **kwargs):
        self.c_data = &self.c_ovrPerfStatsPerCompositorFrame

    @property
    def HmdVsyncIndex(self):
        return self.c_data[0].HmdVsyncIndex

    @property
    def AppFrameIndex(self):
        return self.c_data[0].AppFrameIndex

    @property
    def AppDroppedFrameCount(self):
        return self.c_data[0].AppDroppedFrameCount

    @property
    def AppQueueAheadTime(self):
        return self.c_data[0].AppQueueAheadTime

    @property
    def AppCpuElapsedTime(self):
        return self.c_data[0].AppCpuElapsedTime

    @property
    def AppGpuElapsedTime(self):
        return self.c_data[0].AppGpuElapsedTime

    @property
    def CompositorFrameIndex(self):
        return self.c_data[0].CompositorFrameIndex

    @property
    def CompositorLatency(self):
        return self.c_data[0].CompositorLatency

    @property
    def CompositorCpuElapsedTime(self):
        return self.c_data[0].CompositorCpuElapsedTime

    @property
    def CompositorGpuElapsedTime(self):
        return self.c_data[0].CompositorGpuElapsedTime

    @property
    def CompositorCpuStartToGpuEndElapsedTime(self):
        return self.c_data[0].CompositorCpuStartToGpuEndElapsedTime

    @property
    def CompositorGpuEndToVsyncElapsedTime(self):
        return self.c_data[0].CompositorGpuEndToVsyncElapsedTime

cdef class ovrPerfStats(object):
    cdef ovr_capi.ovrPerfStats*c_data
    cdef ovr_capi.ovrPerfStats  c_ovrPerfStats
    cdef list perf_stats

    def __cinit__(self, *args, **kwargs):
        self.c_data = &self.c_ovrPerfStats

        # initialize performance stats list
        self.perf_stats = list()
        cdef int i, N
        N = <int> ovr_capi.ovrMaxProvidedFrameStats
        for i in range(N):
            self.perf_stats.append(ovrPerfStatsPerCompositorFrame())
            (<ovrPerfStatsPerCompositorFrame> self.perf_stats[i]).c_data[0] = \
                self.c_data[0].FrameStats[i]

    @property
    def FrameStatsCount(self):
        return self.c_data[0].FrameStatsCount

    @property
    def AnyFrameStatsDropped(self):
        return <bint> self.c_data[0].AnyFrameStatsDropped

    @property
    def FrameStats(self):
        cdef int i, N
        N = self.c_data[0].FrameStatsCount
        for i in range(N):
            (<ovrPerfStatsPerCompositorFrame> self.perf_stats[i]).c_data[0] = \
                self.c_data[0].FrameStats[i]

        return self.perf_stats

    @property
    def AdaptiveGpuPerformanceScale(self):
        return <bint> self.c_data[0].AdaptiveGpuPerformanceScale

    @property
    def AswIsAvailable(self):
        return <bint> self.c_data[0].AswIsAvailable

cpdef ovrPerfStats getFrameStats():
    """Get most recent performance stats, returns an object with fields
    corresponding to various performance stats reported by the SDK.
    
    :return: dict 
    
    """
    global _ptrSession_

    cdef ovrPerfStats to_return = ovrPerfStats()
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetPerfStats(
        _ptrSession_,
        &(<ovrPerfStats> to_return).c_data[0])

    if debug_mode:
        check_result(result)

    return to_return

cpdef void resetFrameStats():
    """Flushes backlog of frame stats.
    
    :return: None 
    
    """
    global _ptrSession_
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_ResetPerfStats(
        _ptrSession_)

    if debug_mode:
        check_result(result)

# List of available performance HUD modes.
#
available_hud_modes = [
    'Off',
    'PerfSummary',
    'LatencyTiming',
    'AppRenderTiming',
    'CompRenderTiming',
    'AswStats',
    'VersionInfo']

cpdef void perfHudMode(LibOVRSession session, str mode='Off'):
    """Display a performance HUD with a specified mode.
    
    :param mode: str 
    :return: None
    
    """
    global _ptrSession_
    cdef int perf_hud_mode = 0

    if mode == 'Off':
        perf_hud_mode = <int> ovr_capi.ovrPerfHud_Off
    elif mode == 'PerfSummary':
        perf_hud_mode = <int> ovr_capi.ovrPerfHud_PerfSummary
    elif mode == 'LatencyTiming':
        perf_hud_mode = <int> ovr_capi.ovrPerfHud_LatencyTiming
    elif mode == 'AppRenderTiming':
        perf_hud_mode = <int> ovr_capi.ovrPerfHud_AppRenderTiming
    elif mode == 'CompRenderTiming':
        perf_hud_mode = <int> ovr_capi.ovrPerfHud_CompRenderTiming
    elif mode == 'AswStats':
        perf_hud_mode = <int> ovr_capi.ovrPerfHud_AswStats
    elif mode == 'VersionInfo':
        perf_hud_mode = <int> ovr_capi.ovrPerfHud_VersionInfo

    cdef ovr_capi.ovrBool ret = ovr_capi.ovr_SetInt(
        session.ptrSession, b"PerfHudMode", perf_hud_mode)

# -----------------------------
# Boundary and Safety Functions
# -----------------------------
#
cdef ovr_capi.ovrBoundaryLookAndFeel _boundary_style_

cpdef void setBoundryColor(float r, float g, float b):
    global _ptrSession_, _boundary_style_

    cdef ovr_capi.ovrColorf color
    color.r = r
    color.g = g
    color.b = b

    _boundary_style_.Color = color

    cdef ovr_capi.ovrResult result = ovr_capi.ovr_SetBoundaryLookAndFeel(
        _ptrSession_,
        &_boundary_style_)

    if debug_mode:
        check_result(result)

cpdef void resetBoundryColor():
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_ResetBoundaryLookAndFeel(
        _ptrSession_)

    if debug_mode:
        check_result(result)

cpdef bint isBoundryVisible():
    cdef ovr_capi.ovrBool is_visible
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetBoundaryVisible(
        _ptrSession_, &is_visible)

    if debug_mode:
        check_result(result)

    return <bint> is_visible

cpdef void showBoundry(bint show=True):
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_RequestBoundaryVisible(
        _ptrSession_, <ovr_capi.ovrBool> show)

    if debug_mode:
        check_result(result)
