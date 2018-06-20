#  =============================================================================
#  Oculus(TM) LibOVR Python Interface Module
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
from . cimport ovr_capi
from . cimport ovr_capi_gl
from . cimport ovr_errorcode
from . cimport ovr_capi_util
from . cimport ovr_math
from .math cimport *

cimport libc.math as cmath
from libc.stdint cimport int32_t
import OpenGL.GL as GL

# -----------------
# Initialize module
# -----------------
#
cdef ovr_capi.ovrInitParams _init_params_  # initialization parameters

# HMD descriptor storing information about the HMD being used.
cdef ovr_capi.ovrHmdDesc _hmdDesc_

# Since we are only using one session per module instance, so we are going to
# create our session pointer here and use it module-wide.
#
cdef ovr_capi.ovrSession _ptr_session_
cdef ovr_capi.ovrGraphicsLuid _ptr_luid_

# Frame index
#
cdef long long _frame_index_ = 0

# create an array of texture swap chains
#
cdef ovr_capi.ovrTextureSwapChain _swap_chain_[32]

# mirror texture swap chain, we only create one here
#
cdef ovr_capi.ovrMirrorTexture _mirror_texture_ = NULL

# Persistent VR related structures to store head pose and other data used across
# frames.
#
cdef ovr_capi.ovrEyeRenderDesc[2] _eye_render_desc_
cdef ovr_capi.ovrPosef[2] _hmd_to_eye_view_pose_

# Render layer
#
cdef ovr_capi.ovrLayerEyeFov _eye_layer_

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
    "RThumb" : ovr_capi.ovrButton_RThumb,
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
    "RThumb" : ovr_capi.ovrTouch_RThumb,
    "RThumbRest": ovr_capi.ovrTouch_RThumbRest,
    "RIndexTrigger" : ovr_capi.ovrTouch_RThumb,
    "X": ovr_capi.ovrTouch_X,
    "Y": ovr_capi.ovrTouch_Y,
    "LThumb": ovr_capi.ovrTouch_LThumb,
    "LThumbRest": ovr_capi.ovrTouch_LThumbRest,
    "LIndexTrigger" : ovr_capi.ovrTouch_LIndexTrigger,
    "RIndexPointing": ovr_capi.ovrTouch_RIndexPointing,
    "RThumbUp": ovr_capi.ovrTouch_RThumbUp,
    "LIndexPointing": ovr_capi.ovrTouch_LIndexPointing,
    "LThumbUp": ovr_capi.ovrTouch_LThumbUp}

# Python accessible list of valid touch names.
touch_names = list(ctrl_touch_lut.keys())

# Performance information for profiling.
#
cdef ovr_capi.ovrPerfStats _perf_stats_


cdef class ovrColorf:
    cdef ovr_capi.ovrColorf* c_data
    cdef ovr_capi.ovrColorf  c_ovrColorf

    def __cinit__(self, float r=0.0, float g=0.0, float b=0.0, float a=0.0):
        self.c_data = &self.c_ovrColorf

        self.c_data.r = r
        self.c_data.g = g
        self.c_data.b = b
        self.c_data.a = a

    @property
    def r(self):
        return self.c_data.r

    @r.setter
    def r(self, float value):
        self.c_data.r = value

    @property
    def g(self):
        return self.c_data.g

    @g.setter
    def g(self, float value):
        self.c_data.g = value

    @property
    def b(self):
        return self.c_data.b

    @b.setter
    def b(self, float value):
        self.c_data.b = value

    @property
    def a(self):
        return self.c_data.a

    @a.setter
    def a(self, float value):
        self.c_data.a = value

    def as_tuple(self):
        return self.c_data.r, self.c_data.g, self.c_data.b, self.c_data.a

# ---------------------
# Oculus SDK Math Types
# ---------------------
#


# --------------------
# Swap Chain Functions
# --------------------
#
def allocSwapChain(ovrTextureSwapChainDesc swap_desc):
    """Allocate a new swap chain object with the specified parameters. If
    successful, an integer is returned which is used to reference the swap
    chain. You can allocate up-to 32 swap chains.

    :param width: int
    :param height: int
    :return: int

    """
    global _swap_chain_, _ptr_session_
    # get the first available swap chain, unallocated chains will test as NULL
    cdef int i, sc
    for i in range(32):
        if _swap_chain_[i] is NULL:
            sc = i
            break
    else:
        raise IndexError("Maximum number of swap chains initialized!")

    # create the swap chain
    cdef ovr_capi.ovrResult result = ovr_capi_gl.ovr_CreateTextureSwapChainGL(
        _ptr_session_, &swap_desc.c_ovrTextureSwapChainDesc, &_swap_chain_[sc])

    if debug_mode:
        check_result(result)

    # return the handle
    return sc

# Free or destroy a swap chain. The handle will be made available after this
# call.
#
def freeSwapChain(int sc):
    """Free or destroy a swap chain. The handle will be made available after
    this call.

    :param sc: int
    :return:

    """
    global _swap_chain_, _ptr_session_
    ovr_capi.ovr_DestroyTextureSwapChain(_ptr_session_, _swap_chain_[sc])
    _swap_chain_[sc] = NULL

# Get the next available texture in the specified swap chain. Use the returned
# value as a frame buffer texture.
#
def getTextureSwapChainBufferGL(int sc):
    cdef int current_idx = 0
    cdef unsigned int tex_id = 0
    cdef ovr_capi.ovrResult result

    global _swap_chain_

    # get the current texture index within the swap chain
    result = ovr_capi.ovr_GetTextureSwapChainCurrentIndex(
        _ptr_session_, _swap_chain_[sc], &current_idx)

    if debug_mode:
        check_result(result)

    # get the next available texture ID from the swap chain
    result = ovr_capi_gl.ovr_GetTextureSwapChainBufferGL(
        _ptr_session_, _swap_chain_[sc], current_idx, &tex_id)

    if debug_mode:
        check_result(result)

    return tex_id

# -----------------
# Session Functions
# -----------------
#
cpdef bint isOculusServiceRunning(int timeout_milliseconds=100):
    cdef ovr_capi_util.ovrDetectResult result = ovr_capi_util.ovr_Detect(
        timeout_milliseconds)

    return <bint>result.IsOculusServiceRunning

cpdef bint isHmdConnected(int timeout_milliseconds=100):
    cdef ovr_capi_util.ovrDetectResult result = ovr_capi_util.ovr_Detect(
        timeout_milliseconds)

    return <bint>result.IsOculusHMDConnected

cpdef void startSession():
    """Start a new session. Control is handed over to the application from
    Oculus Home. 
    
    :return: None 
    
    """
    global _ptr_session_
    cdef ovr_capi.ovrResult result = 0
    result = ovr_capi.ovr_Initialize(NULL)
    result = ovr_capi.ovr_Create(&_ptr_session_, &_ptr_luid_)
    if ovr_errorcode.OVR_FAILURE(result):
        ovr_capi.ovr_Shutdown()

    # get HMD descriptor
    global _hmdDesc_
    _hmdDesc_ = ovr_capi.ovr_GetHmdDesc(_ptr_session_)

    # configure VR data with HMD descriptor information
    #global _eye_render_desc_, _hmd_to_eye_view_pose_
    #_eye_render_desc_[0] = ovr_capi.ovr_GetRenderDesc(
    #    _ptr_session_, ovr_capi.ovrEye_Left, _hmd_desc_.DefaultEyeFov[0])
    #_eye_render_desc_[1] = ovr_capi.ovr_GetRenderDesc(
    #    _ptr_session_, ovr_capi.ovrEye_Right, _hmd_desc_.DefaultEyeFov[1])
    #_hmd_to_eye_view_pose_[0] = _eye_render_desc_[0].HmdToEyePose
    #_hmd_to_eye_view_pose_[1] = _eye_render_desc_[1].HmdToEyePose

    # prepare the render layer
    global _eye_layer_
    _eye_layer_.Header.Type = ovr_capi.ovrLayerType_EyeFov
    _eye_layer_.Header.Flags = \
        ovr_capi.ovrLayerFlag_TextureOriginAtBottomLeft | \
        ovr_capi.ovrLayerFlag_HighQuality
    _eye_layer_.ColorTexture[0] = NULL
    _eye_layer_.ColorTexture[1] = NULL

    # setup layer FOV settings, these are computed earlier
    #_eye_layer_.Fov[0] = _eye_render_desc_[0].Fov
    #_eye_layer_.Fov[1] = _eye_render_desc_[1].Fov

cpdef void endSession():
    """End the current session. 
    
    Clean-up routines are executed that destroy all swap chains and mirror 
    texture buffers, afterwards control is returned to Oculus Home. This must be 
    called after every successful 'create_session' call.
    
    :return: None 
    
    """
    # free all swap chains
    global _ptr_session_, _swap_chain_, _mirror_texture_
    cdef int i = 0
    for i in range(32):
        if not _swap_chain_[i] is NULL:
            ovr_capi.ovr_DestroyTextureSwapChain(
                _ptr_session_, _swap_chain_[i])
            _swap_chain_[i] = NULL

    # destroy the mirror texture
    ovr_capi.ovr_DestroyMirrorTexture(_ptr_session_, _mirror_texture_)

    # destroy the current session and shutdown
    ovr_capi.ovr_Destroy(_ptr_session_)
    ovr_capi.ovr_Shutdown()

cdef class ovrHmdDesc(object):
    cdef ovr_capi.ovrHmdDesc c_ovrHmdDesc

    def __cinit__(self, *args, **kwargs):
        pass

    @property
    def type(self):
        return <int>self.c_data[0].Type

    @property
    def ProductName(self):
        return self.c_ovrHmdDesc.ProductName.decode('utf-8')

    @property
    def Manufacturer(self):
        return self.c_ovrHmdDesc.Manufacturer.decode('utf-8')

    @property
    def VendorId(self):
        return <int>self.c_ovrHmdDesc.VendorId

    @property
    def ProductId(self):
        return <int>self.c_ovrHmdDesc.ProductId

    @property
    def SerialNumber(self):
        return self.c_ovrHmdDesc.SerialNumber.decode('utf-8')

    @property
    def FirmwareMajor(self):
        return <int>self.c_ovrHmdDesc.FirmwareMajor

    @property
    def FirmwareMinor(self):
        return <int>self.c_ovrHmdDesc.FirmwareMinor

    @property
    def AvailableHmdCaps(self):
        return <int>self.c_ovrHmdDesc.AvailableHmdCaps

    @property
    def DefaultHmdCaps(self):
        return <int>self.c_ovrHmdDesc.DefaultHmdCaps

    @property
    def AvailableTrackingCaps(self):
        return <int>self.c_ovrHmdDesc.AvailableTrackingCaps

    @property
    def DefaultTrackingCaps(self):
        return <int>self.c_ovrHmdDesc.DefaultTrackingCaps

    @property
    def DefaultEyeFov(self):
        cdef ovrFovPort default_fov_left = ovrFovPort()
        cdef ovrFovPort default_fov_right = ovrFovPort()

        (<ovrFovPort>default_fov_left).c_data[0] = \
            self.c_ovrHmdDesc.DefaultEyeFov[0]
        (<ovrFovPort>default_fov_right).c_data[0] = \
            self.c_ovrHmdDesc.DefaultEyeFov[1]

        return default_fov_left, default_fov_right

    @property
    def MaxEyeFov(self):
        cdef ovrFovPort max_fov_left = ovrFovPort()
        cdef ovrFovPort max_fov_right = ovrFovPort()

        (<ovrFovPort>max_fov_left).c_data[0] = self.c_ovrHmdDesc.MaxEyeFov[0]
        (<ovrFovPort>max_fov_right).c_data[0] = self.c_ovrHmdDesc.MaxEyeFov[1]

        return max_fov_left, max_fov_right

    @property
    def Resolution(self):
        cdef ovr_capi.ovrSizei resolution = self.c_ovrHmdDesc.Resolution

        return resolution.x, resolution.y

    @property
    def DisplayRefreshRate(self):
        return self.c_ovrHmdDesc.DisplayRefreshRate


cpdef ovrHmdDesc getHmdDesc():
    """Get general information about the connected HMD. Information such as the
    serial number can identify a specific unit, etc.
    
    :return: dict 
    
    """
    global _ptr_session_
    cdef ovrHmdDesc to_return = ovrHmdDesc()
    (<ovrHmdDesc>to_return).c_ovrHmdDesc = ovr_capi.ovr_GetHmdDesc(
        _ptr_session_)

    return to_return

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

cdef class ovrTextureSwapChainDesc:
    """ovrTextureSwapChainDesc

    """
    # no data pointer here
    cdef ovr_capi.ovrTextureSwapChainDesc c_ovrTextureSwapChainDesc

    def __cinit__(
            self,
            int type=ovrTexture_2D,
            int _format=OVR_FORMAT_R8G8B8A8_UNORM_SRGB,
            int width=800,
            int height=600,
            int array_size=1,
            int mip_levels=1,
            int sample_count=1,
            bint static_image=False):

        self.c_ovrTextureSwapChainDesc.Type = <ovr_capi.ovrTextureType>type
        self.c_ovrTextureSwapChainDesc.Format = <ovr_capi.ovrTextureFormat>_format
        self.c_ovrTextureSwapChainDesc.ArraySize = array_size
        self.c_ovrTextureSwapChainDesc.Width = width
        self.c_ovrTextureSwapChainDesc.Height = height
        self.c_ovrTextureSwapChainDesc.MipLevels = mip_levels
        self.c_ovrTextureSwapChainDesc.SampleCount = sample_count
        self.c_ovrTextureSwapChainDesc.StaticImage = <ovr_capi.ovrBool>static_image

        # these can't be set right now
        self.c_ovrTextureSwapChainDesc.MiscFlags = ovr_capi.ovrTextureMisc_None
        self.c_ovrTextureSwapChainDesc.BindFlags = ovr_capi.ovrTextureBind_None

    @property
    def Type(self):
        return <int>self.c_ovrTextureSwapChainDesc.Type

    @Type.setter
    def Type(self, int value):
        self.c_ovrTextureSwapChainDesc.Type = <ovr_capi.ovrTextureType>value

    @property
    def Format(self):
        return <int>self.c_ovrTextureSwapChainDesc.Format

    @Format.setter
    def Format(self, int value):
        self.c_ovrTextureSwapChainDesc.Format = <ovr_capi.ovrTextureFormat>value

    @property
    def ArraySize(self):
        return <int>self.c_ovrTextureSwapChainDesc.ArraySize

    @ArraySize.setter
    def ArraySize(self, int value):
        self.c_ovrTextureSwapChainDesc.ArraySize = value

    @property
    def Width(self):
        return <int>self.c_ovrTextureSwapChainDesc.Width

    @Width.setter
    def Width(self, int value):
        self.c_ovrTextureSwapChainDesc.Width = value

    @property
    def Height(self):
        return <int>self.c_ovrTextureSwapChainDesc.Height

    @Height.setter
    def Height(self, int value):
        self.c_ovrTextureSwapChainDesc.Height = value

    @property
    def MipLevels(self):
        return <int>self.c_ovrTextureSwapChainDesc.MipLevels

    @MipLevels.setter
    def MipLevels(self, int value):
        self.c_ovrTextureSwapChainDesc.MipLevels = value

    @property
    def SampleCount(self):
        return <int>self.c_ovrTextureSwapChainDesc.SampleCount

    @SampleCount.setter
    def SampleCount(self, int value):
        self.c_ovrTextureSwapChainDesc.SampleCount = value

    @property
    def StaticImage(self):
        return <bint>self.c_ovrTextureSwapChainDesc.StaticImage

    @StaticImage.setter
    def StaticImage(self, bint value):
        self.c_ovrTextureSwapChainDesc.StaticImage = <ovr_capi.ovrBool>value

cpdef int createTextureSwapChainGL(
        ovrTextureSwapChainDesc swap_desc):
    """Allocate a new swap chain object with the specified parameters. If
    successful, an integer is returned which is used to reference the swap
    chain. You can allocate up-to 32 swap chains.
    
    The swap chain is configured by applying settings to a 
    ovrTextureSwapChainDesc object and passing it as 'swap_desc'. 

    :param swap_desc: ovrTextureSwapChainDesc
    :return: int

    """
    global _swap_chain_, _ptr_session_
    # get the first available swap chain, unallocated chains will test as NULL
    cdef int i, sc
    for i in range(32):
        if _swap_chain_[i] is NULL:
            sc = i
            break
    else:
        raise IndexError("Maximum number of swap chains initialized!")

    # create the swap chain
    cdef ovr_capi.ovrResult result = ovr_capi_gl.ovr_CreateTextureSwapChainGL(
        _ptr_session_, &swap_desc.c_ovrTextureSwapChainDesc, &_swap_chain_[sc])

    if debug_mode:
        check_result(result)

    # return the handle
    return sc

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
    (<ovrSizei>to_return).c_data[0] = ovr_capi.ovr_GetFovTextureSize(
        _ptr_session_,
        <ovr_capi.ovrEyeType>eye_type,
        fov.c_data[0],
        texels_per_pixel)

    return to_return

cpdef void configEyeRenderDesc(int eye_type, ovrFovPort fov):
    """Compute eye render descriptors for a given eye. 
    
    Each eye has an internal 'ovrEyeRenderDesc' structure which stores computed
    information which is not accessible directly from Python. You must call this
    function twice (for each eye) to fully configure the descriptors.

    :param eye_type: int
    :param fov: ovrFovPort
    :return: None
    
    """
    global _eye_render_desc_, _eye_layer_, _hmd_to_eye_view_pose_
    _eye_render_desc_[eye_type] = ovr_capi.ovr_GetRenderDesc(
        _ptr_session_,
        <ovr_capi.ovrEyeType>eye_type,
        fov.c_data[0])

    # set the initial eye pose
    _hmd_to_eye_view_pose_[eye_type] = _eye_render_desc_[eye_type].HmdToEyePose

    # set the render layer FOV to what is computed
    _eye_layer_.Fov[eye_type] = _eye_render_desc_[eye_type].Fov

cpdef list getHmdToEyePose():
    """Get the HMD to eye poses from the internal eye render descriptor.
    
    :return: 
    
    """
    global _eye_render_desc_
    cdef ovrPosef hmdToEyePoseLeft = ovrPosef()
    cdef ovrPosef hmdToEyePoseRight = ovrPosef()

    (<ovrPosef>hmdToEyePoseLeft).c_data[0] = \
        <ovr_math.Posef>_eye_render_desc_[0].HmdToEyePose
    (<ovrPosef>hmdToEyePoseRight).c_data[0] = \
        <ovr_math.Posef>_eye_render_desc_[1].HmdToEyePose

    return [hmdToEyePoseLeft, hmdToEyePoseRight]

cpdef tuple getBufferSize(str fov_type='recommended',
                          float texel_per_pixel=1.0):
    """Compute the recommended buffer (texture) size for a specified 
    configuration.
    
    Returns a tuple with the dimensions of the required texture (w, h). The 
    values can be used when configuring a render buffer which will ultimately
    be used to draw to the HMD buffers.
    
    :return: None 
    
    """
    # get the buffer size for the specified FOV type and buffer layout
    cdef ovr_capi.ovrSizei rec_tex0_size, rec_tex1_size, buffer_size
    if fov_type == 'recommended':
        rec_tex0_size = ovr_capi.ovr_GetFovTextureSize(
            _ptr_session_,
            ovr_capi.ovrEye_Left,
            _hmdDesc_.DefaultEyeFov[0],
            texel_per_pixel)
        rec_tex1_size = ovr_capi.ovr_GetFovTextureSize(
            _ptr_session_,
            ovr_capi.ovrEye_Right,
            _hmdDesc_.DefaultEyeFov[1],
            texel_per_pixel)
    elif fov_type == 'max':
        rec_tex0_size = ovr_capi.ovr_GetFovTextureSize(
            _ptr_session_,
            ovr_capi.ovrEye_Left,
            _hmdDesc_.MaxEyeFov[0],
            texel_per_pixel)
        rec_tex1_size = ovr_capi.ovr_GetFovTextureSize(
            _ptr_session_,
            ovr_capi.ovrEye_Right,
            _hmdDesc_.MaxEyeFov[1],
            texel_per_pixel)

    buffer_size.w  = rec_tex0_size.w + rec_tex1_size.w
    buffer_size.h = max(rec_tex0_size.h, rec_tex1_size.h)

    return buffer_size.w, buffer_size.h

cpdef void setRenderSwapChain(int eye, object swap_chain):
    """Set the swap chain for the render layer.

    :param eye: str
    :param swap_chain: int or None
    :return: None
    
    """
    # set the swap chain textures
    global _eye_layer_
    if not swap_chain is None:
        _eye_layer_.ColorTexture[eye] = _swap_chain_[<int>swap_chain]
    else:
        _eye_layer_.ColorTexture[eye] = NULL

cpdef ovrRecti getRenderViewport(int eye):
    """Get the viewport rectangle for a given eye view. These will return the
    viewports set by the previous 'setRenderViewport' call.
    
    :param eye: int
    :return: None
    
    """
    global _ptr_session_, _eye_layer_
    cdef ovrRecti to_return = ovrRecti()
    (<ovrRecti>to_return).c_data[0] = _eye_layer_.Viewport[eye]

    return to_return

cpdef void setRenderViewport(int eye, ovrRecti viewPortRect):
    """Set the viewport rectangle for a specified eye view. This defines where
    on the swap texture the eye view is to be drawn/retrieved.
    
    :param eye: int
    :param viewPortRect: ovrRecti
    :return: None
    
    """
    global _eye_layer_
    _eye_layer_.Viewport[eye] = viewPortRect.c_data[0]

cpdef int getRenderLayerFlags():
    """Get the render layer's header flags.
    
    :return: int
    
    """
    global _eye_layer_
    return <int>_eye_layer_.Header.Flags

cpdef void setRenderLayerFlags(int layerHeaderFlags):
    """Set the render layer's header flags.
    
    :param layerHeaderFlags: 
    :return: None
    
    """
    global _eye_layer_
    _eye_layer_.Header.Flags = layerHeaderFlags

# ---------------------------------
# VR Tracking Classes and Functions
# ---------------------------------
#
cdef class ovrPoseStatef(object):
    """Pose state data.

    """
    cdef ovr_capi.ovrPoseStatef* c_data
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
        (<ovrPosef>to_return).c_data[0] = <ovr_math.Posef>self.c_data[0].ThePose

        return to_return

    @property
    def AngularVelocity(self):
        self.field_angular_velocity.c_data[0] = \
            (<ovr_math.Vector3f>self.c_data[0].AngularVelocity)

        return self.field_angular_velocity

    @property
    def LinearVelocity(self):
        self.field_linear_velocity.c_data[0] = \
            (<ovr_math.Vector3f>self.c_data[0].LinearVelocity)

        return self.field_linear_velocity

    @property
    def AngularAcceleration(self):
        self.field_angular_acceleration.c_data[0] = \
            (<ovr_math.Vector3f>self.c_data[0].AngularAcceleration)

        return self.field_angular_acceleration

    @property
    def LinearAcceleration(self):
        self.field_linear_acceleration.c_data[0] = \
            (<ovr_math.Vector3f>self.c_data[0].LinearAcceleration)

        return self.field_linear_acceleration

    @property
    def TimeInSeconds(self):
        return <double>self.c_data[0].TimeInSeconds


ovrStatus_OrientationTracked = ovr_capi.ovrStatus_OrientationTracked
ovrStatus_PositionTracked = ovr_capi.ovrStatus_PositionTracked

cdef class TrackingStateData(object):
    """Structure which stores tracking state information. All attributes are
    read-only, returning a copy of the data in the accessed field.

    """
    cdef ovr_capi.ovrTrackingState* c_data
    cdef ovr_capi.ovrTrackingState  c_ovrTrackingState

    def __cinit__(self, *args, **kwargs):
        self.c_data = &self.c_ovrTrackingState

    @property
    def HeadPose(self):
        cdef ovrPoseStatef to_return = ovrPoseStatef()
        (<ovrPoseStatef>to_return).c_data[0] = self.c_data[0].HeadPose

        return to_return

    @property
    def StatusFlags(self):
        return <unsigned int>self.c_data[0].StatusFlags

    @property
    def HandPoses(self):
        cdef ovrPoseStatef left_hand_pose = ovrPoseStatef()
        (<ovrPoseStatef>left_hand_pose).c_data[0] = self.c_data[0].HandPoses[0]

        cdef ovrPoseStatef right_hand_pose = ovrPoseStatef()
        (<ovrPoseStatef>right_hand_pose).c_data[0] = self.c_data[0].HandPoses[1]

        return left_hand_pose, right_hand_pose

    @property
    def HandStatusFlags(self):
        return <unsigned int>self.c_data[0].HandStatusFlags[0], \
               <unsigned int>self.c_data[0].HandStatusFlags[1]

cpdef TrackingStateData getTrackingState(
        double abs_time,
        bint latency_marker=True):

    cdef ovr_capi.ovrBool use_marker = \
        ovr_capi.ovrTrue if latency_marker else ovr_capi.ovrFalse

    cdef ovr_capi.ovrTrackingState ts = ovr_capi.ovr_GetTrackingState(
        _ptr_session_, abs_time, use_marker)

    cdef TrackingStateData to_return = TrackingStateData()
    (<TrackingStateData>to_return).c_data[0] = ts

    return to_return

cpdef void setTrackingOriginType(str origin='floor'):
    """Set the tracking origin type. Can either be 'floor' or 'eye'.
    
    :param origin: str
    :return: 
    """
    global _ptr_session_
    cdef ovr_capi.ovrResult result
    if origin == 'floor':
        result = ovr_capi.ovr_SetTrackingOriginType(
            _ptr_session_, ovr_capi.ovrTrackingOrigin_FloorLevel)
    elif origin == 'eye':
        result = ovr_capi.ovr_SetTrackingOriginType(
            _ptr_session_, ovr_capi.ovrTrackingOrigin_EyeLevel)

    if debug_mode:
        check_result(result)

cpdef str getTrackingOriginType():
    """Get the current tracking origin type.
    
    :return: str
    """
    global _ptr_session_
    cdef ovr_capi.ovrTrackingOrigin origin = ovr_capi.ovr_GetTrackingOriginType(
        _ptr_session_)

    if origin == ovr_capi.ovrTrackingOrigin_FloorLevel:
        return 'floor'
    elif origin == ovr_capi.ovrTrackingOrigin_EyeLevel:
        return 'eye'

cpdef void recenterTrackingOrigin():
    """Recenter the tracking origin.
    
    :return: None
    
    """
    global _ptr_session_
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_RecenterTrackingOrigin(
        _ptr_session_)

    if debug_mode:
        check_result(result)

cpdef void specifyTrackingOrigin(ovrPosef originPose):
    """Specify a custom tracking origin.
    
    :param origin_pose: ovrVector3f
    :return: 
    """
    global _ptr_session_
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_SpecifyTrackingOrigin(
        _ptr_session_, <ovr_capi.ovrPosef>originPose.c_data[0])

    if debug_mode:
        check_result(result)

cpdef list calcEyePoses(TrackingStateData trackingState):
    """Calculate eye poses from tracking state data.
    
    Poses are stored internally for conversion to transformation matrices by 
    calling 'get_eye_view_matrix'. Should be called at least once per frame 
    after 'wait_to_begin_frame' but before 'begin_frame' to minimize 
    motion-to-photon latency.
    
    :param tracking_state: TrackingStateData
    :return: 
    
    """
    global _hmd_to_eye_view_pose_, _eye_layer_

    ovr_capi_util.ovr_CalcEyePoses2(
        trackingState.c_data[0].HeadPose.ThePose,
        _hmd_to_eye_view_pose_,
        _eye_layer_.RenderPose)

    cdef ovrPosef eye_pose0 = ovrPosef()
    cdef ovrPosef eye_pose1 = ovrPosef()
    (<ovrPosef>eye_pose0).c_data[0] = <ovr_math.Posef>_eye_layer_.RenderPose[0]
    (<ovrPosef>eye_pose1).c_data[0] = <ovr_math.Posef>_eye_layer_.RenderPose[1]

    return [eye_pose0, eye_pose1]

cpdef ovrMatrix4f getEyeViewMatrix(ovrPosef eyePose):
    """Get the view matrix from the last calculated head pose. This should be
    called once per frame if real-time head tracking is desired.
    
    :param eye: str
    :return: 
    
    """
    cdef ovrVector3f pos = ovrVector3f()
    cdef ovrMatrix4f rot = ovrMatrix4f()
    pos.c_data[0] = <ovr_math.Vector3f>eyePose.c_data.Translation
    rot.c_data[0] = ovr_math.Matrix4f(<ovr_math.Quatf>eyePose.c_data.Rotation)

    cdef ovrVector3f final_up = \
        (<ovrVector3f>rot).transform(ovrVector3f(0, 1, 0))
    cdef ovrVector3f final_forward = \
        (<ovrVector3f>rot).transform(ovrVector3f(0, 0, -1))
    cdef ovrMatrix4f viewMatrix = \
        ovrMatrix4f.lookAt(pos, pos + final_forward, final_up)

    return viewMatrix

cpdef ovrMatrix4f getEyeProjectionMatrix(
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
    global _eye_layer_

    cdef ovrMatrix4f projectionMatrix = ovrMatrix4f()
    (<ovrMatrix4f>projectionMatrix).c_data[0] = \
        <ovr_math.Matrix4f>ovr_capi_util.ovrMatrix4f_Projection(
            _eye_layer_.Fov[eye],
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
            _ptr_session_, frameIndex)
    else:
        t_secs = ovr_capi.ovr_GetTimeInSeconds()

    return t_secs

cpdef int waitToBeginFrame(unsigned int frameIndex=0):
    cdef ovr_capi.ovrResult result = 0
    result = ovr_capi.ovr_WaitToBeginFrame(_ptr_session_, frameIndex)

    return <int>result

cpdef int beginFrame(unsigned int frameIndex=0):
    result = ovr_capi.ovr_BeginFrame(_ptr_session_, frameIndex)

    return <int>result

cpdef void commitSwapChain(int sc):
    global _ptr_session_, _swap_chain_
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_CommitTextureSwapChain(
        _ptr_session_,
        _swap_chain_[sc])

    if debug_mode:
        check_result(result)

cpdef void endFrame(unsigned int frameIndex=0):
    global _eye_layer_
    cdef ovr_capi.ovrLayerHeader* layers = &_eye_layer_.Header
    result = ovr_capi.ovr_EndFrame(
        _ptr_session_,
        frameIndex,
        NULL,
        &layers,
        <unsigned int>1)

    if debug_mode:
        check_result(result)

    global _frame_index_
    _frame_index_ += 1

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

        self.c_ovrMirrorTextureDesc.Format = <ovr_capi.ovrTextureFormat>_format
        self.c_ovrMirrorTextureDesc.Width = width
        self.c_ovrMirrorTextureDesc.Height = height
        self.c_ovrMirrorTextureDesc.MiscFlags = ovr_capi.ovrTextureMisc_None
        self.c_ovrMirrorTextureDesc.MirrorOptions = <int32_t>mirrorOptions

    @property
    def Format(self):
        return <int>self.c_ovrMirrorTextureDesc.Format

    @Format.setter
    def Format(self, int value):
        self.c_ovrMirrorTextureDesc.Format = <ovr_capi.ovrTextureFormat>value

    @property
    def Width(self):
        return <int>self.c_ovrMirrorTextureDesc.Width

    @Width.setter
    def Width(self, int value):
        self.c_ovrMirrorTextureDesc.Width = value

    @property
    def Height(self):
        return <int>self.c_ovrMirrorTextureDesc.Height

    @Height.setter
    def Height(self, int value):
        self.c_ovrMirrorTextureDesc.Height = value

    @property
    def MirrorOptions(self):
        return <int>self.c_ovrMirrorTextureDesc.MirrorOptions

    @MirrorOptions.setter
    def MirrorOptions(self, int value):
        self.c_ovrMirrorTextureDesc.MirrorOptions = <int32_t>value


cpdef void setupMirrorTexture(ovrMirrorTextureDesc mirrorDesc):
    """Create a mirror texture buffer.
    
    :param width: int 
    :param height: int 
    :return: None
    
    """
    global _mirror_texture_
    cdef ovr_capi.ovrResult result = ovr_capi_gl.ovr_CreateMirrorTextureGL(
        _ptr_session_, &mirrorDesc.c_ovrMirrorTextureDesc, &_mirror_texture_)

    if debug_mode:
        check_result(result)

cpdef unsigned int getMirrorTexture():
    """Get the mirror texture handle.
    
    :return: 
    """
    cdef unsigned int out_tex_id
    cdef ovr_capi.ovrResult result = \
        ovr_capi_gl.ovr_GetMirrorTextureBufferGL(
            _ptr_session_,
            _mirror_texture_,
            &out_tex_id)

    return <unsigned int>out_tex_id

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
    cdef ovr_capi.ovrSessionStatus* c_data
    cdef ovr_capi.ovrSessionStatus  c_ovrSessionStatus

    def __cinit__(self, *args, **kwargs):
        self.c_data = &self.c_ovrSessionStatus

    def IsVisible(self):
        return <bint>self.c_data.IsVisible

    def HmdPresent(self):
        return <bint>self.c_data.HmdPresent

    def DisplayLost(self):
        return <bint>self.c_data.DisplayLost

    def ShouldQuit(self):
        return <bint>self.c_data.ShouldQuit

    def ShouldRecenter(self):
        return <bint>self.c_data.ShouldRecenter

    def HasInputFocus(self):
        return <bint>self.c_data.HasInputFocus

    def OverlayPresent(self):
        return <bint>self.c_data.OverlayPresent

    def DepthRequested(self):
        return <bint>self.c_data.DepthRequested


cpdef ovrSessionStatus getSessionStatus():
    """Get the current session status.
    
    :return: ovrSessionStatus
    
    """
    global _ptr_session_
    cdef ovrSessionStatus to_return = ovrSessionStatus()
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetSessionStatus(
        _ptr_session_, &(<ovrSessionStatus>to_return).c_data[0])

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
    cdef ovr_capi.ovrInputState* c_data
    cdef ovr_capi.ovrInputState c_ovrInputState

    def __cinit__(self, *args, **kwargs):
        self.c_data = &self.c_ovrInputState

    @property
    def TimeInSeconds(self):
        return <double>self.c_data.TimeInSeconds

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
        cdef int ctrl_type = <int>self.c_data[0].ControllerType
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
        cdef float index_trigger_right = self.c_data[0].IndexTriggerNoDeadzone[1]

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
    global _ptr_session_
    cdef ovr_capi.ovrInputState* ptr_state
    cdef ovrInputState to_return = ovrInputState()

    if stateOut is None:
        ptr_state = &(<ovrInputState>to_return).c_ovrInputState
    else:
        ptr_state = &(<ovrInputState>stateOut).c_ovrInputState

    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetInputState(
        _ptr_session_,
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
    global _ptr_session_, _ctrl_states_, _ctrl_states_prev_
    cdef ovr_capi.ovrInputState* ptr_ctrl = NULL
    cdef ovr_capi.ovrInputState* ptr_ctrl_prev = NULL

    cdef ovr_capi.ovrControllerType ctrl_type
    if controller == 'xbox':
        ctrl_type = ovr_capi.ovrControllerType_XBox
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.xbox]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.xbox]
    elif controller == 'remote':
        ctrl_type = ovr_capi.ovrControllerType_Remote
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.remote]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.remote]
    elif controller == 'touch':
        ctrl_type = ovr_capi.ovrControllerType_Touch
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.touch]
    elif controller == 'left_touch':
        ctrl_type = ovr_capi.ovrControllerType_LTouch
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.left_touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.left_touch]
    elif controller == 'right_touch':
        ctrl_type = ovr_capi.ovrControllerType_RTouch
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.right_touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.right_touch]

    # copy the previous control state
    ptr_ctrl_prev[0] = ptr_ctrl[0]

    # update the current controller state
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetInputState(
        _ptr_session_,
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
    cdef ovr_capi.ovrInputState* ptr_ctrl_state = NULL
    if controller == 'xbox':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.xbox]
    elif controller == 'remote':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.remote]
    elif controller == 'touch':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.touch]
    elif controller == 'left_touch':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.left_touch]
    elif controller == 'right_touch':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.right_touch]

    return ptr_ctrl_state[0].TimeInSeconds

cpdef tuple getIndexTriggerValues(str controller, bint deadZone=False):
    """Get index trigger values for a specified controller.
    
    :param controller: str
    :param deadZone: boolean
    :return: tuple
    
    """
    # get pointer to control state
    global _ctrl_states_
    cdef ovr_capi.ovrInputState* ptr_ctrl_state = NULL
    if controller == 'xbox':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.xbox]
    elif controller == 'remote':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.remote]
    elif controller == 'touch':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.touch]
    elif controller == 'left_touch':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.left_touch]
    elif controller == 'right_touch':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.right_touch]

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
    cdef ovr_capi.ovrInputState* ptr_ctrl_state = NULL
    if controller == 'xbox':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.xbox]
    elif controller == 'remote':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.remote]
    elif controller == 'touch':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.touch]
    elif controller == 'left_touch':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.left_touch]
    elif controller == 'right_touch':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.right_touch]

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
    cdef ovr_capi.ovrInputState* ptr_ctrl = NULL
    cdef ovr_capi.ovrInputState* ptr_ctrl_prev = NULL
    if controller == 'xbox':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.xbox]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.xbox]
    elif controller == 'remote':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.remote]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.remote]
    elif controller == 'touch':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.touch]
    elif controller == 'left_touch':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.left_touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.left_touch]
    elif controller == 'right_touch':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.right_touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.right_touch]

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

cpdef bint getButtons(str controller, object buttonNames, str trigger='continuous'):
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
    cdef ovr_capi.ovrInputState* ptr_ctrl = NULL
    cdef ovr_capi.ovrInputState* ptr_ctrl_prev = NULL
    if controller == 'xbox':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.xbox]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.xbox]
    elif controller == 'remote':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.remote]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.remote]
    elif controller == 'touch':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.touch]
    elif controller == 'left_touch':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.left_touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.left_touch]
    elif controller == 'right_touch':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.right_touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.right_touch]

    cdef unsigned int button_bits = 0x00000000
    cdef int i, N
    if isinstance(buttonNames, str):  # don't loop if a string is specified
        button_bits |= ctrl_button_lut[buttonNames]
    elif isinstance(buttonNames, (tuple, list)):
        # loop over all names and combine them
        N = <int>len(buttonNames)
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

cpdef bint getTouches(str controller, object touchNames, str trigger='continuous'):
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
    cdef ovr_capi.ovrInputState* ptr_ctrl = NULL
    cdef ovr_capi.ovrInputState* ptr_ctrl_prev = NULL
    if controller == 'xbox':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.xbox]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.xbox]
    elif controller == 'remote':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.remote]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.remote]
    elif controller == 'touch':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.touch]
    elif controller == 'left_touch':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.left_touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.left_touch]
    elif controller == 'right_touch':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.right_touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.right_touch]

    cdef unsigned int touch_bits = 0x00000000
    cdef int i, N
    if isinstance(touchNames, str):  # don't loop if a string is specified
        touch_bits |= ctrl_button_lut[touchNames]
    elif isinstance(touchNames, (tuple, list)):
        # loop over all names and combine them
        N = <int>len(touchNames)
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
        _ptr_session_)

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
    cdef ovr_capi.ovrPerfStatsPerCompositorFrame* c_data
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
    cdef ovr_capi.ovrPerfStats* c_data
    cdef ovr_capi.ovrPerfStats  c_ovrPerfStats
    cdef list perf_stats

    def __cinit__(self, *args, **kwargs):
        self.c_data = &self.c_ovrPerfStats

        # initialize performance stats list
        self.perf_stats = list()
        cdef int i, N
        N = <int>ovr_capi.ovrMaxProvidedFrameStats
        for i in range(N):
            self.perf_stats.append(ovrPerfStatsPerCompositorFrame())
            (<ovrPerfStatsPerCompositorFrame>self.perf_stats[i]).c_data[0] = \
                self.c_data[0].FrameStats[i]

    @property
    def FrameStatsCount(self):
        return self.c_data[0].FrameStatsCount

    @property
    def AnyFrameStatsDropped(self):
        return <bint>self.c_data[0].AnyFrameStatsDropped

    @property
    def FrameStats(self):
        cdef int i, N
        N = self.c_data[0].FrameStatsCount
        for i in range(N):
            (<ovrPerfStatsPerCompositorFrame>self.perf_stats[i]).c_data[0] = \
                self.c_data[0].FrameStats[i]

        return self.perf_stats

    @property
    def AdaptiveGpuPerformanceScale(self):
        return <bint>self.c_data[0].AdaptiveGpuPerformanceScale

    @property
    def AswIsAvailable(self):
        return <bint>self.c_data[0].AswIsAvailable


cpdef ovrPerfStats getFrameStats():
    """Get most recent performance stats, returns an object with fields
    corresponding to various performance stats reported by the SDK.
    
    :return: dict 
    
    """
    global _ptr_session_

    cdef ovrPerfStats to_return = ovrPerfStats()
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetPerfStats(
        _ptr_session_,
        &(<ovrPerfStats>to_return).c_data[0])

    if debug_mode:
        check_result(result)

    return to_return

cpdef void resetFrameStats():
    """Flushes backlog of frame stats.
    
    :return: None 
    
    """
    global _ptr_session_
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_ResetPerfStats(
        _ptr_session_)

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

cpdef void perfHudMode(str mode='Off'):
    """Display a performance HUD with a specified mode.
    
    :param mode: str 
    :return: None
    
    """
    global _ptr_session_
    cdef int perf_hud_mode = 0

    if mode == 'Off':
        perf_hud_mode = <int>ovr_capi.ovrPerfHud_Off
    elif mode == 'PerfSummary':
        perf_hud_mode = <int>ovr_capi.ovrPerfHud_PerfSummary
    elif mode == 'LatencyTiming':
        perf_hud_mode = <int>ovr_capi.ovrPerfHud_LatencyTiming
    elif mode == 'AppRenderTiming':
        perf_hud_mode = <int>ovr_capi.ovrPerfHud_AppRenderTiming
    elif mode == 'CompRenderTiming':
        perf_hud_mode = <int>ovr_capi.ovrPerfHud_CompRenderTiming
    elif mode == 'AswStats':
        perf_hud_mode = <int>ovr_capi.ovrPerfHud_AswStats
    elif mode == 'VersionInfo':
        perf_hud_mode = <int>ovr_capi.ovrPerfHud_VersionInfo

    cdef ovr_capi.ovrBool ret = ovr_capi.ovr_SetInt(
        _ptr_session_, b"PerfHudMode", perf_hud_mode)

# -----------------------------
# Boundary and Safety Functions
# -----------------------------
#
cdef ovr_capi.ovrBoundaryLookAndFeel _boundary_style_

cpdef void setBoundryColor(float r, float g, float b):
    global _ptr_session_, _boundary_style_

    cdef ovr_capi.ovrColorf color
    color.r = r
    color.g = g
    color.b = b

    _boundary_style_.Color = color

    cdef ovr_capi.ovrResult result = ovr_capi.ovr_SetBoundaryLookAndFeel(
        _ptr_session_,
        &_boundary_style_)

    if debug_mode:
        check_result(result)

cpdef void resetBoundryColor():
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_ResetBoundaryLookAndFeel(
        _ptr_session_)

    if debug_mode:
        check_result(result)

cpdef bint isBoundryVisible():
    cdef ovr_capi.ovrBool is_visible
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetBoundaryVisible(
        _ptr_session_, &is_visible)

    if debug_mode:
        check_result(result)

    return <bint>is_visible

cpdef void showBoundry(bint show=True):
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_RequestBoundaryVisible(
        _ptr_session_, <ovr_capi.ovrBool>show)

    if debug_mode:
        check_result(result)

# -----------------------
# Miscellaneous Functions
# -----------------------
#
cpdef float getPlayerHeight():
    global _ptr_session_
    cdef float to_return  = ovr_capi.ovr_GetFloat(
        _ptr_session_,
        b"PlayerHeight",
        <float>1.778)

    return to_return

cpdef float getEyeHeight():
    global _ptr_session_
    cdef float to_return  = ovr_capi.ovr_GetFloat(
        _ptr_session_,
        b"EyeHeight",
        <float>1.675)

    return to_return

cpdef tuple getNeckEyeDistance():
    global _ptr_session_
    cdef float vals[2]

    cdef unsigned int ret  = ovr_capi.ovr_GetFloatArray(
        _ptr_session_,
        b"NeckEyeDistance",
        vals,
        <unsigned int>2)

    return <float>vals[0], <float>vals[1]

cpdef tuple getEyeToNoseDist():
    global _ptr_session_
    cdef float vals[2]

    cdef unsigned int ret  = ovr_capi.ovr_GetFloatArray(
        _ptr_session_,
        b"EyeToNoseDist",
        vals,
        <unsigned int>2)

    return <float>vals[0], <float>vals[1]