cimport ovr_capi, ovr_capi_gl, ovr_errorcode, ovr_capi_util
from libc.stdint cimport uintptr_t, uint32_t, int32_t
from libcpp cimport nullptr
from libc.stdlib cimport malloc, free

# -----------------
# Initialize module
# -----------------
#
cdef ovr_capi.ovrInitParams _init_params_  # initialization parameters

# HMD descriptor storing information about the HMD being used.
cdef ovr_capi.ovrHmdDesc _hmd_desc_

# Since we are only using one session per module instance, so we are going to
# create our session pointer here and use it module-wide.
#
cdef ovr_capi.ovrSession _ptr_session_
cdef ovr_capi.ovrGraphicsLuid _ptr_luid_

# Frame index
#
cdef long long _frame_index_ = 0

# texture swap chain
#
cdef ovr_capi.ovrTextureSwapChain _swap_chain_ = NULL
cdef ovr_capi.ovrMirrorTexture _mirror_texture_ = NULL

# VR related structures to store head pose and other data used across frames.
#
cdef ovr_capi.ovrEyeRenderDesc[2] _eye_render_desc_
cdef ovr_capi.ovrPosef[2] _hmd_to_eye_view_pose_
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

# ----------------
# Module Functions
# ----------------
#
cpdef dict start_session():
    cdef ovr_capi.ovrResult result = 0
    result = ovr_capi.ovr_Initialize(NULL)
    result = ovr_capi.ovr_Create(&_ptr_session_, &_ptr_luid_)
    if ovr_errorcode.OVR_FAILURE(result):
        ovr_capi.ovr_Shutdown()

    # get HMD descriptor
    global _hmd_desc_
    _hmd_desc_ = ovr_capi.ovr_GetHmdDesc(_ptr_session_)

    # configure VR data with HMD descriptor information
    global _eye_render_desc_, _hmd_to_eye_view_pose_
    _eye_render_desc_[0] = ovr_capi.ovr_GetRenderDesc(
        _ptr_session_, ovr_capi.ovrEye_Left, _hmd_desc_.DefaultEyeFov[0])
    _eye_render_desc_[1] = ovr_capi.ovr_GetRenderDesc(
        _ptr_session_, ovr_capi.ovrEye_Right, _hmd_desc_.DefaultEyeFov[1])
    _hmd_to_eye_view_pose_[0] = _eye_render_desc_[0].HmdToEyePose
    _hmd_to_eye_view_pose_[1] = _eye_render_desc_[1].HmdToEyePose

    # return HMD information from descriptor
    cdef dict hmd_info = dict()
    hmd_info["ProductName"] = _hmd_desc_.ProductName.decode('utf-8')
    hmd_info["Manufacturer"] = _hmd_desc_.Manufacturer.decode('utf-8')
    hmd_info["VendorId"] = <int>_hmd_desc_.VendorId
    hmd_info["ProductId"] = <int>_hmd_desc_.ProductId
    hmd_info["SerialNumber"] = _hmd_desc_.SerialNumber.decode('utf-8')
    hmd_info["FirmwareMajor"] = <int>_hmd_desc_.FirmwareMajor
    hmd_info["FirmwareMinor"] = <int>_hmd_desc_.FirmwareMinor
    hmd_info["DisplayRefreshRate"] = <float>_hmd_desc_.DisplayRefreshRate

    return hmd_info

cpdef void end_session():
    ovr_capi.ovr_DestroyTextureSwapChain(_ptr_session_, _swap_chain_)
    ovr_capi.ovr_Destroy(_ptr_session_)
    ovr_capi.ovr_Shutdown()

cpdef tuple get_buffer_size(float texels_per_pixel=1.0):
    cdef ovr_capi.ovrSizei rec_tex0_size, rec_tex1_size, buffer_size

    rec_tex0_size = ovr_capi.ovr_GetFovTextureSize(
        _ptr_session_,
        ovr_capi.ovrEye_Left,
        _hmd_desc_.DefaultEyeFov[0],
        texels_per_pixel)
    rec_tex1_size = ovr_capi.ovr_GetFovTextureSize(
        _ptr_session_,
        ovr_capi.ovrEye_Right,
        _hmd_desc_.DefaultEyeFov[1],
        texels_per_pixel)

    buffer_size.w  = rec_tex0_size.w + rec_tex1_size.w
    buffer_size.h = max(rec_tex0_size.h, rec_tex1_size.h)

    return buffer_size.w, buffer_size.h

cpdef void setup_render_layer(object buffer_size):
    # setup the texture swap chain
    cdef ovr_capi.ovrTextureSwapChainDesc tmp_desc
    tmp_desc.Type = ovr_capi.ovrTexture_2D
    tmp_desc.Format = ovr_capi.OVR_FORMAT_R8G8B8A8_UNORM_SRGB
    tmp_desc.Width = buffer_size[0]
    tmp_desc.Height = buffer_size[1]
    tmp_desc.ArraySize = tmp_desc.MipLevels = tmp_desc.SampleCount = 1
    tmp_desc.StaticImage = ovr_capi.ovrFalse
    tmp_desc.BindFlags = tmp_desc.MiscFlags = 0

    # create the texture swap chain
    global _swap_chain_
    cdef ovr_capi.ovrResult result = 0
    result = ovr_capi_gl.ovr_CreateTextureSwapChainGL(
        _ptr_session_, &tmp_desc, &_swap_chain_)

    if debug_mode:
        check_result(result)

    # check if a texture ID is returned after creating the swap chain
    cdef unsigned int out_tex_id
    result = ovr_capi_gl.ovr_GetTextureSwapChainBufferGL(
        _ptr_session_, _swap_chain_, 0, &out_tex_id)

    if debug_mode:
        check_result(result)

    if <int>out_tex_id == 0:
        pass  # raise an error, the texture ID returned is invalid

    # setup the render layer
    _eye_layer_.Header.Type = ovr_capi.ovrLayerType_EyeFov
    _eye_layer_.Header.Flags = ovr_capi.ovrLayerFlag_TextureOriginAtBottomLeft
    _eye_layer_.ColorTexture[0] = _swap_chain_
    _eye_layer_.ColorTexture[1] = NULL
    _eye_layer_.Fov[0] = _eye_render_desc_[0].Fov
    _eye_layer_.Fov[1] = _eye_render_desc_[1].Fov
    _eye_layer_.Viewport[0].Pos.x = 0
    _eye_layer_.Viewport[0].Pos.y = 0
    _eye_layer_.Viewport[0].Size.w = buffer_size[0] / 2
    _eye_layer_.Viewport[0].Size.h = buffer_size[1]
    _eye_layer_.Viewport[1].Pos.x = buffer_size[0] / 2
    _eye_layer_.Viewport[1].Pos.y = 0
    _eye_layer_.Viewport[1].Size.w = buffer_size[0] / 2
    _eye_layer_.Viewport[1].Size.h = buffer_size[1]

cpdef void setup_mirror_texture(int width=800, int height=600):
    cdef ovr_capi.ovrMirrorTextureDesc mirror_desc
    mirror_desc.Format = ovr_capi.OVR_FORMAT_R8G8B8A8_UNORM_SRGB
    mirror_desc.Width = width
    mirror_desc.Height = height
    mirror_desc.MiscFlags = ovr_capi.ovrTextureMisc_None
    mirror_desc.MirrorOptions = ovr_capi.ovrMirrorOption_PostDistortion

    global _mirror_texture_
    cdef ovr_capi.ovrResult result = ovr_capi_gl.ovr_CreateMirrorTextureGL(
        _ptr_session_, &mirror_desc, &_mirror_texture_)

    if debug_mode:
        check_result(result)

cpdef double get_display_time(unsigned int frame_index=0, bint predicted=True):
    cdef double t_secs
    if predicted:
        t_secs = ovr_capi.ovr_GetPredictedDisplayTime(
            _ptr_session_, frame_index)
    else:
        t_secs = ovr_capi.ovr_GetTimeInSeconds()

    return t_secs

cpdef int wait_to_begin_frame(unsigned int frame_index=0):
    cdef ovr_capi.ovrResult result = 0
    result = ovr_capi.ovr_WaitToBeginFrame(_ptr_session_, frame_index)

    return <int>result

cpdef void calc_eye_poses(double abs_time, bint time_stamp=True):

    cdef ovr_capi.ovrBool use_marker = 0
    if time_stamp:
        use_marker = ovr_capi.ovrTrue
    else:
        use_marker = ovr_capi.ovrFalse

    cpdef ovr_capi.ovrTrackingState hmd_state = ovr_capi.ovr_GetTrackingState(
        _ptr_session_, abs_time, use_marker)

    ovr_capi_util.ovr_CalcEyePoses2(
        hmd_state.HeadPose.ThePose,
        _hmd_to_eye_view_pose_,
        _eye_layer_.RenderPose)

cpdef int begin_frame(unsigned int frame_index=0):
    result = ovr_capi.ovr_BeginFrame(_ptr_session_, frame_index)

    return <int>result

cpdef unsigned int get_texture_swap_buffer():
    cdef int current_idx = 0
    cdef unsigned int tex_id = 0
    ovr_capi.ovr_GetTextureSwapChainCurrentIndex(
        _ptr_session_, _swap_chain_, &current_idx)
    ovr_capi_gl.ovr_GetTextureSwapChainBufferGL(
        _ptr_session_, _swap_chain_, current_idx, &tex_id)

    return tex_id

cpdef unsigned int get_mirror_texture():
    cdef unsigned int out_tex_id
    cdef ovr_capi.ovrResult result = \
        ovr_capi_gl.ovr_GetMirrorTextureBufferGL(
            _ptr_session_,
            _mirror_texture_,
            &out_tex_id)

    return <unsigned int>out_tex_id

cpdef void end_frame(unsigned int frame_index=0):
    ovr_capi.ovr_CommitTextureSwapChain(_ptr_session_, _swap_chain_)
    cdef ovr_capi.ovrLayerHeader* layers = &_eye_layer_.Header
    result = ovr_capi.ovr_EndFrame(
        _ptr_session_,
        frame_index,
        NULL,
        &layers,
        <unsigned int>1)

    if debug_mode:
        check_result(result)

    global _frame_index_
    _frame_index_ += 1