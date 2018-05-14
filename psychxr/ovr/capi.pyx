#  =============================================================================
#  OVR C API Extension Library for PsychXR
#  =============================================================================
#
#  capi.pyx
#
#  Copyright 2018 Matthew D. Cutone <cutonem(a)yorku.ca> and Laurie M. Wilcox
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
"""This file exposes Oculus Rift(TM) C API types and functions to the Python
environment. The declarations in the file are contemporaneous with version 1.24
(retrieved 04.15.2018) of the Oculus Rift(TM) PC SDK.

"""
cimport ovr_capi
from libc.stdint cimport uintptr_t, uint32_t, int32_t
from libc.stdlib cimport malloc, free

# --- ENUMS/CONSTANTS ---

# enum ovrHmdType
ovrHmd_None = ovr_capi.ovrHmd_None
ovrHmd_DK1 = ovr_capi.ovrHmd_DK1
ovrHmd_DKHD = ovr_capi.ovrHmd_DKHD
ovrHmd_DK2 = ovr_capi.ovrHmd_DK2
ovrHmd_CB = ovr_capi.ovrHmd_CB
ovrHmd_Other = ovr_capi.ovrHmd_Other
ovrHmd_E3_2015 = ovr_capi.ovrHmd_E3_2015
ovrHmd_ES06 = ovr_capi.ovrHmd_ES06
ovrHmd_ES09 = ovr_capi.ovrHmd_ES09
ovrHmd_ES11 = ovr_capi.ovrHmd_ES11
ovrHmd_CV1 = ovr_capi.ovrHmd_CV1

# enum ovrHmdCaps
ovrHmdCap_DebugDevice = ovr_capi.ovrHmdCap_DebugDevice

# enum ovrTrackingCaps
ovrTrackingCap_Orientation = ovr_capi.ovrTrackingCap_Orientation
ovrTrackingCap_MagYawCorrection = ovr_capi.ovrTrackingCap_MagYawCorrection
ovrTrackingCap_Position = ovr_capi.ovrTrackingCap_Position

# enum ovrExtensions
ovrExtension_TextureLayout_Octilinear = \
    ovr_capi.ovrExtension_TextureLayout_Octilinear

# enum ovrEyeType
ovrEye_Left = ovr_capi.ovrEye_Left
ovrEye_Right = ovr_capi.ovrEye_Right
ovrEye_Count = ovr_capi.ovrEye_Count

# enum ovrTrackingOrigin
ovrTrackingOrigin_EyeLevel = ovr_capi.ovrTrackingOrigin_EyeLevel
ovrTrackingOrigin_FloorLevel = ovr_capi.ovrTrackingOrigin_FloorLevel
ovrTrackingOrigin_Count = ovr_capi.ovrTrackingOrigin_Count

# enum ovrStatusBits
ovrStatus_OrientationTracked = ovr_capi.ovrStatus_OrientationTracked
ovrStatus_PositionTracked = ovr_capi.ovrStatus_PositionTracked

# enum ovrTrackerFlags
ovrTracker_Connected = ovr_capi.ovrTracker_Connected
ovrTracker_PoseTracked = ovr_capi.ovrTracker_PoseTracked

# enum ovrTextureType
ovrTexture_2D = ovr_capi.ovrTexture_2D
ovrTexture_2D_External = ovr_capi.ovrTexture_2D_External
ovrTexture_Cube = ovr_capi.ovrTexture_Cube
ovrTexture_Count = ovr_capi.ovrTexture_Count

# enum ovrTextureBindFlags
ovrTextureBind_None  = ovr_capi.ovrTextureBind_None
ovrTextureBind_DX_RenderTarget  = ovr_capi.ovrTextureBind_DX_RenderTarget
ovrTextureBind_DX_UnorderedAccess  = ovr_capi.ovrTextureBind_DX_UnorderedAccess
ovrTextureBind_DX_DepthStencil  = ovr_capi.ovrTextureBind_DX_DepthStencil

# enum ovrTextureFormat
OVR_FORMAT_UNKNOWN = ovr_capi.OVR_FORMAT_UNKNOWN
OVR_FORMAT_B5G6R5_UNORM = ovr_capi.OVR_FORMAT_B5G6R5_UNORM
OVR_FORMAT_B5G5R5A1_UNORM = ovr_capi.OVR_FORMAT_B5G5R5A1_UNORM
OVR_FORMAT_B4G4R4A4_UNORM = ovr_capi.OVR_FORMAT_B4G4R4A4_UNORM
OVR_FORMAT_R8G8B8A8_UNORM = ovr_capi.OVR_FORMAT_R8G8B8A8_UNORM
OVR_FORMAT_R8G8B8A8_UNORM_SRGB = ovr_capi.OVR_FORMAT_R8G8B8A8_UNORM_SRGB
OVR_FORMAT_B8G8R8A8_UNORM = ovr_capi.OVR_FORMAT_B8G8R8A8_UNORM
OVR_FORMAT_B8G8R8_UNORM = ovr_capi.OVR_FORMAT_B8G8R8_UNORM
OVR_FORMAT_B8G8R8A8_UNORM_SRGB = ovr_capi.OVR_FORMAT_B8G8R8A8_UNORM_SRGB
OVR_FORMAT_B8G8R8X8_UNORM = ovr_capi.OVR_FORMAT_B8G8R8X8_UNORM
OVR_FORMAT_B8G8R8X8_UNORM_SRGB = ovr_capi.OVR_FORMAT_B8G8R8X8_UNORM_SRGB
OVR_FORMAT_R16G16B16A16_FLOAT = ovr_capi.OVR_FORMAT_R16G16B16A16_FLOAT
OVR_FORMAT_R11G11B10_FLOAT = ovr_capi.OVR_FORMAT_R11G11B10_FLOAT
OVR_FORMAT_D16_UNORM = ovr_capi.OVR_FORMAT_D16_UNORM
OVR_FORMAT_D24_UNORM_S8_UINT = ovr_capi.OVR_FORMAT_D24_UNORM_S8_UINT
OVR_FORMAT_D32_FLOAT = ovr_capi.OVR_FORMAT_D32_FLOAT
OVR_FORMAT_D32_FLOAT_S8X24_UINT = ovr_capi.OVR_FORMAT_D32_FLOAT_S8X24_UINT
OVR_FORMAT_BC1_UNORM = ovr_capi.OVR_FORMAT_BC1_UNORM
OVR_FORMAT_BC1_UNORM_SRGB = ovr_capi.OVR_FORMAT_BC1_UNORM_SRGB
OVR_FORMAT_BC2_UNORM = ovr_capi.OVR_FORMAT_BC2_UNORM
OVR_FORMAT_BC2_UNORM_SRGB = ovr_capi.OVR_FORMAT_BC2_UNORM_SRGB
OVR_FORMAT_BC3_UNORM = ovr_capi.OVR_FORMAT_BC3_UNORM
OVR_FORMAT_BC3_UNORM_SRGB = ovr_capi.OVR_FORMAT_BC3_UNORM_SRGB
OVR_FORMAT_BC6H_UF16 = ovr_capi.OVR_FORMAT_BC6H_UF16
OVR_FORMAT_BC6H_SF16 = ovr_capi.OVR_FORMAT_BC6H_SF16
OVR_FORMAT_BC7_UNORM = ovr_capi.OVR_FORMAT_BC7_UNORM
OVR_FORMAT_BC7_UNORM_SRGB = ovr_capi.OVR_FORMAT_BC7_UNORM_SRGB

# enum ovrTextureMiscFlags
ovrTextureMisc_None = ovr_capi.ovrTextureMisc_None
ovrTextureMisc_DX_Typeless = ovr_capi.ovrTextureMisc_DX_Typeless
ovrTextureMisc_AllowGenerateMips = ovr_capi.ovrTextureMisc_AllowGenerateMips
ovrTextureMisc_ProtectedContent = ovr_capi.ovrTextureMisc_ProtectedContent
ovrTextureMisc_AutoGenerateMips = ovr_capi.ovrTextureMisc_AutoGenerateMips

# enum ovrMirrorOptions
ovrMirrorOption_Default = ovr_capi.ovrMirrorOption_Default
ovrMirrorOption_PostDistortion = ovr_capi.ovrMirrorOption_PostDistortion
ovrMirrorOption_LeftEyeOnly = ovr_capi.ovrMirrorOption_LeftEyeOnly
ovrMirrorOption_RightEyeOnly = ovr_capi.ovrMirrorOption_RightEyeOnly
ovrMirrorOption_IncludeGuardian = ovr_capi.ovrMirrorOption_IncludeGuardian
ovrMirrorOption_IncludeNotifications = \
    ovr_capi.ovrMirrorOption_IncludeNotifications
ovrMirrorOption_IncludeSystemGui = ovr_capi.ovrMirrorOption_IncludeSystemGui

# enum ovrButton
ovrButton_A = ovr_capi.ovrButton_A
ovrButton_B = ovr_capi.ovrButton_B
ovrButton_RThumb = ovr_capi.ovrButton_RThumb
ovrButton_RShoulder = ovr_capi.ovrButton_RShoulder
ovrButton_X = ovr_capi.ovrButton_X
ovrButton_Y = ovr_capi.ovrButton_Y
ovrButton_LThumb = ovr_capi.ovrButton_LThumb
ovrButton_LShoulder = ovr_capi.ovrButton_LShoulder
ovrButton_Up = ovr_capi.ovrButton_Up
ovrButton_Down = ovr_capi.ovrButton_Down
ovrButton_Left = ovr_capi.ovrButton_Left
ovrButton_Right = ovr_capi.ovrButton_Right
ovrButton_Enter = ovr_capi.ovrButton_Enter
ovrButton_Back = ovr_capi.ovrButton_Back
ovrButton_VolUp = ovr_capi.ovrButton_VolUp
ovrButton_VolDown = ovr_capi.ovrButton_VolDown
ovrButton_Home = ovr_capi.ovrButton_Home
ovrButton_Private = ovr_capi.ovrButton_Private
ovrButton_RMask = ovr_capi.ovrButton_RMask
ovrButton_LMask = ovr_capi.ovrButton_LMask

# enum ovrTouch
ovrTouch_A = ovr_capi.ovrTouch_A
ovrTouch_B = ovr_capi.ovrTouch_B
ovrTouch_RThumb = ovr_capi.ovrTouch_RThumb
ovrTouch_RThumbRest = ovr_capi.ovrTouch_RThumbRest
ovrTouch_RIndexTrigger = ovr_capi.ovrTouch_RIndexTrigger
ovrTouch_RButtonMask = ovr_capi.ovrTouch_RButtonMask
ovrTouch_X = ovr_capi.ovrTouch_X
ovrTouch_Y = ovr_capi.ovrTouch_Y
ovrTouch_LThumb = ovr_capi.ovrTouch_LThumb
ovrTouch_LThumbRest = ovr_capi.ovrTouch_LThumbRest
ovrTouch_LIndexTrigger = ovr_capi.ovrTouch_LIndexTrigger
ovrTouch_LButtonMask = ovr_capi.ovrTouch_LButtonMask
ovrTouch_RIndexPointing = ovr_capi.ovrTouch_RIndexPointing
ovrTouch_RThumbUp = ovr_capi.ovrTouch_RThumbUp
ovrTouch_LIndexPointing = ovr_capi.ovrTouch_LIndexPointing
ovrTouch_LThumbUp = ovr_capi.ovrTouch_LThumbUp
ovrTouch_RPoseMask = ovr_capi.ovrTouch_RPoseMask
ovrTouch_LPoseMask = ovr_capi.ovrTouch_LPoseMask

# enum ovrControllerType
ovrControllerType_None = ovr_capi.ovrControllerType_None
ovrControllerType_LTouch = ovr_capi.ovrControllerType_LTouch
ovrControllerType_RTouch = ovr_capi.ovrControllerType_RTouch
ovrControllerType_Touch = ovr_capi.ovrControllerType_Touch
ovrControllerType_Remote = ovr_capi.ovrControllerType_Remote
ovrControllerType_XBox = ovr_capi.ovrControllerType_XBox
ovrControllerType_Object0 = ovr_capi.ovrControllerType_Object0
ovrControllerType_Object1 = ovr_capi.ovrControllerType_Object1
ovrControllerType_Object2 = ovr_capi.ovrControllerType_Object2
ovrControllerType_Object3 = ovr_capi.ovrControllerType_Object3
ovrControllerType_Active = ovr_capi.ovrControllerType_Active

# enum ovrHapticsBufferSubmitMode
ovrHapticsBufferSubmit_Enqueue = ovr_capi.ovrHapticsBufferSubmit_Enqueue

OVR_HAPTICS_BUFFER_SAMPLES_MAX = ovr_capi.OVR_HAPTICS_BUFFER_SAMPLES_MAX

# enum ovrTrackedDeviceType
ovrTrackedDevice_None = ovr_capi.ovrTrackedDevice_None
ovrTrackedDevice_HMD = ovr_capi.ovrTrackedDevice_HMD
ovrTrackedDevice_LTouch = ovr_capi.ovrTrackedDevice_LTouch
ovrTrackedDevice_RTouch = ovr_capi.ovrTrackedDevice_RTouch
ovrTrackedDevice_Touch = ovr_capi.ovrTrackedDevice_Touch
ovrTrackedDevice_Object0 = ovr_capi.ovrTrackedDevice_Object0
ovrTrackedDevice_Object1 = ovr_capi.ovrTrackedDevice_Object1
ovrTrackedDevice_Object2 = ovr_capi.ovrTrackedDevice_Object2
ovrTrackedDevice_Object3 = ovr_capi.ovrTrackedDevice_Object3

# enum ovrBoundaryType
ovrBoundary_PlayArea = ovr_capi.ovrBoundary_PlayArea

# enum ovrHandType
ovrHand_Left = ovr_capi.ovrHand_Left
ovrHand_Right = ovr_capi.ovrHand_Right
ovrHand_Count = ovr_capi.ovrHand_Count

# enum ovrCameraStatusFlags
ovrCameraStatus_None = ovr_capi.ovrCameraStatus_None
ovrCameraStatus_Connected = ovr_capi.ovrCameraStatus_Connected
ovrCameraStatus_Calibrating = ovr_capi.ovrCameraStatus_Calibrating
ovrCameraStatus_CalibrationFailed = ovr_capi.ovrCameraStatus_CalibrationFailed
ovrCameraStatus_Calibrated = ovr_capi.ovrCameraStatus_Calibrated
ovrCameraStatus_Capturing = ovr_capi.ovrCameraStatus_Capturing

OVR_MAX_EXTERNAL_CAMERA_COUNT = ovr_capi.OVR_MAX_EXTERNAL_CAMERA_COUNT
OVR_EXTERNAL_CAMERA_NAME_SIZE = ovr_capi.OVR_EXTERNAL_CAMERA_NAME_SIZE

# enum ovrInitFlags
ovrInit_Debug = ovr_capi.ovrInit_Debug
ovrInit_RequestVersion = ovr_capi.ovrInit_RequestVersion
ovrInit_Invisible = ovr_capi.ovrInit_Invisible
ovrInit_MixedRendering = ovr_capi.ovrInit_MixedRendering
ovrInit_FocusAware = ovr_capi.ovrInit_FocusAware
ovrinit_WritableBits = ovr_capi.ovrinit_WritableBits

# enum ovrLogLevel
ovrLogLevel_Debug = ovr_capi.ovrLogLevel_Debug
ovrLogLevel_Info = ovr_capi.ovrLogLevel_Info
ovrLogLevel_Error = ovr_capi.ovrLogLevel_Error

# enum ovrLayerType:
ovrLayerType_Disabled = ovr_capi.ovrLayerType_Disabled
ovrLayerType_EyeFov = ovr_capi.ovrLayerType_EyeFov
ovrLayerType_EyeFovDepth = ovr_capi.ovrLayerType_EyeFovDepth
ovrLayerType_Quad = ovr_capi.ovrLayerType_Quad
ovrLayerType_EyeMatrix = ovr_capi.ovrLayerType_EyeMatrix
ovrLayerType_EyeFovMultires = ovr_capi.ovrLayerType_EyeFovMultires
ovrLayerType_Cylinder = ovr_capi.ovrLayerType_Cylinder
ovrLayerType_Cube = ovr_capi.ovrLayerType_Cube

# enum ovrLayerFlags:
ovrLayerFlag_HighQuality = ovr_capi.ovrLayerFlag_HighQuality
ovrLayerFlag_TextureOriginAtBottomLeft = ovr_capi.ovrLayerFlag_TextureOriginAtBottomLeft
ovrLayerFlag_HeadLocked = ovr_capi.ovrLayerFlag_HeadLocked

# enum ovrTextureLayout
ovrTextureLayout_Rectilinear = ovr_capi.ovrTextureLayout_Rectilinear
ovrTextureLayout_Octilinear = ovr_capi.ovrTextureLayout_Rectilinear

# --- C-LEVEL STRUCTURE EXTENSION TYPES ---
#
# C-level structures are wrapped as Cython extension types which allows them to
# be treated like regular Python objects. Data contained in structure fields are
# accessible via properties with the same identifier/name.
#
# Extension types can reference C data in other extension types, allowing access
# to fields which contained in nested structures.
#

cdef class ovrErrorInfo:
    cdef ovr_capi.ovrErrorInfo* c_data
    cdef ovr_capi.ovrErrorInfo  c_ovrErrorInfo

    def __cinit__(self):
        self.c_data = &self.c_ovrErrorInfo

    @property
    def Result(self):
        return self.c_data.Result

    @property
    def ErrorString(self):
        return self.c_data.ErrorString


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


cdef class ovrVector2i:
    cdef ovr_capi.ovrVector2i* c_data
    cdef ovr_capi.ovrVector2i  c_ovrVector2i

    def __cinit__(self, int x=0, int y=0):
        self.c_data = &self.c_ovrVector2i

        self.c_data.x = x
        self.c_data.y = y

    @property
    def x(self):
        return self.c_data.x

    @x.setter
    def x(self, int value):
        self.c_data.x = value

    @property
    def y(self):
        return self.c_data.y

    @y.setter
    def y(self, int value):
        self.c_data.y = value


cdef class ovrSizei:
    cdef ovr_capi.ovrSizei* c_data
    cdef ovr_capi.ovrSizei  c_ovrSizei

    def __cinit__(self, int w=0, int h=0):
        self.c_data = &self.c_ovrSizei

        self.c_data.w = w
        self.c_data.h = h

    @property
    def w(self):
        return self.c_data.w

    @w.setter
    def w(self, int value):
        self.c_data.w = value

    @property
    def h(self):
        return self.c_data.h

    @h.setter
    def h(self, int value):
        self.c_data.h = value


cdef class ovrRecti:
    cdef ovr_capi.ovrRecti* c_data
    cdef ovr_capi.ovrRecti  c_ovrRecti

    # nested field objects
    cdef ovrVector2i obj_pos
    cdef ovrSizei obj_size

    def __cinit__(self, int x=0, int y=0, int w=0, int h=0):
        self.c_data = &self.c_ovrRecti
        self.obj_pos = ovrVector2i()
        self.obj_size = ovrSizei()

        self.obj_pos.c_data = &self.c_data.Pos
        self.obj_size.c_data = &self.c_data.Size

    @property
    def Pos(self):
        return <ovrVector2i>self.obj_pos

    @property
    def Size(self):
        return <ovrSizei>self.obj_size


cdef class ovrQuatf:
    cdef ovr_capi.ovrQuatf* c_data
    cdef ovr_capi.ovrQuatf  c_ovrQuatf

    def __cinit__(self, float x=0.0, float y=0.0, float z=0.0, float w=0.0):
        self.c_data = &self.c_ovrQuatf

        self.c_data.x = x
        self.c_data.y = y
        self.c_data.z = z
        self.c_data.w = w

    @property
    def x(self):
        return self.c_data.x

    @x.setter
    def x(self, float value):
        self.c_data.x = value

    @property
    def y(self):
        return self.c_data.y

    @y.setter
    def y(self, float value):
        self.c_data.y = value

    @property
    def z(self):
        return self.c_data.z

    @z.setter
    def z(self, float value):
        self.c_data.z = value

    @property
    def w(self):
        return self.c_data.w

    @w.setter
    def w(self, float value):
        self.c_data.w = value


cdef class ovrVector2f:
    cdef ovr_capi.ovrVector2f* c_data
    cdef ovr_capi.ovrVector2f  c_ovrVector2f

    def __cinit__(self, float x=0.0, float y=0.0):
        self.c_data = &self.c_ovrVector2f

        self.c_data.x = x
        self.c_data.y = y

    @property
    def x(self):
        return self.c_data.x

    @x.setter
    def x(self, float value):
        self.c_data.x = value

    @property
    def y(self):
        return self.c_data.y

    @y.setter
    def y(self, float value):
        self.c_data.y = value


cdef class ovrVector3f:
    cdef ovr_capi.ovrVector3f* c_data
    cdef ovr_capi.ovrVector3f  c_ovrVector3f

    def __cinit__(self, float x=0.0, float y=0.0, float z=0.0):
        self.c_data = &self.c_ovrVector3f

        self.c_data.x = x
        self.c_data.y = y
        self.c_data.z = z

    @property
    def x(self):
        return self.c_data.x

    @x.setter
    def x(self, float value):
        self.c_data.x = value

    @property
    def y(self):
        return self.c_data.y

    @y.setter
    def y(self, float value):
        self.c_data.y = value

    @property
    def z(self):
        return self.c_data.z

    @z.setter
    def z(self, float value):
        self.c_data.z = value


cdef class ovrMatrix4f:
    cdef ovr_capi.ovrMatrix4f* c_data
    cdef ovr_capi.ovrMatrix4f  c_ovrMatrix4f

    def __cinit__(self):
        self.c_data = &self.c_ovrMatrix4f

    @property
    def M(self):
        return self.c_data.M


cdef class ovrPosef:
    cdef ovr_capi.ovrPosef* c_data
    cdef ovr_capi.ovrPosef  c_ovrPosef

    # nested field objects
    cdef ovrQuatf obj_Orientation
    cdef ovrVector3f obj_Position

    def __cinit__(self):
        self.c_data = &self.c_ovrPosef

        self.obj_Orientation = ovrQuatf()
        self.obj_Position = ovrVector3f()
        self.obj_Orientation.c_data = &self.c_data.Orientation
        self.obj_Position.c_data = &self.c_data.Position

    @property
    def Orientation(self):
        return <ovrQuatf>self.obj_Orientation

    @property
    def Position(self):
        return <ovrVector3f>self.obj_Position


cdef class ovrPoseStatef:
    cdef ovr_capi.ovrPoseStatef* c_data
    cdef ovr_capi.ovrPoseStatef  c_ovrPoseStatef

    # nested field objects
    cdef ovrPosef obj_ThePose
    cdef ovrVector3f obj_AngularVelocity
    cdef ovrVector3f obj_LinearVelocity
    cdef ovrVector3f obj_AngularAcceleration
    cdef ovrVector3f obj_LinearAcceleration

    def __cinit__(self):
        self.c_data = &self.c_ovrPoseStatef

        self.obj_ThePose = ovrPosef()
        self.obj_AngularVelocity = ovrVector3f()
        self.obj_LinearVelocity = ovrVector3f()
        self.obj_AngularAcceleration = ovrVector3f()
        self.obj_LinearAcceleration = ovrVector3f()

        self.obj_ThePose.c_data = &self.c_data.ThePose
        self.obj_AngularVelocity.c_data = &self.c_data.AngularVelocity
        self.obj_LinearVelocity.c_data = &self.c_data.LinearVelocity
        self.obj_AngularAcceleration.c_data = &self.c_data.AngularAcceleration
        self.obj_LinearAcceleration.c_data = &self.c_data.LinearAcceleration

    @property
    def ThePose(self):
        return <ovrPosef>self.obj_ThePose

    @property
    def AngularVelocity(self):
        return <ovrVector3f>self.obj_AngularVelocity

    @property
    def LinearVelocity(self):
        return <ovrVector3f>self.obj_LinearVelocity

    @property
    def AngularAcceleration(self):
        return <ovrVector3f>self.obj_AngularAcceleration

    @property
    def LinearAcceleration(self):
        return <ovrVector3f>self.obj_LinearAcceleration

    @property
    def TimeInSeconds(self):
        return <double>self.c_data.TimeInSeconds


cdef class ovrFovPort:
    cdef ovr_capi.ovrFovPort* c_data
    cdef ovr_capi.ovrFovPort  c_ovrFovPort

    def __cinit__(self,
                  float up=0.0,
                  float down=0.0,
                  float left=0.0,
                  float right=0.0):

        self.c_data = &self.c_ovrFovPort

        self.c_data.UpTan = up
        self.c_data.DownTan = down
        self.c_data.LeftTan = left
        self.c_data.RightTan = right

    @property
    def UpTan(self):
        return self.c_data.UpTan

    @UpTan.setter
    def UpTan(self, float value):
        self.c_data.UpTan = value

    @property
    def DownTan(self):
        return self.c_data.DownTan

    @DownTan.setter
    def DownTan(self, float value):
        self.c_data.DownTan = value

    @property
    def LeftTan(self):
        return self.c_data.LeftTan

    @LeftTan.setter
    def LeftTan(self, float value):
        self.c_data.LeftTan = value

    @property
    def RightTan(self):
        return self.c_data.RightTan

    @RightTan.setter
    def RightTan(self, float value):
        self.c_data.RightTan = value


cdef class ovrGraphicsLuid:
    cdef ovr_capi.ovrGraphicsLuid* c_data
    cdef ovr_capi.ovrGraphicsLuid  c_ovrGraphicsLuid

    def __cinit__(self):
        self.c_data = &self.c_ovrGraphicsLuid

    @property
    def Reserved(self):
        return self.c_data.Reserved


cdef class ovrHmdDesc:
    cdef ovr_capi.ovrHmdDesc* c_data
    cdef ovr_capi.ovrHmdDesc  c_ovrHmdDesc

    # nested fields
    cdef tuple obj_DefaultEyeFov
    cdef tuple obj_MaxEyeFov
    cdef ovrFovPort obj_DefaultEyeFov0
    cdef ovrFovPort obj_DefaultEyeFov1
    cdef ovrFovPort obj_MaxEyeFov0
    cdef ovrFovPort obj_MaxEyeFov1
    cdef ovrSizei obj_Resolution

    def __cinit__(self):
        self.c_data = &self.c_ovrHmdDesc

        self.obj_DefaultEyeFov0 = ovrFovPort()
        self.obj_DefaultEyeFov1 = ovrFovPort()
        self.obj_MaxEyeFov0 = ovrFovPort()
        self.obj_MaxEyeFov1 = ovrFovPort()
        self.obj_Resolution = ovrSizei()

        self.obj_DefaultEyeFov0.c_data = &self.c_data.DefaultEyeFov[0]
        self.obj_DefaultEyeFov1.c_data = &self.c_data.DefaultEyeFov[1]
        self.obj_MaxEyeFov0.c_data = &self.c_data.MaxEyeFov[0]
        self.obj_MaxEyeFov1.c_data = &self.c_data.MaxEyeFov[1]
        self.obj_Resolution.c_data = &self.c_data.Resolution

        # tuples for arrayed objects
        self.obj_DefaultEyeFov = (self.obj_DefaultEyeFov0,
                                  self.obj_DefaultEyeFov1)
        self.obj_MaxEyeFov = (self.obj_MaxEyeFov0, self.obj_MaxEyeFov1)

    @property
    def Type(self):
        return self.c_data.Type

    @property
    def ProductName(self):
        return self.c_data.ProductName

    @property
    def Manufacturer(self):
        return self.c_data.Manufacturer

    @property
    def VendorId(self):
        return self.c_data.VendorId

    @property
    def ProductId(self):
        return self.c_data.ProductId

    @property
    def SerialNumber(self):
        return self.c_data.SerialNumber

    @property
    def FirmwareMajor(self):
        return self.c_data.FirmwareMajor

    @property
    def FirmwareMinor(self):
        return self.c_data.FirmwareMinor

    @property
    def AvailableHmdCaps(self):
        return self.c_data.AvailableHmdCaps

    @property
    def DefaultHmdCaps(self):
        return self.c_data.DefaultHmdCaps

    @property
    def AvailableTrackingCaps(self):
        return self.c_data.AvailableTrackingCaps

    @property
    def DefaultTrackingCaps(self):
        return self.c_data.DefaultTrackingCaps

    @property
    def DefaultEyeFov(self):
        return self.obj_DefaultEyeFov

    @property
    def MaxEyeFov(self):
        return self.obj_MaxEyeFov

    @property
    def Resolution(self):
        return self.obj_Resolution

    @property
    def DisplayRefreshRate(self):
        return <float>self.c_data.DisplayRefreshRate


cdef class ovrSession:
    cdef ovr_capi.ovrSession c_data  # opaque pointer for the session


cdef class ovrProcessId:
    cdef ovr_capi.ovrProcessId* c_data
    cdef ovr_capi.ovrProcessId  c_ovrProcessId

    def __cinit__(self):
        self.c_data = &self.c_ovrProcessId


cdef class ovrTrackerDesc:
    cdef ovr_capi.ovrTrackerDesc* c_data
    cdef ovr_capi.ovrTrackerDesc  c_ovrTrackerDesc

    def __cinit__(self):
        self.c_data = &self.c_ovrTrackerDesc

    @property
    def FrustumHFovInRadians(self):
        return self.c_data.FrustumHFovInRadians

    @property
    def FrustumVFovInRadians(self):
        return self.c_data.FrustumVFovInRadians

    @property
    def FrustumNearZInMeters(self):
        return self.c_data.FrustumNearZInMeters

    @property
    def FrustumFarZInMeters(self):
        return self.c_data.FrustumFarZInMeters


cdef class ovrTrackerPose:
    cdef ovr_capi.ovrTrackerPose* c_data
    cdef ovr_capi.ovrTrackerPose  c_ovrTrackerPose

    cdef ovrPosef obj_Pose
    cdef ovrPosef obj_LeveledPose

    def __cinit__(self):
        self.c_data = &self.c_ovrTrackerPose

        self.obj_Pose = ovrPosef()
        self.obj_LeveledPose = ovrPosef()

        self.obj_Pose.c_data = &self.c_data.Pose
        self.obj_LeveledPose.c_data = &self.c_data.LeveledPose

    @property
    def TrackerFlags(self):
        return self.c_data.TrackerFlags

    @property
    def FrustumNearZInMeters(self):
        return self.obj_Pose

    @property
    def FrustumFarZInMeters(self):
        return self.obj_LeveledPose


cdef class ovrTrackingState:
    cdef ovr_capi.ovrTrackingState* c_data
    cdef ovr_capi.ovrTrackingState  c_ovrTrackingState

    # nested fields
    cdef tuple obj_HandPoses
    cdef ovrPoseStatef obj_HeadPose
    cdef ovrPoseStatef obj_HandPoses0
    cdef ovrPoseStatef obj_HandPoses1
    cdef ovrPosef obj_CalibratedOrigin

    def __cinit__(self):
        self.c_data = &self.c_ovrTrackingState

        self.obj_HeadPose = ovrPoseStatef()
        self.obj_HandPoses0 = ovrPoseStatef()
        self.obj_HandPoses1 = ovrPoseStatef()
        self.obj_CalibratedOrigin = ovrPosef()

        self.obj_HeadPose.c_data = &self.c_data.HeadPose
        self.obj_HandPoses0.c_data = &self.c_data.HandPoses[0]
        self.obj_HandPoses1.c_data = &self.c_data.HandPoses[1]
        self.obj_CalibratedOrigin.c_data = &self.c_data.CalibratedOrigin
        self.obj_HandPoses = (self.obj_HandPoses0, self.obj_HandPoses1)

    @property
    def HeadPose(self):
        return self.obj_HeadPose

    @property
    def HandPoses(self):
        return self.obj_HandPoses

    @property
    def HandStatusFlags(self):
        return self.c_data.HandStatusFlags[0], self.c_data.HandStatusFlags[1]

    @property
    def CalibratedOrigin(self):
        return self.obj_CalibratedOrigin


cdef class ovrEyeRenderDesc:
    cdef ovr_capi.ovrEyeRenderDesc* c_data
    cdef ovr_capi.ovrEyeRenderDesc  c_ovrEyeRenderDesc

    # nested fields
    cdef ovrFovPort obj_Fov
    cdef ovrRecti obj_DistortedViewport
    cdef ovrVector2f obj_PixelsPerTanAngleAtCenter
    cdef ovrPosef obj_HmdToEyePose

    def __cinit__(self):
        self.c_data = &self.c_ovrEyeRenderDesc

        self.obj_Fov = ovrFovPort()
        self.obj_CalibratedOrigin = ovrRecti()
        self.obj_PixelsPerTanAngleAtCenter = ovrVector2f()
        self.obj_HmdToEyePose = ovrPosef()

        self.obj_Fov.c_data = &self.c_data.Fov
        self.obj_DistortedViewport.c_data = &self.c_data.DistortedViewport
        self.obj_PixelsPerTanAngleAtCenter.c_data = \
            &self.c_data.PixelsPerTanAngleAtCenter
        self.obj_HmdToEyePose.c_data = &self.c_data.HmdToEyePose

    @property
    def Eye(self):
        return self.c_data.Eye

    @property
    def Fov(self):
        return self.obj_Fov

    @property
    def DistortedViewport(self):
        return self.obj_DistortedViewport

    @property
    def PixelsPerTanAngleAtCenter(self):
        return self.obj_PixelsPerTanAngleAtCenter

    @property
    def HmdToEyePose(self):
        return self.obj_HmdToEyePose


cdef class ovrTimewarpProjectionDesc:
    cdef ovr_capi.ovrTimewarpProjectionDesc* c_data
    cdef ovr_capi.ovrTimewarpProjectionDesc  c_ovrTimewarpProjectionDesc

    def __cinit__(self):
        self.c_data = &self.c_ovrTimewarpProjectionDesc

    @property
    def Projection22(self):
        return self.c_data.Projection22

    @property
    def Projection23(self):
        return self.c_data.Projection23

    @property
    def Projection32(self):
        return self.c_data.Projection32


cdef class ovrViewScaleDesc:
    cdef ovr_capi.ovrViewScaleDesc* c_data
    cdef ovr_capi.ovrViewScaleDesc  c_ovrViewScaleDesc

    cdef tuple obj_HmdToEyePose
    cdef ovrPosef obj_HmdToEyePose0
    cdef ovrPosef obj_HmdToEyePose1

    def __cinit__(self):
        self.c_data = &self.c_ovrViewScaleDesc

        self.obj_HmdToEyePose0 = ovrPosef()
        self.obj_HmdToEyePose1 = ovrPosef()

        self.obj_HmdToEyePose0.c_data = &self.c_data.HmdToEyePose[0]
        self.obj_HmdToEyePose1.c_data = &self.c_data.HmdToEyePose[1]

        self.obj_HmdToEyePose = (self.obj_HmdToEyePose0, self.obj_HmdToEyePose1)

    @property
    def HmdToEyePose(self):
        return self.obj_HmdToEyePose

    @property
    def HmdSpaceToWorldScaleInMeters(self):
        return self.c_data.HmdSpaceToWorldScaleInMeters


cdef class ovrTextureSwapChainDesc:
    cdef ovr_capi.ovrTextureSwapChainDesc* c_data
    cdef ovr_capi.ovrTextureSwapChainDesc  c_ovrTextureSwapChainDesc

    def __cinit__(self):
        self.c_data = &self.c_ovrTextureSwapChainDesc

    @property
    def Type(self):
        return <int>self.c_data.Type

    @Type.setter
    def Type(self, int value):
        self.c_data.Type = <ovr_capi.ovrTextureType>value

    @property
    def Format(self):
        return <int>self.c_data.Format

    @Format.setter
    def Format(self, int value):
        self.c_data.Format = <ovr_capi.ovrTextureFormat>value

    @property
    def ArraySize(self):
        return <int>self.c_data.ArraySize

    @ArraySize.setter
    def ArraySize(self, int value):
        self.c_data.ArraySize = value

    @property
    def Width(self):
        return <int>self.c_data.Width

    @Width.setter
    def Width(self, int value):
        self.c_data.Width = value

    @property
    def Height(self):
        return <int>self.c_data.Height

    @Height.setter
    def Height(self, int value):
        self.c_data.Height = value

    @property
    def MipLevels(self):
        return <int>self.c_data.MipLevels

    @MipLevels.setter
    def MipLevels(self, int value):
        self.c_data.MipLevels = value

    @property
    def SampleCount(self):
        return <int>self.c_data.SampleCount

    @SampleCount.setter
    def SampleCount(self, int value):
        self.c_data.SampleCount = value

    @property
    def StaticImage(self):
        return <bint>self.c_data.StaticImage

    @StaticImage.setter
    def StaticImage(self, bint value):
        self.c_data.StaticImage = <ovr_capi.ovrBool>value

    @property
    def MiscFlags(self):
        return <int>self.c_data.MiscFlags

    @MiscFlags.setter
    def MiscFlags(self, int value):
        self.c_data.MiscFlags = <unsigned int>value

    @property
    def BindFlags(self):
        return <int>self.c_data.BindFlags

    @BindFlags.setter
    def BindFlags(self, int value):
        self.c_data.BindFlags = <unsigned int>value


cdef class ovrMirrorTextureDesc:
    cdef ovr_capi.ovrMirrorTextureDesc* c_data
    cdef ovr_capi.ovrMirrorTextureDesc  c_ovrMirrorTextureDesc

    def __cinit__(self):
        self.c_data = &self.c_ovrMirrorTextureDesc

    @property
    def Format(self):
        return <int>self.c_data.Format

    @Format.setter
    def Format(self, int value):
        self.c_data.Format = <ovr_capi.ovrTextureFormat>value

    @property
    def Width(self):
        return <int>self.c_data.Width

    @Width.setter
    def Width(self, int value):
        self.c_data.Width = value

    @property
    def Height(self):
        return <int>self.c_data.Height

    @Height.setter
    def Height(self, int value):
        self.c_data.Height = value

    @property
    def MiscFlags(self):
        return <int>self.c_data.MiscFlags

    @MiscFlags.setter
    def MiscFlags(self, int value):
        self.c_data.MiscFlags = <unsigned int>value

    @property
    def MirrorOptions(self):
        return <int>self.c_data.MirrorOptions

    @MirrorOptions.setter
    def MirrorOptions(self, int value):
        self.c_data.MirrorOptions = <unsigned int>value


cdef class ovrTextureSwapChain:
    cdef ovr_capi.ovrTextureSwapChain c_data


cdef class ovrMirrorTexture:
    cdef ovr_capi.ovrMirrorTexture c_data


cdef class ovrTouchHapticsDesc:
    cdef ovr_capi.ovrTouchHapticsDesc* c_data
    cdef ovr_capi.ovrTouchHapticsDesc  c_ovrTouchHapticsDesc

    def __cinit__(self):
        self.c_data = &self.c_ovrTouchHapticsDesc

    @property
    def SampleRateHz(self):
        return <int>self.c_data.SampleRateHz

    @property
    def SampleSizeInBytes(self):
        return <int>self.c_data.SampleSizeInBytes

    @property
    def QueueMinSizeToAvoidStarvation(self):
        return <int>self.c_data.QueueMinSizeToAvoidStarvation

    @property
    def SubmitMinSamples(self):
        return <int>self.c_data.SubmitMinSamples

    @property
    def SubmitMaxSamples(self):
        return <int>self.c_data.SubmitMaxSamples

    @property
    def SubmitOptimalSamples(self):
        return <int>self.c_data.SubmitOptimalSamples


cdef class ovrHapticsBuffer:
    cdef ovr_capi.ovrHapticsBuffer* c_data
    cdef ovr_capi.ovrHapticsBuffer  c_ovrHapticsBuffer

    def __cinit__(self):
        self.c_data = &self.c_ovrHapticsBuffer

    @property
    def Samples(self):
        raise NotImplemented("Setting samples from Python not yet supported.")

    @Samples.setter
    def Samples(self, int value):
        raise NotImplemented("Setting samples from Python not yet supported.")

    @property
    def SamplesCount(self):
        return <int>self.c_data.SamplesCount

    @SamplesCount.setter
    def SamplesCount(self, int value):
        self.c_data.SamplesCount = value

    @property
    def SubmitMode(self):
        return <int>self.c_data.SubmitMode

    @SubmitMode.setter
    def SubmitMode(self, int value):
        self.c_data.SubmitMode = <ovr_capi.ovrHapticsBufferSubmitMode>value


cdef class ovrHapticsPlaybackState:
    cdef ovr_capi.ovrHapticsPlaybackState* c_data
    cdef ovr_capi.ovrHapticsPlaybackState  c_ovrHapticsPlaybackState

    def __cinit__(self):
        self.c_data = &self.c_ovrHapticsPlaybackState

    @property
    def RemainingQueueSpace(self):
        return <int>self.c_data.RemainingQueueSpace

    @property
    def SamplesQueued(self):
        return <int>self.c_data.SamplesQueued


cdef class ovrBoundaryLookAndFeel:
    cdef ovr_capi.ovrBoundaryLookAndFeel* c_data
    cdef ovr_capi.ovrBoundaryLookAndFeel  c_ovrBoundaryLookAndFeel

    # nested color
    cdef ovrColorf obj_Color

    def __cinit__(self):
        self.c_data = &self.c_ovrBoundaryLookAndFeel

        self.obj_Color = ovrColorf()
        self.obj_Color.c_data = &self.c_data.Color

    @property
    def Color(self):
        return self.obj_Color

    @Color.setter
    def Color(self, ovrColorf value):
        self.c_data.Color = value.c_data[0]


cdef class ovrInputState:
    cdef ovr_capi.ovrInputState* c_data
    cdef ovr_capi.ovrInputState  c_ovrInputState

    # nested fields
    cdef tuple obj_Thumbstick
    cdef ovrVector2f obj_Thumbstick0
    cdef ovrVector2f obj_Thumbstick1

    cdef tuple obj_ThumbstickNoDeadzone
    cdef ovrVector2f obj_ThumbstickNoDeadzone0
    cdef ovrVector2f obj_ThumbstickNoDeadzone1

    cdef tuple obj_ThumbstickRaw
    cdef ovrVector2f obj_ThumbstickRaw0
    cdef ovrVector2f obj_ThumbstickRaw1


    def __cinit__(self):
        self.c_data = &self.c_ovrInputState

        self.obj_Thumbstick0 = ovrVector2f()
        self.obj_Thumbstick1 = ovrVector2f()
        self.obj_ThumbstickNoDeadzone0 = ovrVector2f()
        self.obj_ThumbstickNoDeadzone1 = ovrVector2f()
        self.obj_ThumbstickRaw0 = ovrVector2f()
        self.obj_ThumbstickRaw1 = ovrVector2f()

        self.obj_Thumbstick0.c_data = &self.c_data.Thumbstick[0]
        self.obj_Thumbstick1.c_data = &self.c_data.Thumbstick[1]
        self.obj_ThumbstickNoDeadzone0.c_data = \
            &self.c_data.ThumbstickNoDeadzone[0]
        self.obj_ThumbstickNoDeadzone1.c_data = \
            &self.c_data.ThumbstickNoDeadzone[1]
        self.obj_ThumbstickRaw0.c_data = &self.c_data.ThumbstickRaw[0]
        self.obj_ThumbstickRaw1.c_data = &self.c_data.ThumbstickRaw[1]

        # tuples for arrayed objects
        self.obj_Thumbstick = (self.obj_Thumbstick0, self.obj_Thumbstick1)
        self.obj_ThumbstickNoDeadzone = (self.obj_ThumbstickNoDeadzone0,
                                         self.obj_ThumbstickNoDeadzone1)
        self.obj_ThumbstickRaw = (self.obj_ThumbstickRaw0,
                                  self.obj_ThumbstickRaw1)

    @property
    def TimeInSeconds(self):
        return <double>self.c_data.TimeInSeconds

    @property
    def Buttons(self):
        return self.c_data.Buttons

    @property
    def Touches(self):
        return self.c_data.Touches

    @property
    def IndexTrigger(self):
        return self.c_data.IndexTrigger

    @property
    def HandTrigger(self):
        return self.c_data.HandTrigger

    @property
    def Thumbstick(self):
        return self.obj_Thumbstick

    @property
    def ControllerType(self):
        return <int>self.c_data.ControllerType

    @property
    def IndexTriggerNoDeadzone(self):
        return self.c_data.IndexTriggerNoDeadzone

    @property
    def HandTriggerNoDeadzone(self):
        return self.c_data.HandTriggerNoDeadzone

    @property
    def ThumbstickNoDeadzone(self):
        return self.obj_ThumbstickNoDeadzone

    @property
    def IndexTriggerRaw(self):
        return self.c_data.IndexTriggerRaw

    @property
    def HandTriggerRaw(self):
        return self.c_data.HandTriggerRaw

    @property
    def ThumbstickRaw(self):
        return self.obj_ThumbstickRaw


cdef class ovrCameraIntrinsics:
    cdef ovr_capi.ovrCameraIntrinsics* c_data
    cdef ovr_capi.ovrCameraIntrinsics  c_ovrCameraIntrinsics

    cdef ovrFovPort obj_FOVPort
    cdef ovrSizei obj_ImageSensorPixelResolution
    cdef ovrMatrix4f obj_LensDistortionMatrix

    def __cinit__(self):
        self.c_data = &self.c_ovrCameraIntrinsics

        self.obj_FOVPort = ovrFovPort()
        self.obj_FOVPort.c_data = &self.c_data.FOVPort

        self.obj_ImageSensorPixelResolution = ovrSizei()
        self.obj_ImageSensorPixelResolution.c_data = \
            &self.c_data.ImageSensorPixelResolution

        self.obj_LensDistortionMatrix = ovrMatrix4f()
        self.obj_LensDistortionMatrix.c_data = &self.c_data.LensDistortionMatrix

    @property
    def LastChangedTime(self):
        return self.c_data.LastChangedTime

    @property
    def FOVPort(self):
        return self.obj_FOVPort

    @property
    def VirtualNearPlaneDistanceMeters(self):
        return self.c_data.VirtualNearPlaneDistanceMeters

    @property
    def VirtualFarPlaneDistanceMeters(self):
        return self.c_data.VirtualFarPlaneDistanceMeters

    @property
    def ImageSensorPixelResolution(self):
        return self.obj_ImageSensorPixelResolution

    @property
    def LensDistortionMatrix(self):
        return self.obj_LensDistortionMatrix

    @property
    def ExposurePeriodSeconds(self):
        return self.c_data.ExposurePeriodSeconds

    @property
    def ExposureDurationSeconds(self):
        return self.c_data.ExposureDurationSeconds


cdef class ovrCameraExtrinsics:
    cdef ovr_capi.ovrCameraExtrinsics* c_data
    cdef ovr_capi.ovrCameraExtrinsics  c_ovrCameraExtrinsics

    cdef ovrPosef obj_RelativePose

    def __cinit__(self):
        self.c_data = &self.c_ovrCameraExtrinsics

        self.obj_RelativePose = ovrPosef()
        self.obj_RelativePose.c_data = &self.c_data.RelativePose

    @property
    def LastChangedTimeSeconds(self):
        return self.c_data.LastChangedTimeSeconds

    @property
    def CameraStatusFlags(self):
        return self.c_data.CameraStatusFlags

    @property
    def AttachedToDevice(self):
        return <int>self.c_data.AttachedToDevice

    @property
    def RelativePose(self):
        return self.obj_RelativePose

    @property
    def LastExposureTimeSeconds(self):
        return self.c_data.LastExposureTimeSeconds

    @property
    def ExposureLatencySeconds(self):
        return self.c_data.ExposureLatencySeconds

    @property
    def AdditionalLatencySeconds(self):
        return self.c_data.AdditionalLatencySeconds


cdef class ovrExternalCamera:
    cdef ovr_capi.ovrExternalCamera* c_data
    cdef ovr_capi.ovrExternalCamera  c_ovrExternalCamera

    cdef ovrCameraIntrinsics obj_Intrinsics
    cdef ovrCameraExtrinsics obj_Extrinsics

    def __cinit__(self):
        self.c_data = &self.c_ovrExternalCamera

        self.obj_Intrinsics = ovrCameraIntrinsics()
        self.obj_Intrinsics.c_data = &self.c_data.Intrinsics

        self.obj_Extrinsics = ovrCameraExtrinsics()
        self.obj_Extrinsics.c_data = &self.c_data.Extrinsics

    @property
    def Name(self):
        # TODO - make a string
        return self.c_data.Name

    @property
    def Intrinsics(self):
        return self.obj_Intrinsics

    @property
    def Extrinsics(self):
        return self.obj_Extrinsics


cdef class ovrInitParams:
    cdef ovr_capi.ovrInitParams* c_data
    cdef ovr_capi.ovrInitParams  c_ovrInitParams

    def __cinit__(self):
        self.c_data = &self.c_ovrInitParams

    # TODO - callback and user data

    @property
    def Flags(self):
        return <int>self.c_data.Flags

    @Flags.setter
    def Flags(self, int value):
        self.c_data.Flags = <uint32_t>value

    @property
    def RequestedMinorVersion(self):
        return <int>self.c_data.RequestedMinorVersion

    @RequestedMinorVersion.setter
    def RequestedMinorVersion(self, int value):
        self.c_data.RequestedMinorVersion = <uint32_t>value

    @property
    def ConnectionTimeoutMS(self):
        return <int>self.c_data.ConnectionTimeoutMS

    @ConnectionTimeoutMS.setter
    def ConnectionTimeoutMS(self, int value):
        self.c_data.ConnectionTimeoutMS = <uint32_t>value


cdef class ovrSessionStatus:
    cdef ovr_capi.ovrSessionStatus* c_data
    cdef ovr_capi.ovrSessionStatus  c_ovrSessionStatus

    def __cinit__(self):
        self.c_data = &self.c_ovrSessionStatus

    @property
    def IsVisible(self):
        return <int>self.c_data.IsVisible

    @property
    def HmdPresent(self):
        return <int>self.c_data.HmdPresent

    @property
    def HmdMounted(self):
        return <int>self.c_data.HmdMounted

    @property
    def DisplayLost(self):
        return <int>self.c_data.DisplayLost

    @property
    def ShouldQuit(self):
        return <int>self.c_data.ShouldQuit

    @property
    def ShouldRecenter(self):
        return <int>self.c_data.ShouldRecenter

    @property
    def HasInputFocus(self):
        return <int>self.c_data.HasInputFocus

    @property
    def OverlayPresent(self):
        return <int>self.c_data.OverlayPresent

    @property
    def DepthRequested(self):
        return <int>self.c_data.DepthRequested


cdef class ovrLayerHeader:
    cdef ovr_capi.ovrLayerHeader* c_data
    cdef ovr_capi.ovrLayerHeader  c_ovrLayerHeader

    def __cinit__(self):
        self.c_data = &self.c_ovrLayerHeader

    @property
    def Type(self):
        return <int>self.c_data.Type

    @Type.setter
    def Type(self, int value):
        self.c_data.Type = <ovr_capi.ovrLayerType>value

    @property
    def Flags(self):
        return <int>self.c_data.Flags

    @Flags.setter
    def Flags(self, int value):
        self.c_data.Flags = <unsigned int>value


cdef class ovrLayerEyeFov:
    cdef ovr_capi.ovrLayerEyeFov* c_data
    cdef ovr_capi.ovrLayerEyeFov  c_ovrLayerEyeFov

    cdef ovrLayerHeader obj_Header
    cdef tuple obj_ColorTexture
    cdef ovrTextureSwapChain obj_ColorTexture0
    cdef ovrTextureSwapChain obj_ColorTexture1
    cdef tuple obj_Viewport
    cdef ovrRecti obj_Viewport0
    cdef ovrRecti obj_Viewport1
    cdef tuple obj_Fov
    cdef ovrFovPort obj_Fov0
    cdef ovrFovPort obj_Fov1
    cdef tuple obj_RenderPose
    cdef ovrPosef obj_RenderPose0
    cdef ovrPosef obj_RenderPose1

    def __cinit__(self):
        self.c_data = &self.c_ovrLayerEyeFov

        self.obj_Header = ovrLayerHeader()
        self.obj_Header.c_data = &self.c_data.Header

        self.obj_ColorTexture0 = ovrTextureSwapChain()
        self.obj_ColorTexture0.c_data = self.c_data.ColorTexture[0]
        self.obj_ColorTexture1 = ovrTextureSwapChain()
        self.obj_ColorTexture1.c_data = self.c_data.ColorTexture[1]
        self.obj_ColorTexture = (self.obj_ColorTexture0, self.obj_ColorTexture1)

        self.obj_Viewport0 = ovrRecti()
        self.obj_Viewport0.c_data = &self.c_data.Viewport[0]
        self.obj_Viewport1 = ovrRecti()
        self.obj_Viewport1.c_data = &self.c_data.Viewport[1]
        self.obj_Viewport = (self.obj_Viewport0, self.obj_Viewport1)

        self.obj_Fov0 = ovrFovPort()
        self.obj_Fov0.c_data = &self.c_data.Fov[0]
        self.obj_Fov1 = ovrFovPort()
        self.obj_Fov1.c_data = &self.c_data.Fov[1]
        self.obj_Fov = (self.obj_Fov0, self.obj_Fov1)

        self.obj_RenderPose0 = ovrPosef()
        self.obj_RenderPose0.c_data = &self.c_data.RenderPose[0]
        self.obj_RenderPose1 = ovrPosef()
        self.obj_RenderPose1.c_data = &self.c_data.RenderPose[1]
        self.obj_RenderPose = (self.obj_RenderPose0, self.obj_RenderPose1)

    @property
    def Header(self):
        return self.obj_Header

    @Header.setter
    def Header(self, ovrLayerHeader value):
        self.obj_Header.c_data = value.c_data

    @property
    def ColorTexture(self):
        return self.obj_ColorTexture

    @ColorTexture.setter
    def ColorTexture(self, tuple value):
        self.obj_ColorTexture0.c_data = (<ovrTextureSwapChain>value[0]).c_data
        self.obj_ColorTexture1.c_data = (<ovrTextureSwapChain>value[1]).c_data

    @property
    def Viewport(self):
        return self.obj_Viewport

    @Viewport.setter
    def Viewport(self, tuple value):
        self.obj_Viewport0.c_data = (<ovrRecti>value[0]).c_data
        self.obj_Viewport1.c_data = (<ovrRecti>value[1]).c_data

    @property
    def Fov(self):
        return self.obj_Viewport

    @Fov.setter
    def Fov(self, tuple value):
        self.obj_Fov0.c_data = (<ovrFovPort>value[0]).c_data
        self.obj_Fov1.c_data = (<ovrFovPort>value[1]).c_data

    @property
    def RenderPose(self):
        return self.obj_Viewport

    @RenderPose.setter
    def RenderPose(self, tuple value):
        self.obj_RenderPose0.c_data = (<ovrPosef>value[0]).c_data
        self.obj_RenderPose1.c_data = (<ovrPosef>value[1]).c_data

    @property
    def SensorSampleTime(self):
        return <double>self.c_data.SensorSampleTime

    @SensorSampleTime.setter
    def SensorSampleTime(self, double value):
        self.c_data.SensorSampleTime = value


cdef class ovrLayerEyeFovDepth:
    cdef ovr_capi.ovrLayerEyeFovDepth* c_data
    cdef ovr_capi.ovrLayerEyeFovDepth  c_ovrLayerEyeFovDepth

    cdef ovrLayerHeader obj_Header
    cdef tuple obj_ColorTexture
    cdef ovrTextureSwapChain obj_ColorTexture0
    cdef ovrTextureSwapChain obj_ColorTexture1
    cdef tuple obj_Viewport
    cdef ovrRecti obj_Viewport0
    cdef ovrRecti obj_Viewport1
    cdef tuple obj_Fov
    cdef ovrFovPort obj_Fov0
    cdef ovrFovPort obj_Fov1
    cdef tuple obj_RenderPose
    cdef ovrPosef obj_RenderPose0
    cdef ovrPosef obj_RenderPose1
    cdef tuple obj_DepthTexture
    cdef ovrTextureSwapChain obj_DepthTexture0
    cdef ovrTextureSwapChain obj_DepthTexture1
    cdef ovrTimewarpProjectionDesc obj_ProjectionDesc

    def __cinit__(self):
        self.c_data = &self.c_ovrLayerEyeFovDepth

        self.obj_Header = ovrLayerHeader()
        self.obj_Header.c_data = &self.c_data.Header

        self.obj_ColorTexture0 = ovrTextureSwapChain()
        self.obj_ColorTexture0.c_data = self.c_data.ColorTexture[0]
        self.obj_ColorTexture1 = ovrTextureSwapChain()
        self.obj_ColorTexture1.c_data = self.c_data.ColorTexture[1]
        self.obj_ColorTexture = (self.obj_ColorTexture0, self.obj_ColorTexture1)

        self.obj_Viewport0 = ovrRecti()
        self.obj_Viewport0.c_data = &self.c_data.Viewport[0]
        self.obj_Viewport1 = ovrRecti()
        self.obj_Viewport1.c_data = &self.c_data.Viewport[1]
        self.obj_Viewport = (self.obj_Viewport0, self.obj_Viewport1)

        self.obj_Fov0 = ovrFovPort()
        self.obj_Fov0.c_data = &self.c_data.Fov[0]
        self.obj_Fov1 = ovrFovPort()
        self.obj_Fov1.c_data = &self.c_data.Fov[1]
        self.obj_Fov = (self.obj_Fov0, self.obj_Fov1)

        self.obj_RenderPose0 = ovrPosef()
        self.obj_RenderPose0.c_data = &self.c_data.RenderPose[0]
        self.obj_RenderPose1 = ovrPosef()
        self.obj_RenderPose1.c_data = &self.c_data.RenderPose[1]
        self.obj_RenderPose = (self.obj_RenderPose0, self.obj_RenderPose1)

        self.obj_DepthTexture0 = ovrTextureSwapChain()
        self.obj_DepthTexture0.c_data = self.c_data.DepthTexture[0]
        self.obj_DepthTexture1 = ovrTextureSwapChain()
        self.obj_DepthTexture1.c_data = self.c_data.DepthTexture[1]
        self.obj_DepthTexture = (self.obj_DepthTexture0, self.obj_DepthTexture1)

        self.obj_ProjectionDesc = ovrTimewarpProjectionDesc()
        self.obj_ProjectionDesc.c_data = &self.c_data.ProjectionDesc

    @property
    def Header(self):
        return self.obj_Header

    @Header.setter
    def Header(self, ovrLayerHeader value):
        self.obj_Header.c_data = value.c_data

    @property
    def ColorTexture(self):
        return self.obj_ColorTexture

    @ColorTexture.setter
    def ColorTexture(self, object value):
        if not isinstance(value, (tuple, list)):
            raise TypeError(
                "Viewport must be list or tuple of 'ovrTextureSwapChain'.")

        self.obj_ColorTexture0.c_data = (<ovrTextureSwapChain>value[0]).c_data
        self.obj_ColorTexture1.c_data = (<ovrTextureSwapChain>value[1]).c_data

    @property
    def Viewport(self):
        return self.obj_Viewport

    @Viewport.setter
    def Viewport(self, tuple value):
        if not isinstance(value, (tuple, list)):
            raise TypeError(
                "Viewport must be list or tuple of 'ovrRecti'.")

        self.obj_Viewport0.c_data = (<ovrRecti>value[0]).c_data
        self.obj_Viewport1.c_data = (<ovrRecti>value[1]).c_data

    @property
    def Fov(self):
        return self.obj_Viewport

    @Fov.setter
    def Fov(self, tuple value):
        self.obj_Fov0.c_data = (<ovrFovPort>value[0]).c_data
        self.obj_Fov1.c_data = (<ovrFovPort>value[1]).c_data

    @property
    def RenderPose(self):
        return self.obj_Viewport

    @RenderPose.setter
    def RenderPose(self, object value):
        if not isinstance(value, (tuple, list)):
            raise TypeError(
                "RenderPose must be list or tuple of 'ovrPosef'.")

        self.obj_RenderPose0.c_data = (<ovrPosef>value[0]).c_data
        self.obj_RenderPose1.c_data = (<ovrPosef>value[1]).c_data

    @property
    def SensorSampleTime(self):
        return <double>self.c_data.SensorSampleTime

    @SensorSampleTime.setter
    def SensorSampleTime(self, double value):
        self.c_data.SensorSampleTime = value

    @property
    def DepthTexture(self):
        return self.obj_DepthTexture

    @DepthTexture.setter
    def DepthTexture(self, object value):
        if not isinstance(value, (tuple, list)):
            raise TypeError(
                "DepthTexture must be list or tuple of 'ovrTextureSwapChain'.")

        self.obj_DepthTexture0.c_data = (<ovrTextureSwapChain>value[0]).c_data
        self.obj_DepthTexture1.c_data = (<ovrTextureSwapChain>value[1]).c_data

    @property
    def ProjectionDesc(self):
        return self.obj_ProjectionDesc

    @ProjectionDesc.setter
    def ProjectionDesc(self, ovrTimewarpProjectionDesc value):
        self.obj_ProjectionDesc.c_data = value.c_data


cdef class ovrTextureLayoutOctilinear:
    cdef ovr_capi.ovrTextureLayoutOctilinear* c_data
    cdef ovr_capi.ovrTextureLayoutOctilinear  c_ovrTextureLayoutOctilinear

    def __cinit__(self):
        self.c_data = &self.c_ovrTextureLayoutOctilinear

    @property
    def WarpLeft(self):
        return <float>self.c_data.WarpLeft

    @WarpLeft.setter
    def WarpLeft(self, float value):
        self.c_data.WarpLeft = value

    @property
    def WarpRight(self):
        return <float>self.c_data.WarpRight

    @WarpRight.setter
    def WarpRight(self, float value):
        self.c_data.WarpRight = value

    @property
    def WarpUp(self):
        return <float>self.c_data.WarpUp

    @WarpUp.setter
    def WarpUp(self, float value):
        self.c_data.WarpUp = value

    @property
    def WarpDown(self):
        return <float>self.c_data.WarpDown

    @WarpDown.setter
    def WarpDown(self, float value):
        self.c_data.WarpDown = value

    @property
    def SizeLeft(self):
        return <float>self.c_data.SizeLeft

    @SizeLeft.setter
    def SizeLeft(self, float value):
        self.c_data.SizeLeft = value

    @property
    def SizeRight(self):
        return <float>self.c_data.SizeRight

    @SizeRight.setter
    def SizeRight(self, float value):
        self.c_data.SizeRight = value

    @property
    def SizeUp(self):
        return <float>self.c_data.SizeUp

    @SizeUp.setter
    def SizeUp(self, float value):
        self.c_data.SizeUp = value

    @property
    def SizeDown(self):
        return <float>self.c_data.SizeDown

    @SizeDown.setter
    def SizeDown(self, float value):
        self.c_data.SizeDown = value


cdef class ovrLayerEyeFovMultires:
    cdef ovr_capi.ovrLayerEyeFovMultires* c_data
    cdef ovr_capi.ovrLayerEyeFovMultires  c_ovrLayerEyeFovMultires

    cdef ovrLayerHeader obj_Header
    cdef tuple obj_ColorTexture
    cdef ovrTextureSwapChain obj_ColorTexture0
    cdef ovrTextureSwapChain obj_ColorTexture1
    cdef tuple obj_Viewport
    cdef ovrRecti obj_Viewport0
    cdef ovrRecti obj_Viewport1
    cdef tuple obj_Fov
    cdef ovrFovPort obj_Fov0
    cdef ovrFovPort obj_Fov1
    cdef tuple obj_RenderPose
    cdef ovrPosef obj_RenderPose0
    cdef ovrPosef obj_RenderPose1

    def __cinit__(self):
        self.c_data = &self.c_ovrLayerEyeFovMultires

        self.obj_Header = ovrLayerHeader()
        self.obj_Header.c_data = &self.c_data.Header

        self.obj_ColorTexture0 = ovrTextureSwapChain()
        self.obj_ColorTexture0.c_data = self.c_data.ColorTexture[0]
        self.obj_ColorTexture1 = ovrTextureSwapChain()
        self.obj_ColorTexture1.c_data = self.c_data.ColorTexture[1]
        self.obj_ColorTexture = (self.obj_ColorTexture0, self.obj_ColorTexture1)

        self.obj_Viewport0 = ovrRecti()
        self.obj_Viewport0.c_data = &self.c_data.Viewport[0]
        self.obj_Viewport1 = ovrRecti()
        self.obj_Viewport1.c_data = &self.c_data.Viewport[1]
        self.obj_Viewport = (self.obj_Viewport0, self.obj_Viewport1)

        self.obj_Fov0 = ovrFovPort()
        self.obj_Fov0.c_data = &self.c_data.Fov[0]
        self.obj_Fov1 = ovrFovPort()
        self.obj_Fov1.c_data = &self.c_data.Fov[1]
        self.obj_Fov = (self.obj_Fov0, self.obj_Fov1)

        self.obj_RenderPose0 = ovrPosef()
        self.obj_RenderPose0.c_data = &self.c_data.RenderPose[0]
        self.obj_RenderPose1 = ovrPosef()
        self.obj_RenderPose1.c_data = &self.c_data.RenderPose[1]
        self.obj_RenderPose = (self.obj_RenderPose0, self.obj_RenderPose1)

    @property
    def Header(self):
        return self.obj_Header

    @Header.setter
    def Header(self, ovrLayerHeader value):
        self.obj_Header.c_data = value.c_data

    @property
    def ColorTexture(self):
        return self.obj_ColorTexture

    @ColorTexture.setter
    def ColorTexture(self, tuple value):
        self.obj_ColorTexture0.c_data = (<ovrTextureSwapChain>value[0]).c_data
        self.obj_ColorTexture1.c_data = (<ovrTextureSwapChain>value[1]).c_data

    @property
    def Viewport(self):
        return self.obj_Viewport

    @Viewport.setter
    def Viewport(self, tuple value):
        self.obj_Viewport0.c_data = (<ovrRecti>value[0]).c_data
        self.obj_Viewport1.c_data = (<ovrRecti>value[1]).c_data

    @property
    def Fov(self):
        return self.obj_Viewport

    @Fov.setter
    def Fov(self, tuple value):
        self.obj_Fov0.c_data = (<ovrFovPort>value[0]).c_data
        self.obj_Fov1.c_data = (<ovrFovPort>value[1]).c_data

    @property
    def RenderPose(self):
        return self.obj_RenderPose

    @RenderPose.setter
    def RenderPose(self, tuple value):
        self.obj_RenderPose0.c_data = (<ovrPosef>value[0]).c_data
        self.obj_RenderPose1.c_data = (<ovrPosef>value[1]).c_data

    @property
    def SensorSampleTime(self):
        return <double>self.c_data.SensorSampleTime

    @SensorSampleTime.setter
    def SensorSampleTime(self, double value):
        self.c_data.SensorSampleTime = value

    @property
    def TextureLayout(self):
        return <int>self.c_data.TextureLayout

    @TextureLayout.setter
    def TextureLayout(self, int value):
        self.c_data.TextureLayout = <ovr_capi.ovrTextureLayout>value


cdef class ovrLayerEyeMatrix:
    cdef ovr_capi.ovrLayerEyeMatrix* c_data
    cdef ovr_capi.ovrLayerEyeMatrix  c_ovrLayerEyeMatrix

    cdef ovrLayerHeader obj_Header
    cdef tuple obj_ColorTexture
    cdef ovrTextureSwapChain obj_ColorTexture0
    cdef ovrTextureSwapChain obj_ColorTexture1
    cdef tuple obj_Viewport
    cdef ovrRecti obj_Viewport0
    cdef ovrRecti obj_Viewport1
    cdef tuple obj_RenderPose
    cdef ovrPosef obj_RenderPose0
    cdef ovrPosef obj_RenderPose1
    cdef tuple obj_Matrix
    cdef ovrMatrix4f obj_Matrix0
    cdef ovrMatrix4f obj_Matrix1

    def __cinit__(self):
        self.c_data = &self.c_ovrLayerEyeMatrix

        self.obj_Header = ovrLayerHeader()
        self.obj_Header.c_data = &self.c_data.Header

        self.obj_ColorTexture0 = ovrTextureSwapChain()
        self.obj_ColorTexture0.c_data = self.c_data.ColorTexture[0]
        self.obj_ColorTexture1 = ovrTextureSwapChain()
        self.obj_ColorTexture1.c_data = self.c_data.ColorTexture[1]
        self.obj_ColorTexture = (self.obj_ColorTexture0, self.obj_ColorTexture1)

        self.obj_Viewport0 = ovrRecti()
        self.obj_Viewport0.c_data = &self.c_data.Viewport[0]
        self.obj_Viewport1 = ovrRecti()
        self.obj_Viewport1.c_data = &self.c_data.Viewport[1]
        self.obj_Viewport = (self.obj_Viewport0, self.obj_Viewport1)

        self.obj_RenderPose0 = ovrPosef()
        self.obj_RenderPose0.c_data = &self.c_data.RenderPose[0]
        self.obj_RenderPose1 = ovrPosef()
        self.obj_RenderPose1.c_data = &self.c_data.RenderPose[1]
        self.obj_RenderPose = (self.obj_RenderPose0, self.obj_RenderPose1)

        self.obj_Matrix0 = ovrMatrix4f()
        self.obj_Matrix0.c_data = &self.c_data.Matrix[0]
        self.obj_Matrix1 = ovrMatrix4f()
        self.obj_Matrix1.c_data = &self.c_data.Matrix[1]
        self.obj_Matrix = (self.obj_Matrix0, self.obj_Matrix1)

    @property
    def Header(self):
        return self.obj_Header

    @Header.setter
    def Header(self, ovrLayerHeader value):
        self.obj_Header.c_data = value.c_data

    @property
    def ColorTexture(self):
        return self.obj_ColorTexture

    @ColorTexture.setter
    def ColorTexture(self, tuple value):
        self.obj_ColorTexture0.c_data = (<ovrTextureSwapChain>value[0]).c_data
        self.obj_ColorTexture1.c_data = (<ovrTextureSwapChain>value[1]).c_data

    @property
    def Viewport(self):
        return self.obj_Viewport

    @Viewport.setter
    def Viewport(self, tuple value):
        self.obj_Viewport0.c_data = (<ovrRecti>value[0]).c_data
        self.obj_Viewport1.c_data = (<ovrRecti>value[1]).c_data

    @property
    def RenderPose(self):
        return self.obj_RenderPose

    @RenderPose.setter
    def RenderPose(self, tuple value):
        self.obj_RenderPose0.c_data = (<ovrPosef>value[0]).c_data
        self.obj_RenderPose1.c_data = (<ovrPosef>value[1]).c_data

    @property
    def Matrix(self):
        return self.obj_Matrix

    @Matrix.setter
    def Matrix(self, tuple value):
        self.obj_Matrix0.c_data = (<ovrMatrix4f>value[0]).c_data
        self.obj_Matrix1.c_data = (<ovrMatrix4f>value[1]).c_data

    @property
    def SensorSampleTime(self):
        return <double>self.c_data.SensorSampleTime

    @SensorSampleTime.setter
    def SensorSampleTime(self, double value):
        self.c_data.SensorSampleTime = value

    # TODO - ovrTextureLayoutDesc_Union


cdef class ovrLayerQuad:
    cdef ovr_capi.ovrLayerQuad* c_data
    cdef ovr_capi.ovrLayerQuad  c_ovrLayerQuad

    cdef ovrLayerHeader obj_Header
    cdef ovrTextureSwapChain obj_ColorTexture
    cdef ovrRecti obj_Viewport
    cdef ovrPosef obj_QuadPoseCenter
    cdef ovrVector2f obj_QuadSize

    def __cinit__(self):
        self.c_data = &self.c_ovrLayerQuad

        self.obj_Header = ovrLayerHeader()
        self.obj_Header.c_data = &self.c_data.Header

        self.obj_ColorTexture = ovrTextureSwapChain()
        self.obj_ColorTexture.c_data = self.c_data.ColorTexture

        self.obj_Viewport = ovrRecti()
        self.obj_Viewport.c_data = &self.c_data.Viewport

        self.obj_QuadPoseCenter = ovrPosef()
        self.obj_QuadPoseCenter.c_data = &self.c_data.QuadPoseCenter

        self.obj_QuadSize = ovrVector2f()
        self.obj_QuadSize.c_data = &self.c_data.QuadSize

    @property
    def Header(self):
        return self.obj_Header

    @Header.setter
    def Header(self, ovrLayerHeader value):
        self.obj_Header.c_data = value.c_data

    @property
    def ColorTexture(self):
        return self.obj_ColorTexture

    @ColorTexture.setter
    def ColorTexture(self, ovrTextureSwapChain value):
        self.obj_ColorTexture.c_data = value.c_data

    @property
    def Viewport(self):
        return self.obj_Viewport

    @Viewport.setter
    def Viewport(self, ovrRecti value):
        self.obj_Viewport.c_data = value.c_data

    @property
    def QuadPoseCenter(self):
        return self.obj_QuadPoseCenter

    @QuadPoseCenter.setter
    def QuadPoseCenter(self, ovrPosef value):
        self.obj_QuadPoseCenter.c_data = value.c_data

    @property
    def QuadSize(self):
        return self.obj_QuadSize

    @QuadSize.setter
    def QuadSize(self, ovrVector2f value):
        self.obj_QuadSize.c_data = value.c_data


cdef class ovrLayerCylinder:
    cdef ovr_capi.ovrLayerCylinder* c_data
    cdef ovr_capi.ovrLayerCylinder  c_ovrLayerCylinder

    cdef ovrLayerHeader obj_Header
    cdef ovrTextureSwapChain obj_ColorTexture
    cdef ovrRecti obj_Viewport
    cdef ovrPosef obj_CylinderPoseCenter

    def __cinit__(self):
        self.c_data = &self.c_ovrLayerCylinder

        self.obj_Header = ovrLayerHeader()
        self.obj_Header.c_data = &self.c_data.Header

        self.obj_ColorTexture = ovrTextureSwapChain()
        self.obj_ColorTexture.c_data = self.c_data.ColorTexture

        self.obj_Viewport = ovrRecti()
        self.obj_Viewport.c_data = &self.c_data.Viewport

        self.obj_CylinderPoseCenter = ovrPosef()
        self.obj_CylinderPoseCenter.c_data = &self.c_data.CylinderPoseCenter

    @property
    def Header(self):
        return self.obj_Header

    @Header.setter
    def Header(self, ovrLayerHeader value):
        self.obj_Header.c_data = value.c_data

    @property
    def ColorTexture(self):
        return self.obj_ColorTexture

    @ColorTexture.setter
    def ColorTexture(self, ovrTextureSwapChain value):
        self.obj_ColorTexture.c_data = value.c_data

    @property
    def Viewport(self):
        return self.obj_Viewport

    @Viewport.setter
    def Viewport(self, ovrRecti value):
        self.obj_Viewport.c_data = value.c_data

    @property
    def CylinderPoseCenter(self):
        return self.obj_CylinderPoseCenter

    @CylinderPoseCenter.setter
    def CylinderPoseCenter(self, ovrPosef value):
        self.obj_CylinderPoseCenter.c_data = value.c_data


cdef class ovrLayerCube:
    cdef ovr_capi.ovrLayerCube* c_data
    cdef ovr_capi.ovrLayerCube  c_ovrLayerCube

    cdef ovrLayerHeader obj_Header
    cdef ovrQuatf obj_Orientation
    cdef ovrTextureSwapChain obj_CubeMapTexture

    cdef ovrPosef obj_CylinderPoseCenter

    def __cinit__(self):
        self.c_data = &self.c_ovrLayerCube

        self.obj_Header = ovrLayerHeader()
        self.obj_Header.c_data = &self.c_data.Header

        self.obj_Orientation = ovrQuatf()
        self.obj_Orientation.c_data = &self.c_data.Orientation

        self.obj_CubeMapTexture = ovrTextureSwapChain()
        self.obj_CubeMapTexture.c_data = self.c_data.CubeMapTexture

    @property
    def Header(self):
        return self.obj_Header

    @Header.setter
    def Header(self, ovrLayerHeader value):
        self.obj_Header.c_data = value.c_data

    @property
    def Orientation(self):
        return self.obj_Orientation

    @Orientation.setter
    def Orientation(self, ovrQuatf value):
        self.obj_Orientation.c_data = value.c_data

    @property
    def CubeMapTexture(self):
        return self.obj_CubeMapTexture

    @CubeMapTexture.setter
    def CubeMapTexture(self, ovrTextureSwapChain value):
        self.obj_CubeMapTexture.c_data = value.c_data


# --- API EXPORTED FUNCTIONS ---
#
cpdef int ovr_Initialize(ovrInitParams params):
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_Initialize(params.c_data)

    return result

cpdef void ovr_Shutdown():
    ovr_capi.ovr_Shutdown()

cpdef void ovr_GetLastErrorInfo(ovrErrorInfo errorInfo):
    ovr_capi.ovr_GetLastErrorInfo(errorInfo.c_data)

cpdef str ovr_GetVersionString():
    cdef const char* version_string = ovr_capi.ovr_GetVersionString()

    return version_string.decode("utf-8")

cpdef tuple ovr_TraceMessage(int level):
    cdef const char* message_string = b""
    cdef int result = ovr_capi.ovr_TraceMessage(level, message_string)

    return result, message_string.decode("utf-8")

cpdef tuple ovr_IdentifyClient():
    cdef const char* identity_string = b""
    cdef int result = ovr_capi.ovr_IdentifyClient(identity_string)

    return result, identity_string.decode("utf-8")

cpdef ovrHmdDesc ovr_GetHmdDesc(ovrSession session):
    cdef ovrHmdDesc to_return = ovrHmdDesc()
    (<ovrHmdDesc>to_return).c_data[0] = ovr_capi.ovr_GetHmdDesc(session.c_data)

    return to_return

cpdef int ovr_GetTrackerCount(ovrSession session):
    cdef unsigned int result = ovr_capi.ovr_GetTrackerCount(session.c_data)

    return <int>result

cpdef ovrTrackerDesc ovr_GetTrackerDesc(ovrSession session,
                                        int trackerDescIndex):
    cdef ovrTrackerDesc to_return = ovrTrackerDesc()
    (<ovrTrackerDesc>to_return).c_data[0] = ovr_capi.ovr_GetTrackerDesc(
        session.c_data, <unsigned int>trackerDescIndex)

    return to_return

cpdef int ovr_Create(ovrSession pPession,
                     ovrGraphicsLuid pLuid):
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_Create(&pPession.c_data,
                                                         pLuid.c_data)

    return <int>result

cpdef void ovr_Destroy(ovrSession session):
    ovr_capi.ovr_Destroy(session.c_data)

cpdef int ovr_GetSessionStatus(ovrSession session,
                               ovrSessionStatus sessionStatus):
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetSessionStatus(
        session.c_data,
        sessionStatus.c_data)

    return <int>result

cpdef int ovr_IsExtensionSupported(ovrSession session,
                                   int extension,
                                   bint outExtensionSupported):
    cdef ovr_capi.ovrBool ext_supported = 0
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_IsExtensionSupported(
        session.c_data,
        <ovr_capi.ovrExtensions>extension,
        &ext_supported)

    return <int>ext_supported

cpdef int ovr_EnableExtension(ovrSession session, int extension):
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_EnableExtension(
        session.c_data,
        <ovr_capi.ovrExtensions>extension)

    return <int>result

cpdef int ovr_SetTrackingOriginType(ovrSession session, int origin):
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_SetTrackingOriginType(
        session.c_data,
        <ovr_capi.ovrTrackingOrigin>origin)

    return <int>result

cpdef int ovr_GetTrackingOriginType(ovrSession session):
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetTrackingOriginType(
        session.c_data)

    return <int>result

cpdef int ovr_RecenterTrackingOrigin(ovrSession session):
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_RecenterTrackingOrigin(
        session.c_data)

    return <int>result

cpdef int ovr_SpecifyTrackingOrigin(ovrSession session, ovrPosef originPose):
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_SpecifyTrackingOrigin(
        session.c_data,
        originPose.c_data[0])

    return <int>result

cpdef void ovr_ClearShouldRecenterFlag(ovrSession session):
    ovr_capi.ovr_ClearShouldRecenterFlag(session.c_data)

cpdef ovrTrackingState ovr_GetTrackingState(ovrSession session,
                                            double absTime,
                                            bint latencyMarker):
    cdef ovrTrackingState to_return = ovrTrackingState()
    (<ovrTrackingState>to_return).c_data[0] = ovr_capi.ovr_GetTrackingState(
        session.c_data,
        absTime,
        <ovr_capi.ovrBool>latencyMarker)

    return to_return

cpdef int ovr_GetDevicePoses(ovrSession session,
                             list deviceTypes,
                             int deviceCount,
                             double absTime,
                             list outDevicePoses):

    # create deviceTypes array
    cdef ovr_capi.ovrTrackedDeviceType* c_in_devices = \
        <ovr_capi.ovrTrackedDeviceType*>malloc(
            deviceCount * sizeof(ovr_capi.ovrTrackedDeviceType))
    if not c_in_devices:
        raise MemoryError()

    # create outDevicePoses array
    cdef ovr_capi.ovrPoseStatef* c_out_poses = <ovr_capi.ovrPoseStatef*>malloc(
        deviceCount * sizeof(ovr_capi.ovrPoseStatef))
    if not c_out_poses:
        raise MemoryError()

    # convert the Python list to C, this is a list of integers
    cdef size_t i
    for i in range(<size_t>deviceCount):
        c_in_devices[i] = <ovr_capi.ovrTrackedDeviceType>(deviceTypes[i])

    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetDevicePoses(
        session.c_data,
        c_in_devices,
        deviceCount,
        absTime,
        c_out_poses)

    # populate output list
    for i in range(<size_t>deviceCount):
        this_device = ovrPoseStatef()
        (<ovrPoseStatef>this_device).c_data[0] = c_out_poses[i]
        outDevicePoses.append(this_device)

    # free C arrays
    free(c_in_devices)
    free(c_out_poses)

    return <int>result

cpdef ovrTrackerPose ovr_GetTrackerPose(ovrSession session, int trackerPoseIndex):
    cdef ovrTrackerPose tracker_pose = ovrTrackerPose()
    (<ovrTrackerPose>tracker_pose).c_data[0] = ovr_capi.ovr_GetTrackerPose(
        session.c_data,
        <unsigned int>trackerPoseIndex)

    return tracker_pose

cpdef int ovr_GetInputState(ovrSession session,
                            int controllerType,
                            ovrInputState inputState):

    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetInputState(
        session.c_data,
        <ovr_capi.ovrControllerType>controllerType,
        inputState.c_data)

    return <int>result

cpdef int ovr_GetConnectedControllerTypes(ovrSession session):
    cdef unsigned int conn_contr = \
        ovr_capi.ovr_GetConnectedControllerTypes(session.c_data)

    return <int>conn_contr

cpdef ovrTouchHapticsDesc ovr_GetTouchHapticsDesc(ovrSession session,
                                                  int controllerType):
    cdef ovrTouchHapticsDesc haptics_desc = ovrTouchHapticsDesc()
    (<ovrTrackerPose>haptics_desc).c_data[0] = ovr_capi.ovr_GetTrackerPose(
        session.c_data,
        <ovr_capi.ovrControllerType>controllerType)

    return haptics_desc

cpdef double ovr_GetPredictedDisplayTime(ovrSession session, int frameIndex):
    cdef double pred_time = ovr_capi.ovr_GetPredictedDisplayTime(
        session.c_data, <long long>frameIndex)

    return pred_time

cpdef int ovr_GetTextureSwapChainLength(ovrSession session,
                                        ovrTextureSwapChain chain,
                                        int out_Length):

    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetTextureSwapChainLength(
        session.c_data,
        chain.c_data,
        &out_Length)

    return <int>result