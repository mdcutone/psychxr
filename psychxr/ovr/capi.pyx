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


# --- C-LEVEL STRUCTURE EXTENSION TYPES ---
#
# C-level structures are wrapped as Cython extension types which allows them to
# be treated like regular Python objects. Data contained in structure fields are
# accessible via properties with the same identifier/name.
#
# Extension types can reference C data in other extension types, allowing access
# to fields which contained in nested structures.
#

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

        self.obj_orientation = ovrQuatf()
        self.obj_position = ovrVector3f()
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


