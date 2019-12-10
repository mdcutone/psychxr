# distutils: language=c++
#  =============================================================================
#  OVR C API Cython Declaration File
#  =============================================================================
#
#  libovr_capi.pxd
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
"""This file exposes Oculus Rift C API types and functions, allowing Cython
extensions to access them. The declarations in the file are contemporaneous
with version 1.40 (retrieved 02.01.2019) of the Oculus Rift PC SDK.

The Oculus PC SDK is Copyright (c) Facebook Technologies, LLC and its
affiliates. All rights reserved.

"""
from libc.stdint cimport uintptr_t, uint32_t, int32_t, uint16_t

cdef extern from "OVR_Version.h":
    cdef int OVR_PRODUCT_VERSION
    cdef int OVR_MAJOR_VERSION
    cdef int OVR_MINOR_VERSION
    cdef int OVR_PATCH_VERSION
    cdef int OVR_BUILD_NUMBER
    cdef int OVR_DLL_COMPATIBLE_VERSION
    cdef int OVR_MIN_REQUESTABLE_MINOR_VERSION
    cdef int OVR_FEATURE_VERSION

cdef extern from "OVR_CAPI_Keys.h":
    cdef const char* OVR_KEY_USER
    cdef const char* OVR_KEY_NAME
    cdef const char* OVR_KEY_GENDER
    cdef const char* OVR_DEFAULT_GENDER
    cdef const char* OVR_KEY_PLAYER_HEIGHT
    cdef float OVR_DEFAULT_PLAYER_HEIGHT
    cdef const char* OVR_KEY_EYE_HEIGHT
    cdef float OVR_DEFAULT_EYE_HEIGHT
    cdef const char* OVR_KEY_NECK_TO_EYE_DISTANCE
    cdef float OVR_DEFAULT_NECK_TO_EYE_HORIZONTAL
    cdef float OVR_DEFAULT_NECK_TO_EYE_VERTICAL
    cdef const char* OVR_KEY_EYE_TO_NOSE_DISTANCE
    cdef const char* OVR_PERF_HUD_MODE
    cdef const char* OVR_LAYER_HUD_MODE
    cdef const char* OVR_LAYER_HUD_CURRENT_LAYER
    cdef const char* OVR_LAYER_HUD_SHOW_ALL_LAYERS
    cdef const char* OVR_DEBUG_HUD_STEREO_MODE
    cdef const char* OVR_DEBUG_HUD_STEREO_GUIDE_INFO_ENABLE
    cdef const char* OVR_DEBUG_HUD_STEREO_GUIDE_SIZE
    cdef const char* OVR_DEBUG_HUD_STEREO_GUIDE_POSITION
    cdef const char* OVR_DEBUG_HUD_STEREO_GUIDE_YAWPITCHROLL
    cdef const char* OVR_DEBUG_HUD_STEREO_GUIDE_COLOR

cdef extern from "OVR_ErrorCode.h":
    ctypedef int32_t ovrResult

    ctypedef enum ovrSuccessType:
        ovrSuccess = 0

    ctypedef enum ovrSuccessTypes:
        ovrSuccess_NotVisible = 1000,
        ovrSuccess_BoundaryInvalid = 1001,
        ovrSuccess_DeviceUnavailable = 1002

    ctypedef enum ovrErrorType:
        ovrError_MemoryAllocationFailure = -1000,
        ovrError_InvalidSession = -1002,
        ovrError_Timeout = -1003,
        ovrError_NotInitialized = -1004,
        ovrError_InvalidParameter = -1005,
        ovrError_ServiceError = -1006,
        ovrError_NoHmd = -1007,
        ovrError_Unsupported = -1009,
        ovrError_DeviceUnavailable = -1010,
        ovrError_InvalidHeadsetOrientation = -1011,
        ovrError_ClientSkippedDestroy = -1012,
        ovrError_ClientSkippedShutdown = -1013,
        ovrError_ServiceDeadlockDetected = -1014,
        ovrError_InvalidOperation = -1015,
        ovrError_InsufficientArraySize = -1016,
        ovrError_NoExternalCameraInfo = -1017,
        ovrError_LostTracking = -1018,
        ovrError_ExternalCameraInitializedFailed = -1019,
        ovrError_ExternalCameraCaptureFailed = -1020,
        ovrError_ExternalCameraNameListsBufferSize = -1021,
        ovrError_ExternalCameraNameListsMistmatch = -1022,
        ovrError_ExternalCameraNotCalibrated = -1023,
        ovrError_ExternalCameraNameWrongSize = -1024,
        ovrError_AudioDeviceNotFound = -2001,
        ovrError_AudioComError = -2002,
        ovrError_Initialize = -3000,
        ovrError_LibLoad = -3001,
        ovrError_LibVersion = -3002,
        ovrError_ServiceConnection = -3003,
        ovrError_ServiceVersion = -3004,
        ovrError_IncompatibleOS = -3005,
        ovrError_DisplayInit = -3006,
        ovrError_ServerStart = -3007,
        ovrError_Reinitialization = -3008,
        ovrError_MismatchedAdapters = -3009,
        ovrError_LeakingResources = -3010,
        ovrError_ClientVersion = -3011,
        ovrError_OutOfDateOS = -3012,
        ovrError_OutOfDateGfxDriver = -3013,
        ovrError_IncompatibleGPU = -3014,
        ovrError_NoValidVRDisplaySystem = -3015,
        ovrError_Obsolete = -3016,
        ovrError_DisabledOrDefaultAdapter = -3017,
        ovrError_HybridGraphicsNotSupported = -3018,
        ovrError_DisplayManagerInit = -3019,
        ovrError_TrackerDriverInit = -3020,
        ovrError_LibSignCheck = -3021,
        ovrError_LibPath = -3022,
        ovrError_LibSymbols = -3023,
        ovrError_RemoteSession = -3024,
        ovrError_InitializeVulkan = -3025,
        ovrError_BlacklistedGfxDriver = -3026,
        ovrError_DisplayLost = -6000,
        ovrError_TextureSwapChainFull = -6001,
        ovrError_TextureSwapChainInvalid = -6002,
        ovrError_GraphicsDeviceReset = -6003,
        ovrError_DisplayRemoved = -6004,
        ovrError_ContentProtectionNotAvailable = -6005,
        ovrError_ApplicationInvisible = -6006,
        ovrError_Disallowed = -6007,
        ovrError_DisplayPluggedIncorrectly = -6008,
        ovrError_DisplayLimitReached = -6009,
        ovrError_RuntimeException = -7000,
        ovrError_NoCalibration = -9000,
        ovrError_OldVersion = -9001,
        ovrError_MisformattedBlock = -9002

    ctypedef struct ovrErrorInfo:
        ovrResult Result
        char[512] ErrorString


cdef inline int OVR_SUCCESS(ovrResult result):
    return result >= ovrSuccessType.ovrSuccess

cdef inline int OVR_UNQUALIFIED_SUCCESS(ovrResult result):
    return result == ovrSuccessType.ovrSuccess

cdef inline int OVR_FAILURE(ovrResult result):
    return not OVR_SUCCESS(result)


cdef extern from "OVR_CAPI.h":
    ctypedef char ovrBool
    cdef ovrBool ovrFalse = 0
    cdef ovrBool ovrTrue = 1

    ctypedef struct ovrColorf:
        float r
        float g
        float b
        float a

    ctypedef struct ovrVector2i:
        int x
        int y

    ctypedef struct ovrSizei:
        int w 
        int h

    ctypedef struct ovrRecti:
        ovrVector2i Pos
        ovrSizei Size

    ctypedef struct ovrQuatf:
        float x
        float y
        float z 
        float w

    ctypedef struct ovrVector2f:
        float x
        float y

    ctypedef struct ovrVector3f:
        float x 
        float y 
        float z

    ctypedef struct ovrMatrix4f:
        float[4][4] M

    ctypedef struct ovrPosef:
        ovrQuatf Orientation
        ovrVector3f Position

    ctypedef struct ovrPoseStatef:
        ovrPosef ThePose
        ovrVector3f AngularVelocity 
        ovrVector3f LinearVelocity 
        ovrVector3f AngularAcceleration
        ovrVector3f LinearAcceleration 
        double TimeInSeconds 

    ctypedef struct ovrFovPort:
        float UpTan 
        float DownTan 
        float LeftTan 
        float RightTan 

    ctypedef enum ovrHmdType:
        ovrHmd_None = 0,
        ovrHmd_DK1 = 3,
        ovrHmd_DKHD = 4,
        ovrHmd_DK2 = 6,
        ovrHmd_CB = 8,
        ovrHmd_Other = 9,
        ovrHmd_E3_2015 = 10,
        ovrHmd_ES06 = 11,
        ovrHmd_ES09 = 12,
        ovrHmd_ES11 = 13,
        ovrHmd_CV1 = 14,
        ovrHmd_RiftS = 16

    ctypedef enum ovrHmdCaps:
        ovrHmdCap_DebugDevice = 0x0010

    ctypedef enum ovrTrackingCaps:
        ovrTrackingCap_Orientation = 0x0010,
        ovrTrackingCap_MagYawCorrection = 0x0020,
        ovrTrackingCap_Position = 0x0040

    ctypedef enum ovrExtensions:
        ovrExtension_TextureLayout_Octilinear = 0 
        
    ctypedef enum ovrEyeType:
        ovrEye_Left = 0, 
        ovrEye_Right = 1,
        ovrEye_Count = 2

    ctypedef enum ovrTrackingOrigin:
        ovrTrackingOrigin_EyeLevel = 0,
        ovrTrackingOrigin_FloorLevel = 1,
        ovrTrackingOrigin_Count = 2 

    ctypedef struct ovrGraphicsLuid:
        char[8] Reserved

    ctypedef struct ovrHmdDesc:
        ovrHmdType Type
        char[64] ProductName
        char[64] Manufacturer
        short VendorId 
        short ProductId 
        char[24] SerialNumber
        short FirmwareMajor
        short FirmwareMinor 
        unsigned int AvailableHmdCaps 
        unsigned int DefaultHmdCaps 
        unsigned int AvailableTrackingCaps
        unsigned int DefaultTrackingCaps 
        ovrFovPort[2] DefaultEyeFov
        ovrFovPort[2] MaxEyeFov
        ovrSizei Resolution 
        float DisplayRefreshRate 

    ctypedef struct ovrHmdStruct
    ctypedef ovrHmdStruct* ovrSession
    ctypedef uint32_t ovrProcessId

    ctypedef enum ovrStatusBits:
        ovrStatus_OrientationTracked = 0x0001, 
        ovrStatus_PositionTracked = 0x0002,
        ovrStatus_OrientationValid = 0x0004,
        ovrStatus_PositionValid = 0x0008

    ctypedef struct ovrTrackerDesc:
        float FrustumHFovInRadians 
        float FrustumVFovInRadians 
        float FrustumNearZInMeters
        float FrustumFarZInMeters 

    ctypedef enum ovrTrackerFlags:
        ovrTracker_Connected = 0x0020,
        ovrTracker_PoseTracked = 0x0004

    ctypedef struct ovrTrackerPose:
        unsigned int TrackerFlags
        ovrPosef Pose
        ovrPosef LeveledPose

    ctypedef struct ovrTrackingState:
        ovrPoseStatef HeadPose
        unsigned int StatusFlags
        ovrPoseStatef[2] HandPoses
        unsigned int[2] HandStatusFlags
        ovrPosef CalibratedOrigin

    ctypedef struct ovrEyeRenderDesc:
        ovrEyeType Eye  
        ovrFovPort Fov  
        ovrRecti DistortedViewport  
        ovrVector2f PixelsPerTanAngleAtCenter
        ovrPosef HmdToEyePose  

    ctypedef struct ovrTimewarpProjectionDesc:
        float Projection22 
        float Projection23 
        float Projection32 

    ctypedef struct ovrViewScaleDesc:
        ovrPosef[2] HmdToEyePose
        float HmdSpaceToWorldScaleInMeters

    ctypedef enum ovrTextureType:
        ovrTexture_2D,
        ovrTexture_2D_External, 
        ovrTexture_Cube,
        ovrTexture_Count

    ctypedef enum ovrTextureBindFlags:
        ovrTextureBind_None,
        ovrTextureBind_DX_RenderTarget = 0x0001,
        ovrTextureBind_DX_UnorderedAccess = 0x0002,
        ovrTextureBind_DX_DepthStencil = 0x0004

    ctypedef enum ovrTextureFormat:
        OVR_FORMAT_UNKNOWN = 0,
        OVR_FORMAT_B5G6R5_UNORM = 1, 
        OVR_FORMAT_B5G5R5A1_UNORM = 2, 
        OVR_FORMAT_B4G4R4A4_UNORM = 3, 
        OVR_FORMAT_R8G8B8A8_UNORM = 4,
        OVR_FORMAT_R8G8B8A8_UNORM_SRGB = 5,
        OVR_FORMAT_B8G8R8A8_UNORM = 6,
        OVR_FORMAT_B8G8R8_UNORM = 27,
        OVR_FORMAT_B8G8R8A8_UNORM_SRGB = 7, 
        OVR_FORMAT_B8G8R8X8_UNORM = 8, 
        OVR_FORMAT_B8G8R8X8_UNORM_SRGB = 9, 
        OVR_FORMAT_R16G16B16A16_FLOAT = 10,
        OVR_FORMAT_R11G11B10_FLOAT = 25,
        OVR_FORMAT_D16_UNORM = 11,
        OVR_FORMAT_D24_UNORM_S8_UINT = 12,
        OVR_FORMAT_D32_FLOAT = 13,
        OVR_FORMAT_D32_FLOAT_S8X24_UINT = 14,
        OVR_FORMAT_BC1_UNORM = 15,
        OVR_FORMAT_BC1_UNORM_SRGB = 16,
        OVR_FORMAT_BC2_UNORM = 17,
        OVR_FORMAT_BC2_UNORM_SRGB = 18,
        OVR_FORMAT_BC3_UNORM = 19,
        OVR_FORMAT_BC3_UNORM_SRGB = 20,
        OVR_FORMAT_BC6H_UF16 = 21,
        OVR_FORMAT_BC6H_SF16 = 22,
        OVR_FORMAT_BC7_UNORM = 23,
        OVR_FORMAT_BC7_UNORM_SRGB = 24

    ctypedef enum ovrTextureFlags:
        ovrTextureMisc_None,
        ovrTextureMisc_DX_Typeless = 0x0001,
        ovrTextureMisc_AllowGenerateMips = 0x0002,
        ovrTextureMisc_ProtectedContent = 0x0004,
        ovrTextureMisc_AutoGenerateMips = 0x0008

    ctypedef struct ovrTextureSwapChainDesc:
        ovrTextureType Type
        ovrTextureFormat Format
        int ArraySize 
        int Width
        int Height
        int MipLevels
        int SampleCount
        ovrBool StaticImage 
        unsigned int MiscFlags
        unsigned int BindFlags 

    ctypedef enum ovrMirrorOptions:
        ovrMirrorOption_Default = 0x0000,
        ovrMirrorOption_PostDistortion = 0x0001,
        ovrMirrorOption_LeftEyeOnly = 0x0002,
        ovrMirrorOption_RightEyeOnly = 0x0004,
        ovrMirrorOption_IncludeGuardian = 0x0008,
        ovrMirrorOption_IncludeNotifications = 0x0010,
        ovrMirrorOption_IncludeSystemGui = 0x0020,
        ovrMirrorOption_ForceSymmetricFov = 0x0040

    ctypedef struct ovrMirrorTextureDesc:
        ovrTextureFormat Format
        int Width
        int Height
        unsigned int MiscFlags 
        unsigned int MirrorOptions

    ctypedef struct ovrTextureSwapChainData
    ctypedef ovrTextureSwapChainData* ovrTextureSwapChain
    ctypedef struct ovrMirrorTextureData
    ctypedef ovrMirrorTextureData* ovrMirrorTexture

    ctypedef enum ovrFovStencilType:
        ovrFovStencil_HiddenArea = 0,
        ovrFovStencil_VisibleArea = 1,
        ovrFovStencil_BorderLine = 2,
        ovrFovStencil_VisibleRectangle = 3

    ctypedef enum ovrFovStencilFlags:
        ovrFovStencilFlag_MeshOriginAtBottomLeft = 0x01

    ctypedef struct ovrFovStencilDesc:
        ovrFovStencilType StencilType
        uint32_t StencilFlags
        ovrEyeType Eye
        ovrFovPort FovPort
        ovrQuatf HmdToEyeRotation

    ctypedef struct ovrFovStencilMeshBuffer:
        int AllocVertexCount
        int UsedVertexCount
        ovrVector2f* VertexBuffer
        int AllocIndexCount
        int UsedIndexCount
        uint16_t* IndexBuffer

    cdef ovrResult ovr_GetFovStencil(ovrSession session, const ovrFovStencilDesc* fovStencilDesc, ovrFovStencilMeshBuffer* meshBuffer)

    ctypedef enum ovrButton:
        ovrButton_A = 0x00000001,
        ovrButton_B = 0x00000002,
        ovrButton_RThumb = 0x00000004,
        ovrButton_RShoulder = 0x00000008,
        ovrButton_X = 0x00000100,
        ovrButton_Y = 0x00000200,
        ovrButton_LThumb = 0x00000400,
        ovrButton_LShoulder = 0x00000800,
        ovrButton_Up = 0x00010000,
        ovrButton_Down = 0x00020000,
        ovrButton_Left = 0x00040000,
        ovrButton_Right = 0x00080000,
        ovrButton_Enter = 0x00100000,
        ovrButton_Back = 0x00200000,
        ovrButton_VolUp = 0x00400000,
        ovrButton_VolDown = 0x00800000,
        ovrButton_Home = 0x01000000,
        ovrButton_Private = ovrButton_VolUp | ovrButton_VolDown | ovrButton_Home,
        ovrButton_RMask = ovrButton_A | ovrButton_B | ovrButton_RThumb | ovrButton_RShoulder,
        ovrButton_LMask = ovrButton_X | ovrButton_Y | ovrButton_LThumb | ovrButton_LShoulder | ovrButton_Enter

    ctypedef enum ovrTouch:
        ovrTouch_A = ovrButton_A,
        ovrTouch_B = ovrButton_B,
        ovrTouch_RThumb = ovrButton_RThumb,
        ovrTouch_RThumbRest = 0x00000008,
        ovrTouch_RIndexTrigger = 0x00000010,
        ovrTouch_RButtonMask = ovrTouch_A | ovrTouch_B | ovrTouch_RThumb | ovrTouch_RThumbRest | ovrTouch_RIndexTrigger,
        ovrTouch_X = ovrButton_X,
        ovrTouch_Y = ovrButton_Y,
        ovrTouch_LThumb = ovrButton_LThumb,
        ovrTouch_LThumbRest = 0x00000800,
        ovrTouch_LIndexTrigger = 0x00001000,
        ovrTouch_LButtonMask = ovrTouch_X | ovrTouch_Y | ovrTouch_LThumb | ovrTouch_LThumbRest | ovrTouch_LIndexTrigger,
        ovrTouch_RIndexPointing = 0x00000020,
        ovrTouch_RThumbUp = 0x00000040,
        ovrTouch_LIndexPointing = 0x00002000,
        ovrTouch_LThumbUp = 0x00004000,
        ovrTouch_RPoseMask = ovrTouch_RIndexPointing | ovrTouch_RThumbUp,
        ovrTouch_LPoseMask = ovrTouch_LIndexPointing | ovrTouch_LThumbUp

    ctypedef struct ovrTouchHapticsDesc:
        int SampleRateHz
        int SampleSizeInBytes
        int QueueMinSizeToAvoidStarvation
        int SubmitMinSamples
        int SubmitMaxSamples
        int SubmitOptimalSamples

    ctypedef enum ovrControllerType:
        ovrControllerType_None = 0x0000,
        ovrControllerType_LTouch = 0x0001,
        ovrControllerType_RTouch = 0x0002,
        ovrControllerType_Touch = (ovrControllerType_LTouch | ovrControllerType_RTouch),
        ovrControllerType_Remote = 0x0004,
        ovrControllerType_XBox = 0x0010,
        ovrControllerType_Object0 = 0x0100,
        ovrControllerType_Object1 = 0x0200,
        ovrControllerType_Object2 = 0x0400,
        ovrControllerType_Object3 = 0x0800,
        ovrControllerType_Active = 0xffffffff

    ctypedef enum ovrHapticsBufferSubmitMode:
        ovrHapticsBufferSubmit_Enqueue

    cdef int OVR_HAPTICS_BUFFER_SAMPLES_MAX

    ctypedef struct ovrHapticsBuffer:
        const void* Samples
        int SamplesCount
        ovrHapticsBufferSubmitMode SubmitMode

    ctypedef struct ovrHapticsPlaybackState:
        int RemainingQueueSpace
        int SamplesQueued

    ctypedef enum ovrTrackedDeviceType:
        ovrTrackedDevice_None = 0x0000,
        ovrTrackedDevice_HMD = 0x0001,
        ovrTrackedDevice_LTouch = 0x0002,
        ovrTrackedDevice_RTouch = 0x0004,
        ovrTrackedDevice_Touch = (ovrTrackedDevice_LTouch | ovrTrackedDevice_RTouch),
        ovrTrackedDevice_Object0 = 0x0010,
        ovrTrackedDevice_Object1 = 0x0020,
        ovrTrackedDevice_Object2 = 0x0040,
        ovrTrackedDevice_Object3 = 0x0080

    ctypedef enum ovrBoundaryType:
        ovrBoundary_Outer = 0x0001
        ovrBoundary_PlayArea = 0x0100

    ctypedef struct ovrBoundaryLookAndFeel:
        ovrColorf Color

    ctypedef struct ovrBoundaryTestResult:
        ovrBool IsTriggering
        float ClosestDistance
        ovrVector3f ClosestPoint
        ovrVector3f ClosestPointNormal

    ctypedef enum ovrHandType:
        ovrHand_Left = 0,
        ovrHand_Right = 1,
        ovrHand_Count = 2

    ctypedef struct ovrInputState:
        double TimeInSeconds
        unsigned int Buttons
        unsigned int Touches
        float[2] IndexTrigger
        float[2] HandTrigger
        ovrVector2f[2] Thumbstick
        ovrControllerType ControllerType
        float[2] IndexTriggerNoDeadzone
        float[2] HandTriggerNoDeadzone
        ovrVector2f[2] ThumbstickNoDeadzone
        float[2] IndexTriggerRaw
        float[2] HandTriggerRaw
        ovrVector2f[2] ThumbstickRaw

    ctypedef struct ovrCameraIntrinsics:
        double LastChangedTime
        ovrFovPort FOVPort
        float VirtualNearPlaneDistanceMeters
        float VirtualFarPlaneDistanceMeters
        ovrSizei ImageSensorPixelResolution
        ovrMatrix4f LensDistortionMatrix
        double ExposurePeriodSeconds
        double ExposureDurationSeconds

    ctypedef enum ovrCameraStatusFlags:
        ovrCameraStatus_None = 0x0,
        ovrCameraStatus_Connected = 0x1,
        ovrCameraStatus_Calibrating = 0x2,
        ovrCameraStatus_CalibrationFailed = 0x4,
        ovrCameraStatus_Calibrated = 0x8,
        ovrCameraStatus_Capturing = 0x10

    ctypedef struct ovrCameraExtrinsics:
        double LastChangedTimeSeconds
        unsigned int CameraStatusFlags
        ovrTrackedDeviceType AttachedToDevice
        ovrPosef RelativePose
        double LastExposureTimeSeconds
        double ExposureLatencySeconds
        double AdditionalLatencySeconds

    ctypedef enum:
        OVR_MAX_EXTERNAL_CAMERA_COUNT = 16
        OVR_EXTERNAL_CAMERA_NAME_SIZE = 32
        
    ctypedef struct ovrExternalCamera:
        char[OVR_EXTERNAL_CAMERA_NAME_SIZE] Name
        ovrCameraIntrinsics Intrinsics
        ovrCameraExtrinsics Extrinsics

    ctypedef enum ovrInitFlags:
        ovrInit_Debug = 0x00000001,
        ovrInit_RequestVersion = 0x00000004,
        ovrInit_Invisible = 0x00000010,
        ovrInit_MixedRendering = 0x00000020,
        ovrInit_FocusAware = 0x00000040,
        ovrinit_WritableBits = 0x00ffffff

    ctypedef enum ovrLogLevel:
        ovrLogLevel_Debug = 0,
        ovrLogLevel_Info = 1, 
        ovrLogLevel_Error = 2 

    ctypedef void(*ovrLogCallback)(uintptr_t userData, int level, const char* message)

    ctypedef struct ovrInitParams:
        uint32_t Flags
        uint32_t RequestedMinorVersion
        ovrLogCallback LogCallback
        uintptr_t UserData
        uint32_t ConnectionTimeoutMS

    cdef ovrResult ovr_Initialize(const ovrInitParams* params)
    cdef void ovr_Shutdown()
    cdef void ovr_GetLastErrorInfo(ovrErrorInfo* errorInfo)
    cdef const char* ovr_GetVersionString()
    cdef int ovr_TraceMessage(int level, const char* message)
    cdef ovrResult ovr_IdentifyClient(const char* identity)
    cdef ovrHmdDesc ovr_GetHmdDesc(ovrSession session)
    cdef unsigned int ovr_GetTrackerCount(ovrSession session)
    cdef ovrTrackerDesc ovr_GetTrackerDesc(ovrSession session, unsigned int trackerDescIndex)
    cdef ovrResult ovr_Create(ovrSession* pSession, ovrGraphicsLuid* pLuid)
    cdef void ovr_Destroy(ovrSession session)

    ctypedef struct ovrSessionStatus:
        ovrBool IsVisible
        ovrBool HmdPresent
        ovrBool HmdMounted
        ovrBool DisplayLost
        ovrBool ShouldQuit
        ovrBool ShouldRecenter
        ovrBool HasInputFocus
        ovrBool OverlayPresent
        ovrBool DepthRequested

    cdef ovrResult ovr_GetSessionStatus(ovrSession session, ovrSessionStatus* sessionStatus)
    cdef ovrResult ovr_IsExtensionSupported(ovrSession session, ovrExtensions extension, ovrBool* outExtensionSupported);
    cdef ovrResult ovr_EnableExtension(ovrSession session, ovrExtensions extension)
    cdef ovrResult ovr_SetTrackingOriginType(ovrSession session, ovrTrackingOrigin origin)
    cdef ovrTrackingOrigin ovr_GetTrackingOriginType(ovrSession session)
    cdef ovrResult ovr_RecenterTrackingOrigin(ovrSession session)
    cdef ovrResult ovr_SpecifyTrackingOrigin(ovrSession session, ovrPosef originPose)
    cdef void ovr_ClearShouldRecenterFlag(ovrSession session)
    cdef ovrTrackingState ovr_GetTrackingState(ovrSession session, double absTime, ovrBool latencyMarker)
    cdef ovrResult ovr_GetDevicePoses(ovrSession session, ovrTrackedDeviceType* deviceTypes, int deviceCount, double absTime, ovrPoseStatef* outDevicePoses)
    cdef ovrTrackerPose ovr_GetTrackerPose(ovrSession session, unsigned int trackerPoseIndex)
    cdef ovrResult ovr_GetInputState(ovrSession session, ovrControllerType controllerType, ovrInputState* inputState)
    cdef unsigned int ovr_GetConnectedControllerTypes(ovrSession session)
    cdef ovrTouchHapticsDesc ovr_GetTouchHapticsDesc(ovrSession session, ovrControllerType controllerType)
    cdef ovrResult ovr_SetControllerVibration(ovrSession session, ovrControllerType controllerType, float frequency, float amplitude)
    cdef ovrResult ovr_SubmitControllerVibration(ovrSession session, ovrControllerType controllerType, const ovrHapticsBuffer* buffer)
    cdef ovrResult ovr_GetControllerVibrationState(ovrSession session, ovrControllerType controllerType, ovrHapticsPlaybackState* outState)
    cdef ovrResult ovr_TestBoundary(ovrSession session, ovrTrackedDeviceType deviceBitmask, ovrBoundaryType boundaryType, ovrBoundaryTestResult* outTestResult)
    cdef ovrResult ovr_TestBoundaryPoint(ovrSession session, const ovrVector3f* point, ovrBoundaryType singleBoundaryType, ovrBoundaryTestResult* outTestResult)
    cdef ovrResult ovr_SetBoundaryLookAndFeel(ovrSession session, const ovrBoundaryLookAndFeel* lookAndFeel)
    cdef ovrResult ovr_ResetBoundaryLookAndFeel(ovrSession session)
    cdef ovrResult ovr_GetBoundaryGeometry(ovrSession session, ovrBoundaryType boundaryType, ovrVector3f* outFloorPoints, int* outFloorPointsCount)
    cdef ovrResult ovr_GetBoundaryDimensions(ovrSession session, ovrBoundaryType boundaryType, ovrVector3f* outDimensions)
    cdef ovrResult ovr_GetBoundaryVisible(ovrSession session, ovrBool* outIsVisible)
    cdef ovrResult ovr_RequestBoundaryVisible(ovrSession session, ovrBool visible)
    cdef ovrResult ovr_GetExternalCameras(ovrSession session, ovrExternalCamera* cameras, unsigned int* inoutCameraCount)
    cdef ovrResult ovr_SetExternalCameraProperties(ovrSession session, const char* name, const ovrCameraIntrinsics* const intrinsics, const ovrCameraExtrinsics* const extrinsics)

    cdef int ovrMaxLayerCount = 16

    ctypedef enum ovrLayerType:
        ovrLayerType_Disabled = 0,
        ovrLayerType_EyeFov = 1,
        ovrLayerType_EyeFovDepth = 2,
        ovrLayerType_Quad = 3,
        ovrLayerType_EyeMatrix = 5,
        ovrLayerType_EyeFovMultires = 7,
        ovrLayerType_Cylinder = 8,
        ovrLayerType_Cube = 10

    ctypedef enum ovrLayerFlags:
        ovrLayerFlag_HighQuality = 0x01,
        ovrLayerFlag_TextureOriginAtBottomLeft = 0x02,
        ovrLayerFlag_HeadLocked = 0x04

    ctypedef struct ovrLayerHeader:
        ovrLayerType Type
        unsigned int Flags

    ctypedef struct ovrLayerEyeFov:
        ovrLayerHeader Header
        ovrTextureSwapChain[2] ColorTexture
        ovrRecti[2] Viewport
        ovrFovPort[2] Fov
        ovrPosef[2] RenderPose
        double SensorSampleTime

    ctypedef struct ovrLayerEyeFovDepth:
        ovrLayerHeader Header
        ovrTextureSwapChain[2] ColorTexture
        ovrRecti[2] Viewport
        ovrFovPort[2] Fov
        ovrPosef[2] RenderPose
        double SensorSampleTime
        ovrTextureSwapChain[2] DepthTexture
        ovrTimewarpProjectionDesc ProjectionDesc

    ctypedef enum ovrTextureLayout:
        ovrTextureLayout_Rectilinear = 0, 
        ovrTextureLayout_Octilinear = 1 

    ctypedef struct ovrTextureLayoutOctilinear:
        float WarpLeft
        float WarpRight
        float WarpUp
        float WarpDown
        float SizeLeft
        float SizeRight
        float SizeUp
        float SizeDown

    ctypedef union ovrTextureLayoutDesc_Union:
        ovrTextureLayoutOctilinear[2] Octilinear

    ctypedef struct ovrLayerEyeFovMultires:
        ovrLayerHeader Header
        ovrTextureSwapChain[2] ColorTexture
        ovrRecti[2] Viewport
        ovrFovPort[2] Fov
        ovrPosef[2] RenderPose
        double SensorSampleTime
        ovrTextureLayout TextureLayout
        ovrTextureLayoutDesc_Union TextureLayoutDesc

    ctypedef struct ovrLayerEyeMatrix:
        ovrLayerHeader Header
        ovrTextureSwapChain[2] ColorTexture
        ovrRecti[2] Viewport
        ovrPosef[2] RenderPose
        ovrMatrix4f[2] Matrix
        double SensorSampleTime

    ctypedef struct ovrLayerQuad:
        ovrLayerHeader Header
        ovrTextureSwapChain ColorTexture
        ovrRecti Viewport
        ovrPosef QuadPoseCenter
        ovrVector2f QuadSize

    ctypedef struct ovrLayerCylinder:
        ovrLayerHeader Header
        ovrTextureSwapChain ColorTexture
        ovrRecti Viewport
        ovrPosef CylinderPoseCenter
        float CylinderRadius
        float CylinderAngle
        float CylinderAspectRatio

    ctypedef struct ovrLayerCube:
        ovrLayerHeader Header
        ovrQuatf Orientation
        ovrTextureSwapChain CubeMapTexture

    ctypedef union ovrLayer_Union:
        ovrLayerHeader Header
        ovrLayerEyeFov EyeFov
        ovrLayerEyeFovDepth EyeFovDepth
        ovrLayerQuad Quad
        ovrLayerEyeFovMultires Multires
        ovrLayerCylinder Cylinder
        ovrLayerCube Cube

    cdef ovrResult ovr_GetTextureSwapChainLength(ovrSession session, ovrTextureSwapChain chain, int* out_Length)
    cdef ovrResult ovr_GetTextureSwapChainCurrentIndex(ovrSession session, ovrTextureSwapChain chain, int* out_Index)
    cdef ovrResult ovr_GetTextureSwapChainDesc(ovrSession session, ovrTextureSwapChain chain, ovrTextureSwapChainDesc* out_Desc)
    cdef ovrResult ovr_CommitTextureSwapChain(ovrSession session, ovrTextureSwapChain chain)
    cdef void ovr_DestroyTextureSwapChain(ovrSession session, ovrTextureSwapChain chain)
    cdef void ovr_DestroyMirrorTexture(ovrSession session, ovrMirrorTexture mirrorTexture)
    cdef ovrSizei ovr_GetFovTextureSize(ovrSession session, ovrEyeType eye, ovrFovPort fov, float pixelsPerDisplayPixel)
    cdef ovrEyeRenderDesc ovr_GetRenderDesc(ovrSession session, ovrEyeType eyeType, ovrFovPort fov)
    cdef ovrResult ovr_WaitToBeginFrame(ovrSession session, long long frameIndex)
    cdef ovrResult ovr_BeginFrame(ovrSession session, long long frameIndex)
    # note edited argument "ovrLayerHeader const* const* layerPtrList"
    cdef ovrResult ovr_EndFrame(ovrSession session, long long frameIndex, const ovrViewScaleDesc* viewScaleDesc, const ovrLayerHeader *const *layerPtrList, unsigned int layerCount)
    cdef ovrResult ovr_SubmitFrame(ovrSession session, long long frameIndex, const ovrViewScaleDesc* viewScaleDesc, const ovrLayerHeader *const *layerPtrList, unsigned int layerCount)

    ctypedef struct ovrPerfStatsPerCompositorFrame:
        int HmdVsyncIndex
        int AppFrameIndex
        int AppDroppedFrameCount
        float AppMotionToPhotonLatency
        float AppQueueAheadTime
        float AppCpuElapsedTime
        float AppGpuElapsedTime
        int CompositorFrameIndex
        int CompositorDroppedFrameCount
        float CompositorLatency
        float CompositorCpuElapsedTime
        float CompositorGpuElapsedTime
        float CompositorCpuStartToGpuEndElapsedTime
        float CompositorGpuEndToVsyncElapsedTime
        ovrBool AswIsActive
        int AswActivatedToggleCount
        int AswPresentedFrameCount
        int AswFailedFrameCount

    cdef int ovrMaxProvidedFrameStats = 5

    ctypedef struct ovrPerfStats:
        ovrPerfStatsPerCompositorFrame[5] FrameStats
        int FrameStatsCount
        ovrBool AnyFrameStatsDropped
        float AdaptiveGpuPerformanceScale
        ovrBool AswIsAvailable
        ovrProcessId VisibleProcessId

    cdef ovrResult ovr_GetPerfStats(ovrSession session, ovrPerfStats* outStats)
    cdef ovrResult ovr_ResetPerfStats(ovrSession session)
    cdef double ovr_GetPredictedDisplayTime(ovrSession session, long long frameIndex)
    cdef double ovr_GetTimeInSeconds()

    ctypedef enum ovrPerfHudMode:
        ovrPerfHud_Off = 0,
        ovrPerfHud_PerfSummary = 1,
        ovrPerfHud_LatencyTiming = 2, 
        ovrPerfHud_AppRenderTiming = 3,
        ovrPerfHud_CompRenderTiming = 4,
        ovrPerfHud_AswStats = 6, 
        ovrPerfHud_VersionInfo = 5, 
        ovrPerfHud_Count = 7 

    ctypedef enum ovrLayerHudMode:
        ovrLayerHud_Off = 0,
        ovrLayerHud_Info = 1

    ctypedef enum ovrDebugHudStereoMode:
        ovrDebugHudStereo_Off = 0,
        ovrDebugHudStereo_Quad = 1,
        ovrDebugHudStereo_QuadWithCrosshair = 2,
        ovrDebugHudStereo_CrosshairAtInfinity = 3,
        ovrDebugHudStereo_Count

    cdef ovrBool ovr_GetBool(ovrSession session, const char* propertyName, ovrBool defaultVal)
    cdef ovrBool ovr_SetBool(ovrSession session, const char* propertyName, ovrBool value)
    cdef int ovr_GetInt(ovrSession session, const char* propertyName, int defaultVal)
    cdef ovrBool ovr_SetInt(ovrSession session, const char* propertyName, int value)
    cdef float ovr_GetFloat(ovrSession session, const char* propertyName, float defaultVal)
    cdef ovrBool ovr_SetFloat(ovrSession session, const char* propertyName, float value)
    cdef unsigned int ovr_GetFloatArray(ovrSession session, const char* propertyName, float values[], unsigned int valuesCapacity)
    cdef ovrBool ovr_SetFloatArray(ovrSession session, const char* propertyName, const float values[], unsigned int valuesSize)
    cdef const char* ovr_GetString(ovrSession session, const char* propertyName, const char* defaultVal)
    cdef ovrBool ovr_SetString(ovrSession session, const char* propertyName, const char* value)


cdef extern from "OVR_CAPI_Util.h":
    ctypedef enum ovrProjectionModifier:
        ovrProjection_None = 0x00,
        ovrProjection_LeftHanded = 0x01,
        ovrProjection_FarLessThanNear = 0x02,
        ovrProjection_FarClipAtInfinity = 0x04,
        ovrProjection_ClipRangeOpenGL = 0x08

    ctypedef struct ovrDetectResult:
        ovrBool IsOculusServiceRunning
        ovrBool IsOculusHMDConnected

    ctypedef enum ovrHapticsGenMode:
        ovrHapticsGenMode_PointSample,
        ovrHapticsGenMode_Count

    ctypedef struct ovrAudioChannelData:
        const float* Samples
        int SamplesCount
        int Frequency

    ctypedef struct ovrHapticsClip:
        const void* Samples
        int SamplesCount

    cdef ovrDetectResult ovr_Detect(int timeoutMilliseconds)
    cdef ovrMatrix4f ovrMatrix4f_Projection(ovrFovPort fov, float znear, float zfar, unsigned int projectionModFlags)
    cdef ovrTimewarpProjectionDesc ovrTimewarpProjectionDesc_FromProjection(ovrMatrix4f projection, unsigned int projectionModFlags)
    cdef ovrMatrix4f ovrMatrix4f_OrthoSubProjection(ovrMatrix4f projection, ovrVector2f orthoScale, float orthoDistance, float HmdToEyeOffsetX)
    cdef void ovr_CalcEyePoses(ovrPosef headPose, const ovrVector3f hmdToEyeOffset[2], ovrPosef outEyePoses[2])
    cdef void ovr_CalcEyePoses2(ovrPosef headPose, const ovrPosef HmdToEyePose[2], ovrPosef outEyePoses[2])
    cdef void ovr_GetEyePoses(ovrSession session, long long frameIndex, ovrBool latencyMarker, const ovrVector3f hmdToEyeOffset[2], ovrPosef outEyePoses[2], double* outSensorSampleTime)
    cdef void ovr_GetEyePoses2(ovrSession session, long long frameIndex, ovrBool latencyMarker, const ovrPosef HmdToEyePose[2], ovrPosef outEyePoses[2], double* outSensorSampleTime)
    cdef void ovrPosef_FlipHandedness(const ovrPosef* inPose, ovrPosef* outPose)
    cdef ovrResult ovr_ReadWavFromBuffer(ovrAudioChannelData* outAudioChannel, const void* inputData, int dataSizeInBytes, int stereoChannelToUse)
    cdef ovrResult ovr_GenHapticsFromAudioData(ovrHapticsClip* outHapticsClip, const ovrAudioChannelData* audioChannel, ovrHapticsGenMode genMode)
    cdef void ovr_ReleaseAudioChannelData(ovrAudioChannelData* audioChannel)
    cdef void ovr_ReleaseHapticsClip(ovrHapticsClip* hapticsClip)


cdef extern from "OVR_CAPI_GL.h":
    cdef ovrResult ovr_CreateTextureSwapChainGL(ovrSession session, const ovrTextureSwapChainDesc* desc, ovrTextureSwapChain* out_TextureSwapChain)
    cdef ovrResult ovr_GetTextureSwapChainBufferGL(ovrSession session, ovrTextureSwapChain chain, int index, unsigned int* out_TexId)
    cdef ovrResult ovr_CreateMirrorTextureWithOptionsGL(ovrSession session, const ovrMirrorTextureDesc* desc, ovrMirrorTexture* out_MirrorTexture)
    cdef ovrResult ovr_CreateMirrorTextureGL(ovrSession session, const ovrMirrorTextureDesc* desc, ovrMirrorTexture* out_MirrorTexture)
    cdef ovrResult ovr_GetMirrorTextureBufferGL(ovrSession session, ovrMirrorTexture mirrorTexture, unsigned int* out_TexId)


cdef extern from "OVR_CAPI_Audio.h":
    ctypedef enum:
        OVR_AUDIO_MAX_DEVICE_STR_SIZE

