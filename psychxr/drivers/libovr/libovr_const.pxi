#  =============================================================================
#  libovr_const.pxi - Module level constants
#  =============================================================================
#
#  libovr_const.pxi
#
#  Copyright 2020 Matthew Cutone <cutonem(a)yorku.ca> and Laurie M. Wilcox
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
STATUS_ORIENTATION_VALID = capi.ovrStatus_OrientationValid
STATUS_POSITION_VALID = capi.ovrStatus_PositionValid

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
PERF_HUD_COUNT = capi.ovrPerfHud_Count  # for cycling

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
TEXTURE_SWAP_CHAIN_COUNT = 16

# boundary modes
BOUNDARY_PLAY_AREA = capi.ovrBoundary_PlayArea
BOUNDARY_OUTER = capi.ovrBoundary_Outer

# layer header flags
LAYER_FLAG_HIGH_QUALITY = capi.ovrLayerFlag_HighQuality
LAYER_FLAG_TEXTURE_ORIGIN_AT_BOTTOM_LEFT = \
    capi.ovrLayerFlag_TextureOriginAtBottomLeft
LAYER_FLAG_HEAD_LOCKED = capi.ovrLayerFlag_HeadLocked

# HMD types
HMD_NONE = capi.ovrHmd_None
HMD_DK1 = capi.ovrHmd_DK1
HMD_DKHD = capi.ovrHmd_DKHD
HMD_DK2 = capi.ovrHmd_DK2
HMD_CB = capi.ovrHmd_CB
HMD_OTHER = capi.ovrHmd_Other
HMD_E3_2015  = capi.ovrHmd_E3_2015
HMD_ES06 = capi.ovrHmd_ES06
HMD_ES09 = capi.ovrHmd_ES09
HMD_ES11 = capi.ovrHmd_ES11
HMD_CV1 = capi.ovrHmd_CV1
HMD_RIFTS = capi.ovrHmd_RiftS

# haptics buffer
HAPTICS_BUFFER_SAMPLES_MAX = capi.OVR_HAPTICS_BUFFER_SAMPLES_MAX

# mirror texture options
MIRROR_OPTION_DEFAULT = capi.ovrMirrorOption_Default
MIRROR_OPTION_POST_DISTORTION = capi.ovrMirrorOption_PostDistortion
MIRROR_OPTION_LEFT_EYE_ONLY = capi.ovrMirrorOption_LeftEyeOnly
MIRROR_OPTION_RIGHT_EYE_ONLY = capi.ovrMirrorOption_RightEyeOnly
MIRROR_OPTION_INCLUDE_GUARDIAN = capi.ovrMirrorOption_IncludeGuardian
MIRROR_OPTION_INCLUDE_NOTIFICATIONS = capi.ovrMirrorOption_IncludeNotifications
MIRROR_OPTION_INCLUDE_SYSTEM_GUI = capi.ovrMirrorOption_IncludeSystemGui
MIRROR_OPTION_FORCE_SYMMETRIC_FOV = capi.ovrMirrorOption_ForceSymmetricFov

# logging levels
LOG_LEVEL_DEBUG = capi.ovrLogLevel_Debug
LOG_LEVEL_INFO = capi.ovrLogLevel_Info
LOG_LEVEL_ERROR = capi.ovrLogLevel_Error