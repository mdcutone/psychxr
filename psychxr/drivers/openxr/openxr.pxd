# distutils: language=c++
#  =============================================================================
#  openxr.pxd - Cython definitions for `openxr.h`
#  =============================================================================
#
#  Copyright 2021 Matthew Cutone <mcutone@opensciencetools.com>
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

from libc.stdint cimport uint32_t, uint64_t, int64_t, uint8_t, int32_t


cdef extern from "openxr.h":
    cdef int XR_VERSION_1_0
    cdef int XR_MAKE_VERSION
    cdef int XR_CURRENT_API_VERSION
    cdef int XR_VERSION_MAJOR
    cdef int XR_VERSION_MINOR
    cdef int XR_VERSION_PATCH
    cdef int XR_NULL_HANDLE

    cdef int XR_NULL_SYSTEM_ID
    cdef int XR_NULL_PATH
    cdef int XR_SUCCEEDED
    cdef int XR_FAILED
    cdef int XR_UNQUALIFIED_SUCCESS
    cdef int XR_NO_DURATION
    cdef int XR_INFINITE_DURATION
    cdef int XR_MIN_HAPTIC_DURATION
    cdef int XR_FREQUENCY_UNSPECIFIED
    cdef int XR_MAX_EVENT_DATA_SIZE

    ctypedef uint64_t XrVersion
    ctypedef uint64_t XrFlags64
    ctypedef uint64_t XrSystemId
    ctypedef uint32_t XrBool32
    ctypedef uint64_t XrPath
    ctypedef int64_t XrTime
    ctypedef int64_t XrDuration

    ctypedef struct XrInstance
    ctypedef XrInstance* XrInstance
    ctypedef struct XrSession
    ctypedef XrSession* XrSession
    ctypedef struct XrSpace
    ctypedef XrSpace* XrSpace
    ctypedef struct XrAction
    ctypedef XrAction* XrAction
    ctypedef struct XrSwapchain
    ctypedef XrSwapchain* XrSwapchain
    ctypedef struct XrActionSet
    ctypedef XrActionSet* XrActionSet

    ctypedef XR_TRUE
    ctypedef XR_FALSE
    ctypedef XR_MAX_EXTENSION_NAME_SIZE
    ctypedef XR_MAX_API_LAYER_NAME_SIZE
    ctypedef XR_MAX_API_LAYER_DESCRIPTION_SIZE
    ctypedef XR_MAX_SYSTEM_NAME_SIZE
    ctypedef XR_MAX_APPLICATION_NAME_SIZE
    ctypedef XR_MAX_ENGINE_NAME_SIZE
    ctypedef XR_MAX_RUNTIME_NAME_SIZE
    ctypedef XR_MAX_PATH_LENGTH
    ctypedef XR_MAX_STRUCTURE_NAME_SIZE
    ctypedef XR_MAX_RESULT_STRING_SIZE
    ctypedef XR_MIN_COMPOSITION_LAYERS_SUPPORTED
    ctypedef XR_MAX_ACTION_SET_NAME_SIZE
    ctypedef XR_MAX_LOCALIZED_ACTION_SET_NAME_SIZE
    ctypedef XR_MAX_ACTION_NAME_SIZE
    ctypedef XR_MAX_LOCALIZED_ACTION_NAME_SIZE

    ctypedef enum XrResult:
        XR_SUCCESS = 0,
        XR_TIMEOUT_EXPIRED = 1,
        XR_SESSION_LOSS_PENDING = 3,
        XR_EVENT_UNAVAILABLE = 4,
        XR_SPACE_BOUNDS_UNAVAILABLE = 7,
        XR_SESSION_NOT_FOCUSED = 8,
        XR_FRAME_DISCARDED = 9,
        XR_ERROR_VALIDATION_FAILURE = -1,
        XR_ERROR_RUNTIME_FAILURE = -2,
        XR_ERROR_OUT_OF_MEMORY = -3,
        XR_ERROR_API_VERSION_UNSUPPORTED = -4,
        XR_ERROR_INITIALIZATION_FAILED = -6,
        XR_ERROR_FUNCTION_UNSUPPORTED = -7,
        XR_ERROR_FEATURE_UNSUPPORTED = -8,
        XR_ERROR_EXTENSION_NOT_PRESENT = -9,
        XR_ERROR_LIMIT_REACHED = -10,
        XR_ERROR_SIZE_INSUFFICIENT = -11,
        XR_ERROR_HANDLE_INVALID = -12,
        XR_ERROR_INSTANCE_LOST = -13,
        XR_ERROR_SESSION_RUNNING = -14,
        XR_ERROR_SESSION_NOT_RUNNING = -16,
        XR_ERROR_SESSION_LOST = -17,
        XR_ERROR_SYSTEM_INVALID = -18,
        XR_ERROR_PATH_INVALID = -19,
        XR_ERROR_PATH_COUNT_EXCEEDED = -20,
        XR_ERROR_PATH_FORMAT_INVALID = -21,
        XR_ERROR_PATH_UNSUPPORTED = -22,
        XR_ERROR_LAYER_INVALID = -23,
        XR_ERROR_LAYER_LIMIT_EXCEEDED = -24,
        XR_ERROR_SWAPCHAIN_RECT_INVALID = -25,
        XR_ERROR_SWAPCHAIN_FORMAT_UNSUPPORTED = -26,
        XR_ERROR_ACTION_TYPE_MISMATCH = -27,
        XR_ERROR_SESSION_NOT_READY = -28,
        XR_ERROR_SESSION_NOT_STOPPING = -29,
        XR_ERROR_TIME_INVALID = -30,
        XR_ERROR_REFERENCE_SPACE_UNSUPPORTED = -31,
        XR_ERROR_FILE_ACCESS_ERROR = -32,
        XR_ERROR_FILE_CONTENTS_INVALID = -33,
        XR_ERROR_FORM_FACTOR_UNSUPPORTED = -34,
        XR_ERROR_FORM_FACTOR_UNAVAILABLE = -35,
        XR_ERROR_API_LAYER_NOT_PRESENT = -36,
        XR_ERROR_CALL_ORDER_INVALID = -37,
        XR_ERROR_GRAPHICS_DEVICE_INVALID = -38,
        XR_ERROR_POSE_INVALID = -39,
        XR_ERROR_INDEX_OUT_OF_RANGE = -40,
        XR_ERROR_VIEW_CONFIGURATION_TYPE_UNSUPPORTED = -41,
        XR_ERROR_ENVIRONMENT_BLEND_MODE_UNSUPPORTED = -42,
        XR_ERROR_NAME_DUPLICATED = -44,
        XR_ERROR_NAME_INVALID = -45,
        XR_ERROR_ACTIONSET_NOT_ATTACHED = -46,
        XR_ERROR_ACTIONSETS_ALREADY_ATTACHED = -47,
        XR_ERROR_LOCALIZED_NAME_DUPLICATED = -48,
        XR_ERROR_LOCALIZED_NAME_INVALID = -49,
        XR_ERROR_GRAPHICS_REQUIREMENTS_CALL_MISSING = -50,
        XR_ERROR_RUNTIME_UNAVAILABLE = -51,
        XR_ERROR_ANDROID_THREAD_SETTINGS_ID_INVALID_KHR = -1000003000,
        XR_ERROR_ANDROID_THREAD_SETTINGS_FAILURE_KHR = -1000003001,
        XR_ERROR_CREATE_SPATIAL_ANCHOR_FAILED_MSFT = -1000039001,
        XR_ERROR_SECONDARY_VIEW_CONFIGURATION_TYPE_NOT_ENABLED_MSFT = -1000053000,
        XR_ERROR_CONTROLLER_MODEL_KEY_INVALID_MSFT = -1000055000,
        XR_ERROR_REPROJECTION_MODE_UNSUPPORTED_MSFT = -1000066000,
        XR_ERROR_COMPUTE_NEW_SCENE_NOT_COMPLETED_MSFT = -1000097000,
        XR_ERROR_SCENE_COMPONENT_ID_INVALID_MSFT = -1000097001,
        XR_ERROR_SCENE_COMPONENT_TYPE_MISMATCH_MSFT = -1000097002,
        XR_ERROR_SCENE_MESH_BUFFER_ID_INVALID_MSFT = -1000097003,
        XR_ERROR_SCENE_COMPUTE_FEATURE_INCOMPATIBLE_MSFT = -1000097004,
        XR_ERROR_SCENE_COMPUTE_CONSISTENCY_MISMATCH_MSFT = -1000097005,
        XR_ERROR_DISPLAY_REFRESH_RATE_UNSUPPORTED_FB = -1000101000,
        XR_ERROR_COLOR_SPACE_UNSUPPORTED_FB = -1000108000,
        XR_RESULT_MAX_ENUM = 0x7FFFFFFF

    ctypedef enum XrStructureType:
        XR_TYPE_UNKNOWN = 0,
        XR_TYPE_API_LAYER_PROPERTIES = 1,
        XR_TYPE_EXTENSION_PROPERTIES = 2,
        XR_TYPE_INSTANCE_CREATE_INFO = 3,
        XR_TYPE_SYSTEM_GET_INFO = 4,
        XR_TYPE_SYSTEM_PROPERTIES = 5,
        XR_TYPE_VIEW_LOCATE_INFO = 6,
        XR_TYPE_VIEW = 7,
        XR_TYPE_SESSION_CREATE_INFO = 8,
        XR_TYPE_SWAPCHAIN_CREATE_INFO = 9,
        XR_TYPE_SESSION_BEGIN_INFO = 10,
        XR_TYPE_VIEW_STATE = 11,
        XR_TYPE_FRAME_END_INFO = 12,
        XR_TYPE_HAPTIC_VIBRATION = 13,
        XR_TYPE_EVENT_DATA_BUFFER = 16,
        XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING = 17,
        XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED = 18,
        XR_TYPE_ACTION_STATE_BOOLEAN = 23,
        XR_TYPE_ACTION_STATE_FLOAT = 24,
        XR_TYPE_ACTION_STATE_VECTOR2F = 25,
        XR_TYPE_ACTION_STATE_POSE = 27,
        XR_TYPE_ACTION_SET_CREATE_INFO = 28,
        XR_TYPE_ACTION_CREATE_INFO = 29,
        XR_TYPE_INSTANCE_PROPERTIES = 32,
        XR_TYPE_FRAME_WAIT_INFO = 33,
        XR_TYPE_COMPOSITION_LAYER_PROJECTION = 35,
        XR_TYPE_COMPOSITION_LAYER_QUAD = 36,
        XR_TYPE_REFERENCE_SPACE_CREATE_INFO = 37,
        XR_TYPE_ACTION_SPACE_CREATE_INFO = 38,
        XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING = 40,
        XR_TYPE_VIEW_CONFIGURATION_VIEW = 41,
        XR_TYPE_SPACE_LOCATION = 42,
        XR_TYPE_SPACE_VELOCITY = 43,
        XR_TYPE_FRAME_STATE = 44,
        XR_TYPE_VIEW_CONFIGURATION_PROPERTIES = 45,
        XR_TYPE_FRAME_BEGIN_INFO = 46,
        XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW = 48,
        XR_TYPE_EVENT_DATA_EVENTS_LOST = 49,
        XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING = 51,
        XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED = 52,
        XR_TYPE_INTERACTION_PROFILE_STATE = 53,
        XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO = 55,
        XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO = 56,
        XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO = 57,
        XR_TYPE_ACTION_STATE_GET_INFO = 58,
        XR_TYPE_HAPTIC_ACTION_INFO = 59,
        XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO = 60,
        XR_TYPE_ACTIONS_SYNC_INFO = 61,
        XR_TYPE_BOUND_SOURCES_FOR_ACTION_ENUMERATE_INFO = 62,
        XR_TYPE_INPUT_SOURCE_LOCALIZED_NAME_GET_INFO = 63,
        XR_TYPE_COMPOSITION_LAYER_CUBE_KHR = 1000006000,
        XR_TYPE_INSTANCE_CREATE_INFO_ANDROID_KHR = 1000008000,
        XR_TYPE_COMPOSITION_LAYER_DEPTH_INFO_KHR = 1000010000,
        XR_TYPE_VULKAN_SWAPCHAIN_FORMAT_LIST_CREATE_INFO_KHR = 1000014000,
        XR_TYPE_EVENT_DATA_PERF_SETTINGS_EXT = 1000015000,
        XR_TYPE_COMPOSITION_LAYER_CYLINDER_KHR = 1000017000,
        XR_TYPE_COMPOSITION_LAYER_EQUIRECT_KHR = 1000018000,
        XR_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT = 1000019000,
        XR_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT = 1000019001,
        XR_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT = 1000019002,
        XR_TYPE_DEBUG_UTILS_LABEL_EXT = 1000019003,
        XR_TYPE_GRAPHICS_BINDING_OPENGL_WIN32_KHR = 1000023000,
        XR_TYPE_GRAPHICS_BINDING_OPENGL_XLIB_KHR = 1000023001,
        XR_TYPE_GRAPHICS_BINDING_OPENGL_XCB_KHR = 1000023002,
        XR_TYPE_GRAPHICS_BINDING_OPENGL_WAYLAND_KHR = 1000023003,
        XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR = 1000023004,
        XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_KHR = 1000023005,
        XR_TYPE_GRAPHICS_BINDING_OPENGL_ES_ANDROID_KHR = 1000024001,
        XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_ES_KHR = 1000024002,
        XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_ES_KHR = 1000024003,
        XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR = 1000025000,
        XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR = 1000025001,
        XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR = 1000025002,
        XR_TYPE_GRAPHICS_BINDING_D3D11_KHR = 1000027000,
        XR_TYPE_SWAPCHAIN_IMAGE_D3D11_KHR = 1000027001,
        XR_TYPE_GRAPHICS_REQUIREMENTS_D3D11_KHR = 1000027002,
        XR_TYPE_GRAPHICS_BINDING_D3D12_KHR = 1000028000,
        XR_TYPE_SWAPCHAIN_IMAGE_D3D12_KHR = 1000028001,
        XR_TYPE_GRAPHICS_REQUIREMENTS_D3D12_KHR = 1000028002,
        XR_TYPE_SYSTEM_EYE_GAZE_INTERACTION_PROPERTIES_EXT = 1000030000,
        XR_TYPE_EYE_GAZE_SAMPLE_TIME_EXT = 1000030001,
        XR_TYPE_VISIBILITY_MASK_KHR = 1000031000,
        XR_TYPE_EVENT_DATA_VISIBILITY_MASK_CHANGED_KHR = 1000031001,
        XR_TYPE_SESSION_CREATE_INFO_OVERLAY_EXTX = 1000033000,
        XR_TYPE_EVENT_DATA_MAIN_SESSION_VISIBILITY_CHANGED_EXTX = 1000033003,
        XR_TYPE_COMPOSITION_LAYER_COLOR_SCALE_BIAS_KHR = 1000034000,
        XR_TYPE_SPATIAL_ANCHOR_CREATE_INFO_MSFT = 1000039000,
        XR_TYPE_SPATIAL_ANCHOR_SPACE_CREATE_INFO_MSFT = 1000039001,
        XR_TYPE_VIEW_CONFIGURATION_DEPTH_RANGE_EXT = 1000046000,
        XR_TYPE_GRAPHICS_BINDING_EGL_MNDX = 1000048004,
        XR_TYPE_SPATIAL_GRAPH_NODE_SPACE_CREATE_INFO_MSFT = 1000049000,
        XR_TYPE_SYSTEM_HAND_TRACKING_PROPERTIES_EXT = 1000051000,
        XR_TYPE_HAND_TRACKER_CREATE_INFO_EXT = 1000051001,
        XR_TYPE_HAND_JOINTS_LOCATE_INFO_EXT = 1000051002,
        XR_TYPE_HAND_JOINT_LOCATIONS_EXT = 1000051003,
        XR_TYPE_HAND_JOINT_VELOCITIES_EXT = 1000051004,
        XR_TYPE_SYSTEM_HAND_TRACKING_MESH_PROPERTIES_MSFT = 1000052000,
        XR_TYPE_HAND_MESH_SPACE_CREATE_INFO_MSFT = 1000052001,
        XR_TYPE_HAND_MESH_UPDATE_INFO_MSFT = 1000052002,
        XR_TYPE_HAND_MESH_MSFT = 1000052003,
        XR_TYPE_HAND_POSE_TYPE_INFO_MSFT = 1000052004,
        XR_TYPE_SECONDARY_VIEW_CONFIGURATION_SESSION_BEGIN_INFO_MSFT = 1000053000,
        XR_TYPE_SECONDARY_VIEW_CONFIGURATION_STATE_MSFT = 1000053001,
        XR_TYPE_SECONDARY_VIEW_CONFIGURATION_FRAME_STATE_MSFT = 1000053002,
        XR_TYPE_SECONDARY_VIEW_CONFIGURATION_FRAME_END_INFO_MSFT = 1000053003,
        XR_TYPE_SECONDARY_VIEW_CONFIGURATION_LAYER_INFO_MSFT = 1000053004,
        XR_TYPE_SECONDARY_VIEW_CONFIGURATION_SWAPCHAIN_CREATE_INFO_MSFT = 1000053005,
        XR_TYPE_CONTROLLER_MODEL_KEY_STATE_MSFT = 1000055000,
        XR_TYPE_CONTROLLER_MODEL_NODE_PROPERTIES_MSFT = 1000055001,
        XR_TYPE_CONTROLLER_MODEL_PROPERTIES_MSFT = 1000055002,
        XR_TYPE_CONTROLLER_MODEL_NODE_STATE_MSFT = 1000055003,
        XR_TYPE_CONTROLLER_MODEL_STATE_MSFT = 1000055004,
        XR_TYPE_VIEW_CONFIGURATION_VIEW_FOV_EPIC = 1000059000,
        XR_TYPE_HOLOGRAPHIC_WINDOW_ATTACHMENT_MSFT = 1000063000,
        XR_TYPE_COMPOSITION_LAYER_REPROJECTION_INFO_MSFT = 1000066000,
        XR_TYPE_COMPOSITION_LAYER_REPROJECTION_PLANE_OVERRIDE_MSFT = 1000066001,
        XR_TYPE_ANDROID_SURFACE_SWAPCHAIN_CREATE_INFO_FB = 1000070000,
        XR_TYPE_INTERACTION_PROFILE_ANALOG_THRESHOLD_VALVE = 1000079000,
        XR_TYPE_HAND_JOINTS_MOTION_RANGE_INFO_EXT = 1000080000,
        XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR = 1000089000,
        XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR = 1000090000,
        XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR = 1000090001,
        XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR = 1000090003,
        XR_TYPE_COMPOSITION_LAYER_EQUIRECT2_KHR = 1000091000,
        XR_TYPE_SCENE_OBSERVER_CREATE_INFO_MSFT = 1000097000,
        XR_TYPE_SCENE_CREATE_INFO_MSFT = 1000097001,
        XR_TYPE_NEW_SCENE_COMPUTE_INFO_MSFT = 1000097002,
        XR_TYPE_VISUAL_MESH_COMPUTE_LOD_INFO_MSFT = 1000097003,
        XR_TYPE_SCENE_COMPONENTS_MSFT = 1000097004,
        XR_TYPE_SCENE_COMPONENTS_GET_INFO_MSFT = 1000097005,
        XR_TYPE_SCENE_COMPONENT_LOCATIONS_MSFT = 1000097006,
        XR_TYPE_SCENE_COMPONENTS_LOCATE_INFO_MSFT = 1000097007,
        XR_TYPE_SCENE_OBJECTS_MSFT = 1000097008,
        XR_TYPE_SCENE_COMPONENT_PARENT_FILTER_INFO_MSFT = 1000097009,
        XR_TYPE_SCENE_OBJECT_TYPES_FILTER_INFO_MSFT = 1000097010,
        XR_TYPE_SCENE_PLANES_MSFT = 1000097011,
        XR_TYPE_SCENE_PLANE_ALIGNMENT_FILTER_INFO_MSFT = 1000097012,
        XR_TYPE_SCENE_MESHES_MSFT = 1000097013,
        XR_TYPE_SCENE_MESH_BUFFERS_GET_INFO_MSFT = 1000097014,
        XR_TYPE_SCENE_MESH_BUFFERS_MSFT = 1000097015,
        XR_TYPE_SCENE_MESH_VERTEX_BUFFER_MSFT = 1000097016,
        XR_TYPE_SCENE_MESH_INDICES_UINT32_MSFT = 1000097017,
        XR_TYPE_SCENE_MESH_INDICES_UINT16_MSFT = 1000097018,
        XR_TYPE_SERIALIZED_SCENE_FRAGMENT_DATA_GET_INFO_MSFT = 1000098000,
        XR_TYPE_SCENE_DESERIALIZE_INFO_MSFT = 1000098001,
        XR_TYPE_EVENT_DATA_DISPLAY_REFRESH_RATE_CHANGED_FB = 1000101000,
        XR_TYPE_SYSTEM_COLOR_SPACE_PROPERTIES_FB = 1000108000,
        XR_TYPE_BINDING_MODIFICATIONS_KHR = 1000120000,
        XR_TYPE_VIEW_LOCATE_FOVEATED_RENDERING_VARJO = 1000121000,
        XR_TYPE_FOVEATED_VIEW_CONFIGURATION_VIEW_VARJO = 1000121001,
        XR_TYPE_SYSTEM_FOVEATED_RENDERING_PROPERTIES_VARJO = 1000121002,
        XR_TYPE_COMPOSITION_LAYER_DEPTH_TEST_VARJO = 1000122000,
        XR_TYPE_SWAPCHAIN_STATE_ANDROID_SURFACE_DIMENSIONS_FB = 1000161000,
        XR_TYPE_SWAPCHAIN_STATE_SAMPLER_OPENGL_ES_FB = 1000162000,
        XR_TYPE_SWAPCHAIN_STATE_SAMPLER_VULKAN_FB = 1000163000,
        XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR = XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR,
        XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR = XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR,
        XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR = XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR,
        XR_STRUCTURE_TYPE_MAX_ENUM = 0x7FFFFFFF

    ctypedef enum XrFormFactor:
        XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY = 1,
        XR_FORM_FACTOR_HANDHELD_DISPLAY = 2,
        XR_FORM_FACTOR_MAX_ENUM = 0x7FFFFFFF

    ctypedef enum XrViewConfigurationType:
        XR_VIEW_CONFIGURATION_TYPE_PRIMARY_MONO = 1,
        XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO = 2,
        XR_VIEW_CONFIGURATION_TYPE_PRIMARY_QUAD_VARJO = 1000037000,
        XR_VIEW_CONFIGURATION_TYPE_SECONDARY_MONO_FIRST_PERSON_OBSERVER_MSFT = 1000054000,
        XR_VIEW_CONFIGURATION_TYPE_MAX_ENUM = 0x7FFFFFFF

    ctypedef enum XrEnvironmentBlendMode:
        XR_ENVIRONMENT_BLEND_MODE_OPAQUE = 1,
        XR_ENVIRONMENT_BLEND_MODE_ADDITIVE = 2,
        XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND = 3,
        XR_ENVIRONMENT_BLEND_MODE_MAX_ENUM = 0x7FFFFFFF

    ctypedef enum XrReferenceSpaceType:
        XR_REFERENCE_SPACE_TYPE_VIEW = 1,
        XR_REFERENCE_SPACE_TYPE_LOCAL = 2,
        XR_REFERENCE_SPACE_TYPE_STAGE = 3,
        XR_REFERENCE_SPACE_TYPE_UNBOUNDED_MSFT = 1000038000,
        XR_REFERENCE_SPACE_TYPE_COMBINED_EYE_VARJO = 1000121000,
        XR_REFERENCE_SPACE_TYPE_MAX_ENUM = 0x7FFFFFFF

    ctypedef enum XrActionType:
        XR_ACTION_TYPE_BOOLEAN_INPUT = 1,
        XR_ACTION_TYPE_FLOAT_INPUT = 2,
        XR_ACTION_TYPE_VECTOR2F_INPUT = 3,
        XR_ACTION_TYPE_POSE_INPUT = 4,
        XR_ACTION_TYPE_VIBRATION_OUTPUT = 100,
        XR_ACTION_TYPE_MAX_ENUM = 0x7FFFFFFF

    ctypedef enum XrEyeVisibility:
        XR_EYE_VISIBILITY_BOTH = 0,
        XR_EYE_VISIBILITY_LEFT = 1,
        XR_EYE_VISIBILITY_RIGHT = 2,
        XR_EYE_VISIBILITY_MAX_ENUM = 0x7FFFFFFF

    ctypedef enum XrSessionState:
        XR_SESSION_STATE_UNKNOWN = 0,
        XR_SESSION_STATE_IDLE = 1,
        XR_SESSION_STATE_READY = 2,
        XR_SESSION_STATE_SYNCHRONIZED = 3,
        XR_SESSION_STATE_VISIBLE = 4,
        XR_SESSION_STATE_FOCUSED = 5,
        XR_SESSION_STATE_STOPPING = 6,
        XR_SESSION_STATE_LOSS_PENDING = 7,
        XR_SESSION_STATE_EXITING = 8,
        XR_SESSION_STATE_MAX_ENUM = 0x7FFFFFFF

    ctypedef enum XrObjectType:
        XR_OBJECT_TYPE_UNKNOWN = 0,
        XR_OBJECT_TYPE_INSTANCE = 1,
        XR_OBJECT_TYPE_SESSION = 2,
        XR_OBJECT_TYPE_SWAPCHAIN = 3,
        XR_OBJECT_TYPE_SPACE = 4,
        XR_OBJECT_TYPE_ACTION_SET = 5,
        XR_OBJECT_TYPE_ACTION = 6,
        XR_OBJECT_TYPE_DEBUG_UTILS_MESSENGER_EXT = 1000019000,
        XR_OBJECT_TYPE_SPATIAL_ANCHOR_MSFT = 1000039000,
        XR_OBJECT_TYPE_HAND_TRACKER_EXT = 1000051000,
        XR_OBJECT_TYPE_SCENE_OBSERVER_MSFT = 1000097000,
        XR_OBJECT_TYPE_SCENE_MSFT = 1000097001,
        XR_OBJECT_TYPE_MAX_ENUM = 0x7FFFFFFF

    ctypedef XrFlags64 XrInstanceCreateFlags
    ctypedef XrFlags64 XrSessionCreateFlags
    ctypedef XrFlags64 XrSpaceVelocityFlags

    cdef XrSpaceVelocityFlags XR_SPACE_VELOCITY_LINEAR_VALID_BIT
    cdef XrSpaceVelocityFlags XR_SPACE_VELOCITY_ANGULAR_VALID_BIT

    ctypedef XrFlags64 XrSpaceLocationFlags

    cdef XrSpaceLocationFlags XR_SPACE_LOCATION_ORIENTATION_VALID_BIT
    cdef XrSpaceLocationFlags XR_SPACE_LOCATION_POSITION_VALID_BIT
    cdef XrSpaceLocationFlags XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT
    cdef XrSpaceLocationFlags XR_SPACE_LOCATION_POSITION_TRACKED_BIT

    ctypedef XrFlags64 XrSwapchainCreateFlags

    # Flag bits for XrSwapchainCreateFlags
    cdef XrSwapchainCreateFlags XR_SWAPCHAIN_CREATE_PROTECTED_CONTENT_BIT
    cdef XrSwapchainCreateFlags XR_SWAPCHAIN_CREATE_STATIC_IMAGE_BIT

    ctypedef XrFlags64 XrSwapchainUsageFlags

    cdef XrSwapchainUsageFlags XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT
    cdef XrSwapchainUsageFlags XR_SWAPCHAIN_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
    cdef XrSwapchainUsageFlags XR_SWAPCHAIN_USAGE_UNORDERED_ACCESS_BIT
    cdef XrSwapchainUsageFlags XR_SWAPCHAIN_USAGE_TRANSFER_SRC_BIT
    cdef XrSwapchainUsageFlags XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT
    cdef XrSwapchainUsageFlags XR_SWAPCHAIN_USAGE_SAMPLED_BIT
    cdef XrSwapchainUsageFlags XR_SWAPCHAIN_USAGE_MUTABLE_FORMAT_BIT
    cdef XrSwapchainUsageFlags XR_SWAPCHAIN_USAGE_INPUT_ATTACHMENT_BIT_MND

    ctypedef XrFlags64 XrCompositionLayerFlags

    # Flag bits for XrCompositionLayerFlags
    cdef XrCompositionLayerFlags XR_COMPOSITION_LAYER_CORRECT_CHROMATIC_ABERRATION_BIT
    cdef XrCompositionLayerFlags XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT
    cdef XrCompositionLayerFlags XR_COMPOSITION_LAYER_UNPREMULTIPLIED_ALPHA_BIT

    ctypedef XrFlags64 XrViewStateFlags;

    # Flag bits for XrViewStateFlags
    cdef XrViewStateFlags XR_VIEW_STATE_ORIENTATION_VALID_BIT
    cdef XrViewStateFlags XR_VIEW_STATE_POSITION_VALID_BIT
    cdef XrViewStateFlags XR_VIEW_STATE_ORIENTATION_TRACKED_BIT
    cdef XrViewStateFlags XR_VIEW_STATE_POSITION_TRACKED_BIT

    ctypedef XrFlags64 XrInputSourceLocalizedNameFlags;

    # Flag bits for XrInputSourceLocalizedNameFlags
    cdef XrInputSourceLocalizedNameFlags XR_INPUT_SOURCE_LOCALIZED_NAME_USER_PATH_BIT
    cdef XrInputSourceLocalizedNameFlags XR_INPUT_SOURCE_LOCALIZED_NAME_INTERACTION_PROFILE_BIT
    cdef XrInputSourceLocalizedNameFlags XR_INPUT_SOURCE_LOCALIZED_NAME_COMPONENT_BIT

    ctypedef struct XrApiLayerProperties:
        XrStructureType type
        void* next
        char layerName[XR_MAX_API_LAYER_NAME_SIZE];
        XrVersion specVersion
        uint32_t layerVersion
        char description[XR_MAX_API_LAYER_DESCRIPTION_SIZE]

    ctypedef struct XrExtensionProperties:
        XrStructureType type
        void* next;
        char extensionName[XR_MAX_EXTENSION_NAME_SIZE]
        uint32_t extensionVersion

    ctypedef struct XrApplicationInfo:
        char applicationName[XR_MAX_APPLICATION_NAME_SIZE]
        uint32_t applicationVersion
        char engineName[XR_MAX_ENGINE_NAME_SIZE]
        uint32_t engineVersion
        XrVersion apiVersion

    ctypedef struct XrInstanceCreateInfo:
        XrStructureType type;
        const void* next
        XrInstanceCreateFlags createFlags
        XrApplicationInfo applicationInfo
        uint32_t enabledApiLayerCount
        const char* const* enabledApiLayerNames
        uint32_t enabledExtensionCount
        const char* const* enabledExtensionNames

    ctypedef struct XrInstanceProperties:
        XrStructureType type
        void* next
        XrVersion runtimeVersion
        char runtimeName[XR_MAX_RUNTIME_NAME_SIZE]

    ctypedef struct XrEventDataBuffer:
        XrStructureType type
        const void* next
        uint8_t varying[4000]

    ctypedef struct XrSystemGetInfo:
        XrStructureType type
        const void* next
        XrFormFactor formFactor

    ctypedef struct XrSystemGraphicsProperties:
        uint32_t maxSwapchainImageHeight
        uint32_t maxSwapchainImageWidth
        uint32_t maxLayerCount

    ctypedef struct XrSystemTrackingProperties:
        XrBool32 orientationTracking
        XrBool32 positionTracking

    ctypedef struct XrSystemProperties:
        XrStructureType type
        void* next
        XrSystemId systemId
        uint32_t vendorId
        char systemName[XR_MAX_SYSTEM_NAME_SIZE]
        XrSystemGraphicsProperties graphicsProperties
        XrSystemTrackingProperties trackingProperties

    ctypedef struct XrSessionCreateInfo:
        XrStructureType type
        const void* next
        XrSessionCreateFlags createFlags
        XrSystemId systemId

    ctypedef struct XrVector3f:
        float x
        float y
        float z

    # XrSpaceVelocity extends XrSpaceLocation
    ctypedef struct XrSpaceVelocity:
        XrStructureType type
        void* next
        XrSpaceVelocityFlags velocityFlags
        XrVector3f linearVelocity
        XrVector3f angularVelocity

    ctypedef struct XrQuaternionf:
        float x
        float y
        float z
        float w

    ctypedef struct XrPosef:
        XrQuaternionf orientation
        XrVector3f position

    ctypedef struct XrReferenceSpaceCreateInfo:
        XrStructureType type
        const void* next
        XrReferenceSpaceType referenceSpaceType
        XrPosef poseInReferenceSpace

    ctypedef struct XrExtent2Df:
        float width
        float height

    ctypedef struct XrActionSpaceCreateInfo:
        XrStructureType type
        const void* next
        XrAction action
        XrPath subactionPath
        XrPosef poseInActionSpace

    ctypedef struct XrSpaceLocation:
        XrStructureType type
        void* next
        XrSpaceLocationFlags locationFlags
        XrPosef pose

    ctypedef struct XrViewConfigurationProperties:
        XrStructureType type
        void* next
        XrViewConfigurationType viewConfigurationType
        XrBool32 fovMutable

    ctypedef struct XrViewConfigurationView:
        XrStructureType type
        void* next
        uint32_t recommendedImageRectWidth;
        uint32_t maxImageRectWidt
        uint32_t recommendedImageRectHeight
        uint32_t maxImageRectHeight
        uint32_t recommendedSwapchainSampleCount
        uint32_t maxSwapchainSampleCount

    ctypedef struct XrSwapchainCreateInfo:
        XrStructureType type
        const void* next
        XrSwapchainCreateFlags createFlags
        XrSwapchainUsageFlags usageFlags
        int64_t format
        uint32_t sampleCount
        uint32_t width
        uint32_t height
        uint32_t faceCount
        uint32_t arraySize
        uint32_t mipCount

    ctypedef struct XrSwapchainImageBaseHeader:
        XrStructureType type
        void* next
    
    ctypedef struct XrSwapchainImageAcquireInfo:
        XrStructureType type
        const void* next
    
    ctypedef struct XrSwapchainImageWaitInfo:
        XrStructureType type
        const void* next
        XrDuration timeout
    
    ctypedef struct XrSwapchainImageReleaseInfo:
        XrStructureType type
        const void* next
    
    ctypedef struct XrSessionBeginInfo:
        XrStructureType type
        const void* next
        XrViewConfigurationType     primaryViewConfigurationType
    
    ctypedef struct XrFrameWaitInfo:
        XrStructureType type
        const void* next
    
    ctypedef struct XrFrameState:
        XrStructureType type
        void* next
        XrTime predictedDisplayTime
        XrDuration  predictedDisplayPeriod
        XrBool32 shouldRender
    
    ctypedef struct XrFrameBeginInfo:
        XrStructureType type
        const void* next
    
    ctypedef struct  XrCompositionLayerBaseHeader:
        XrStructureType type
        const void* next
        XrCompositionLayerFlags layerFlags
        XrSpace space
   
    ctypedef struct XrFrameEndInfo:
        XrStructureType type
        const void*  next
        XrTime displayTime
        XrEnvironmentBlendMode environmentBlendMode
        uint32_t layerCount
        const XrCompositionLayerBaseHeader* const* layers
    
    ctypedef struct XrViewLocateInfo:
        XrStructureType type
        const void* next
        XrViewConfigurationType viewConfigurationType
        XrTime displayTime
        XrSpace space
    
    ctypedef struct XrViewState:
        XrStructureType type
        void* next
        XrViewStateFlags viewStateFlags
    
    ctypedef struct XrFovf:
        float angleLeft
        float angleRight
        float angleUp
        float angleDown
    
    ctypedef struct XrView:
        XrStructureType type
        void* next
        XrPosef pose
        XrFovf fov
    
    ctypedef struct XrActionSetCreateInfo:
        XrStructureType type
        const void* next
        char actionSetName[XR_MAX_ACTION_SET_NAME_SIZE]
        char localizedActionSetName[XR_MAX_LOCALIZED_ACTION_SET_NAME_SIZE]
        uint32_t priority
    
    ctypedef struct XrActionCreateInfo:
        XrStructureType type
        const void* next
        char actionName[XR_MAX_ACTION_NAME_SIZE]
        XrActionType actionType
        uint32_t countSubactionPaths
        const XrPath* subactionPaths
        char localizedActionName[XR_MAX_LOCALIZED_ACTION_NAME_SIZE]
    
    ctypedef struct XrActionSuggestedBinding:
        XrAction    action
        XrPath      binding
    
    ctypedef struct XrInteractionProfileSuggestedBinding:
        XrStructureType                    type
        const void*            next
        XrPath                             interactionProfile
        uint32_t                           countSuggestedBindings
        const XrActionSuggestedBinding*    suggestedBindings
    
    ctypedef struct XrSessionActionSetsAttachInfo:
        XrStructureType             type
        const void*     next
        uint32_t                    countActionSets
        const XrActionSet*          actionSets
    
    ctypedef struct XrInteractionProfileState:
        XrStructureType       type
        void*     next
        XrPath                interactionProfile
    
    ctypedef struct XrActionStateGetInfo:
        XrStructureType             type
        const void*     next
        XrAction                    action
        XrPath                      subactionPath
    
    ctypedef struct XrActionStateBoolean:
        XrStructureType       type
        void*     next
        XrBool32              currentState
        XrBool32              changedSinceLastSync
        XrTime                lastChangeTime
        XrBool32              isActive
    
    ctypedef struct XrActionStateFloat:
        XrStructureType       type
        void*     next
        float                 currentState
        XrBool32              changedSinceLastSync
        XrTime                lastChangeTime
        XrBool32              isActive
    
    ctypedef struct XrVector2f:
        float    x
        float    y
    
    ctypedef struct XrActionStateVector2f:
        XrStructureType       type
        void*     next
        XrVector2f            currentState
        XrBool32              changedSinceLastSync
        XrTime                lastChangeTime
        XrBool32              isActive
    
    ctypedef struct XrActionStatePose:
        XrStructureType       type
        void*     next
        XrBool32              isActive
    
    ctypedef struct XrActiveActionSet:
        XrActionSet    actionSet
        XrPath         subactionPath
    
    ctypedef struct XrActionsSyncInfo:
        XrStructureType             type
        const void*     next
        uint32_t                    countActiveActionSets
        const XrActiveActionSet*    activeActionSets
    
    ctypedef struct XrBoundSourcesForActionEnumerateInfo:
        XrStructureType             type
        const void*     next
        XrAction                    action
    
    ctypedef struct XrInputSourceLocalizedNameGetInfo:
        XrStructureType                    type
        const void*            next
        XrPath                             sourcePath
        XrInputSourceLocalizedNameFlags    whichComponents
    
    ctypedef struct XrHapticActionInfo:
        XrStructureType             type
        const void*     next
        XrAction                    action
        XrPath                      subactionPath
    
    ctypedef struct  XrHapticBaseHeader:
        XrStructureType             type
        const void*     next

    ctypedef struct _XrBaseInStructure
    ctypedef struct  XrBaseInStructure:
        XrStructureType type
        _XrBaseInStructure* next

    ctypedef struct _XrBaseOutStructure
    ctypedef struct  XrBaseOutStructure:
        XrStructureType type
        _XrBaseOutStructure* next
    
    ctypedef struct XrOffset2Di:
        int32_t x
        int32_t y
    
    ctypedef struct XrExtent2Di:
        int32_t width
        int32_t height
    
    ctypedef struct XrRect2Di:
        XrOffset2Di offset
        XrExtent2Di extent
    
    ctypedef struct XrSwapchainSubImage:
        XrSwapchain swapchain
        XrRect2Di imageRect
        uint32_t imageArrayIndex
    
    ctypedef struct XrCompositionLayerProjectionView:
        XrStructureType type
        const void* next
        XrPosef pose
        XrFovf fov
        XrSwapchainSubImage subImage
    
    ctypedef struct XrCompositionLayerProjection:
        XrStructureType type
        const void* next
        XrCompositionLayerFlags layerFlags
        XrSpace space
        uint32_t viewCount
        const XrCompositionLayerProjectionView* views
    
    ctypedef struct XrCompositionLayerQuad:
        XrStructureType type
        const void* next
        XrCompositionLayerFlags layerFlags
        XrSpace space
        XrEyeVisibility eyeVisibility
        XrSwapchainSubImage subImage
        XrPosef pose
        XrExtent2Df size
   
    ctypedef struct  XrEventDataBaseHeader:
        XrStructureType type
        const void* next
    
    ctypedef struct XrEventDataEventsLost:
        XrStructureType type
        const void* next
        uint32_t lostEventCount
    
    ctypedef struct XrEventDataInstanceLossPending:
        XrStructureType type
        const void* next
        XrTime lossTime
    
    ctypedef struct XrEventDataSessionStateChanged:
        XrStructureType  type
        const void* next
        XrSession session
        XrSessionState state
        XrTime time
    
    ctypedef struct XrEventDataReferenceSpaceChangePending:
        XrStructureType type
        const void* next
        XrSession session
        XrReferenceSpaceType referenceSpaceType
        XrTime changeTime
        XrBool32 poseValid
        XrPosef poseInPreviousSpace
    
    ctypedef struct XrEventDataInteractionProfileChanged:
        XrStructureType type
        const void* next
        XrSession session
    
    ctypedef struct XrHapticVibration:
        XrStructureType type
        const void* next
        XrDuration duration
        float frequency
        float amplitude
    
    ctypedef struct XrOffset2Df:
        float x
        float y
    
    ctypedef struct XrRect2Df:
        XrOffset2Df    offset
        XrExtent2Df    extent
    
    ctypedef struct XrVector4f:
        float x
        float y
        float z
        float w
    
    ctypedef struct XrColor4f:
        float r
        float g
        float b
        float a
