# distutils: language=c++
#  =============================================================================
#  openhmd.pxd - Cython definitions for `openhmd.h`
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

cdef extern from "openhmd.h":
    cdef int OHMD_STR_SIZE = 256

    ctypedef enum ohmd_status:
        OHMD_S_OK = 0
        OHMD_S_UNKNOWN_ERROR = -1
        OHMD_S_INVALID_PARAMETER = -2
        OHMD_S_UNSUPPORTED = -3
        OHMD_S_INVALID_OPERATION = -4
        OHMD_S_USER_RESERVED = -16384

    ctypedef enum ohmd_string_value:
        OHMD_VENDOR = 0
        OHMD_PRODUCT = 1
        OHMD_PATH = 2

    ctypedef enum ohmd_string_description:
        OHMD_GLSL_DISTORTION_VERT_SRC = 0
        OHMD_GLSL_DISTORTION_FRAG_SRC = 1
        OHMD_GLSL_330_DISTORTION_VERT_SRC = 2
        OHMD_GLSL_330_DISTORTION_FRAG_SRC = 3
        OHMD_GLSL_ES_DISTORTION_VERT_SRC = 4
        OHMD_GLSL_ES_DISTORTION_FRAG_SRC = 5

    ctypedef enum ohmd_control_hint:
        OHMD_GENERIC = 0
        OHMD_TRIGGER = 1
        OHMD_TRIGGER_CLICK = 2
        OHMD_SQUEEZE = 3
        OHMD_MENU = 4
        OHMD_HOME = 5
        OHMD_ANALOG_X = 6
        OHMD_ANALOG_Y = 7
        OHMD_ANALOG_PRESS = 8
        OHMD_BUTTON_A = 9
        OHMD_BUTTON_B = 10
        OHMD_BUTTON_X = 11
        OHMD_BUTTON_Y = 12
        OHMD_VOLUME_PLUS = 13
        OHMD_VOLUME_MINUS = 14
        OHMD_MIC_MUTE = 15

    ctypedef enum ohmd_float_value:
        OHMD_ROTATION_QUAT = 1
        OHMD_LEFT_EYE_GL_MODELVIEW_MATRIX = 2
        OHMD_RIGHT_EYE_GL_MODELVIEW_MATRIX = 3
        OHMD_LEFT_EYE_GL_PROJECTION_MATRIX = 4
        OHMD_RIGHT_EYE_GL_PROJECTION_MATRIX = 5
        OHMD_POSITION_VECTOR = 6
        OHMD_SCREEN_HORIZONTAL_SIZE = 7
        OHMD_SCREEN_VERTICAL_SIZE = 8
        OHMD_LENS_HORIZONTAL_SEPARATION = 9
        OHMD_LENS_VERTICAL_POSITION = 10
        OHMD_LEFT_EYE_FOV = 11
        OHMD_LEFT_EYE_ASPECT_RATIO = 12
        OHMD_RIGHT_EYE_FOV = 13
        OHMD_RIGHT_EYE_ASPECT_RATIO = 14
        OHMD_EYE_IPD = 15
        OHMD_PROJECTION_ZFAR = 16
        OHMD_PROJECTION_ZNEAR = 17
        OHMD_DISTORTION_K = 18
        OHMD_EXTERNAL_SENSOR_FUSION = 19
        OHMD_UNIVERSAL_DISTORTION_K = 20
        OHMD_UNIVERSAL_ABERRATION_K = 21
        OHMD_CONTROLS_STATE = 22

    ctypedef enum ohmd_int_value:
        OHMD_SCREEN_HORIZONTAL_RESOLUTION = 0
        OHMD_SCREEN_VERTICAL_RESOLUTION = 1
        OHMD_DEVICE_CLASS = 2
        OHMD_DEVICE_FLAGS = 3
        OHMD_CONTROL_COUNT = 4
        OHMD_CONTROLS_HINTS = 5
        OHMD_CONTROLS_TYPES = 6

    ctypedef enum ohmd_data_value:
        OHMD_DRIVER_DATA = 0
        OHMD_DRIVER_PROPERTIES = 1

    ctypedef enum ohmd_int_settings:
        OHMD_IDS_AUTOMATIC_UPDATE = 0

    ctypedef enum ohmd_device_class:
        OHMD_DEVICE_CLASS_HMD = 0
        OHMD_DEVICE_CLASS_CONTROLLER = 1
        OHMD_DEVICE_CLASS_GENERIC_TRACKER = 2

    ctypedef enum ohmd_device_flags:
        OHMD_DEVICE_FLAGS_NULL_DEVICE = 1
        OHMD_DEVICE_FLAGS_POSITIONAL_TRACKING = 2
        OHMD_DEVICE_FLAGS_ROTATIONAL_TRACKING = 3
        OHMD_DEVICE_FLAGS_LEFT_CONTROLLER = 4
        OHMD_DEVICE_FLAGS_RIGHT_CONTROLLER = 5

    ctypedef struct ohmd_context:
        pass

    ctypedef struct ohmd_device:
        pass

    ctypedef struct ohmd_device_settings:
        pass

    cdef ohmd_context* ohmd_ctx_create()
    cdef void ohmd_ctx_destroy(ohmd_context* ctx)
    cdef const char* ohmd_ctx_get_error(ohmd_context* ctx)
    cdef void ohmd_ctx_update(ohmd_context* ctx)
    cdef int ohmd_ctx_probe(ohmd_context* ctx)
    cdef int ohmd_gets(ohmd_string_description type, const char** out)
    cdef const char* ohmd_list_gets(ohmd_context* ctx, int index, ohmd_string_value type)
    cdef int ohmd_list_geti(ohmd_context* ctx, int index, ohmd_int_value type, int*out)
    cdef ohmd_device* ohmd_list_open_device(ohmd_context* ctx, int index)
    cdef ohmd_device* ohmd_list_open_device_s(ohmd_context* ctx, int index, ohmd_device_settings* settings)
    cdef ohmd_status ohmd_device_settings_seti(ohmd_device_settings* settings, ohmd_int_settings key, const int*val)
    cdef ohmd_device_settings* ohmd_device_settings_create(ohmd_context* ctx)
    cdef void ohmd_device_settings_destroy(ohmd_device_settings* settings)
    cdef int ohmd_close_device(ohmd_device* device)
    cdef int ohmd_device_getf(ohmd_device*, ohmd_float_value, float*)
    cdef int ohmd_device_setf(ohmd_device*, ohmd_float_value, const float*)
    cdef int ohmd_device_geti(ohmd_device*, ohmd_int_value, int* out)
    cdef int ohmd_device_seti(ohmd_device*, ohmd_int_value, const int*)
    cdef int ohmd_device_set_data(ohmd_device*, ohmd_data_value, const void*)
    cdef void ohmd_get_version(int* out_major, int* out_minor, int* out_patch)
    cdef ohmd_status ohmd_require_version(int major, int minor, int patch)
    cdef void ohmd_sleep(double time)


# custom descriptors and enums
ctypedef enum ohmdEyeType:
    OHMD_EYE_LEFT = 0
    OHMD_EYE_RIGHT = 1
    OHMD_EYE_COUNT = 2

ctypedef enum ohmdHandType:
    OHMD_HAND_LEFT = 0
    OHMD_HAND_RIGHT = 1
    OHMD_HAND_COUNT = 2

ctypedef struct ohmdDeviceInfo:
    int deviceIdx
    const char* vendorName
    const char* productName
    char* path
    int deviceClass
    int deviceFlags
    int isOpened

ctypedef struct ohmdDisplayInfo:
    int deviceIdx
    float ipd
    float[2] screenResolution
    float[2] screenSize
    float[2] eyeFov
    float[2] eyeAspect

ctypedef struct ohmdControllerInfo:
    int deviceIdx
    int controlCount

