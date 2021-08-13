# distutils: language=c++
#  =============================================================================
#  _openxr.pyx - Python Interface Module for OpenXR
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
"""This extension module exposes the OpenXR driver interface.
"""

# ------------------------------------------------------------------------------
# Module information
#
__author__ = "Matthew D. Cutone"
__credits__ = ["Matthew D. Cutone"]
__copyright__ = "Copyright 2021 Matthew D. Cutone"
__license__ = "MIT"
__version__ = '0.2.4rc2'
__status__ = "Stable"
__maintainer__ = "Matthew D. Cutone"
__email__ = "mcutone@opensciencetools.org"

__all__ = [
    'XR_CURRENT_API_VERSION',
    'XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY',
    'XR_FORM_FACTOR_HANDHELD_DISPLAY',
    'XR_NULL_SYSTEM_ID',
    'XR_VIEW_CONFIGURATION_TYPE_PRIMARY_MONO',
    'XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO',
    'XR_REFERENCE_SPACE_TYPE_VIEW',
    'XR_REFERENCE_SPACE_TYPE_LOCAL',
    'XR_REFERENCE_SPACE_TYPE_STAGE',
    'XR_SWAPCHAIN_USAGE_SAMPLED_BIT',
    'XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT',
    'OpenXRPose',
    'OpenXRApplicationInfo',
    'OpenXRSystemInfo',
    'OpenXRViewConfigInfo',
    'createInstance',
    'instanceStarted',
    'destroyInstance',
    'findSystem',
    'getViewConfigurations',
    'getGraphicsRequirementsOpenGL',
    'createGraphicsBindingOpenGLWin32',
    'createSession',
    'createSwapChainColorOpenGL',
    'createSpace',
    'destroySpace'
]

# ------------------------------------------------------------------------------
# Imports
#

from libc.stdint cimport uint32_t, uint16_t, uint64_t, uintptr_t, int64_t
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from . cimport openxr
cimport numpy as np
import numpy as np
np.import_array()
import ctypes


# ------------------------------------------------------------------------------
# Module level constants
#

cdef openxr.XrInstance _ptrInstance = NULL  # pointer to instance
cdef openxr.XrSession _ptrSession = NULL  # pointer to session

# system to use
cdef openxr.XrSystemProperties _system
_system.type = openxr.XR_TYPE_SYSTEM_PROPERTIES
_system.next = NULL

# view information for each system
cdef uint32_t _systemViewCount = 0
cdef openxr.XrViewConfigurationView* _systemViewConfigs = NULL

# Graphics binding data. We will need handles for the window and GL context from
# the application before the window is initialized.
cdef openxr.XrGraphicsBindingOpenGLWin32KHR _gfxBinding
_gfxBinding.type = openxr.XR_TYPE_GRAPHICS_BINDING_OPENGL_WIN32_KHR  # windows only
_gfxBinding.next = NULL
_gfxBinding.hDC = NULL  # device context for the window
_gfxBinding.hGLRC = NULL  # GL context handle

# Swapchains for color and depth buffers, allocated when creating a session
cdef uint32_t numSupportedSwapChainFormats = 0
cdef int64_t* supportedSwapChainFormats = NULL
cdef int64_t colorSwapChainFormat = 0
cdef int64_t depthSwapChainFormat = 0
cdef openxr.XrSwapchain* colorSwapChains = NULL
cdef openxr.XrSwapchain* depthSwapChains = NULL
cdef uint32_t* colorSwapChainLengths = NULL
cdef uint32_t* depthSwapChainLengths = NULL
cdef openxr.XrSwapchainImageOpenGLKHR** colorSwapChainImagesGL = NULL
cdef openxr.XrSwapchainImageOpenGLKHR** depthSwapChainImagesGL = NULL

# reference space data, just one like LibOVR
cdef openxr.XrSpace _refSpace = NULL

# Python accessible constants
XR_CURRENT_API_VERSION = openxr.XR_CURRENT_API_VERSION
XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY = openxr.XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY
XR_FORM_FACTOR_HANDHELD_DISPLAY = openxr.XR_FORM_FACTOR_HANDHELD_DISPLAY
XR_NULL_SYSTEM_ID = openxr.XR_NULL_SYSTEM_ID
XR_MIN_COMPOSITION_LAYERS_SUPPORTED = openxr.XR_MIN_COMPOSITION_LAYERS_SUPPORTED
XR_VIEW_CONFIGURATION_TYPE_PRIMARY_MONO = openxr.XR_VIEW_CONFIGURATION_TYPE_PRIMARY_MONO
XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO = openxr.XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO
XR_REFERENCE_SPACE_TYPE_VIEW = openxr.XR_REFERENCE_SPACE_TYPE_VIEW
XR_REFERENCE_SPACE_TYPE_LOCAL = openxr.XR_REFERENCE_SPACE_TYPE_LOCAL
XR_REFERENCE_SPACE_TYPE_STAGE = openxr.XR_REFERENCE_SPACE_TYPE_STAGE
XR_SWAPCHAIN_USAGE_SAMPLED_BIT = openxr.XR_SWAPCHAIN_USAGE_SAMPLED_BIT
XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT = openxr.XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT

# ------------------------------------------------------------------------------
# Helper functions
#

cdef char* str2bytes(str strIn):
    """Convert UTF-8 encoded strings to bytes."""
    py_bytes = strIn.encode('UTF-8')
    cdef char* to_return = py_bytes

    return to_return


cdef str bytes2str(char* bytesIn):
    """Convert UTF-8 encoded strings to bytes."""
    return bytesIn.decode('UTF-8')


cdef xr_get_version(openxr.XrVersion version):
    """Convert the version into major, minor and patch format.
    
    Parameters
    ----------
    version : XrVersion
        Version in OpenXR format.
    
    """
    cdef uint16_t major = (version >> 48) & 0xffffULL
    cdef uint16_t minor = (version >> 32) & 0xffffULL
    cdef uint32_t patch = version & 0xffffffffULL

    return major, minor, patch


cdef np.npy_intp[1] VEC3_SHAPE = [3]
cdef np.npy_intp[1] QUAT_SHAPE = [4]

cdef np.ndarray _wrap_XrVector3f_as_ndarray(openxr.XrVector3f* prtVec):
    """Wrap an XrVector3f object with a NumPy array."""
    return np.PyArray_SimpleNewFromData(
        1, VEC3_SHAPE, np.NPY_FLOAT32, <void*>prtVec)


cdef np.ndarray _wrap_XrQuaternionf_as_ndarray(openxr.XrQuaternionf* prtVec):
    """Wrap an XrQuaternionf object with a NumPy array."""
    return np.PyArray_SimpleNewFromData(
        1, QUAT_SHAPE, np.NPY_FLOAT32, <void*>prtVec)


# ------------------------------------------------------------------------------
# Exceptions
#

class OpenXRError(BaseException):
    """Base exception for OpenXR related errors."""
    pass


class OpenXRValidationFailureError(OpenXRError):
    pass


class OpenXRRuntimeFailureError(OpenXRError):
    pass


class OpenXROutOfMemoryError(OpenXRError):
    pass


class OpenXRHandleInvalidError(OpenXRError):
    pass


class OpenXRInstanceLostError(OpenXRError):
    pass


class OpenXRSystemInvalidError(OpenXRError):
    pass


class OpenXRLimitReachedError(OpenXRError):
    pass


class OpenXRRuntimeUnavailableError(OpenXRError):
    pass


class OpenXRNameInvalidError(OpenXRError):
    pass


class OpenXRNameInitializationFailedError(OpenXRError):
    pass


class OpenXRNameExtensionNotPresentError(OpenXRError):
    pass


class OpenXRApiVersionNotSupportedError(OpenXRError):
    pass


class OpenXRApiLayerNotPresentError(OpenXRError):
    pass


class OpenXRFormFactorUnsupportedError(OpenXRError):
    pass


class OpenXRFormFactorUnavailableError(OpenXRError):
    pass


class OpenXRSizeInsufficientError(OpenXRError):
    pass


class OpenXRViewConfigurationNotSupportedError(OpenXRError):
    pass


class OpenXRFunctionUnsupportedError(OpenXRError):
    pass


class OpenXRGraphicsRequirementsCallMissingError(OpenXRError):
    pass


class OpenXRGraphicsDeviceInvalidError(OpenXRError):
    pass



# lookup table of exceptions
cdef dict openxr_error_lut = {
    openxr.XR_ERROR_VALIDATION_FAILURE: OpenXRValidationFailureError,
    openxr.XR_ERROR_RUNTIME_FAILURE: OpenXRRuntimeFailureError,
    openxr.XR_ERROR_OUT_OF_MEMORY: OpenXROutOfMemoryError,
    openxr.XR_ERROR_HANDLE_INVALID: OpenXRHandleInvalidError,
    openxr.XR_ERROR_INSTANCE_LOST: OpenXRInstanceLostError,
    openxr.XR_ERROR_SYSTEM_INVALID: OpenXRSystemInvalidError,
    openxr.XR_ERROR_LIMIT_REACHED: OpenXRLimitReachedError,
    openxr.XR_ERROR_RUNTIME_UNAVAILABLE: OpenXRRuntimeUnavailableError,
    openxr.XR_ERROR_NAME_INVALID: OpenXRNameInvalidError,
    openxr.XR_ERROR_INITIALIZATION_FAILED: OpenXRNameInitializationFailedError,
    openxr.XR_ERROR_EXTENSION_NOT_PRESENT: OpenXRNameExtensionNotPresentError,
    openxr.XR_ERROR_API_VERSION_UNSUPPORTED: OpenXRApiVersionNotSupportedError,
    openxr.XR_ERROR_API_LAYER_NOT_PRESENT: OpenXRApiLayerNotPresentError,
    openxr.XR_ERROR_FORM_FACTOR_UNSUPPORTED: OpenXRFormFactorUnsupportedError,
    openxr.XR_ERROR_FORM_FACTOR_UNAVAILABLE: OpenXRFormFactorUnavailableError,
    openxr.XR_ERROR_SIZE_INSUFFICIENT: OpenXRSizeInsufficientError,
    openxr.XR_ERROR_VIEW_CONFIGURATION_TYPE_UNSUPPORTED:
        OpenXRViewConfigurationNotSupportedError,
    openxr.XR_ERROR_FUNCTION_UNSUPPORTED: OpenXRFunctionUnsupportedError,
    openxr.XR_ERROR_GRAPHICS_REQUIREMENTS_CALL_MISSING: OpenXRGraphicsRequirementsCallMissingError,
    openxr.XR_ERROR_GRAPHICS_DEVICE_INVALID: OpenXRGraphicsDeviceInvalidError
}


def checkResult(int result):
    """Check the result of an OpenXR API return.
    
    If the result is anything less than `XR_SUCCESS`, an exception will be
    raised. Users should catch these exceptions to handle abnormal results from 
    the OpenXR API.
    
    Parameters
    ----------
    result : openxr.XrResult
        Returned value from an OpenXR API call.
    
    """
    if result >= openxr.XR_SUCCESS:
        return

    try:
        raise openxr_error_lut[result]()
    except KeyError:
        raise RuntimeError(
            'Caught unhandled exception ({}).'.format(<int>result))


# ------------------------------------------------------------------------------
# Classes and Functions to interface with OpenXR
#

cdef class OpenXRApplicationInfo:
    """Application information descriptor.

    This descriptor contains information about the application which can be
    passed to :func:`createInstance`.

    Parameters
    ----------
    applicationName : str
        Application name. Cannot be empty and must be shorter than
        ``XR_MAX_APPLICATION_NAME_SIZE``.
    applicationVersion : int
        Application version as an unsigned integer provided by the developer.
    engineName : str
        Name of the engine in use by the application. This is optional, leave
        empty if you wish not to specify an engine. Must be shorter than
        ``XR_MAX_ENGINE_NAME_SIZE``.
    engineVersion : int
        Engine version as an unsigned integer.
    apiVersion : int
        OpenXR API version this application will run. Uses
        ``XR_CURRENT_API_VERSION`` by default, which is the version PsychXR
        was built against.

    """
    cdef openxr.XrApplicationInfo c_data

    def __init__(self,
                 applicationName="OpenXR Application",
                 applicationVersion=1,
                 engineName="PsychXR",
                 engineVersion=0,
                 apiVersion=XR_CURRENT_API_VERSION):

        self.applicationName = applicationName
        self.applicationVersion = applicationVersion
        self.engineName = engineName
        self.engineVersion = engineVersion
        self.apiVersion = apiVersion

    def __repr__(self):
        return (
            f"OpenXRApplicationInfo("
            f"applicationName='{self.applicationName}', "
            f"applicationVersion={self.applicationVersion}, "
            f"engineName='{self.engineName}', "
            f"engineVersion={self.engineVersion}, "
            f"apiVersion={self.apiVersion})")

    def __cinit__(self, *args, **kwargs):
        pass

    @property
    def applicationName(self):
        """Application name (`str`). Must not be empty when passed to
        `createInstance`.
        """
        return bytes2str(self.c_data.applicationName)

    @applicationName.setter
    def applicationName(self, value):
        if not isinstance(value, str):  # check type
            raise TypeError(
                'Property `XrApplicationInfo.applicationName` must be type '
                '`str`.')

        if len(value) > openxr.XR_MAX_APPLICATION_NAME_SIZE:
            raise ValueError(
                'Value for `XrApplicationInfo.applicationName` must have '
                'length <{}'.format(openxr.XR_MAX_APPLICATION_NAME_SIZE))

        self.c_data.applicationName = str2bytes(value)

    @property
    def applicationVersion(self):
        """Application version number (`int`).
        """
        return int(self.c_data.applicationVersion)

    @applicationVersion.setter
    def applicationVersion(self, value):
        if not isinstance(value, int):  # check type
            raise TypeError(
                'Property `XrApplicationInfo.applicationVersion` must be type '
                '`int`.')

        self.c_data.applicationVersion = <int>value

    @property
    def engineName(self):
        """Engine name (`str`). Default is "PsychXR".
        """
        return bytes2str(self.c_data.engineName)

    @engineName.setter
    def engineName(self, value):
        if not isinstance(value, str):  # check type
            raise TypeError(
                'Property `XrApplicationInfo.engineName` must be type '
                '`str`.')

        if len(value) > openxr.XR_MAX_ENGINE_NAME_SIZE:
            raise ValueError(
                'Value for `XrApplicationInfo.engineName` must have '
                'length <{}'.format(openxr.XR_MAX_ENGINE_NAME_SIZE))

        self.c_data.engineName = str2bytes(value)

    @property
    def engineVersion(self):
        """Engine version number (`int`).
        """
        return int(self.c_data.engineVersion)

    @engineVersion.setter
    def engineVersion(self, value):
        if not isinstance(value, int):  # check type
            raise TypeError(
                'Property `XrApplicationInfo.engineVersion` must be type '
                '`int`.')

        self.c_data.engineVersion = <int>value

    @property
    def apiVersion(self):
        """OpenXR API version number (`int`).
        """
        return int(self.c_data.apiVersion)

    @apiVersion.setter
    def apiVersion(self, value):
        if not isinstance(value, int):  # check type
            raise TypeError(
                'Property `XrApplicationInfo.apiVersion` must be type '
                '`int`.')

        self.c_data.apiVersion = <openxr.XrVersion>value


cdef class OpenXRSystemInfo:
    """Descriptor for a system (head-mounted or handheld, etc.) available to
    OpenXR.

    These are instanced and returned when calling :func:`getSystem`, users
    should not instance this themselves during regular use.

    """
    cdef openxr.XrSystemProperties c_data

    def __init__(self,
                 systemId=XR_NULL_SYSTEM_ID,
                 vendorId=0,
                 systemName="",
                 maxSwapchainImageHeight=0,
                 maxSwapchainImageWidth=0,
                 maxLayerCount=XR_MIN_COMPOSITION_LAYERS_SUPPORTED,
                 orientationTracking=False,
                 positionTracking=False):

        self.systemId = systemId
        self.vendorId = vendorId
        self.systemName = systemName
        self.maxSwapchainImageHeight = maxSwapchainImageHeight
        self.maxSwapchainImageWidth = maxSwapchainImageWidth
        self.maxLayerCount = maxLayerCount
        self.orientationTracking = orientationTracking
        self.positionTracking = positionTracking

    def __cinit__(self, *args, **kwargs):
        self.c_data.type = openxr.XR_TYPE_SYSTEM_PROPERTIES

    def __repr__(self):
        return (
            f"OpenXRSystemInfo("
            f"systemId={self.systemId}, "
            f"vendorId={self.vendorId}, "
            f"systemName='{self.systemName}', "
            f"maxSwapchainImageHeight={self.maxSwapchainImageHeight}, "
            f"maxSwapchainImageWidth={self.maxSwapchainImageWidth}, "
            f"maxLayerCount={self.maxLayerCount}, "
            f"orientationTracking={self.orientationTracking}, "
            f"positionTracking={self.positionTracking})"
        )

    @property
    def systemId(self):
        """System ID assigned by OpenXR (`int`)."""
        return self.c_data.systemId

    @systemId.setter
    def systemId(self, value):
        if not isinstance(value, int):  # check type
            raise TypeError(
                'Property `OpenXRSystem.systemId` must be type `int`.')

        self.c_data.systemId = <openxr.XrSystemId>value

    @property
    def vendorId(self):
        """Vendor ID assigned by OpenXR (`int`)."""
        return self.c_data.vendorId

    @vendorId.setter
    def vendorId(self, value):
        if not isinstance(value, int):  # check type
            raise TypeError(
                'Property `OpenXRSystem.vendorId` must be type `int`.')

        self.c_data.vendorId = <int>value

    @property
    def systemName(self):
        """System name (`str`). Name must not exceed the length of
        ``XR_MAX_SYSTEM_NAME_SIZE``.
        """
        return bytes2str(self.c_data.systemName)

    @systemName.setter
    def systemName(self, value):
        if not isinstance(value, str):  # check type
            raise TypeError(
                'Property `OpenXRSystem.systemName` must be type '
                '`str`.')

        if len(value) > openxr.XR_MAX_SYSTEM_NAME_SIZE:
            raise ValueError(
                'Value for `OpenXRSystem.systemName` must have '
                'length <{}'.format(openxr.XR_MAX_SYSTEM_NAME_SIZE))

        self.c_data.systemName = str2bytes(value)

    @property
    def maxSwapchainImageHeight(self):
        """Maximum swap chain image height in pixels (`int`)."""
        return self.c_data.graphicsProperties.maxSwapchainImageHeight

    @maxSwapchainImageHeight.setter
    def maxSwapchainImageHeight(self, value):
        if not isinstance(value, int):  # check type
            raise TypeError(
                'Property `OpenXRSystem.maxSwapchainImageHeight` must be type '
                '`int`.')

        self.c_data.graphicsProperties.maxSwapchainImageHeight = <int>value

    @property
    def maxSwapchainImageWidth(self):
        """Maximum supported swap chain image width in pixels (`int`)."""
        return self.c_data.graphicsProperties.maxSwapchainImageHeight

    @maxSwapchainImageWidth.setter
    def maxSwapchainImageWidth(self, value):
        if not isinstance(value, int):  # check type
            raise TypeError(
                'Property `OpenXRSystem.maxSwapchainImageWidth` must be type '
                '`int`.')

        self.c_data.graphicsProperties.maxSwapchainImageWidth = <int>value

    @property
    def maxSwapchainSize(self):
        """Maximum supported width and height of a swapchain image in pixels
        (`tuple`)."""
        return self.maxSwapchainImageWidth, self.maxSwapchainImageHeight

    @property
    def maxLayerCount(self):
        """Maximum number of layers allowed (`int`)."""
        return self.c_data.graphicsProperties.maxLayerCount

    @maxLayerCount.setter
    def maxLayerCount(self, value):
        if not isinstance(value, int):  # check type
            raise TypeError(
                'Property `OpenXRSystem.maxLayerCount` must be type '
                '`int`.')

        if value < openxr.XR_MIN_COMPOSITION_LAYERS_SUPPORTED:
            raise ValueError(
                'Property `OpenXRSystem.maxLayerCount` must be greater than '
                '`XR_MIN_COMPOSITION_LAYERS_SUPPORTED`.'
            )

        self.c_data.graphicsProperties.maxLayerCount = <int>value

    @property
    def orientationTracking(self):
        """`True` if the system is capable of tracking orientation (`bool`)."""
        return <bint>self.c_data.trackingProperties.orientationTracking

    @orientationTracking.setter
    def orientationTracking(self, value):
        if not isinstance(value, (bool, int)):  # check type, ints are okay
            raise TypeError(
                'Property `OpenXRSystem.orientationTracking` must be type '
                '`bool`.'
            )

        self.c_data.trackingProperties.orientationTracking = \
            <openxr.XrBool32>value

    @property
    def positionTracking(self):
        """`True` if the system is capable of tracking position (`bool`)."""
        return <bint>self.c_data.trackingProperties.positionTracking

    @positionTracking.setter
    def positionTracking(self, value):
        if not isinstance(value, (bool, int)):  # check type, ints are okay
            raise TypeError(
                'Property `OpenXRSystem.positionTracking` must be type '
                '`bool`.'
            )

        self.c_data.trackingProperties.positionTracking = <openxr.XrBool32>value

    @property
    def fullTracking(self):
        """`True` if this system has both orientation and position tracking
        capabilities (`bool`).
        """
        return self.positionTracking and self.orientationTracking


cdef class OpenXRViewConfigInfo:
    """Descriptor describing the configuration of a system view.
    """
    cdef openxr.XrViewConfigurationView c_data
    def __init__(self,
                 recommendedImageRectWidth=0,
                 recommendedImageRectHeight=0,
                 maxImageRectWidth=0,
                 maxImageRectHeight=0,
                 recommendedSwapchainSampleCount=0,
                 maxSwapchainSampleCount=0):

        self.recommendedImageRectWidth = recommendedImageRectWidth
        self.maxImageRectWidth = maxImageRectWidth
        self.recommendedImageRectHeight = recommendedImageRectHeight
        self.maxImageRectHeight = maxImageRectHeight
        self.recommendedSwapchainSampleCount = recommendedSwapchainSampleCount
        self.maxSwapchainSampleCount = maxSwapchainSampleCount

    def __cinit__(self, *args, **kwargs):
        self.c_data.type = openxr.XR_TYPE_VIEW_CONFIGURATION_VIEW
        self.c_data.next = NULL

    def __repr__(self):
        return (
            f"OpenXRViewConfigInfo("
            f"recommendedImageRectWidth={self.recommendedImageRectWidth}, "
            f"recommendedImageRectHeight={self.recommendedImageRectHeight}, "
            f"maxImageRectWidth={self.maxImageRectWidth}, "
            f"maxImageRectHeight={self.maxImageRectHeight}, "
            f"recommendedSwapchainSampleCount={self.recommendedSwapchainSampleCount}, "
            f"maxSwapchainSampleCount={self.maxSwapchainSampleCount})"
        )

    @property
    def recommendedImageRectWidth(self):
        return <int>self.c_data.recommendedImageRectWidth

    @recommendedImageRectWidth.setter
    def recommendedImageRectWidth(self, value):
        if not isinstance(value, int):  # check type
            raise TypeError(
                'Property `OpenXRViewConfig.recommendedImageRectWidth` must be '
                'type `int`.'
            )

        self.c_data.recommendedImageRectWidth = <uint32_t>value

    @property
    def recommendedImageRectHeight(self):
        return <int>self.c_data.recommendedImageRectHeight

    @recommendedImageRectHeight.setter
    def recommendedImageRectHeight(self, value):
        if not isinstance(value, int):  # check type
            raise TypeError(
                'Property `OpenXRViewConfig.recommendedImageRectHeight` must '
                'be type `int`.'
            )

        self.c_data.recommendedImageRectHeight = <uint32_t>value

    @property
    def recommendedImageRectSize(self):
        return (<int>self.c_data.recommendedImageRectWidth,
                <int>self.c_data.recommendedImageRectHeight)

    @property
    def maxImageRectWidth(self):
        return <int>self.c_data.maxImageRectWidth

    @maxImageRectWidth.setter
    def maxImageRectWidth(self, value):
        if not isinstance(value, int):  # check type
            raise TypeError(
                'Property `OpenXRViewConfig.maxImageRectWidth` must be type '
                '`int`.'
            )

        self.c_data.maxImageRectWidth = <uint32_t>value

    @property
    def maxImageRectHeight(self):
        return <int>self.c_data.maxImageRectWidth

    @maxImageRectHeight.setter
    def maxImageRectHeight(self, value):
        if not isinstance(value, int):  # check type
            raise TypeError(
                'Property `OpenXRViewConfig.maxImageRectHeight` must be type '
                '`int`.'
            )

        self.c_data.maxImageRectHeight = <uint32_t>value

    @property
    def maxImageRectSize(self):
        return (<int>self.c_data.maxImageRectWidth,
                <int>self.c_data.maxImageRectHeight)

    @property
    def recommendedSwapchainSampleCount(self):
        return <int>self.c_data.recommendedSwapchainSampleCount

    @recommendedSwapchainSampleCount.setter
    def recommendedSwapchainSampleCount(self, value):
        if not isinstance(value, int):  # check type
            raise TypeError(
                'Property `OpenXRViewConfig.recommendedSwapchainSampleCount` '
                'must be type `int`.'
            )

        self.c_data.recommendedSwapchainSampleCount = <uint32_t>value

    @property
    def maxSwapchainSampleCount(self):
        return <int>self.c_data.maxSwapchainSampleCount

    @maxSwapchainSampleCount.setter
    def maxSwapchainSampleCount(self, value):
        if not isinstance(value, int):  # check type
            raise TypeError(
                'Property `OpenXRViewConfig.maxSwapchainSampleCount` must be '
                'type `int`.'
            )

        self.c_data.maxSwapchainSampleCount = <uint32_t>value


cdef class OpenXRPose(object):
    """Class for representing rigid body poses.

    Parameters
    ----------
    pos : array_like
        Initial position vector (x, y, z).
    ori : array_like
        Initial orientation quaternion (x, y, z, w).

    """
    cdef openxr.XrPosef* c_data
    cdef bint ptr_owner

    cdef np.ndarray _pos
    cdef np.ndarray _ori

    def __init__(self, pos=(0., 0., 0.), ori=(0., 0., 0., 1.)):
        self._new_struct(pos, ori)

    def __cinit__(self, *args, **kwargs):
        self.ptr_owner = False

    @staticmethod
    cdef OpenXRPose fromPtr(openxr.XrPosef* ptr, bint owner=False):
        cdef OpenXRPose wrapper = OpenXRPose.__new__(OpenXRPose)
        wrapper.c_data = ptr
        wrapper.ptr_owner = owner

        wrapper._pos = _wrap_XrVector3f_as_ndarray(&ptr.position)
        wrapper._ori = _wrap_XrQuaternionf_as_ndarray(&ptr.orientation)

        return wrapper

    cdef void _new_struct(self, object pos, object ori):
        if self.c_data is not NULL:
            return

        cdef openxr.XrPosef* ptr = <openxr.XrPosef*>PyMem_Malloc(
            sizeof(openxr.XrPosef))

        if ptr is NULL:
            raise MemoryError

        # clear memory
        ptr.position.x = <float>pos[0]
        ptr.position.y = <float>pos[1]
        ptr.position.z = <float>pos[2]
        ptr.orientation.x = <float>ori[0]
        ptr.orientation.y = <float>ori[1]
        ptr.orientation.z = <float>ori[2]
        ptr.orientation.w = <float>ori[3]

        self.c_data = ptr
        self.ptr_owner = True

        self._pos = _wrap_XrVector3f_as_ndarray(&ptr.position)
        self._ori = _wrap_XrQuaternionf_as_ndarray(&ptr.orientation)

    def __dealloc__(self):
        # don't do anything crazy like set c_data=NULL without deallocating!
        if self.c_data is not NULL:
            if self.ptr_owner is True:
                PyMem_Free(self.c_data)
                self.c_data = NULL

    def __repr__(self):
        return f'OpenXRPose(pos={repr(self.pos)}, ori={repr(self.ori)})'

    @property
    def pos(self):
        """ndarray : Position vector [X, Y, Z].
        """
        return self._pos

    @pos.setter
    def pos(self, object value):
        self._pos[:] = value

    def getPos(self, np.ndarray[np.float32_t, ndim=1] out=None):
        """Position vector X, Y, Z.

        Parameters
        ----------
        out : ndarray or None
            Optional array to write values to. Must have a float32 data type.

        Returns
        -------
        ndarray
            Position coordinate of this pose.

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = out

        toReturn[0] = self.c_data[0].position.x
        toReturn[1] = self.c_data[0].position.y
        toReturn[2] = self.c_data[0].position.z

        return toReturn

    def setPos(self, object pos):
        """Set the position of the pose in a scene.

        Parameters
        ----------
        pos : array_like
            Position vector [X, Y, Z].

        """
        self.c_data[0].position.x = <float>pos[0]
        self.c_data[0].position.y = <float>pos[1]
        self.c_data[0].position.z = <float>pos[2]

    @property
    def ori(self):
        """ndarray : Orientation quaternion [X, Y, Z, W].
        """
        return self._ori

    @ori.setter
    def ori(self, object value):
        self._ori[:] = value

    def getOri(self, np.ndarray[np.float32_t, ndim=1] out=None):
        """Orientation quaternion X, Y, Z, W. Components X, Y, Z are imaginary
        and W is real.

        Parameters
        ----------
        out : ndarray  or None
            Optional array to write values to. Must have a float32 data type.

        Returns
        -------
        ndarray
            Orientation quaternion of this pose.

        Notes
        -----

        * The orientation quaternion should be normalized.

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((4,), dtype=np.float32)
        else:
            toReturn = out

        toReturn[0] = self.c_data[0].orientation.x
        toReturn[1] = self.c_data[0].orientation.y
        toReturn[2] = self.c_data[0].orientation.z
        toReturn[3] = self.c_data[0].orientation.w

        return toReturn

    def setOri(self, object ori):
        """Set the orientation of the pose in a scene.

        Parameters
        ----------
        ori : array_like
            Orientation quaternion [X, Y, Z, W].

        """
        self.c_data[0].orientation.x = <float>ori[0]
        self.c_data[0].orientation.y = <float>ori[1]
        self.c_data[0].orientation.z = <float>ori[2]
        self.c_data[0].orientation.w = <float>ori[3]


def createInstance(OpenXRApplicationInfo applicationInfo):
    """Create an OpenXR instance.

    PsychXR currently only supports creating one instance at a time.

    Parameters
    ----------
    applicationInfo : OpenXRApplicationInfo
        Application info descriptor.

    """
    global _ptrInstance

    if _ptrInstance is not NULL:
        return

    # only one extension is in use here, we'll allow the user to set these later
    cdef const char* enabled_exts[1]
    enabled_exts[0] = openxr.XR_KHR_OPENGL_ENABLE_EXTENSION_NAME

    # crete descriptor for instance information
    cdef openxr.XrInstanceCreateInfo instance_create_info
    instance_create_info.type = openxr.XR_TYPE_INSTANCE_CREATE_INFO
    instance_create_info.next = NULL  # not used, but must be set
    instance_create_info.createFlags = 0  # must be zero
    instance_create_info.enabledExtensionCount = 1
    instance_create_info.enabledExtensionNames = enabled_exts
    instance_create_info.enabledApiLayerCount = 0
    instance_create_info.enabledApiLayerNames = NULL

    # pass along application information specified by the user
    instance_create_info.applicationInfo = applicationInfo.c_data

    # create the instance
    cdef openxr.XrResult result = openxr.xrCreateInstance(
        &instance_create_info,
        &_ptrInstance)

    checkResult(result)


def instanceStarted():
    """Check if we have already created an OpenXR instance.

    Returns
    -------
    bool
        `True` if an instance has been created, otherwise `False`.

    """
    global _ptrInstance
    cdef bint result = _ptrInstance != NULL

    return result


def destroyInstance():
    """Destroy the current OpenXR instance. This function does nothing if no
    instance as previously created.

    Raises
    ------
    OpenXRHandleInvalidError
        Raised if `createInstance` was not previously called or instance handle
        it set is not valid.

    Examples
    --------
    Destroy an instance::

        try:
            destroyInstance()
        except OpenXRHandleInvalidError:
            print("`createInstance` was not previously called!")

    """
    global _ptrInstance
    # if _ptrInstance == NULL:
    #     return openxr.XR_ERROR_HANDLE_INVALID

    cdef openxr.XrResult result = openxr.xrDestroyInstance(_ptrInstance)

    if result == openxr.XR_SUCCESS:
        _ptrInstance = NULL  # reset to NULL if successful
    else:
        raise openxr_error_lut[result]()


def findSystem(formFactor):
    """Query OpenXR for a system with the specified form factor.

    Parameters
    ----------
    formFactor : int
        Symbolic constant representing the form factor to fetch. Can be either
        ``XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY`` or
        ``XR_FORM_FACTOR_HANDHELD_DISPLAY``.

    Returns
    -------
    OpenXRSystemInfo
        Descriptor containing information about the system.

    Examples
    --------
    Get a system ID handle for an OpenXR supported HMD::

        try:
            system_id = getSystem(XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY)
        except (OpenXRFormFactorUnsupportedError, OpenXRFormFactorUnavailableError):
            print('Specified form factor cannot be used.')

    """
    global _ptrInstance
    cdef openxr.XrSystemId system_id
    cdef openxr.XrResult result
    cdef openxr.XrSystemGetInfo systemGetInfo

    # set the form factor to get
    systemGetInfo.type = openxr.XR_TYPE_SYSTEM_GET_INFO
    systemGetInfo.formFactor = formFactor
    systemGetInfo.next = NULL

    # get the system
    result = openxr.xrGetSystem(
        _ptrInstance,
        &systemGetInfo,
        &system_id)

    checkResult(result)

    # get system properties
    cdef openxr.XrSystemProperties system_props
    system_props.type = openxr.XR_TYPE_SYSTEM_PROPERTIES
    system_props.next = NULL
    result = openxr.xrGetSystemProperties(
        _ptrInstance,
        system_id,
        &system_props)

    checkResult(result)

    cdef OpenXRSystemInfo system_desc = OpenXRSystemInfo()
    system_desc.c_data = system_props

    return system_desc


def getViewConfigurations(OpenXRSystemInfo system, int viewType):
    """Get configuration for each view supported by the provided system.

    Parameters
    ----------
    system : OpenXRSystemInfo
        Descriptor for the system to query view information.
    viewType : int
        Symbolic constant representing the type of view data to retrieve. Value
        may be one of ``XR_VIEW_CONFIGURATION_TYPE_PRIMARY_MONO`` or
        ``XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO``.

    Returns
    -------
    list of OpenXRViewConfigInfo
        View configurations for each view provided by the system.

    Examples
    --------
    Get view configurations for a given system using a stereoscopic display::

        mySystem = getSystem(formFactor=XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY)
        leftEyeView, rightEyeView = getViewConfigurations(
            system=mySystem,
            viewType=XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO)

        # get the recommended framebuffer size for each eye
        fboLeftSize = (
            leftEyeView.recommendedImageRectWidth,
            leftEyeView.recommendedImageRectHeight)
        fboRightSize = (
            rightEyeView.recommendedImageRectWidth,
            rightEyeView.recommendedImageRectHeight
        )

    """
    global _ptrInstance

    cdef openxr.XrViewConfigurationType view_type = \
        <openxr.XrViewConfigurationType>viewType  # return value
    cdef openxr.XrSystemId system_id = system.c_data.systemId

    # get the number of views in the first pass
    cdef uint32_t view_count = 0
    cdef openxr.XrResult result
    result = openxr.xrEnumerateViewConfigurationViews(
        _ptrInstance,
        system_id,
        view_type,
        0,  # write no values
        &view_count,
        NULL)  # pass NULL to just get the count right now

    checkResult(result)

    # allocate arrays to hold view config data
    cdef openxr.XrViewConfigurationView* c_view_configs = \
        <openxr.XrViewConfigurationView*>PyMem_Malloc(
            sizeof(openxr.XrViewConfigurationView) * view_count)

    if c_view_configs is NULL:
        raise MemoryError("Failed to allocate array `c_view_configs`.")

    cdef Py_ssize_t i = 0
    for i in range(<Py_ssize_t>view_count):
        c_view_configs[i].type = openxr.XR_TYPE_VIEW_CONFIGURATION_VIEW
        c_view_configs[i].next = NULL

    # call again with the array to write values
    result = openxr.xrEnumerateViewConfigurationViews(
        _ptrInstance,
        system_id,
        view_type,
        view_count,
        &view_count,
        c_view_configs)

    checkResult(result)

    # loop over returned view configs and write them to Python objects
    cdef list to_return = []  # returns a list of view settings
    cdef OpenXRViewConfigInfo view_desc
    for i in range(<Py_ssize_t>view_count):
        view_desc = OpenXRViewConfigInfo()
        view_desc.c_data = c_view_configs[i]
        to_return.append(view_desc)

    PyMem_Free(c_view_configs)  # free the array

    return to_return


def getGraphicsRequirementsOpenGL(OpenXRSystemInfo system):
    """Get the graphics requirements for the given system.

    This provides information about the requirements for the OpenGL context
    version needed by the driver for rendering. Keep in mind that you graphics
    driver may allow for functionality outside of the recommended core
    profiles specified by OpenXR for the given system. However, your graphics
    driver must at least support the OpenGL versions returned.

    Parameters
    ----------
    system : OpenXRSystemInfo
        System to query for graphics requirements.

    Returns
    -------
    tuple
        Minimum and maximum OpenGL API versions supported by the system driver
        in (major, minor, patch) format.

    Examples
    --------
    Check the minimum OpenGL version required by the system::

        minVersionGL, maxVersionGL = getGraphicsRequirementsGL(mySystem)
        major, minor, patch = minVersionGL  # something like (3, 3, 0)

    """
    global _ptrInstance
    cdef openxr.XrGraphicsRequirementsOpenGLKHR opengl_reqs
    opengl_reqs.next = NULL
    opengl_reqs.type = openxr.XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_KHR

    # load the function to get the version
    cdef openxr.PFN_xrGetOpenGLGraphicsRequirementsKHR \
        pfnGetOpenGLGraphicsRequirementsKHR = NULL
    cdef openxr.XrResult result = openxr.xrGetInstanceProcAddr(
        _ptrInstance,
        "xrGetOpenGLGraphicsRequirementsKHR",
	    <openxr.PFN_xrVoidFunction*>&pfnGetOpenGLGraphicsRequirementsKHR)

    checkResult(result)

    # query the system for the recommended versions
    result = pfnGetOpenGLGraphicsRequirementsKHR(
        _ptrInstance, system.c_data.systemId, &opengl_reqs)

    checkResult(result)

    return (xr_get_version(opengl_reqs.minApiVersionSupported),
        xr_get_version(opengl_reqs.maxApiVersionSupported))


def createGraphicsBindingOpenGLWin32(hDC, hGLRC):
    """Create a graphics binding for OpenGL (MS Windows).

    Call this after creating a window, passing handles for the window (HDC) and
    OpenGL context (HGLRC). You can get these values from SDL2, Pyglet, GLFW,
    etc.

    Once bound, OpenXR will be able to allocate resources using the provided
    OpenGL context after starting a session. You cannot start a session without
    binding a window first.

    Parameters
    ----------
    hDC : ctypes.c_void_p
        Handle for the window device context.
    hGLRC : ctypes.c_void_p
        Handle for an OpenGL rendering context.

    Examples
    --------
    Create a graphics binding using a Pyglet window::

        # create a Pyglet window
        window = pyglet.window.Window()

        # call this, OpenXR requires it even if you don't use the info returned
        _, _ = getGraphicsRequirementsOpenGL()

        # set the context bindings
        createGraphicsBindingOpenGLWin32(window._dc, window._wgl_context)

        # create a session
        createSession(system)

    """
    # We could move this logic into `createSession`, having the user pass the
    # binding data prior to creating a session. - mdc
    global _gfxBinding

    # need to copy over values
    cdef void* c_hDC = <void*>hDC
    cdef void* c_hGLRC = <void*>hGLRC

    # set the bindings
    _gfxBinding.hDC = <openxr.HDC>c_hDC
    _gfxBinding.hGLRC = <openxr.HGLRC>c_hGLRC


def createSession(OpenXRSystemInfo system, int viewType):
    """Create an OpenXR session.

    Parameters
    ----------
    system : OpenXRSystemInfo
        System to use for the session. This cannot be changed once the session
        is created until `destroySession` is called.
    viewType : int
        Symbolic constant representing the view type to use for the session.
        Value may be one of ``XR_VIEW_CONFIGURATION_TYPE_PRIMARY_MONO`` or
        ``XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO``. View type must be
        supported by the `system`.

    """
    global _ptrInstance
    global _ptrSession
    global _gfxBinding
    global _system  # system to use
    global _systemViewConfigs
    global _systemViewCount
    global numSupportedSwapChainFormats

    cdef openxr.XrResult result  # for API return values

    if _ptrInstance == NULL:  # check if we have an instance
        raise RuntimeError(
            'Cannot create an OpenXR session, must create an instance first.')

    # check if the user supplied window binding data
    if _gfxBinding.hDC == NULL or _gfxBinding.hGLRC == NULL:
        raise RuntimeError(
            'Attempted to create a session without creating an OpenGL binding '
            'first.')

    # set the system
    _system = system.c_data

    # get the number of views in the first pass
    _systemViewCount = 0
    result = openxr.xrEnumerateViewConfigurationViews(
        _ptrInstance,
        _system.systemId,
        <openxr.XrViewConfigurationType>viewType,
        0,  # write no values
        &_systemViewCount,
        NULL)  # pass NULL to just get the count right now

    checkResult(result)

    # allocate arrays to hold view config data
    _systemViewConfigs = <openxr.XrViewConfigurationView*>PyMem_Malloc(
        sizeof(openxr.XrViewConfigurationView) * _systemViewCount)

    if _systemViewConfigs is NULL:
        raise MemoryError("Failed to allocate array `_systemViewConfigs`.")

    # array to hold view config data
    cdef Py_ssize_t i = 0
    for i in range(<Py_ssize_t>_systemViewCount):
        _systemViewConfigs[i].type = openxr.XR_TYPE_VIEW_CONFIGURATION_VIEW
        _systemViewConfigs[i].next = NULL

    # call again with the array to write values
    result = openxr.xrEnumerateViewConfigurationViews(
        _ptrInstance,
        _system.systemId,
        <openxr.XrViewConfigurationType>viewType,
        _systemViewCount,
        &_systemViewCount,
        _systemViewConfigs)

    checkResult(result)

    # session info
    cdef openxr.XrSessionCreateInfo sessionCreateInfo
    sessionCreateInfo.type = openxr.XR_TYPE_SESSION_CREATE_INFO
    sessionCreateInfo.next = &_gfxBinding
    sessionCreateInfo.systemId = system.c_data.systemId

    # call this to have the API load the required pointer functions for OpenGL
    getGraphicsRequirementsOpenGL(system)

    # create a session
    result = openxr.xrCreateSession(
        _ptrInstance,
        &sessionCreateInfo,
        &_ptrSession)

    checkResult(result)

    # get number of swap chain formats
    result = openxr.xrEnumerateSwapchainFormats(
        _ptrSession,
        0,
        &numSupportedSwapChainFormats,
        NULL)

    checkResult(result)


def createSpace(int referenceSpaceType, OpenXRPose poseInReferenceSpace):
    """Create a reference space.

    Parameters
    ----------
    referenceSpaceType : int
        Symbolic constant representing a reference space type to use. Can be one
        of ``XR_REFERENCE_SPACE_TYPE_VIEW``, ``XR_REFERENCE_SPACE_TYPE_LOCAL``,
        or ``XR_REFERENCE_SPACE_TYPE_STAGE``.
    poseInReferenceSpace : OpenXRPose
        Base pose in the reference space.

    """
    global _ptrSession
    global _refSpace

    if _ptrSession == NULL:
        raise RuntimeError('Cannot create a reference space without a session.')

    cdef openxr.XrReferenceSpaceCreateInfo ref_space_info
    ref_space_info.type = openxr.XR_TYPE_REFERENCE_SPACE_CREATE_INFO
    ref_space_info.next = NULL
    ref_space_info.referenceSpaceType = \
        <openxr.XrReferenceSpaceType>referenceSpaceType
    ref_space_info.poseInReferenceSpace = poseInReferenceSpace.c_data[0]

    cdef openxr.XrResult result = openxr.xrCreateReferenceSpace(
        _ptrSession,
        &ref_space_info,
        &_refSpace)

    checkResult(result)


def destroySpace():
    """Destroy a previously created reference space.
    """
    global _refSpace

    if _refSpace == NULL:
        return  # nop if there is no reference space

    cdef openxr.XrResult result = openxr.xrDestroySpace(_refSpace)

    checkResult(result)

    _refSpace = NULL  # reset


def createSwapChainColorOpenGL(
        int width,
        int height,
        int format,
        int viewCount,
        int sampleCount=1,
        int mipCount=1,
        int faceCount=1,
        int arraySize=1,
        int usageFlags=0):
    """Create a swap for color images."""
    # Need to figure out if we want to go about creating swap chains this way.
    global _ptrSession
    global colorSwapChains
    global colorSwapChainImagesGL

    cdef openxr.XrResult result
    colorSwapChains = <openxr.XrSwapchain*>PyMem_Malloc(
        sizeof(openxr.XrSwapchain) * viewCount)

    if colorSwapChains is NULL:
        raise MemoryError

    # create a swap chain for each view
    cdef openxr.XrSwapchainCreateInfo swapChainCreateInfo
    cdef int i = 0
    for i in range(viewCount):
        # parameters for swap chain
        swapChainCreateInfo.type = openxr.XR_TYPE_SWAPCHAIN_CREATE_INFO
        swapChainCreateInfo.format = format
        swapChainCreateInfo.width = width
        swapChainCreateInfo.height = height
        swapChainCreateInfo.sampleCount = sampleCount
        swapChainCreateInfo.faceCount = faceCount
        swapChainCreateInfo.arraySize = arraySize
        swapChainCreateInfo.mipCount = mipCount
        swapChainCreateInfo.usageFlags = usageFlags

        # create the swap chains for each eye
        result = openxr.xrCreateSwapchain(
            _ptrSession, &swapChainCreateInfo, &colorSwapChains[i])

        checkResult(result)


