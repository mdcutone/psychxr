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
    'XR_SUCCESS',
    'XR_ERROR_HANDLE_INVALID',
    'XR_CURRENT_API_VERSION',
    'OpenXRApplicationInfo',
    'OpenXRSystem',
    'createInstance',
    'hasInstance',
    'destroyInstance',
    'getSystem'
]

# ------------------------------------------------------------------------------
# Imports
#

from . cimport openxr
cimport numpy as np
import numpy as np
np.import_array()
import warnings


# ------------------------------------------------------------------------------
# Module level constants
#

cdef openxr.XrInstance _ptrInstance = NULL  # pointer to instance
cdef openxr.XrSession _ptrSession = NULL  # pointer to session
cdef openxr.XrSystemId _systemId = openxr.XR_NULL_SYSTEM_ID

XR_SUCCESS = openxr.XR_SUCCESS
XR_ERROR_HANDLE_INVALID = openxr.XR_ERROR_HANDLE_INVALID
XR_CURRENT_API_VERSION = openxr.XR_CURRENT_API_VERSION
XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY = openxr.XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY
XR_FORM_FACTOR_HANDHELD_DISPLAY = openxr.XR_FORM_FACTOR_HANDHELD_DISPLAY


cdef char* str2bytes(str strIn):
    """Convert UTF-8 encoded strings to bytes."""
    py_bytes = strIn.encode('UTF-8')
    cdef char* to_return = py_bytes

    return to_return


cdef str bytes2str(char* bytesIn):
    """Convert UTF-8 encoded strings to bytes."""
    return bytesIn.decode('UTF-8')


# ------------------------------------------------------------------------------
# Exceptions
#

class OpenXRError(BaseException):
    """Base exception for OpenXR related errors."""


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


# lookup table of exceptions
cdef dict openxr_error_lut = {
    openxr.XR_ERROR_VALIDATION_FAILURE: OpenXRValidationFailureError,
    openxr.XR_ERROR_RUNTIME_FAILURE: OpenXRRuntimeFailureError,
    openxr.XR_ERROR_OUT_OF_MEMORY: OpenXROutOfMemoryError,
    openxr.XR_ERROR_HANDLE_INVALID: OpenXRHandleInvalidError,
    openxr.XR_ERROR_INSTANCE_LOST: OpenXRInstanceLostError,
    openxr.XR_ERROR_SYSTEM_INVALID: OpenXRSystemInvalidError,
    openxr.XR_ERROR_LIMIT_REACHED: OpenXRLimitReachedError,
    openxr.XR_ERROR_RUNTIME_UNAVAILABLE : OpenXRRuntimeUnavailableError,
    openxr.XR_ERROR_NAME_INVALID : OpenXRNameInvalidError,
    openxr.XR_ERROR_INITIALIZATION_FAILED : OpenXRNameInitializationFailedError,
    openxr.XR_ERROR_EXTENSION_NOT_PRESENT : OpenXRNameExtensionNotPresentError,
    openxr.XR_ERROR_API_VERSION_UNSUPPORTED : OpenXRApiVersionNotSupportedError,
    openxr.XR_ERROR_API_LAYER_NOT_PRESENT : OpenXRApiLayerNotPresentError
}


# ------------------------------------------------------------------------------
# Classes and Functions
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
        return (f"XrApplicationInfo(applicationName='{self.applicationName}', "
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


cdef class OpenXRSystem:
    """Descriptor for a system (HMD, controllers, etc.) available to OpenXR.

    These are instanced when calling :func:`getSystem`, users should not
    instance this themselves during regular use.

    """
    cdef openxr.XrSystemProperties c_data

    def __init__(self,
                 systemId=0,
                 vendorId=0,
                 systemName=""):

        self.systemId = systemId
        self.vendorId = vendorId
        self.systemName = systemName
        # self.graphicsProperties = graphicsProperties
        # self.trackingProperties = trackingProperties

    def __cinit__(self, *args, **kwargs):
        self.c_data.type = openxr.XR_TYPE_SYSTEM_PROPERTIES

    def __repr__(self):
        return (f"OpenXRSystem(systemId='{self.systemId}', "
            f"vendorId={self.vendorId}, "
            f"engineName='{self.systemName}')")

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


def createInstance(OpenXRApplicationInfo applicationInfo):
    """Create an OpenXR instance.

    PsychXR currently only supports creating one instance at a time.

    Parameters
    ----------
    applicationInfo : OpenXRApplicationInfo
        Application info descriptor.

    Returns
    -------
    int
        Result of the ``xrCreateInstance`` OpenXR API call. A value >=0
        indicates success.

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
    instance_create_info.next = NULL  # not used
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

    return result


def hasInstance():
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


def getSystem(formFactor):
    """Query OpenXR for a system with the specified form factor.

    Parameters
    ----------
    formFactor : int
        Symbolic constant representing the form factor to fetch. Can be either
        ``XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY`` or
        ``XR_FORM_FACTOR_HANDHELD_DISPLAY``.

    Returns
    -------
    tuple
        API call result for `xrGetSystem` (`int`) and the enumerated system ID
        (`int`) matching the desired `formFactor`. If no system was found, the
        second value will be `None`.

    Examples
    --------
    Get a system ID handle for an OpenXR supported HMD::

        result, system_id =

    """
    global _ptrInstance
    cdef openxr.XrSystemId system_id
    cdef openxr.XrSystemGetInfo system_get_info

    # set the form factor to get
    system_get_info.formFactor = formFactor

    # get the system
    cdef openxr.XrResult result = openxr.xrGetSystem(
        _ptrInstance,
        &system_get_info,
        &system_id)

    if result == openxr.XR_SUCCESS:  # failed to get a system, return nothing
        return system_id
    else:
        raise openxr_error_lut[result]()
