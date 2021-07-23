# distutils: language=c++
#  =============================================================================
#  _openxr.pyx - Python Interface Module for OpenXR
#  =============================================================================
#
#  Copyright 2021 Matthew Cutone <mcutone@opensciencetools.com> and Laurie M.
#  Wilcox <lmwilcox(a)yorku.ca>; The Centre For Vision Research, York
#  University, Toronto, Canada
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
    'XrApplicationInfo',
    'createInstance',
    'hasInstance',
    'destroyInstance'
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


cdef char* str2bytes(str strIn):
    """Convert UTF-8 encoded strings to bytes."""
    py_bytes = strIn.encode('UTF-8')
    cdef char* to_return = py_bytes

    return to_return


cdef str bytes2str(char* bytesIn):
    """Convert UTF-8 encoded strings to bytes."""
    return bytesIn.decode('UTF-8')


# ------------------------------------------------------------------------------
# Classes and Functions
#

cdef class XrApplicationInfo:
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


def createInstance(XrApplicationInfo applicationInfo):
    """Create an OpenXR instance.

    PsychXR currently only supports creating one instance at a time.

    Parameters
    ----------
    applicationInfo : XrApplicationInfo
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

    Returns
    -------
    int
        Returns ``XR_SUCCESS`` if the instance was successfully destroyed.
        Otherwise, expect ``XR_ERROR_HANDLE_INVALID`` if the handle was invalid
        or :func:`createInstance` was not previously called.

    """
    global _ptrInstance
    if _ptrInstance == NULL:
        return openxr.XR_ERROR_HANDLE_INVALID

    cdef openxr.XrResult result = openxr.xrDestroyInstance(_ptrInstance)

    return result
