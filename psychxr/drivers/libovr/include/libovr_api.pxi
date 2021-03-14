#  =============================================================================
#  libovr_api.pxi - LibOVR API
#  =============================================================================
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

# API version information
PRODUCT_VERSION = capi.OVR_PRODUCT_VERSION
MAJOR_VERSION = capi.OVR_MAJOR_VERSION
MINOR_VERSION = capi.OVR_MINOR_VERSION
PATCH_VERSION = capi.OVR_PATCH_VERSION
# BUILD_NUMBER = capi.OVR_BUILD_NUMBER
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

def getBool(bytes propertyName, bint defaultVal=False):
    """Read a LibOVR boolean property.

    Parameters
    ----------
    propertyName : bytes
        Name of the property to get.
    defaultVal : bool, optional
        Return value if the property could not be set. The default value is
        ``False``.

    Returns
    -------
    bool
        Value of the property. Returns `defaultVal` if the property does not
        exist.

    """
    global _ptrSession
    cdef capi.ovrBool val = capi.ovrTrue if defaultVal else capi.ovrFalse

    cdef capi.ovrBool to_return = capi.ovr_GetBool(
        _ptrSession,
        propertyName,
        defaultVal)

    return to_return == capi.ovrTrue


def setBool(bytes propertyName, bint value=True):
    """Write a LibOVR boolean property.

    Parameters
    ----------
    propertyName : bytes
        Name of the property to set.
    value : bool
        Value to write.

    Returns
    -------
    bool
        ``True`` if the property was set successfully, ``False`` if the property
        was read-only or does not exist.

    """
    global _ptrSession
    cdef capi.ovrBool val = capi.ovrTrue if value else capi.ovrFalse

    cdef capi.ovrBool to_return = capi.ovr_SetBool(
        _ptrSession,
        propertyName,
        val)

    return to_return == capi.ovrTrue


def getInt(bytes propertyName, int defaultVal=0):
    """Read a LibOVR integer property.

    Parameters
    ----------
    propertyName : bytes
        Name of the property to get.
    defaultVal : int, optional
        Return value if the property could not be set.

    Returns
    -------
    int
        Value of the property. Returns `defaultVal` if the property does not
        exist.

    """
    global _ptrSession

    cdef int to_return = capi.ovr_GetInt(
        _ptrSession,
        propertyName,
        defaultVal)

    return to_return


def setInt(bytes propertyName, int value):
    """Write a LibOVR integer property.

    Parameters
    ----------
    propertyName : bytes
        Name of the property to set.
    value : int
        Value to write.

    Returns
    -------
    bool
        ``True`` if the property was set successfully, ``False`` if the property
        was read-only or does not exist.

    Examples
    --------

    Set the performance HUD mode to show summary information::

        setInt(PERF_HUD_MODE, PERF_HUD_PERF_SUMMARY)

    Switch off the performance HUD::

        setInt(PERF_HUD_MODE, PERF_OFF)

    """
    global _ptrSession

    cdef capi.ovrBool to_return = capi.ovr_SetInt(
        _ptrSession,
        propertyName,
        value)

    return to_return == capi.ovrTrue


def getFloat(bytes propertyName, float defaultVal=0.0):
    """Read a LibOVR floating point number property.

    Parameters
    ----------
    propertyName : bytes
        Name of the property to get.
    defaultVal : float, optional
        Return value if the property could not be set. Returns 0.0 if not
        specified.

    Returns
    -------
    float
        Value of the property. Returns `defaultVal` if the property does not
        exist.

    """
    global _ptrSession

    cdef float to_return = capi.ovr_GetFloat(
        _ptrSession,
        propertyName,
        defaultVal)

    return to_return


def setFloat(bytes propertyName, float value):
    """Write a LibOVR floating point number property.

    Parameters
    ----------
    propertyName : bytes
        Name of the property to set.
    value : float
        Value to write.

    Returns
    -------
    bool
        ``True`` if the property was set successfully, ``False`` if the property
        was read-only or does not exist.

    """
    global _ptrSession

    cdef capi.ovrBool to_return = capi.ovr_SetFloat(
        _ptrSession,
        propertyName,
        value)

    return to_return == capi.ovrTrue


def setFloatArray(bytes propertyName, np.ndarray[np.float32_t, ndim=1] values):
    """Write a LibOVR floating point number property.

    Parameters
    ----------
    propertyName : bytes
        Name of the property to set.
    values : ndarray
        Value to write, must be 1-D and have dtype=float32.

    Returns
    -------
    bool
        ``True`` if the property was set successfully, ``False`` if the property
        was read-only or does not exist.

    Examples
    --------

    Set the position of the stereo debug guide::

        guidePos = numpy.asarray([0., 0., -10.0], dtype=np.float32)
        setFloatArray(DEBUG_HUD_STEREO_GUIDE_POSITION, guidePos)

    """
    global _ptrSession

    cdef Py_ssize_t valuesCapacity = len(values)
    cdef capi.ovrBool to_return = capi.ovr_SetFloatArray(
        _ptrSession,
        propertyName,
        &values[0],
        <unsigned int>valuesCapacity)

    return to_return == capi.ovrTrue


def getFloatArray(bytes propertyName, np.ndarray[np.float32_t, ndim=1] values not None):
    """Read a LibOVR float array property.

    Parameters
    ----------
    propertyName : bytes
        Name of the property to get.
    values : ndarray
        Output array array for values, must be 1-D and have dtype=float32.

    Returns
    -------
    int
        Number of values successfully read from the property.

    Examples
    --------

    Get the position of the stereo debug guide::

        guidePos = numpy.zeros((3,), dtype=np.float32)  # array to write to
        result = getFloatArray(DEBUG_HUD_STEREO_GUIDE_POSITION, guidePos)

        # check if the array we specified was long enough to store the values
        if result <= len(guidePos):
            # success

    """
    global _ptrSession

    cdef Py_ssize_t valuesCapacity = len(values)
    cdef unsigned int to_return = capi.ovr_GetFloatArray(
        _ptrSession,
        propertyName,
        &values[0],
        <unsigned int>valuesCapacity)

    return to_return


def setString(bytes propertyName, object value):
    """Write a LibOVR floating point number property.

    Parameters
    ----------
    propertyName : bytes
        Name of the property to set.
    value : str or bytes
        Value to write.

    Returns
    -------
    bool
        ``True`` if the property was set successfully, ``False`` if the property
        was read-only or does not exist.

    """
    global _ptrSession

    cdef object value_in = None
    if isinstance(value, str):
        value_in = value.encode('UTF-8')
    elif isinstance(value, bytes):
        value_in = value
    else:
        raise TypeError("Default value must be type 'str' or 'bytes'.")

    cdef capi.ovrBool to_return = capi.ovr_SetString(
        _ptrSession,
        propertyName,
        value_in)

    return to_return == capi.ovrTrue


def getString(bytes propertyName, object defaultVal=''):
    """Read a LibOVR string property.

    Parameters
    ----------
    propertyName : bytes
        Name of the property to get.
    defaultVal : bytes, optional
        Return value if the property could not be set. Returns 0.0 if not
        specified.

    Returns
    -------
    str
        Value of the property. Returns `defaultVal` if the property does not
        exist.

    Notes
    -----

    * Strings passed to this function are converted to bytes before being passed
      to ``OVR::ovr_GetString``.

    """
    global _ptrSession

    cdef object value_in = None
    if isinstance(defaultVal, str):
        value_in = defaultVal.encode('UTF-8')
    elif isinstance(defaultVal, bytes):
        value_in = defaultVal
    else:
        raise TypeError("Default value must be type 'str' or 'bytes'.")

    cdef const char* to_return = capi.ovr_GetString(
        _ptrSession,
        propertyName,
        value_in)

    return to_return.decode('UTF-8')