#  =============================================================================
#  libovr_error.pxi - LibOVR API errors
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

# error information
cdef capi.ovrErrorInfo _errorInfo  # store our last error here


# Function to check for errors returned by OVRLib functions
#
cdef capi.ovrErrorInfo _last_error_info_  # store our last error here
def check_result(result):
    if capi.OVR_FAILURE(result):
        capi.ovr_GetLastErrorInfo(&_last_error_info_)
        raise RuntimeError(
            str(result) + ": " + _last_error_info_.ErrorString.decode("utf-8"))


def success(int result):
    """Check if an API return indicates success.

    Returns
    -------
    bool
        ``True`` if API call was an successful (`result` > 0).

    """
    return <bint>capi.OVR_SUCCESS(result)


def unqualifiedSuccess(int result):
    """Check if an API return indicates unqualified success.

    Returns
    -------
    bool
        ``True`` if API call was an unqualified success (`result` == 0).

    """
    return <bint>capi.OVR_UNQUALIFIED_SUCCESS(result)


def failure(int result):
    """Check if an API return indicates failure (error).

    Returns
    -------
    bool
        ``True`` if API call returned an error (`result` < 0).

    """
    return <bint>capi.OVR_FAILURE(result)


def getLastErrorInfo():
    """Get the last error code and information string reported by the API.

    This function can be used when implementing custom error handlers for
    dealing with exceptions raised by LibOVR. You must call
    :func:`getLastErrorInfo` every time after any function which makes an LibOVR
    API call if you wish to catch all errors, since only the most recent is
    returned.

    Returns
    -------
    tuple (int, str)
        Tuple of the API call result and error string. If there was no API
        error, the function will return tuple (0, '<unknown>').

    Examples
    --------

    Raise a Python exception if LibOVR reports an error::

        result = create()
        if failure(result):
            errorVal, errorMsg = getLastErrorInfo()
            raise RuntimeError(errorMsg)  # Python generic runtime error!

    """
    cdef capi.ovrErrorInfo lastErrorInfo  # store our last error here
    capi.ovr_GetLastErrorInfo(&lastErrorInfo)

    cdef capi.ovrResult result = lastErrorInfo.Result
    cdef str errorString = lastErrorInfo.ErrorString.decode("utf-8")

    return <int>result, errorString
