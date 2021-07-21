#  =============================================================================
#  libovr_session.pxi - VR session and status
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

# return success codes, values other than 'SUCCESS' are conditional
SUCCESS = capi.ovrSuccess
SUCCESS_NOT_VISIBLE = capi.ovrSuccess_NotVisible
SUCCESS_DEVICE_UNAVAILABLE = capi.ovrSuccess_DeviceUnavailable
SUCCESS_BOUNDARY_INVALID = capi.ovrSuccess_BoundaryInvalid

cdef capi.ovrInitParams _initParams  # initialization parameters
cdef capi.ovrSession _ptrSession  # session pointer
cdef capi.ovrGraphicsLuid _gfxLuid  # LUID
cdef capi.ovrSessionStatus _sessionStatus


cdef class LibOVRSessionStatus(object):
    """Class for storing session status information. An instance of this class
    is returned when :func:`getSessionStatus` is called.

    One can check if there was a status change between calls of
    :func:`getSessionStatus` by using the ``==`` and ``!=`` operators on the
    returned :py:class:`LibOVRSessionStatus` instances.

    """
    cdef capi.ovrSessionStatus c_data

    def __eq__(self, LibOVRSessionStatus other):
        """Equality test between status objects. Use this to check if the status
        is unchanged between two :func:`getSessionStatus` calls.

        """
        return (self.c_data.IsVisible == other.c_data.IsVisible and
               self.c_data.HmdPresent == other.c_data.HmdPresent and
               self.c_data.hmdMounted == other.c_data.hmdMounted and
               self.c_data.displayLost == other.c_data.displayLost and
               self.c_data.shouldQuit == other.c_data.shouldQuit and
               self.c_data.shouldRecenter == other.c_data.shouldRecenter and
               self.c_data.hasInputFocus == other.c_data.hasInputFocus and
               self.c_data.overlayPresent == other.c_data.overlayPresent and
               self.c_data.depthRequested == other.c_data.depthRequested)

    def __ne__(self, LibOVRSessionStatus other):
        """Equality test between status objects. Use this to check if the status
        differs between two :func:`getSessionStatus` calls.

        """
        return not (self.c_data.IsVisible == other.c_data.IsVisible and
               self.c_data.HmdPresent == other.c_data.HmdPresent and
               self.c_data.hmdMounted == other.c_data.hmdMounted and
               self.c_data.displayLost == other.c_data.displayLost and
               self.c_data.shouldQuit == other.c_data.shouldQuit and
               self.c_data.shouldRecenter == other.c_data.shouldRecenter and
               self.c_data.hasInputFocus == other.c_data.hasInputFocus and
               self.c_data.overlayPresent == other.c_data.overlayPresent and
               self.c_data.depthRequested == other.c_data.depthRequested)

    @property
    def isVisible(self):
        """``True`` the application has focus and visible in the HMD."""
        return self.c_data.IsVisible == capi.ovrTrue

    @property
    def hmdPresent(self):
        """``True`` if the HMD is present."""
        return self.c_data.HmdPresent == capi.ovrTrue

    @property
    def hmdMounted(self):
        """``True`` if the HMD is being worn on the user's head."""
        return self.c_data.HmdMounted == capi.ovrTrue

    @property
    def displayLost(self):
        """``True`` the the display was lost.

        If occurs, the HMD was disconnected and the current session is invalid.
        You need to destroy all resources associated with current session and
        call :func:`create` again. Alternatively, you can raise an error and
        shutdown the application.
        """
        return self.c_data.DisplayLost == capi.ovrTrue

    @property
    def shouldQuit(self):
        """``True`` the application was signaled to quit.

        This can occur if the user requests the application exit through the
        system UI menu. You can ignore this flag if needed.
        """
        return self.c_data.ShouldQuit == capi.ovrTrue

    @property
    def shouldRecenter(self):
        """``True`` if the application was signaled to recenter.

        This happens when the user requests the application recenter the VR
        scene on their current physical location through the system UI. You can
        ignore this request or clear it by calling
        :func:`clearShouldRecenterFlag`.

        """
        return self.c_data.ShouldRecenter == capi.ovrTrue

    @property
    def hasInputFocus(self):
        """``True`` if the application has input focus.

        If the application has focus, the statistics presented by the
        performance HUD will reflect the current application's frame statistics.

        """
        return self.c_data.HasInputFocus == capi.ovrTrue

    @property
    def overlayPresent(self):
        """``True`` if the system UI is visible."""
        return self.c_data.OverlayPresent == capi.ovrTrue

    @property
    def depthRequested(self):
        """``True`` if the a depth texture is requested.

        Notes
        -----
        * This feature is currently unused by PsychXR.

        """
        return self.c_data.DepthRequested == capi.ovrTrue


def initialize(bint focusAware=False, int connectionTimeout=0, object logCallback=None):
    """Initialize the session.

    Parameters
    ----------
    focusAware : bool, optional
        Client is focus aware.
    connectionTimeout : bool, optional
        Timeout in milliseconds for connecting to the server.
    logCallback : object, optional
        Python callback function for logging. May be called at anytime from
        any thread until :func:`shutdown` is called. Function must accept
        arguments `level` and `message`. Where `level` is passed the logging
        level and `message` the message string. Callbacks message levels can be
        ``LOG_LEVEL_DEBUG``, ``LOG_LEVEL_INFO``, and ``LOG_LEVEL_ERROR``. The
        application can filter messages accordingly.

    Returns
    -------
    int
        Return code of the LibOVR API call ``OVR::ovr_Initialize``. Returns
        ``SUCCESS`` if completed without errors. In the event of an
        error, possible return values are:

        * ``ERROR_INITIALIZE``: Initialization error.
        * ``ERROR_LIB_LOAD``:  Failed to load LibOVRRT.
        * ``ERROR_LIB_VERSION``:  LibOVRRT version incompatible.
        * ``ERROR_SERVICE_CONNECTION``:  Cannot connect to OVR service.
        * ``ERROR_SERVICE_VERSION``: OVR service version is incompatible.
        * ``ERROR_INCOMPATIBLE_OS``: Operating system version is incompatible.
        * ``ERROR_DISPLAY_INIT``: Unable to initialize the HMD.
        * ``ERROR_SERVER_START``:  Cannot start a server.
        * ``ERROR_REINITIALIZATION``: Reinitialized with a different version.

    Examples
    --------
    Passing a callback function for logging::

        def myLoggingCallback(level, message):
            level_text = {
                LOG_LEVEL_DEBUG: '[DEBUG]:',
                LOG_LEVEL_INFO: '[INFO]:',
                LOG_LEVEL_ERROR: '[ERROR]:'}

            # print message like '[INFO]: IAD changed to 62.1mm'
            print(level_text[level], message)

        result = initialize(logCallback=myLoggingCallback)

    """
    cdef int32_t flags = capi.ovrInit_RequestVersion
    if focusAware is True:
        flags |= capi.ovrInit_FocusAware

    #if debug is True:
    #    flags |= capi.ovrInit_Debug
    global _initParams
    _initParams.Flags = flags
    _initParams.RequestedMinorVersion = capi.OVR_MINOR_VERSION

    if logCallback is not None:
        _initParams.LogCallback = <capi.ovrLogCallback>LibOVRLogCallback
        _initParams.UserData = <uintptr_t>(<void*>logCallback)
    else:
        _initParams.LogCallback = NULL

    _initParams.ConnectionTimeoutMS = <uint32_t>connectionTimeout
    cdef capi.ovrResult result = capi.ovr_Initialize(
        &_initParams)

    return result  # failed to initalize, return error code


def create():
    """Create a new session. Control is handed over to the application from
    Oculus Home.

    Starting a session will initialize and create a new session. Afterwards
    API functions will return valid values. You can only create one session per
    interpreter thread. All other files/modules within the same thread which
    import PsychXR make API calls to the same session after `create` is called.

    Returns
    -------
    int
        Result of the ``OVR::ovr_Create`` API call. A session was successfully
        created if the result is ``SUCCESS``.

    """
    global _ptrSession
    global _gfxLuid
    global _eyeLayer
    global _hmdDesc
    global _eyeRenderDesc

    result = capi.ovr_Create(&_ptrSession, &_gfxLuid)
    check_result(result)
    if capi.OVR_FAILURE(result):
        return result  # failed to create session, return error code

    # if we got to this point, everything should be fine
    # get HMD descriptor
    _hmdDesc = capi.ovr_GetHmdDesc(_ptrSession)

    # configure the eye render descriptor to use the recommended FOV, this
    # can be changed later
    cdef Py_ssize_t i = 0
    for i in range(capi.ovrEye_Count):
        _eyeRenderDesc[i] = capi.ovr_GetRenderDesc(
            _ptrSession,
            <capi.ovrEyeType>i,
            _hmdDesc.DefaultEyeFov[i])

        _eyeLayer.Fov[i] = _eyeRenderDesc[i].Fov

    return result


def checkSessionStarted():
    """Check of a session has been created.

    This value should return `True` between calls of :func:`create` and
    :func:`destroy`. You can use this to determine if you can make API calls
    which require an active session.

    Returns
    -------
    bool
        `True` if a session is present.

    """
    return _ptrSession != NULL


def destroy():
    """Destroy a session.

    Must be called after every successful :func:`create` call. Calling destroy
    will invalidate the current session and all resources must be freed and
    re-created.

    """
    global _ptrSession
    global _eyeLayer
    # null eye textures in eye layer
    _eyeLayer.ColorTexture[0] = _eyeLayer.ColorTexture[1] = NULL

    # destroy the current session and shutdown
    capi.ovr_Destroy(_ptrSession)
    _ptrSession = NULL


def shutdown():
    """End the current session.

    Clean-up routines are executed that destroy all swap chains and mirror
    texture buffers, afterwards control is returned to Oculus Home. This
    must be called after every successful :func:`initialize` call.

    Notes
    -----
    * As of `PsychXR` version `0.2.4`, calling `shutdown` results in an
      access violation (0xC0000005) when Python exits after creating a swap
      chain.

    """
    capi.ovr_Shutdown()


def getGraphicsLUID():
    """The graphics device LUID.

    Returns
    -------
    str
        Reserved graphics LUID.

    """
    global _gfxLuid
    return _gfxLuid.Reserved.decode('utf-8')


def isOculusServiceRunning(int timeoutMs=100):
    """Check if the Oculus Runtime is loaded and running.

    Parameters
    ----------
    timeoutMS : int
        Timeout in milliseconds.

    Returns
    -------
    bool
        True if the Oculus background service is running.

    """
    cdef capi.ovrDetectResult result = capi.ovr_Detect(
        timeoutMs)

    return <bint>result.IsOculusServiceRunning


def isHmdConnected(int timeoutMs=100):
    """Check if an HMD is connected.

    Parameters
    ----------
    timeoutMs : int
        Timeout in milliseconds.

    Returns
    -------
    bool
        True if a LibOVR compatible HMD is connected.

    """
    cdef capi.ovrDetectResult result = capi.ovr_Detect(
        timeoutMs)

    return <bint>result.IsOculusHMDConnected


def getSessionStatus():
    """Get the current session status.

    Returns
    -------
    tuple (int, LibOVRSessionStatus)
        Result of LibOVR API call ``OVR::ovr_GetSessionStatus`` and a
        :py:class:`LibOVRSessionStatus`.

    Examples
    --------

    Check if the display is visible to the user::

        result, sessionStatus = getSessionStatus()
        if sessionStatus.isVisible:
            # begin frame rendering ...

    Quit if the user requests to through the Oculus overlay::

        result, sessionStatus = getSessionStatus()
        if sessionStatus.shouldQuit:
            # destroy any swap chains ...
            destroy()
            shutdown()

    """
    global _ptrSession
    global _sessionStatus

    cdef capi.ovrResult result = capi.ovr_GetSessionStatus(_ptrSession,
                                                           &_sessionStatus)

    cdef LibOVRSessionStatus to_return = LibOVRSessionStatus()
    to_return.c_data = _sessionStatus

    return result, to_return
