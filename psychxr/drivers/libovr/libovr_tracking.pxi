#  =============================================================================
#  libovr_tracking.pxi - Tracking related extension types and functions
#  =============================================================================
#
#  libovr_tracking.pxi
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
# tracking origin types
TRACKING_ORIGIN_EYE_LEVEL = capi.ovrTrackingOrigin_EyeLevel
TRACKING_ORIGIN_FLOOR_LEVEL = capi.ovrTrackingOrigin_FloorLevel

# trackings state status flags
STATUS_ORIENTATION_TRACKED = capi.ovrStatus_OrientationTracked
STATUS_POSITION_TRACKED = capi.ovrStatus_PositionTracked
STATUS_ORIENTATION_VALID = capi.ovrStatus_OrientationValid
STATUS_POSITION_VALID = capi.ovrStatus_PositionValid

# tracked device types
TRACKED_DEVICE_TYPE_HMD = capi.ovrTrackedDevice_HMD
TRACKED_DEVICE_TYPE_LTOUCH = capi.ovrTrackedDevice_LTouch
TRACKED_DEVICE_TYPE_RTOUCH = capi.ovrTrackedDevice_RTouch
TRACKED_DEVICE_TYPE_TOUCH = capi.ovrTrackedDevice_Touch
TRACKED_DEVICE_TYPE_OBJECT0 = capi.ovrTrackedDevice_Object0
TRACKED_DEVICE_TYPE_OBJECT1 = capi.ovrTrackedDevice_Object1
TRACKED_DEVICE_TYPE_OBJECT2 = capi.ovrTrackedDevice_Object2
TRACKED_DEVICE_TYPE_OBJECT3 = capi.ovrTrackedDevice_Object3

cdef class LibOVRTrackingState(object):
    """Class for tracking state information.

    Instances of this class are returned by :func:`getTrackingState` calls, with
    data referenced to the specified absolute time. Pose states with tracked
    position and orientation, as well as first and second order motion
    derivatives, for the head and hands can be accessed through attributes
    :py:attr:`~LibOVRTrackingState.headPose` and
    :py:attr:`~LibOVRTrackingState.handPoses`.

    Status flags describe the status of sensor tracking when a tracking
    state was sampled, accessible for the head and hands through the
    :py:attr:`~LibOVRTrackingState.statusFlags` and
    :py:attr:`~LibOVRTrackingState.handStatusFlags`, respectively. You can
    check each status bit by using the following values:

    * ``STATUS_ORIENTATION_TRACKED``: Orientation is tracked/reported.
    * ``STATUS_ORIENTATION_VALID``: Orientation is valid for application use.
    * ``STATUS_POSITION_TRACKED``: Position is tracked/reported.
    * ``STATUS_POSITION_VALID``: Position is valid for application use.

    As of SDK 1.39, `*_VALID` flags should be used to determine if tracking data
    is usable by the application.

    """
    cdef capi.ovrTrackingState* c_data
    cdef bint ptr_owner

    cdef LibOVRPoseState _headPoseState
    cdef LibOVRPoseState _leftHandPoseState
    cdef LibOVRPoseState _rightHandPoseState
    cdef LibOVRPose _calibratedOrigin

    def __init__(self):
        """
        Attributes
        ----------
        headPose : LibOVRPoseState
        handPoses : tuple
        statusFlags : int
        positionValid : bool
        orientationValid : bool
        handStatusFlags : tuple
        handPositionValid : tuple
        handOrientationValid : tuple
        calibratedOrigin : LibOVRPose
        """
        self._new_struct()

    def __cinit__(self):
        self.ptr_owner = False

    @staticmethod
    cdef LibOVRTrackingState fromPtr(capi.ovrTrackingState* ptr, bint owner=False):
        cdef LibOVRTrackingState wrapper = \
            LibOVRTrackingState.__new__(LibOVRTrackingState)
        wrapper.c_data = ptr
        wrapper.ptr_owner = owner

        wrapper._headPoseState = LibOVRPoseState.fromPtr(&ptr.HeadPose)
        wrapper._leftHandPoseState = LibOVRPoseState.fromPtr(&ptr.HandPoses[0])
        wrapper._rightHandPoseState = LibOVRPoseState.fromPtr(&ptr.HandPoses[1])
        wrapper._calibratedOrigin = LibOVRPose.fromPtr(&ptr.CalibratedOrigin)

        return wrapper

    cdef void _new_struct(self):
        if self.c_data is not NULL:
            return

        cdef capi.ovrTrackingState* ptr = \
            <capi.ovrTrackingState*>PyMem_Malloc(sizeof(capi.ovrTrackingState))

        if ptr is NULL:
            raise MemoryError

        self.c_data = ptr
        self.ptr_owner = True

        self._headPoseState = LibOVRPoseState.fromPtr(&ptr.HeadPose)
        self._leftHandPoseState = LibOVRPoseState.fromPtr(&ptr.HandPoses[0])
        self._rightHandPoseState = LibOVRPoseState.fromPtr(&ptr.HandPoses[1])
        self._calibratedOrigin = LibOVRPose.fromPtr(&ptr.CalibratedOrigin)

    def __dealloc__(self):
        if self.c_data is not NULL:
            if self.ptr_owner is True:
                PyMem_Free(self.c_data)
                self.c_data = NULL

    @property
    def headPose(self):
        """Head pose state (`LibOVRPoseState`)."""
        return self._headPoseState

    @property
    def handPoses(self):
        """Hand pose states (`LibOVRPoseState`, `LibOVRPoseState`).

        Examples
        --------
        Get the left and right hand pose states::

            leftHandPoseState, rightHandPoseState = trackingState.handPoses

        """
        return self._leftHandPoseState, self._rightHandPoseState

    @property
    def statusFlags(self):
        """Head tracking status flags (`int`).

        Examples
        --------
        Check if orientation was tracked and data is valid for use::

            # check if orientation is tracked and valid
            statusFlags = STATUS_ORIENTATION_TRACKED | STATUS_ORIENTATION_VALID
            if (trackingState.statusFlags & statusFlags) == statusFlags:
                print("Orientation is tracked and valid")

        """
        return self.c_data.StatusFlags

    @property
    def positionValid(self):
        """`True` if position tracking is valid."""
        return (self.c_data.StatusFlags & STATUS_POSITION_VALID) == \
               STATUS_POSITION_VALID

    @property
    def orientationValid(self):
        """`True` if orientation tracking is valid."""
        return (self.c_data.StatusFlags & STATUS_ORIENTATION_VALID) == \
               STATUS_ORIENTATION_VALID

    @property
    def handStatusFlags(self):
        """Hand tracking status flags (`int`, `int`)."""
        return self.c_data.HandStatusFlags[0], self.c_data.HandStatusFlags[1]

    @property
    def handPositionValid(self):
        """`True` if position tracking is valid."""
        return (self.c_data.StatusFlags & STATUS_POSITION_VALID) == \
               STATUS_POSITION_VALID

    @property
    def handOrientationValid(self):
        """Hand orientation tracking is valid (`bool`, `bool`).

        Examples
        --------
        Check if orientation is valid for the right hand's tracking state::

            rightHandOriTracked = trackingState.handOrientationValid[HAND_RIGHT]

        """
        cdef bint left_hand = (
            self.c_data.HandStatusFlags[HAND_LEFT] &
                STATUS_ORIENTATION_VALID) == STATUS_ORIENTATION_VALID
        cdef bint right_hand = (
            self.c_data.HandStatusFlags[HAND_RIGHT] &
                STATUS_ORIENTATION_VALID) == STATUS_ORIENTATION_VALID

        return left_hand, right_hand

    @property
    def handPositionValid(self):
        """Hand position tracking is valid (`bool`, `bool`).

        Examples
        --------
        Check if position is valid for the right hand's tracking state::

            rightHandOriTracked = trackingState.handPositionValid[HAND_RIGHT]

        """
        cdef bint left_hand = (
            self.c_data.HandStatusFlags[HAND_LEFT] &
                STATUS_POSITION_VALID) == STATUS_POSITION_VALID
        cdef bint right_hand = (
            self.c_data.HandStatusFlags[HAND_RIGHT] &
                STATUS_POSITION_VALID) == STATUS_POSITION_VALID

        return left_hand, right_hand

    @property
    def calibratedOrigin(self):
        """Pose of the calibrated origin.

        This pose is used to find the calibrated origin in space if
        :func:`recenterTrackingOrigin` or :func:`specifyTrackingOrigin` was
        called. If those functions were never called during a session, this will
        return an identity pose, which reflects the tracking origin type.

        """
        return self._calibratedOrigin


cdef class LibOVRTrackerInfo(object):
    """Class for storing tracker (sensor) information such as pose, status, and
    camera frustum information. This object is returned by calling
    :func:`~psychxr.libovr.getTrackerInfo`. Attributes of this class are
    read-only.

    """
    cdef capi.ovrTrackerPose c_ovrTrackerPose
    cdef capi.ovrTrackerDesc c_ovrTrackerDesc

    cdef LibOVRPose _pose
    cdef LibOVRPose _leveledPose

    cdef unsigned int _trackerIndex

    def __init__(self):
        """
        Attributes
        ----------
        trackerIndex : int
            Tracker index this objects refers to (read-only).
        pose : LibOVRPose
            The pose of the sensor (read-only).
        leveledPose : LibOVRPose
            Gravity aligned pose of the sensor (read-only).
        isConnected : bool
            True if the sensor is connected and available (read-only).
        isPoseTracked : bool
            True if the sensor has a valid pose (read-only).
        horizontalFov : float
            Horizontal FOV of the sensor in radians (read-only).
        verticalFov : float
            Vertical FOV of the sensor in radians (read-only).
        nearZ : float
            Near clipping plane of the sensor frustum in meters (read-only).
        farZ : float
            Far clipping plane of the sensor frustum in meters (read-only).

        """
        pass

    def __cinit__(self):
        self._pose = LibOVRPose.fromPtr(&self.c_ovrTrackerPose.Pose)
        self._leveledPose = LibOVRPose.fromPtr(&self.c_ovrTrackerPose.LeveledPose)

    @property
    def trackerIndex(self):
        """Tracker index this objects refers to (read-only)."""
        return self._trackerIndex

    @property
    def pose(self):
        """The pose of the sensor (read-only)."""
        return self._pose

    @property
    def leveledPose(self):
        """Gravity aligned pose of the sensor (read-only)."""
        return self._leveledPose

    @property
    def isConnected(self):
        """True if the sensor is connected and available (read-only)."""
        return <bint>((capi.ovrTracker_Connected &
             self.c_ovrTrackerPose.TrackerFlags) ==
                      capi.ovrTracker_Connected)

    @property
    def isPoseTracked(self):
        """True if the sensor has a valid pose (read-only)."""
        return <bint>((capi.ovrTracker_PoseTracked &
             self.c_ovrTrackerPose.TrackerFlags) ==
                      capi.ovrTracker_PoseTracked)

    @property
    def horizontalFov(self):
        """Horizontal FOV of the sensor in radians (read-only)."""
        return self.c_ovrTrackerDesc.FrustumHFovInRadians

    @property
    def verticalFov(self):
        """Vertical FOV of the sensor in radians (read-only)."""
        return self.c_ovrTrackerDesc.FrustumVFovInRadians

    @property
    def nearZ(self):
        """Near clipping plane of the sensor frustum in meters (read-only)."""
        return self.c_ovrTrackerDesc.FrustumNearZInMeters

    @property
    def farZ(self):
        """Far clipping plane of the sensor frustum in meters (read-only)."""
        return self.c_ovrTrackerDesc.FrustumFarZInMeters


def getSensorSampleTime():
    """Get the sensor sample timestamp.

    The time when the source data used to compute the render pose was sampled.
    This value is used to compute the motion-to-photon latency. This value is
    set when :func:`getDevicePoses` and :func:`setSensorSampleTime` is called.
    If :func:`getTrackingState` was called with `latencyMarker` set, sensor
    sample time will be 0.0.

    Returns
    -------
    float
        Sample timestamp in seconds.

    See Also
    --------
    setSensorSampleTime : Set sensor sample time.

    """
    global _eyeLayer
    return _eyeLayer.SensorSampleTime


def setSensorSampleTime(double absTime):
    """Set the sensor sample timestamp.

    Specify the sensor sample time of the source data used to compute the render
    poses of each eye. This value is used to compute motion-to-photon latency.

    Parameters
    ----------
    absTime : float
        Time in seconds.

    See Also
    --------
    getSensorSampleTime : Get sensor sample time.
    getTrackingState : Get the current tracking state.
    getDevicePoses : Get device poses.

    Examples
    --------
    Supplying sensor sample time from an external tracking source::

        # get sensor time from the mocal system
        sampleTime = timeInSeconds() - mocap.timeSinceMidExposure

        # set sample time
        setSensorSampleTime(sampleTime)
        calcEyePoses(headRigidBody)

        # get frame perf stats after calling `endFrame` to get last frame
        # motion-to-photon latency
        perfStats = getPerfStats()
        m2p_latency = perfStats.frameStats[0].appMotionToPhotonLatency

    """
    global _eyeLayer
    _eyeLayer.SensorSampleTime = absTime


def getTrackingState(double absTime, bint latencyMarker=True):
    """Get the current tracking state of the head and hands.

    Parameters
    ----------
    absTime : float
        Absolute time in seconds which the tracking state refers to.
    latencyMarker : bool
        Insert a latency marker for motion-to-photon calculation.

    Returns
    -------
    LibOVRTrackingState
        Tracking state at `absTime` for head and hands.

    Examples
    --------
    Getting the head pose and calculating eye render poses::

        t = hmd.getPredictedDisplayTime()
        trackingState = hmd.getTrackingState(t)

        # tracking state flags
        flags = STATUS_ORIENTATION_TRACKED | STATUS_ORIENTATION_TRACKED

        # check if tracking
        if (flags & trackingState.statusFlags) == flags:
            hmd.calcEyePose(trackingState.headPose.pose)  # calculate eye poses

    """
    global _ptrSession
    global _eyeLayer

    cdef capi.ovrBool use_marker = \
        capi.ovrTrue if latencyMarker else capi.ovrFalse

    # tracking state object that is actually returned to Python land
    cdef LibOVRTrackingState to_return = LibOVRTrackingState()
    to_return.c_data[0] = capi.ovr_GetTrackingState(
        _ptrSession, absTime, use_marker)

    return to_return


def getDevicePoses(object deviceTypes, double absTime, bint latencyMarker=True):
    """Get tracked device poses.

    Each pose in the returned array matches the device type at each index
    specified in `deviceTypes`. You need to call this function to get the poses
    for 'objects', which are additional Touch controllers that can be paired and
    tracked in the scene.

    It is recommended that :func:`getTrackingState` is used for obtaining the
    head and hand poses.

    Parameters
    ----------
    deviceTypes : list or tuple of int
        List of device types. Valid device types identifiers are:

        * ``TRACKED_DEVICE_TYPE_HMD`` : The head or HMD.
        * ``TRACKED_DEVICE_TYPE_LTOUCH`` : Left touch controller or hand.
        * ``TRACKED_DEVICE_TYPE_RTOUCH`` : Right touch controller or hand.
        * ``TRACKED_DEVICE_TYPE_TOUCH`` : Both touch controllers.

        Up to four additional touch controllers can be paired and tracked, they
        are assigned as:

        * ``TRACKED_DEVICE_TYPE_OBJECT0``
        * ``TRACKED_DEVICE_TYPE_OBJECT1``
        * ``TRACKED_DEVICE_TYPE_OBJECT2``
        * ``TRACKED_DEVICE_TYPE_OBJECT3``

    absTime : float
        Absolute time in seconds poses refer to.
    latencyMarker: bool, optional
        Insert a marker for motion-to-photon latency calculation. Set this to
        False if :func:`getTrackingState` was previously called and a latency
        marker was set there. The latency marker is set to the absolute time
        this function was called.

    Returns
    -------
    tuple
        Return code (`int`) of the ``OVR::ovr_GetDevicePoses`` API call and list
        of tracked device poses (`list` of :py:class:`LibOVRPoseState`). If a
        device cannot be tracked, the return code will be
        ``ERROR_LOST_TRACKING``.

    Warning
    -------
    If multiple devices were specified with `deviceTypes`, the return code will
    be ``ERROR_LOST_TRACKING`` if ANY of the devices lost tracking.

    Examples
    --------

    Get HMD and touch controller poses::

        deviceTypes = (TRACKED_DEVICE_TYPE_HMD,
                       TRACKED_DEVICE_TYPE_LTOUCH,
                       TRACKED_DEVICE_TYPE_RTOUCH)
        headPose, leftHandPose, rightHandPose = getDevicePoses(
            deviceTypes, absTime)

    """
    # give a success code and empty pose list if an empty list was specified
    global _ptrSession
    global _eyeLayer

    if not deviceTypes:
        if latencyMarker:
            _eyeLayer.SensorSampleTime = capi.ovr_GetTimeInSeconds()
        return capi.ovrSuccess, []

    # allocate arrays to store pose types and poses
    cdef int count = <int>len(deviceTypes)
    cdef capi.ovrTrackedDeviceType* devices = \
        <capi.ovrTrackedDeviceType*>PyMem_Malloc(
            count * sizeof(capi.ovrTrackedDeviceType))
    if not devices:
        raise MemoryError("Failed to allocate array 'devices'.")

    cdef int i = 0
    for i in range(count):
        devices[i] = <capi.ovrTrackedDeviceType>deviceTypes[i]

    cdef capi.ovrPoseStatef* devicePoses = \
        <capi.ovrPoseStatef*>PyMem_Malloc(
            count * sizeof(capi.ovrPoseStatef))
    if not devicePoses:
        raise MemoryError("Failed to allocate array 'devicePoses'.")

    # get the device poses
    cdef capi.ovrResult result = capi.ovr_GetDevicePoses(
        _ptrSession,
        devices,
        count,
        absTime,
        devicePoses)

    # for computing app photon-to-motion latency
    if latencyMarker:
        _eyeLayer.SensorSampleTime = capi.ovr_GetTimeInSeconds()

    # build list of device poses
    cdef list outPoses = list()
    cdef LibOVRPoseState thisPose
    for i in range(count):
        thisPose = LibOVRPoseState()  # new
        thisPose.c_data[0] = devicePoses[i]
        outPoses.append(thisPose)

    # free the allocated arrays
    PyMem_Free(devices)
    PyMem_Free(devicePoses)

    return result, outPoses


def getTrackingOriginType():
    """Get the current tracking origin type.

    The tracking origin type specifies where the origin is placed when computing
    the pose of tracked objects (i.e. the head and touch controllers.) Valid
    values are ``TRACKING_ORIGIN_EYE_LEVEL`` and
    ``TRACKING_ORIGIN_FLOOR_LEVEL``.

    See Also
    --------
    setTrackingOriginType : Set the tracking origin type.

    """
    global _ptrSession
    cdef capi.ovrTrackingOrigin originType = \
        capi.ovr_GetTrackingOriginType(_ptrSession)

    if originType == capi.ovrTrackingOrigin_FloorLevel:
        return TRACKING_ORIGIN_FLOOR_LEVEL
    elif originType == capi.ovrTrackingOrigin_EyeLevel:
        return TRACKING_ORIGIN_EYE_LEVEL


def setTrackingOriginType(int value):
    """Set the tracking origin type.

    Specify the tracking origin to use when computing eye poses. Subsequent
    calls of :func:`calcEyePoses` will use the set tracking origin.

    Parameters
    ----------
    value : int
        Tracking origin type, must be either ``TRACKING_ORIGIN_FLOOR_LEVEL`` or
        ``TRACKING_ORIGIN_EYE_LEVEL``.

    Returns
    -------
    int
        Result of the ``OVR::ovr_SetTrackingOriginType`` LibOVR API call.

    See Also
    --------
    getTrackingOriginType : Get the current tracking origin type.

    """
    cdef capi.ovrResult result
    global _ptrSession
    if value == TRACKING_ORIGIN_FLOOR_LEVEL:
        result = capi.ovr_SetTrackingOriginType(
            _ptrSession, capi.ovrTrackingOrigin_FloorLevel)
    elif value == TRACKING_ORIGIN_EYE_LEVEL:
        result = capi.ovr_SetTrackingOriginType(
            _ptrSession, capi.ovrTrackingOrigin_EyeLevel)
    else:
        raise ValueError("Invalid tracking origin type specified "
                         "must be 'TRACKING_ORIGIN_FLOOR_LEVEL' or "
                         "'TRACKING_ORIGIN_EYE_LEVEL'.")

    return result


def recenterTrackingOrigin():
    """Recenter the tracking origin.

    Returns
    -------
    int
        The result of the LibOVR API call ``OVR::ovr_RecenterTrackingOrigin``.

    Examples
    --------
    Recenter the tracking origin if requested by the session status::

        sessionStatus = getSessionStatus()
        if sessionStatus.shouldRecenter:
            recenterTrackingOrigin()

    """
    global _ptrSession
    cdef capi.ovrResult result = capi.ovr_RecenterTrackingOrigin(
        _ptrSession)

    return result


def specifyTrackingOrigin(LibOVRPose newOrigin):
    """Specify a new tracking origin.

    Parameters
    ----------
    newOrigin : LibOVRPose
        New origin to use.

    """
    global _ptrSession
    cdef capi.ovrResult result = capi.ovr_SpecifyTrackingOrigin(
        _ptrSession,
        newOrigin.c_data[0])

    return result


def clearShouldRecenterFlag():
    """Clear the :py:attr:`LibOVRSessionStatus.shouldRecenter` flag.

    """
    global _ptrSession
    capi.ovr_ClearShouldRecenterFlag(_ptrSession)


def getTrackerCount():
    """Get the number of attached trackers.

    Returns
    -------
    int
        Number of trackers reported by `LibOVR`.

    Notes
    -----
    * The Oculus Rift S uses inside-out tracking, therefore does not have
      external trackers. For compatibility, `LibOVR` will return a tracker count
      of 3.

    """
    global _ptrSession
    cdef unsigned int trackerCount = capi.ovr_GetTrackerCount(
        _ptrSession)

    return <int>trackerCount


def getTrackerInfo(int trackerIndex):
    """Get information about a given tracker.

    Parameters
    ----------
    trackerIndex : int
        The index of the sensor to query. Valid values are between 0 and
        :func:`getTrackerCount` - 1.

    Notes
    -----
    * The Oculus Rift S uses inside-out tracking, therefore does not have
      external trackers. For compatibility, `LibOVR` will dummy tracker objects.

    """
    cdef LibOVRTrackerInfo to_return = LibOVRTrackerInfo()
    global _ptrSession

    # set the tracker index
    to_return._trackerIndex = <unsigned int>trackerIndex

    # set the descriptor data
    to_return.c_ovrTrackerDesc = capi.ovr_GetTrackerDesc(
        _ptrSession, <unsigned int>trackerIndex)
    # get the tracker pose
    to_return.c_ovrTrackerPose = capi.ovr_GetTrackerPose(
        _ptrSession, <unsigned int>trackerIndex)

    return to_return