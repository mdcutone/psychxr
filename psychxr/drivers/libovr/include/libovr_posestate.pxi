#  =============================================================================
#  libovr_posestate.pxi - Wrapper extensions type for ovrPoseStatef
#  =============================================================================
#
#  libovr_posestate.pxi
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

cdef class LibOVRPoseState(object):
    """Class for representing rigid body poses with additional state
    information.

    Pose states contain the pose of the tracked body, but also angular and
    linear motion derivatives experienced by the pose. The pose within a state
    can be accessed via the :py:attr:`~psychxr.libovr.LibOVRPoseState.thePose`
    attribute.

    Velocity and acceleration for linear and angular motion can be used to
    compute forces applied to rigid bodies and predict the future positions of
    objects (see :py:meth:`~psychxr.libovr.LibOVRPoseState.timeIntegrate`). You
    can create `LibOVRPoseState` objects using data from other sources, such as
    *n*DOF IMUs for use with VR environments.

    Parameters
    ----------
    thePose : LibOVRPose, list, tuple or None
        Rigid body pose this state refers to. Can be a `LibOVRPose` pose
        instance or a tuple/list of a position coordinate (x, y, z) and
        orientation quaternion (x, y, z, w). If ``None`` the pose will be
        initialized as an identity pose.
    linearVelocity : array_like
        Linear acceleration vector [vx, vy, vz] in meters/sec.
    angularVelocity : array_like
        Angular velocity vector [vx, vy, vz] in radians/sec.
    linearAcceleration : array_like
        Linear acceleration vector [ax, ay, az] in meters/sec^2.
    angularAcceleration : array_like
        Angular acceleration vector [ax, ay, az] in radians/sec^2.
    timeInSeconds : float
        Time in seconds this state refers to.

    """
    cdef capi.ovrPoseStatef* c_data
    cdef bint ptr_owner  # owns the data

    # these will hold references until this object is de-allocated
    cdef LibOVRPose _thePose
    cdef np.ndarray _linearVelocity
    cdef np.ndarray _angularVelocity
    cdef np.ndarray _linearAcceleration
    cdef np.ndarray _angularAcceleration

    def __init__(self,
                 object thePose=None,
                 object linearVelocity=(0., 0., 0.),
                 object angularVelocity=(0., 0., 0.),
                 object linearAcceleration=(0., 0. ,0.),
                 object angularAcceleration=(0., 0., 0.),
                 double timeInSeconds=0.0):
        """
        Attributes
        ----------
        thePose : LibOVRPose
        angularVelocity : ndarray
        linearVelocity : ndarray
        angularAcceleration : ndarray
        linearAcceleration : ndarray
        timeInSeconds : float

        """
        self._new_struct(
            thePose,
            linearVelocity,
            angularVelocity,
            linearAcceleration,
            angularAcceleration,
            timeInSeconds)

    def __cinit__(self, *args, **kwargs):
        self.ptr_owner = False

    @staticmethod
    cdef LibOVRPoseState fromPtr(capi.ovrPoseStatef* ptr, bint owner=False):
        # bypass __init__ if wrapping a pointer
        cdef LibOVRPoseState wrapper = LibOVRPoseState.__new__(LibOVRPoseState)
        wrapper.c_data = ptr
        wrapper.ptr_owner = owner

        wrapper._thePose = LibOVRPose.fromPtr(&wrapper.c_data.ThePose)
        wrapper._linearVelocity = _wrap_ovrVector3f_as_ndarray(
                &wrapper.c_data.LinearVelocity)
        wrapper._linearAcceleration = _wrap_ovrVector3f_as_ndarray(
                &wrapper.c_data.LinearAcceleration)
        wrapper._angularVelocity = _wrap_ovrVector3f_as_ndarray(
                &wrapper.c_data.AngularVelocity)
        wrapper._angularAcceleration = _wrap_ovrVector3f_as_ndarray(
                &wrapper.c_data.AngularAcceleration)

        return wrapper

    cdef void _new_struct(
            self,
            object pose,
            object linearVelocity,
            object angularVelocity,
            object linearAcceleration,
            object angularAcceleration,
            double timeInSeconds):

        if self.c_data is not NULL:  # already allocated, __init__ called twice?
            return

        cdef capi.ovrPoseStatef* _ptr = \
            <capi.ovrPoseStatef*>PyMem_Malloc(
                sizeof(capi.ovrPoseStatef))

        if _ptr is NULL:
            raise MemoryError

        self.c_data = _ptr
        self.ptr_owner = True

        # setup property wrappers
        self._thePose = LibOVRPose.fromPtr(&self.c_data.ThePose)
        self._linearVelocity = _wrap_ovrVector3f_as_ndarray(
            &self.c_data.LinearVelocity)
        self._linearAcceleration = _wrap_ovrVector3f_as_ndarray(
            &self.c_data.LinearAcceleration)
        self._angularVelocity = _wrap_ovrVector3f_as_ndarray(
            &self.c_data.AngularVelocity)
        self._angularAcceleration = _wrap_ovrVector3f_as_ndarray(
            &self.c_data.AngularAcceleration)

        # set values
        if pose is None:
            _ptr.ThePose.Position = [0., 0., 0.]
            _ptr.ThePose.Orientation = [0., 0., 0., 1.]
        elif isinstance(pose, LibOVRPose):
            _ptr.ThePose.Position = (<LibOVRPose>pose).c_data.Position
            _ptr.ThePose.Orientation = (<LibOVRPose>pose).c_data.Orientation
        elif isinstance(pose, (tuple, list,)):
            self._thePose.posOri = pose
        else:
            raise TypeError('Invalid value for `pose`, must be `LibOVRPose`'
                            ', `list` or `tuple`.')

        self._angularVelocity[:] = angularVelocity
        self._linearVelocity[:] = linearVelocity
        self._angularAcceleration[:] = angularAcceleration
        self._linearAcceleration[:] = linearAcceleration

        _ptr.TimeInSeconds = 0.0

    def __dealloc__(self):
        # don't do anything crazy like set c_data=NULL without deallocating!
        if self.c_data is not NULL:
            if self.ptr_owner is True:
                PyMem_Free(self.c_data)

    def __deepcopy__(self, memo=None):
        """Deep copy returned by :py:func:`copy.deepcopy`.

        New :py:class:`LibOVRPoseState` instance with a copy of the data in a
        separate memory location. Does not increase the reference count of the
        object being copied.

        Examples
        --------

        Deep copy::

            import copy
            a = LibOVRPoseState()
            b = copy.deepcopy(a)  # create independent copy of 'a'

        """
        cdef capi.ovrPoseStatef* ptr = \
            <capi.ovrPoseStatef*>PyMem_Malloc(sizeof(capi.ovrPoseStatef))

        if ptr is NULL:
            raise MemoryError

        cdef LibOVRPoseState to_return = LibOVRPoseState.fromPtr(ptr, True)

        # copy over data
        to_return.c_data[0] = self.c_data[0]

        if memo is not None:
            memo[id(self)] = to_return

        return to_return

    def duplicate(self):
        """Create a deep copy of this object.

        Same as calling `copy.deepcopy` on an instance.

        Returns
        -------
        LibOVRPoseState
            An independent copy of this object.

        """
        return self.__deepcopy__()

    @property
    def thePose(self):
        """Rigid body pose."""
        return self._thePose

    @thePose.setter
    def thePose(self, LibOVRPose value):
        self.c_data.ThePose = value.c_data[0]  # copy into

    @property
    def angularVelocity(self):
        """Angular velocity vector in radians/sec."""
        return self._angularVelocity

    @angularVelocity.setter
    def angularVelocity(self, object value):
        self._angularVelocity[:] = value

    @property
    def linearVelocity(self):
        """Linear velocity vector in meters/sec."""
        return self._linearVelocity

    @linearVelocity.setter
    def linearVelocity(self, object value):
        self._linearVelocity[:] = value

    @property
    def angularAcceleration(self):
        """Angular acceleration vector in radians/s^2."""
        return self._angularAcceleration

    @angularAcceleration.setter
    def angularAcceleration(self, object value):
        self._angularAcceleration[:] = value

    @property
    def linearAcceleration(self):
        """Linear acceleration vector in meters/s^2."""
        return self._linearAcceleration

    @linearAcceleration.setter
    def linearAcceleration(self, object value):
        self._linearAcceleration[:] = value

    @property
    def timeInSeconds(self):
        """Absolute time this data refers to in seconds."""
        return <double>self.c_data[0].TimeInSeconds

    @timeInSeconds.setter
    def timeInSeconds(self, double value):
        self.c_data[0].TimeInSeconds = value

    def timeIntegrate(self, float dt):
        """Time integrate rigid body motion derivatives referenced by the
        current pose.

        Parameters
        ----------
        dt : float
            Time delta in seconds.

        Returns
        -------
        LibOVRPose
            Pose at `dt`.

        Examples
        --------

        Time integrate a pose for 20 milliseconds (note the returned object is a
        :py:mod:`LibOVRPose`, not another :py:class:`LibOVRPoseState`)::

            newPose = oldPose.timeIntegrate(0.02)
            pos, ori = newPose.posOri  # extract components

        Time integration can be used to predict the pose of an object at HMD
        V-Sync if velocity and acceleration are known. Usually we would pass the
        predicted time to `getDevicePoses` or `getTrackingState` for a more
        robust estimate of HMD pose at predicted display time. However, in most
        cases the following will yield the same position and orientation as
        `LibOVR` within a few decimal places::

            tsec = timeInSeconds()
            ptime = getPredictedDisplayTime(frame_index)

            _, headPoseState = getDevicePoses(
                [TRACKED_DEVICE_TYPE_HMD],
                absTime=tsec,  # not the predicted time!
                latencyMarker=True)

            dt = ptime - tsec  # time difference from now and v-sync
            headPoseAtVsync = headPose.timeIntegrate(dt)
            calcEyePoses(headPoseAtVsync)

        """
        cdef libovr_math.Posef res = \
            (<libovr_math.Posef>self.c_data[0].ThePose).TimeIntegrate(
                <libovr_math.Vector3f>self.c_data[0].LinearVelocity,
                <libovr_math.Vector3f>self.c_data[0].AngularVelocity,
                <libovr_math.Vector3f>self.c_data[0].LinearAcceleration,
                <libovr_math.Vector3f>self.c_data[0].AngularAcceleration,
                dt)

        cdef capi.ovrPosef* ptr = \
            <capi.ovrPosef*>PyMem_Malloc(sizeof(capi.ovrPosef))

        if ptr is NULL:
            raise MemoryError(
                "Failed to allocate 'ovrPosef' in 'timeIntegrate'.")

        cdef LibOVRPose to_return = LibOVRPose.fromPtr(ptr, True)

        # copy over data
        to_return.c_data[0] = <capi.ovrPosef>res

        return to_return