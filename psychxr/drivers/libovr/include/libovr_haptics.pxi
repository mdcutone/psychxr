#  =============================================================================
#  libovr_haptics.pxi - Haptics related types and functions
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
HAPTICS_BUFFER_SAMPLES_MAX = capi.OVR_HAPTICS_BUFFER_SAMPLES_MAX


cdef class LibOVRHapticsInfo(object):
    """Class for touch haptics engine information.

    """
    cdef capi.ovrTouchHapticsDesc c_data

    @property
    def sampleRateHz(self):
        """Haptics engine frequency/sample-rate."""
        return self.c_data.SampleRateHz

    @property
    def sampleTime(self):
        """Time in seconds per sample. You can compute the total playback time
        of a haptics buffer with the formula ``sampleTime * samplesCount``.

        """
        return <float>1.0 / <float>self.c_data.SampleRateHz

    @property
    def queueMinSizeToAvoidStarvation(self):
        """Queue size required to prevent starving the haptics engine."""
        return self.c_data.QueueMinSizeToAvoidStarvation

    @property
    def submitMinSamples(self):
        """Minimum number of samples that can be sent to the haptics engine."""
        return self.c_data.SubmitMinSamples

    @property
    def submitMaxSamples(self):
        """Maximum number of samples that can be sent to the haptics engine."""
        return self.c_data.SubmitMinSamples

    @property
    def submitOptimalSamples(self):
        """Optimal number of samples for the haptics engine."""
        return self.c_data.SubmitMinSamples


cdef class LibOVRHapticsBuffer(object):
    """Class for haptics buffer data for controller vibration.

    Instances of this class store a buffer of vibration amplitude values which
    can be passed to the haptics engine for playback using the
    :func:`submitControllerVibration` function. Samples are stored as a 1D array
    of 32-bit floating-point values ranging between 0.0 and 1.0, with a maximum
    length of ``HAPTICS_BUFFER_SAMPLES_MAX - 1``. You can access this buffer
    through the :py:attr:`~LibOVRHapticsBuffer.samples` attribute.

    One can use `Numpy` functions to generate samples for the haptics buffer.
    Here is an example were amplitude ramps down over playback::

        samples = np.linspace(
            1.0, 0.0, num=HAPTICS_BUFFER_SAMPLES_MAX-1, dtype=np.float32)
        hbuff = LibOVRHapticsBuffer(samples)
        # vibrate right Touch controller
        submitControllerVibration(CONTROLLER_TYPE_RTOUCH, hbuff)

    For information about the haptics engine, such as sampling frequency, call
    :func:`getHapticsInfo` and inspect the returned
    :py:class:`LibOVRHapticsInfo` object.

    Parameters
    ----------
    buffer : array_like
        Buffer of samples. Must be a 1D array of floating point values between
        0.0 and 1.0. If an `ndarray` with dtype `float32` is specified, the
        buffer will be set without copying.

    """
    cdef capi.ovrHapticsBuffer c_data
    cdef np.ndarray _samples

    def __init__(self, object buffer):
        """
        Attributes
        ----------
        samples : ndarray
        samplesCount : int

        """
        pass

    def __cinit__(self, object buffer):
        cdef np.ndarray[np.float32_t, ndim=1] array_in = \
            np.asarray(buffer, dtype=np.float32)

        if array_in.ndim > 1:
            raise ValueError(
                "Array has invalid number of dimensions, must be 1.")

        cdef int num_samples = <int>array_in.shape[0]
        if num_samples >= capi.OVR_HAPTICS_BUFFER_SAMPLES_MAX:
            raise ValueError(
                "Array too large, must have length < HAPTICS_BUFFER_SAMPLES_MAX")

        # clip values so range is between 0.0 and 1.0
        np.clip(array_in, 0.0, 1.0, out=array_in)

        self._samples = array_in

        # set samples buffer data
        self.c_data.Samples = <void*>self._samples.data
        self.c_data.SamplesCount = num_samples
        self.c_data.SubmitMode = capi.ovrHapticsBufferSubmit_Enqueue

    @property
    def samples(self):
        """Haptics buffer samples. Each sample specifies the amplitude of
        vibration at a given point of playback. Must have a length less than
        ``HAPTICS_BUFFER_SAMPLES_MAX``.

        Warnings
        --------
        Do not change the value of `samples` during haptic buffer playback. This
        may crash the application. Check the playback status of the haptics
        engine before setting the array.

        """
        return self._samples

    @samples.setter
    def samples(self, object value):
        cdef np.ndarray[np.float32_t, ndim=1] array_in = \
            np.asarray(value, dtype=np.float32)

        if array_in.ndim > 1:
            raise ValueError(
                "Array has invalid number of dimensions, must be 1.")

        cdef int num_samples = <int>array_in.shape[0]
        if num_samples >= capi.OVR_HAPTICS_BUFFER_SAMPLES_MAX:
            raise ValueError(
                "Array too large, must have length < HAPTICS_BUFFER_SAMPLES_MAX")

        # clip values so range is between 0.0 and 1.0
        np.clip(array_in, 0.0, 1.0, out=array_in)

        self._samples = array_in

        # set samples buffer data
        self.c_data.Samples = <void*>self._samples.data
        self.c_data.SamplesCount = num_samples

    @property
    def samplesCount(self):
        """Number of haptic buffer samples stored. This value will always be
        less than ``HAPTICS_BUFFER_SAMPLES_MAX``.

        """
        return self.c_data.SamplesCount


def setControllerVibration(int controller, str frequency, float amplitude):
    """Vibrate a controller.

    Vibration is constant at fixed frequency and amplitude. Vibration lasts
    2.5 seconds, so this function needs to be called more often than that
    for sustained vibration. Only controllers which support vibration can be
    used here.

    There are only two frequencies permitted 'high' and 'low', however,
    amplitude can vary from 0.0 to 1.0. Specifying frequency='off' stops
    vibration.

    Parameters
    ----------
    controller : int
        Controller name. Valid values are:

        * ``CONTROLLER_TYPE_XBOX`` : XBox gamepad.
        * ``CONTROLLER_TYPE_REMOTE`` : Oculus Remote.
        * ``CONTROLLER_TYPE_TOUCH`` : Combined Touch controllers.
        * ``CONTROLLER_TYPE_LTOUCH`` : Left Touch controller.
        * ``CONTROLLER_TYPE_RTOUCH`` : Right Touch controller.
        * ``CONTROLLER_TYPE_OBJECT0`` : Object 0 controller.
        * ``CONTROLLER_TYPE_OBJECT1`` : Object 1 controller.
        * ``CONTROLLER_TYPE_OBJECT2`` : Object 2 controller.
        * ``CONTROLLER_TYPE_OBJECT3`` : Object 3 controller.

    frequency : str
        Vibration frequency. Valid values are: 'off', 'low', or 'high'.
    amplitude : float
        Vibration amplitude in the range of [0.0 and 1.0]. Values outside
        this range are clamped.

    Returns
    -------
    int
        Return value of API call ``OVR::ovr_SetControllerVibration``. Can return
        ``SUCCESS_DEVICE_UNAVAILABLE`` if no device is present.

    """
    global _ptrSession

    # get frequency associated with the string
    cdef float freq = 0.0
    if frequency == 'off':
        freq = amplitude = 0.0
    elif frequency == 'low':
        freq = 0.5
    elif frequency == 'high':
        freq = 1.0
    else:
        raise RuntimeError("Invalid frequency specified.")

    cdef capi.ovrResult result = capi.ovr_SetControllerVibration(
        _ptrSession,
        <capi.ovrControllerType>controller,
        freq,
        amplitude)

    return result


def getHapticsInfo(int controller):
    """Get information about the haptics engine for a particular controller.

    Parameters
    ----------
    controller : int
        Controller name. Valid values are:

        * ``CONTROLLER_TYPE_XBOX`` : XBox gamepad.
        * ``CONTROLLER_TYPE_REMOTE`` : Oculus Remote.
        * ``CONTROLLER_TYPE_TOUCH`` : Combined Touch controllers.
        * ``CONTROLLER_TYPE_LTOUCH`` : Left Touch controller.
        * ``CONTROLLER_TYPE_RTOUCH`` : Right Touch controller.
        * ``CONTROLLER_TYPE_OBJECT0`` : Object 0 controller.
        * ``CONTROLLER_TYPE_OBJECT1`` : Object 1 controller.
        * ``CONTROLLER_TYPE_OBJECT2`` : Object 2 controller.
        * ``CONTROLLER_TYPE_OBJECT3`` : Object 3 controller.

    Returns
    -------
    LibOVRHapticsInfo
        Haptics engine information. Values do not change over the course of a
        session.

    """
    global _ptrSession
    cdef LibOVRHapticsInfo to_return = LibOVRHapticsInfo()
    to_return.c_data = capi.ovr_GetTouchHapticsDesc(
        _ptrSession, <capi.ovrControllerType>controller,)

    return to_return


def submitControllerVibration(int controller, LibOVRHapticsBuffer buffer):
    """Submit a haptics buffer to Touch controllers.

    Parameters
    ----------
    controller : int
        Controller name. Valid values are:

        * ``CONTROLLER_TYPE_TOUCH`` : Combined Touch controllers.
        * ``CONTROLLER_TYPE_LTOUCH`` : Left Touch controller.
        * ``CONTROLLER_TYPE_RTOUCH`` : Right Touch controller.
        * ``CONTROLLER_TYPE_OBJECT0`` : Object 0 controller.
        * ``CONTROLLER_TYPE_OBJECT1`` : Object 1 controller.
        * ``CONTROLLER_TYPE_OBJECT2`` : Object 2 controller.
        * ``CONTROLLER_TYPE_OBJECT3`` : Object 3 controller.

    buffer : LibOVRHapticsBuffer
        Haptics buffer to submit.

    Returns
    -------
    int
        Return value of API call ``OVR::ovr_SubmitControllerVibration``. Can
        return ``SUCCESS_DEVICE_UNAVAILABLE`` if no device is present.

    """
    global _ptrSession

    cdef capi.ovrResult result = capi.ovr_SubmitControllerVibration(
        _ptrSession,
        <capi.ovrControllerType>controller,
        &buffer.c_data)

    return result


def getControllerPlaybackState(int controller):
    """Get the playback state of a touch controller.

    Parameters
    ----------
    controller : int
        Controller name. Valid values are:

        * ``CONTROLLER_TYPE_TOUCH`` : Combined Touch controllers.
        * ``CONTROLLER_TYPE_LTOUCH`` : Left Touch controller.
        * ``CONTROLLER_TYPE_RTOUCH`` : Right Touch controller.
        * ``CONTROLLER_TYPE_OBJECT0`` : Object 0 controller.
        * ``CONTROLLER_TYPE_OBJECT1`` : Object 1 controller.
        * ``CONTROLLER_TYPE_OBJECT2`` : Object 2 controller.
        * ``CONTROLLER_TYPE_OBJECT3`` : Object 3 controller.

    Returns
    -------
    tuple (int, int, int)
        Returns three values, the value of API call
        ``OVR::ovr_GetControllerVibrationState``, the remaining space in the
        haptics buffer available to queue more samples, and the number of
        samples currently queued.

    """
    global _ptrSession

    cdef capi.ovrHapticsPlaybackState playback_state
    cdef capi.ovrResult result = capi.ovr_GetControllerVibrationState(
        _ptrSession,
        <capi.ovrControllerType>controller,
        &playback_state)

    return result, playback_state.RemainingQueueSpace, playback_state.SamplesQueued