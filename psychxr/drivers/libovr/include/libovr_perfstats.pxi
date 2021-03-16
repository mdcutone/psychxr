#  =============================================================================
#  libovr_perfstats.pxi - Performance statistics
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

MAX_PROVIDED_FRAME_STATS = capi.ovrMaxProvidedFrameStats

cdef class LibOVRPerfStatsPerCompositorFrame(object):
    """Class for frame performance statistics per compositor frame.

    Instances of this class are returned by calling :func:`getPerfStats` and
    accessing the :py:attr:`LibOVRPerfStats.frameStats` field of the returned
    :class:`LibOVRPerfStats` instance.

    Data contained in this class provide information about compositor
    performance. Metrics include motion-to-photon latency, dropped frames, and
    elapsed times of various stages of frame processing to the vertical
    synchronization (V-Sync) signal of the HMD.

    Calling :func:`resetFrameStats` will reset integer fields of this class in
    successive calls to :func:`getPerfStats`.

    """
    cdef capi.ovrPerfStatsPerCompositorFrame* c_data
    cdef bint ptr_owner

    def __init__(self):
        self._new_struct()

    def __cinit__(self):
        self.ptr_owner = False

    @staticmethod
    cdef LibOVRPerfStatsPerCompositorFrame fromPtr(
            capi.ovrPerfStatsPerCompositorFrame* ptr, bint owner=False):
        cdef LibOVRPerfStatsPerCompositorFrame wrapper = \
            LibOVRPerfStatsPerCompositorFrame.__new__(
                LibOVRPerfStatsPerCompositorFrame)

        wrapper.c_data = ptr
        wrapper.ptr_owner = owner

        return wrapper

    cdef void _new_struct(self):
        if self.c_data is not NULL:
            return

        cdef capi.ovrPerfStatsPerCompositorFrame* ptr = \
            <capi.ovrPerfStatsPerCompositorFrame*>PyMem_Malloc(
                sizeof(capi.ovrPerfStatsPerCompositorFrame))

        if ptr is NULL:
            raise MemoryError

        self.c_data = ptr
        self.ptr_owner = True

    def __dealloc__(self):
        # don't do anything crazy like set c_data=NULL without deallocating!
        if self.c_data is not NULL:
            if self.ptr_owner is True:
                PyMem_Free(self.c_data)
                self.c_data = NULL

    @property
    def hmdVsyncIndex(self):
        """Increments every HMD vertical sync signal."""
        return self.c_data.HmdVsyncIndex

    @property
    def appFrameIndex(self):
        """Index increments after each call to :func:`endFrame`."""
        return self.c_data.AppFrameIndex

    @property
    def appDroppedFrameCount(self):
        """If :func:`endFrame` is not called on-time, this will increment (i.e.
        missed HMD vertical sync deadline).

        Examples
        --------
        Check if the application dropped a frame::

            framesDropped = frameStats.frameStats[0].appDroppedFrameCount >
                lastFrameStats.frameStats[0].appDroppedFrameCount

        """
        return self.c_data.AppDroppedFrameCount

    @property
    def appMotionToPhotonLatency(self):
        """Motion-to-photon latency in seconds computed using the marker set by
        :func:`getTrackingState` or the sensor sample time set by
        :func:`setSensorSampleTime`.
        """
        return self.c_data.AppMotionToPhotonLatency

    @property
    def appQueueAheadTime(self):
        """Queue-ahead time in seconds. If >11 ms, the CPU is outpacing the GPU
        workload by 1 frame.
        """
        return self.c_data.AppQueueAheadTime

    @property
    def appCpuElapsedTime(self):
        """Time in seconds the CPU spent between calls of :func:`endFrame`. Form
        the point when :func:`endFrame` releases control back to the
        application, to the next time it is called.
        """
        return self.c_data.AppCpuElapsedTime

    @property
    def appGpuElapsedTime(self):
        """Time in seconds the GPU spent between calls of :func:`endFrame`."""
        return self.c_data.AppGpuElapsedTime

    @property
    def compositorFrameIndex(self):
        """Increments when the compositor completes a distortion pass, happens
        regardless if :func:`endFrame` was called late.
        """
        return self.c_data.CompositorFrameIndex

    @property
    def compositorDroppedFrameCount(self):
        """Number of frames dropped by the compositor. This can happen
        spontaneously for reasons not related to application performance.
        """
        return self.c_data.CompositorDroppedFrameCount

    @property
    def compositorLatency(self):
        """Motion-to-photon latency of the compositor, which include the
        latency of 'timewarp' needed to correct for application latency and
        dropped application frames.
        """
        return self.c_data.CompositorLatency

    @property
    def compositorCpuElapsedTime(self):
        """Time in seconds the compositor spends on the CPU."""
        return self.c_data.CompositorCpuElapsedTime

    @property
    def compositorGpuElapsedTime(self):
        """Time in seconds the compositor spends on the GPU."""
        return self.c_data.CompositorGpuElapsedTime

    @property
    def compositorCpuStartToGpuEndElapsedTime(self):
        """Time in seconds between the point the compositor executes and
        completes distortion/timewarp. Value is -1.0 if GPU time is not
        available.
        """
        return self.c_data.CompositorCpuStartToGpuEndElapsedTime

    @property
    def compositorGpuEndToVsyncElapsedTime(self):
        """Time in seconds left between the compositor is complete and the
        target vertical synchronization (v-sync) on the HMD."""
        return self.c_data.CompositorGpuEndToVsyncElapsedTime

    @property
    def timeToVsync(self):
        """Total time elapsed from when CPU control is handed off to the
        compositor to HMD vertical synchronization signal (V-Sync).

        """
        return self.c_data.CompositorCpuStartToGpuEndElapsedTime + \
            self.c_data.CompositorGpuEndToVsyncElapsedTime

    @property
    def aswIsActive(self):
        """``True`` if Asynchronous Space Warp (ASW) was active this frame."""
        return self.c_data.AswIsActive == capi.ovrTrue

    @property
    def aswActivatedToggleCount(self):
        """How many frames ASW activated during the runtime of this application.
        """
        return self.c_data.AswActivatedToggleCount

    @property
    def aswPresentedFrameCount(self):
        """Number of frames the compositor extrapolated using ASW."""
        return self.c_data.AswPresentedFrameCount

    @property
    def aswFailedFrameCount(self):
        """Number of frames the compositor failed to present extrapolated frames
        using ASW.
        """
        return self.c_data.AswFailedFrameCount


cdef class LibOVRPerfStats(object):
    """Class for frame performance statistics.

    Instances of this class are returned by calling :func:`getPerfStats`.

    """
    cdef capi.ovrPerfStats *c_data
    cdef bint ptr_owner

    cdef LibOVRPerfStatsPerCompositorFrame compFrame0
    cdef LibOVRPerfStatsPerCompositorFrame compFrame1
    cdef LibOVRPerfStatsPerCompositorFrame compFrame2
    cdef LibOVRPerfStatsPerCompositorFrame compFrame3
    cdef LibOVRPerfStatsPerCompositorFrame compFrame4

    def __init__(self):
        self._new_struct()

    def __cinit__(self):
        self.ptr_owner = False

    @staticmethod
    cdef LibOVRPerfStats fromPtr(capi.ovrPerfStats* ptr, bint owner=False):
        cdef LibOVRPerfStats wrapper = LibOVRPerfStats.__new__(LibOVRPerfStats)

        wrapper.c_data = ptr
        wrapper.ptr_owner = owner

        wrapper.compFrame0 = LibOVRPerfStatsPerCompositorFrame.fromPtr(
            &wrapper.c_data.FrameStats[0])
        wrapper.compFrame1 = LibOVRPerfStatsPerCompositorFrame.fromPtr(
            &wrapper.c_data.FrameStats[1])
        wrapper.compFrame2 = LibOVRPerfStatsPerCompositorFrame.fromPtr(
            &wrapper.c_data.FrameStats[2])
        wrapper.compFrame3 = LibOVRPerfStatsPerCompositorFrame.fromPtr(
            &wrapper.c_data.FrameStats[3])
        wrapper.compFrame4 = LibOVRPerfStatsPerCompositorFrame.fromPtr(
            &wrapper.c_data.FrameStats[4])

        return wrapper

    cdef void _new_struct(self):
        if self.c_data is not NULL:
            return

        cdef capi.ovrPerfStats* ptr = \
            <capi.ovrPerfStats*>PyMem_Malloc(
                sizeof(capi.ovrPerfStats))

        if ptr is NULL:
            raise MemoryError

        self.c_data = ptr
        self.ptr_owner = True

        self.compFrame0 = LibOVRPerfStatsPerCompositorFrame.fromPtr(
            &self.c_data.FrameStats[0])
        self.compFrame1 = LibOVRPerfStatsPerCompositorFrame.fromPtr(
            &self.c_data.FrameStats[1])
        self.compFrame2 = LibOVRPerfStatsPerCompositorFrame.fromPtr(
            &self.c_data.FrameStats[2])
        self.compFrame3 = LibOVRPerfStatsPerCompositorFrame.fromPtr(
            &self.c_data.FrameStats[3])
        self.compFrame4 = LibOVRPerfStatsPerCompositorFrame.fromPtr(
            &self.c_data.FrameStats[4])

    def __dealloc__(self):
        # don't do anything crazy like set c_data=NULL without deallocating!
        if self.c_data is not NULL:
            if self.ptr_owner is True:
                PyMem_Free(self.c_data)
                self.c_data = NULL

    @property
    def frameStats(self):
        """Performance stats per compositor frame. Statistics are in reverse
        chronological order where the first index is the most recent. Only
        indices 0 to :py:attr:`LibOVRPerfStats.frameStatsCount` are valid.
        """
        return (self.compFrame0,
                self.compFrame1,
                self.compFrame2,
                self.compFrame3,
                self.compFrame4)

    @property
    def frameStatsCount(self):
        """Number of compositor frame statistics available. The maximum number
        of frame statistics is 5. If 1 is returned, the application is calling
        :func:`getFrameStats` at a rate equal to or greater than the refresh
        rate of the display.
        """
        return self.c_data.FrameStatsCount

    @property
    def anyFrameStatsDropped(self):
        """``True`` if compositor frame statistics have been dropped. This
        occurs if :func:`getPerfStats` is called at a rate less than 1/5th the
        refresh rate of the HMD. You can obtain the refresh rate for your model
        of HMD by calling :func:`getHmdInfo` and accessing the
        :py:attr:`LibOVRHmdInfo.refreshRate` field of the returned
        :py:class:`LibOVRHmdInfo` instance.

        """
        return self.c_data.AnyFrameStatsDropped == capi.ovrTrue

    @property
    def adaptiveGpuPerformanceScale(self):
        """Adaptive performance scale value. This value ranges between 0.0 and
        1.0. If the application is taking up too many GPU resources, this value
        will be less than 1.0, indicating the application needs to throttle GPU
        usage somehow to maintain performance. If the value is 1.0, the GPU is
        being utilized the correct amount for the application.

        """
        return self.c_data.AdaptiveGpuPerformanceScale

    @property
    def aswIsAvailable(self):
        """``True`` is ASW is enabled."""
        return self.c_data.AswIsAvailable == capi.ovrTrue

    @property
    def visibleProcessId(self):
        """Visible process ID.

        Since performance stats can be obtained for any application running on
        the LibOVR runtime that has focus, this value should equal the current
        process ID returned by ``os.getpid()`` to ensure the statistics returned
        are for the current application.

        Examples
        --------
        Check if frame statistics are for the present PsychXR application::

            perfStats = getPerfStats()
            if perfStats.visibleProcessId == os.getpid():
                # has focus, performance stats are for this application

        """
        return <int>self.c_data.VisibleProcessId


def getPerfStats():
    """Get detailed compositor frame statistics.

    Returns
    -------
    LibOVRPerfStats
        Frame statistics.

    Examples
    --------
    Get the time spent by the application between :func:`endFrame` calls::

        result = updatePerfStats()

        if getFrameStatsCount() > 0:
            frameStats = getFrameStats(0)  # only the most recent
            appTime = frameStats.appCpuElapsedTime

    """
    global _ptrSession
    cdef LibOVRPerfStats to_return = LibOVRPerfStats()
    cdef capi.ovrResult result = capi.ovr_GetPerfStats(
        _ptrSession, to_return.c_data)

    return to_return


def resetPerfStats():
    """Reset frame performance statistics.

    Calling this will reset frame statistics, which may be needed if the
    application loses focus (eg. when the system UI is opened) and performance
    stats no longer apply to the application.

    Returns
    -------
    int
        Error code returned by ``OVR::ovr_ResetPerfStats``.

    """
    global _ptrSession
    cdef capi.ovrResult result = capi.ovr_ResetPerfStats(_ptrSession)

    return result