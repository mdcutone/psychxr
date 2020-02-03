#  =============================================================================
#  libovr_timing.pxi - Timing related functions
#  =============================================================================
#
#  libovr_timing.pxi
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

# clock offset in seconds
cdef double t_offset = 0.0


def getPredictedDisplayTime(unsigned int frameIndex=0):
    """Get the predicted time a frame will be displayed.

    Parameters
    ----------
    frameIndex : int
        Frame index.

    Returns
    -------
    float
        Absolute frame mid-point time for the given frame index in seconds.

    """
    global _ptrSession
    cdef double t_sec = capi.ovr_GetPredictedDisplayTime(
        _ptrSession,
        frameIndex)

    return t_sec


def timeInSeconds():
    """Absolute time in seconds.

    Returns
    -------
    float
        Time in seconds.

    """
    cdef double t_sec = capi.ovr_GetTimeInSeconds()

    return t_sec


def setReferenceTime(double refTime):
    """Set a reference time to synchronize the time source used by the LibOVR
    driver with an external clock.

    This function computes a time offset between the external clock and the one
    used by the LibOVR driver. The offset is then applied when calling any
    function which requires or retrieves absolute time information (eg.
    :func:`getPredictedDisplayTime`). This is useful for cases where the
    application interfacing with the HMD is using its own time source.

    Parameters
    ----------
    refTime : float
        Current time of the external clock in seconds (must be >=0.0).

    Returns
    -------
    float
        The difference between the external and LibOVR time sources in seconds.

    Notes
    -----
    * If the reference time is changed, any previously reported time will be
      invalid.
    * Allows for some error on the order of a few microseconds when the time
      offset is computed.
    * It is assumed that the an external time source operating on the exact same
      frequency as the time source used by LibOVR.

    """
    global t_offset

    if refTime < 0:
        raise ValueError("Value for `refTime` must be >=0.")

    t_offset = refTime - capi.ovr_GetTimeInSeconds()  # compute the offset

    return t_offset


def getFrameOnsetTime(int frameIndex):
    """Get the estimated frame onset time.

    This function **estimates** the onset time of `frameIndex` by subtracting
    half the display's frequency from the predicted mid-frame display time
    reported by LibOVR.

    Returns
    -------
    float
        Estimated onset time of the next frame in seconds.

    Notes
    -----
    * Onset times are estimated and one should use caution when using the
      value reported by this function.

    """
    global _hmdDesc
    global _ptrSession
    cdef double halfRefresh = (1.0 / <double>_hmdDesc.DisplayRefreshRate) / 2.0

    return capi.ovr_GetPredictedDisplayTime(_ptrSession, frameIndex) - \
           halfRefresh


