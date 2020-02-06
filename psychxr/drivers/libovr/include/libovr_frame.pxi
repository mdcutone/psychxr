#  =============================================================================
#  libovr_frame.pxi - Frame related functions
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

def waitToBeginFrame(unsigned int frameIndex=0):
    """Wait until a buffer is available so frame rendering can begin. Must be
    called before :func:`beginFrame`.

    Parameters
    ----------
    frameIndex : int
        The target frame index.

    Returns
    -------
    int
        Return code of the LibOVR API call ``OVR::ovr_WaitToBeginFrame``.
        Returns ``SUCCESS`` if completed without errors. May return
        ``ERROR_DISPLAY_LOST`` if the device was removed, rendering the current
        session invalid.

    """
    global _ptrSession
    cdef capi.ovrResult result = \
        capi.ovr_WaitToBeginFrame(_ptrSession, frameIndex)

    return <int>result


def beginFrame(unsigned int frameIndex=0):
    """Begin rendering the frame. Must be called prior to drawing and
    :func:`endFrame`.

    Parameters
    ----------
    frameIndex : int
        The target frame index.

    Returns
    -------
    int
        Error code returned by ``OVR::ovr_BeginFrame``.

    """
    global _ptrSession
    cdef capi.ovrResult result = \
        capi.ovr_BeginFrame(_ptrSession, frameIndex)

    return <int>result


def endFrame(unsigned int frameIndex=0):
    """Call when rendering a frame has completed. Buffers which have been
    committed are passed to the compositor for distortion.

    Successful :func:`waitToBeginFrame` and :func:`beginFrame` call must precede
    calling :func:`endFrame`.

    Parameters
    ----------
    frameIndex : int
        The target frame index.

    Returns
    -------
    tuple (int, float)
        Error code returned by API call `OVR::ovr_EndFrame` and the absolute
        time in seconds `OVR::ovr_EndFrame` returned.

    """
    global _ptrSession
    global _eyeLayer
    global _eyeRenderPoses

    # if head locking is enabled, make sure the render poses are fixed to
    # HmdToEyePose
    if (_eyeLayer.Header.Flags & capi.ovrLayerFlag_HeadLocked) != \
            capi.ovrLayerFlag_HeadLocked:
        _eyeLayer.RenderPose = _eyeRenderPoses
    else:
        _eyeLayer.RenderPose[0] = _eyeRenderDesc[0].HmdToEyePose
        _eyeLayer.RenderPose[1] = _eyeRenderDesc[1].HmdToEyePose

    cdef capi.ovrLayerHeader* layers = &_eyeLayer.Header
    cdef capi.ovrResult result = capi.ovr_EndFrame(
        _ptrSession,
        frameIndex,
        NULL,
        &layers,
        <unsigned int>1)

    cdef double absTime = capi.ovr_GetTimeInSeconds()

    return result, absTime