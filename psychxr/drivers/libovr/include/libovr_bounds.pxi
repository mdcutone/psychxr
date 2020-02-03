#  =============================================================================
#  libovr_bounds.pxi - Types and functions for Guardian
#  =============================================================================
#
#  libovr_bounds.pxi
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
# boundary modes
BOUNDARY_PLAY_AREA = capi.ovrBoundary_PlayArea
BOUNDARY_OUTER = capi.ovrBoundary_Outer

cdef capi.ovrBoundaryLookAndFeel _boundaryStyle


cdef class LibOVRBoundaryTestResult(object):
    """Class for boundary collision test data. An instance of this class is
    returned when :func:`testBoundary` is called.

    """
    cdef capi.ovrBoundaryTestResult c_data

    def __init__(self):
        """
        Attributes
        ----------
        isTriggering : bool (read-only)
        closestDistance : float (read-only)
        closestPoint : ndarray (read-only)
        closestPointNormal : ndarray (read-only)

        """
        pass

    def __cinit__(self):
        pass

    @property
    def isTriggering(self):
        """``True`` if the play area boundary is triggering. Since the boundary
        fades-in, it might not be perceptible when this is called.
        """
        return self.c_data.IsTriggering == capi.ovrTrue

    @property
    def closestDistance(self):
        """Closest point to the boundary in meters."""
        return <float>self.c_data.ClosestDistance

    @property
    def closestPoint(self):
        """Closest point on the boundary surface."""
        cdef np.ndarray[float, ndim=1] to_return = np.asarray([
            self.c_data.ClosestPoint.x,
            self.c_data.ClosestPoint.y,
            self.c_data.ClosestPoint.z],
            dtype=np.float32)

        return to_return

    @property
    def closestPointNormal(self):
        """Unit normal of the closest boundary surface."""
        cdef np.ndarray[float, ndim=1] to_return = np.asarray([
            self.c_data.ClosestPointNormal.x,
            self.c_data.ClosestPointNormal.y,
            self.c_data.ClosestPointNormal.z],
            dtype=np.float32)

        return to_return


def setBoundaryColor(float red, float green, float blue):
    """Set the boundary color.

    The boundary is drawn by the compositor which overlays the extents of
    the physical space where the user can safely move.

    Parameters
    ----------
    red : float
        Red component of the color from 0.0 to 1.0.
    green : float
        Green component of the color from 0.0 to 1.0.
    blue : float
        Blue component of the color from 0.0 to 1.0.

    Returns
    -------
    int
        Result of the LibOVR API call ``OVR::ovr_SetBoundaryLookAndFeel``.

    """
    global _boundaryStyle
    global _ptrSession

    cdef capi.ovrColorf color
    color.r = <float>red
    color.g = <float>green
    color.b = <float>blue

    _boundaryStyle.Color = color

    cdef capi.ovrResult result = capi.ovr_SetBoundaryLookAndFeel(
        _ptrSession,
        &_boundaryStyle)

    return result


def resetBoundaryColor():
    """Reset the boundary color to system default.

    Returns
    -------
    int
        Result of the LibOVR API call ``OVR::ovr_ResetBoundaryLookAndFeel``.

    """
    global _ptrSession
    cdef capi.ovrResult result = capi.ovr_ResetBoundaryLookAndFeel(
        _ptrSession)

    return result


def getBoundaryVisible():
    """Check if the Guardian boundary is visible.

    The boundary is drawn by the compositor which overlays the extents of
    the physical space where the user can safely move.

    Returns
    -------
    tuple (int, bool)
        Result of the LibOVR API call ``OVR::ovr_GetBoundaryVisible`` and the
        boundary state.

    Notes
    -----
    * Since the boundary has a fade-in effect, the boundary might be reported as
      visible but difficult to actually see.

    """
    global _ptrSession
    cdef capi.ovrBool isVisible
    cdef capi.ovrResult result = capi.ovr_GetBoundaryVisible(
        _ptrSession, &isVisible)

    return result, isVisible


def showBoundary():
    """Show the boundary.

    The boundary is drawn by the compositor which overlays the extents of
    the physical space where the user can safely move.

    Returns
    -------
    int
        Result of LibOVR API call ``OVR::ovr_RequestBoundaryVisible``.

    """
    global _ptrSession
    cdef capi.ovrResult result = capi.ovr_RequestBoundaryVisible(
        _ptrSession, capi.ovrTrue)

    return result


def hideBoundary():
    """Hide the boundry.

    Returns
    -------
    int
        Result of LibOVR API call ``OVR::ovr_RequestBoundaryVisible``.

    """
    global _ptrSession
    cdef capi.ovrResult result = capi.ovr_RequestBoundaryVisible(
        _ptrSession, capi.ovrFalse)

    return result


def getBoundaryDimensions(int boundaryType):
    """Get the dimensions of the boundary.

    Parameters
    ----------
    boundaryType : int
        Boundary type, can be ``BOUNDARY_OUTER`` or ``BOUNDARY_PLAY_AREA``.

    Returns
    -------
    tuple (int, ndarray)
        Result of the LibOVR APi call ``OVR::ovr_GetBoundaryDimensions`` and the
        dimensions of the boundary in meters [x, y, z].

    """
    global _ptrSession
    cdef capi.ovrBoundaryType btype = <capi.ovrBoundaryType>boundaryType
    if not (boundaryType == capi.ovrBoundary_PlayArea or
            boundaryType == capi.ovrBoundary_Outer):
        raise ValueError("Invalid boundary type specified.")

    cdef capi.ovrVector3f vec_out
    cdef capi.ovrResult result = capi.ovr_GetBoundaryDimensions(
            _ptrSession, btype, &vec_out)

    cdef np.ndarray[np.float32_t, ndim=1] to_return = np.asarray(
        (vec_out.x, vec_out.y, vec_out.z), dtype=np.float32)

    return result, to_return


def testBoundary(int deviceBitmask, int boundaryType):
    """Test collision of tracked devices on boundary.

    Parameters
    ----------
    deviceBitmask : int
        Devices to test. Multiple devices identifiers can be combined
        together. Valid device IDs are:

        * ``TRACKED_DEVICE_TYPE_HMD``: The head or HMD.
        * ``TRACKED_DEVICE_TYPE_LTOUCH``: Left touch controller or hand.
        * ``TRACKED_DEVICE_TYPE_RTOUCH``: Right touch controller or hand.
        * ``TRACKED_DEVICE_TYPE_TOUCH``: Both touch controllers.
        * ``TRACKED_DEVICE_TYPE_OBJECT0``
        * ``TRACKED_DEVICE_TYPE_OBJECT1``
        * ``TRACKED_DEVICE_TYPE_OBJECT2``
        * ``TRACKED_DEVICE_TYPE_OBJECT3``

    boundaryType : int
        Boundary type, can be ``BOUNDARY_OUTER`` or ``BOUNDARY_PLAY_AREA``.

    Returns
    -------
    tuple (int, LibOVRBoundaryTestResult)
        Result of the ``OVR::ovr_TestBoundary`` LibOVR API call and
        collision test results.

    """
    global _ptrSession

    cdef LibOVRBoundaryTestResult testResult = LibOVRBoundaryTestResult()
    cdef capi.ovrResult result = capi.ovr_TestBoundary(
        _ptrSession, <capi.ovrTrackedDeviceType>deviceBitmask,
        <capi.ovrBoundaryType>boundaryType,
        &testResult.c_data)

    return testResult

