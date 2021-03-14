#  =============================================================================
#  libovr_hmdinfo.pxi - Wrapper extensions type for ovrHmdDesc
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

# HMD types
HMD_NONE = capi.ovrHmd_None
HMD_DK1 = capi.ovrHmd_DK1
HMD_DKHD = capi.ovrHmd_DKHD
HMD_DK2 = capi.ovrHmd_DK2
HMD_CB = capi.ovrHmd_CB
HMD_OTHER = capi.ovrHmd_Other
HMD_E3_2015  = capi.ovrHmd_E3_2015
HMD_ES06 = capi.ovrHmd_ES06
HMD_ES09 = capi.ovrHmd_ES09
HMD_ES11 = capi.ovrHmd_ES11
HMD_CV1 = capi.ovrHmd_CV1
HMD_RIFTS = capi.ovrHmd_RiftS
HMD_QUEST = capi.ovrHmd_Quest
HMD_QUEST2 = capi.ovrHmd_Quest2

cdef capi.ovrHmdDesc _hmdDesc  # HMD information descriptor


cdef class LibOVRHmdInfo(object):
    """Class for general HMD information and capabilities. An instance of this
    class is returned by calling :func:`~psychxr.libovr.getHmdInfo`.

    """
    cdef capi.ovrHmdDesc* c_data
    cdef bint ptr_owner

    def __init__(self):
        """
        Attributes
        ----------
        hmdType : int
        hasOrientationTracking : bool
        hasPositionTracking : bool
        hasMagYawCorrection : bool
        isDebugDevice : bool
        productName : str
        manufacturer : str
        serialNumber : str
        resolution : tuple
        refreshRate : float
        hid : tuple
        firmwareVersion : tuple
        defaultEyeFov : tuple (ndarray and ndarray)
        maxEyeFov : tuple (ndarray and ndarray)
        symmetricEyeFov : tuple (ndarray and ndarray)

        """
        self.newStruct()

    def __cinit__(self):
        self.ptr_owner = False

    @staticmethod
    cdef LibOVRHmdInfo fromPtr(capi.ovrHmdDesc* ptr, bint owner=False):
        # bypass __init__ if wrapping a pointer
        cdef LibOVRHmdInfo wrapper = LibOVRHmdInfo.__new__(LibOVRHmdInfo)
        wrapper.c_data = ptr
        wrapper.ptr_owner = owner

        return wrapper

    cdef void newStruct(self):
        if self.c_data is not NULL:  # already allocated, __init__ called twice?
            return

        cdef capi.ovrHmdDesc* _ptr = <capi.ovrHmdDesc*>PyMem_Malloc(
            sizeof(capi.ovrHmdDesc))

        if _ptr is NULL:
            raise MemoryError

        self.c_data = _ptr
        self.ptr_owner = True

    def __dealloc__(self):
        if self.c_data is not NULL and self.ptr_owner is True:
            PyMem_Free(self.c_data)
            self.c_data = NULL

    @property
    def hmdType(self):
        """HMD type currently used.

        Valid values returned are ``HMD_NONE``, ``HMD_DK1``, ``HMD_DKHD``,
        ``HMD_DK2``, ``HMD_CB``, ``HMD_OTHER``, ``HMD_E3_2015``, ``HMD_ES06``,
        ``HMD_ES09``, ``HMD_ES11``, `HMD_CV1``, ``HMD_RIFTS``, ``HMD_QUEST``,
        ``HMD_QUEST2``.

        """
        return <int>self.c_data.Type

    @property
    def hasOrientationTracking(self):
        """``True`` if the HMD is capable of tracking orientation."""
        return (self.c_data.AvailableTrackingCaps &
                capi.ovrTrackingCap_Orientation) == \
               capi.ovrTrackingCap_Orientation

    @property
    def hasPositionTracking(self):
        """``True`` if the HMD is capable of tracking position."""
        return (self.c_data.AvailableTrackingCaps &
                capi.ovrTrackingCap_Position) == capi.ovrTrackingCap_Position

    @property
    def hasMagYawCorrection(self):
        """``True`` if this HMD supports yaw drift correction."""
        return (self.c_data.AvailableTrackingCaps &
                capi.ovrTrackingCap_MagYawCorrection) == \
               capi.ovrTrackingCap_MagYawCorrection

    @property
    def isDebugDevice(self):
        """``True`` if the HMD is a virtual debug device."""
        return (self.c_data.AvailableHmdCaps & capi.ovrHmdCap_DebugDevice) == \
               capi.ovrHmdCap_DebugDevice

    @property
    def productName(self):
        """Get the product name for this device.

        Returns
        -------
        str
            Product name string (utf-8).

        """
        return self.c_data[0].ProductName.decode('utf-8')

    @property
    def manufacturer(self):
        """Get the device manufacturer name.

        Returns
        -------
        str
            Manufacturer name string (utf-8).

        """
        return self.c_data[0].Manufacturer.decode('utf-8')

    @property
    def serialNumber(self):
        """Get the device serial number.

        Returns
        -------
        str
            Serial number (utf-8).

        """
        return self.c_data[0].SerialNumber.decode('utf-8')

    @property
    def resolution(self):
        """Horizontal and vertical resolution of the display in pixels.

        Returns
        -------
        ndarray
            Resolution of the display [w, h].

        """
        return np.asarray((self.c_data[0].Resolution.w,
                           self.c_data[0].Resolution.h), dtype=int)

    @property
    def refreshRate(self):
        """Nominal refresh rate in Hertz of the display.

        Returns
        -------
        ndarray
            Refresh rate in Hz.

        """
        return <float>self.c_data[0].DisplayRefreshRate

    @property
    def hid(self):
        """USB human interface device class identifiers.

        Returns
        -------
        tuple (int, int)
            USB HIDs (vendor, product).

        """
        return <int>self.c_data[0].VendorId, <int>self.c_data[0].ProductId

    @property
    def firmwareVersion(self):
        """Firmware version for this device.

        Returns
        -------
        tuple (int, int)
            Firmware version (major, minor).

        """
        return <int>self.c_data[0].FirmwareMajor, \
               <int>self.c_data[0].FirmwareMinor

    @property
    def defaultEyeFov(self):
        """Default or recommended eye field-of-views (FOVs) provided by the API.

        Returns
        -------
        tuple (ndarray, ndarray)
            Pair of left and right eye FOVs specified as tangent angles [Up,
            Down, Left, Right].

        """
        cdef np.ndarray fovLeft = np.asarray([
            self.c_data[0].DefaultEyeFov[0].UpTan,
            self.c_data[0].DefaultEyeFov[0].DownTan,
            self.c_data[0].DefaultEyeFov[0].LeftTan,
            self.c_data[0].DefaultEyeFov[0].RightTan],
            dtype=np.float32)

        cdef np.ndarray fovRight = np.asarray([
            self.c_data[0].DefaultEyeFov[1].UpTan,
            self.c_data[0].DefaultEyeFov[1].DownTan,
            self.c_data[0].DefaultEyeFov[1].LeftTan,
            self.c_data[0].DefaultEyeFov[1].RightTan],
            dtype=np.float32)

        return fovLeft, fovRight

    @property
    def maxEyeFov(self):
        """Maximum eye field-of-views (FOVs) provided by the API.

        Returns
        -------
        tuple (ndarray, ndarray)
            Pair of left and right eye FOVs specified as tangent angles in
            radians [Up, Down, Left, Right].

        """
        cdef np.ndarray[float, ndim=1] fov_left = np.asarray([
            self.c_data[0].MaxEyeFov[0].UpTan,
            self.c_data[0].MaxEyeFov[0].DownTan,
            self.c_data[0].MaxEyeFov[0].LeftTan,
            self.c_data[0].MaxEyeFov[0].RightTan],
            dtype=np.float32)

        cdef np.ndarray[float, ndim=1] fov_right = np.asarray([
            self.c_data[0].MaxEyeFov[1].UpTan,
            self.c_data[0].MaxEyeFov[1].DownTan,
            self.c_data[0].MaxEyeFov[1].LeftTan,
            self.c_data[0].MaxEyeFov[1].RightTan],
            dtype=np.float32)

        return fov_left, fov_right

    @property
    def symmetricEyeFov(self):
        """Symmetric field-of-views (FOVs) for mono rendering.

        By default, the Rift uses off-axis FOVs. These frustum parameters make
        it difficult to converge monoscopic stimuli.

        Returns
        -------
        tuple (ndarray, ndarray)
            Pair of left and right eye FOVs specified as tangent angles in
            radians [Up, Down, Left, Right]. Both FOV objects will have the same
            values.

        """
        cdef libovr_math.FovPort fov_left = \
            <libovr_math.FovPort>self.c_data[0].DefaultEyeFov[0]
        cdef libovr_math.FovPort fov_right = \
            <libovr_math.FovPort>self.c_data[0].DefaultEyeFov[1]

        cdef libovr_math.FovPort fov_max = libovr_math.FovPort.Max(
            <libovr_math.FovPort>fov_left, <libovr_math.FovPort>fov_right)

        cdef float tan_half_fov_horz = maxf(fov_max.LeftTan, fov_max.RightTan)
        cdef float tan_half_fov_vert = maxf(fov_max.DownTan, fov_max.UpTan)

        cdef capi.ovrFovPort fov_both
        fov_both.LeftTan = fov_both.RightTan = tan_half_fov_horz
        fov_both.UpTan = fov_both.DownTan = tan_half_fov_horz

        cdef np.ndarray[float, ndim=1] fov_left_out = np.asarray([
            fov_both.UpTan,
            fov_both.DownTan,
            fov_both.LeftTan,
            fov_both.RightTan],
            dtype=np.float32)

        cdef np.ndarray[float, ndim=1] fov_right_out = np.asarray([
            fov_both.UpTan,
            fov_both.DownTan,
            fov_both.LeftTan,
            fov_both.RightTan],
            dtype=np.float32)

        return fov_left_out, fov_right_out


def getHmdInfo():
    """Get HMD information.

    Returns
    -------
    LibOVRHmdInfo
        HMD information.

    """
    global _hmdDesc
    cdef LibOVRHmdInfo toReturn = LibOVRHmdInfo()
    toReturn.c_data[0] = _hmdDesc

    return toReturn