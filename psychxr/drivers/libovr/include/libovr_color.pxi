#  =============================================================================
#  libovr_color.pxi - Wrapper extensions type for ovrColorSpaceTypes
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

# color spaces used by various HMDs
COLORSPACE_UNKNOWN = capi.ovrColorSpace_Unknown
COLORSPACE_UNMANAGED = capi.ovrColorSpace_Unmanaged
COLORSPACE_RIFT_CV1 = capi.ovrColorSpace_Rift_CV1
COLORSPACE_RIFT_S = capi.ovrColorSpace_Rift_S
COLORSPACE_QUEST = capi.ovrColorSpace_Quest
COLORSPACE_REC_2020 = capi.ovrColorSpace_Rec_2020
COLORSPACE_REC_709 = capi.ovrColorSpace_Rec_709
COLORSPACE_P3 = capi.ovrColorSpace_P3
COLORSPACE_ADOBE_RGB = capi.ovrColorSpace_Adobe_RGB

# internal store for chromaticity values
cdef np.npy_intp[3] CHROMA_STORE_SHAPE = [capi.ovrColorSpace_Count, 2, 4]
cdef np.ndarray chroma_xys = np.PyArray_SimpleNew(
    3, CHROMA_STORE_SHAPE, np.NPY_FLOAT32)

# internal color indices
cdef Py_ssize_t CHROMA_RED_IDX = 0
cdef Py_ssize_t CHROMA_GREEN_IDX = 1
cdef Py_ssize_t CHROMA_BLUE_IDX = 2
cdef Py_ssize_t CHROMA_WHITE_IDX = 3

# clear store
chroma_xys[:, :, :] = 0.0

# populate values
chroma_xys[COLORSPACE_RIFT_CV1, :, :] = [
    [0.666, 0.334], # red
    [0.238, 0.714], # green
    [0.139, 0.053], # blue
    [0.298, 0.318]  # D75 white point
]
# default for COLORSPACE_UNKNOWN is to use the CV1
chroma_xys[COLORSPACE_UNKNOWN, :, :] = chroma_xys[COLORSPACE_RIFT_CV1, :, :]
chroma_xys[COLORSPACE_RIFT_S, :, :] = [
    [0.640, 0.330],
    [0.292, 0.586],
    [0.156, 0.058],
    [0.156, 0.058]
]
chroma_xys[COLORSPACE_QUEST, :, :] = [
    [0.661, 0.338],
    [0.228, 0.718],
    [0.142, 0.042],
    [0.298, 0.318]
]
chroma_xys[COLORSPACE_REC_2020, :, :] = [
    [0.708, 0.292],
    [0.170, 0.797],
    [0.131, 0.046],
    [0.3127, 0.329]
]
chroma_xys[COLORSPACE_REC_709, :, :] = [
    [0.640, 0.330],
    [0.300, 0.600],
    [0.150, 0.060],
    [0.3127, 0.329]
]
chroma_xys[COLORSPACE_P3, :, :] = [
    [0.680, 0.320],
    [0.265, 0.690],
    [0.150, 0.060],
    [0.150, 0.060]  # D65 white point
]
chroma_xys[COLORSPACE_ADOBE_RGB, :, :] = [
    [0.640, 0.330],
    [0.210, 0.710],
    [0.150, 0.060],
    [0.3127, 0.329]
]
# make sure the user can't overwrite these values by accident
chroma_xys.flags.writeable = False


cdef class LibOVRHmdColorSpace(object):
    """Class for HMD color space data.

    This class is used to store color space information related to the HMD. The
    color space value is a symbolic constant accessed through the `colorSpace`
    property.

    As of version *23.0* of the Oculus PC SDK, the API provides functions for
    specifying and retrieving data about the color space of the display. This is
    needed because the chromaticity coordinates of RGB primaries and the white
    points differ between models, causing differences in perceived color when
    content authored for one platform is viewed on another. To deal with this,
    the API allows you to specify the color space the content was intended for
    and the driver will remap colors to be best represented on the current
    display.

    When developing an application to run across multiple HMD devices, the
    manufacturer recommends that you target the CV1 or Quest HMDs since the
    color gamut on those displays are wider than other HMDs in their product
    lineup (such as the Rift S).

    PsychXR provides additional information about these color spaces, such as
    the chromaticity coordinates used by various devices in the Oculus(tm)
    product lineup. These values can be accessed using properties associated to
    instances of this class.

    """
    cdef capi.ovrHmdColorDesc c_data

    def __init__(self):
        """
        Attributes
        ----------
        colorSpace : int
        red : ndarray
        green : ndarray
        blue : ndarray
        whitePoint : ndarray
        """
        pass

    def __cinit__(self):
        self.c_data.ColorSpace = COLORSPACE_UNKNOWN  # default value to use

    @staticmethod
    def getRGBPrimaries(int colorSpace):
        """Get RGB primaries for a given color model.

        Parameters
        ----------
        colorSpace : int
            Symbolic constant representing a color space (e.g.,
            ``COLORSPACE_RIFT_CV1``).

        Returns
        -------
        ndarray
            3x2 array of RGB primaries corresponding to the specified color
            model.

        """
        cdef np.ndarray[np.float32_t, ndim=2] to_return = np.zeros(
            (2, 3), dtype=np.float32)

        to_return[:, :] = chroma_xys[colorSpace, :3, :]

        return to_return

    @staticmethod
    def getWhitePoint(int colorSpace):
        """Get RGB primaries for a given color model.

        Parameters
        ----------
        colorSpace : int
            Symbolic constant representing a color space (e.g.,
            ``COLORSPACE_RIFT_CV1``).

        Returns
        -------
        ndarray
            3x2 array of RGB primaries corresponding to the specified color
            model.

        """
        cdef np.ndarray[np.float32_t, ndim=1] to_return = np.zeros(
            (2,), dtype=np.float32)

        to_return[:] = chroma_xys[colorSpace, 3, :]

        return to_return

    @property
    def colorSpace(self):
        """The color space (`int`). A symbolic constant representing a color
        space.

        Valid values returned are ``COLORSPACE_UNKNOWN``,
        ``COLORSPACE_UNMANAGED``, ``COLORSPACE_RIFT_CV1``, ``COLORSPACE_RIFT_S``,
        ``COLORSPACE_QUEST``, ``COLORSPACE_REC_2020``, ``COLORSPACE_REC_709``,
        ``COLORSPACE_P3`` or ``COLORSPACE_ADOBE_RGB``.

        Notes
        -----
        If `colorSpace` is set to ``COLORSPACE_UNMANAGED``, the chromaticity
        coordinates will be set to the defaults for the current HMD. For the
        DK2, Rec. 709 coordinates will be used (``COLORSPACE_REC_709``).

        """
        return <int>self.c_data.ColorSpace

    @colorSpace.setter
    def colorSpace(self, object value):
        self.c_data.ColorSpace = <capi.ovrColorSpace>value

    @property
    def red(self):
        """Chromaticity coordinate for the red primary (CIE 1931 xy) used by the
        display (`ndarray`). This is set by the value of
        `LibOVRHmdColorSpace.colorSpace`.

        """
        return chroma_xys[<int>self.c_data.ColorSpace, CHROMA_RED_IDX, :]

    @property
    def green(self):
        """Chromaticity coordinate for the green primary (CIE 1931 xy) used by
        the display (`ndarray`). This is set by the value of
        `LibOVRHmdColorSpace.colorSpace`.

        """
        return chroma_xys[<int>self.c_data.ColorSpace, CHROMA_GREEN_IDX, :]

    @property
    def blue(self):
        """Chromaticity coordinate for the blue primary (CIE 1931 xy) used by
        the display (`ndarray`). This is set by the value of
        `LibOVRHmdColorSpace.colorSpace`.

        """
        return chroma_xys[<int>self.c_data.ColorSpace, CHROMA_BLUE_IDX, :]

    @property
    def whitePoint(self):
        """Chromaticity coordinate for the white point (CIE 1931 xy) used by the
        display (`ndarray`). This is set by the value of
        `LibOVRHmdColorSpace.colorSpace`.

        """
        return chroma_xys[<int>self.c_data.ColorSpace, CHROMA_WHITE_IDX, :]


def getHmdColorSpace():
    """Get HMD colorspace information.

    Upon starting a new session, the default colorspace used is for the CV1. Can
    only be called after :func:`start` was called.

    Returns
    -------
    LibOVRHmdColorSpace
        HMD colorspace information.

    Examples
    --------
    Get the current color space in use::

        colorSpaceInfo = getHmdColorSpace()

    Get the color coordinates of the RGB primaries::

        redX, redY = colorSpaceInfo.red
        greenX, greenY = colorSpaceInfo.red
        blueX, blueY = colorSpaceInfo.red

    Get the white point in use::

        whiteX, whiteY = colorSpaceInfo.whitePoint

    """
    global _ptrSession
    cdef capi.ovrHmdColorDesc desc
    desc = capi.ovr_GetHmdColorDesc(_ptrSession)

    cdef LibOVRHmdColorSpace toReturn = LibOVRHmdColorSpace()
    toReturn.c_data.ColorSpace = desc.ColorSpace

    return toReturn


def setClientColorSpace(object colorSpace):
    """Set the colorspace used by the client.

    This function is used by the driver to transform color values between
    spaces. The purpose of this is to allow content authored for one model of
    HMD to appear correctly on others. Can oly be called after `start()` was
    called. Until this function is not called, the color space will be assumed
    to be ``COLORSPACE_UNKNOWN`` which defaults to ``COLORSPACE_RIFT_CV1``.

    **New as of version 0.2.4**

    Parameters
    ----------
    colorSpace : LibOVRHmdColorSpace or int
        Color space information descriptor or symbolic constant (e.g.,
        ``COLORSPACE_RIFT_CV1``.

    Returns
    -------
    int
        Return code for the `ovr_SetClientColorDesc` call.

    Examples
    --------
    Tell the driver to remap colors for an application authored using the Quest
    to be displayed correctly on the current device::

        result = setClientColorSpace(COLORSPACE_QUEST)

    """
    global _ptrSession
    global _hmdDesc

    cdef capi.ovrHmdColorDesc desc

    if isinstance(colorSpace, LibOVRHmdColorSpace):
        desc.ColorSpace = (<LibOVRHmdColorSpace>colorSpace).c_data.Colorspace
    elif isinstance(colorSpace, int):
        desc.ColorSpace = <capi.ovrColorSpace>colorSpace
    else:
        raise TypeError('Value for `colorSpace` must be type `int` or '
                        '`LibOVRHmdColorSpace`.')

    # deal with unmanaged case
    if desc.ColorSpace == capi.ovrColorSpace_Unmanaged:
        chroma_xys.flags.writeable = True
        if _hmdDesc.Type == capi.ovrHmd_CV1:
            chroma_xys[COLORSPACE_UNMANAGED, :, :] = \
                chroma_xys[COLORSPACE_RIFT_CV1, :, :]
        elif _hmdDesc.Type == capi.ovrHmd_Quest or \
                _hmdDesc.Type == capi.ovrHmd_Quest2:
            chroma_xys[COLORSPACE_UNMANAGED, :, :] = \
                chroma_xys[COLORSPACE_QUEST, :, :]
        elif _hmdDesc.Type == capi.ovrHmd_RiftS:
            chroma_xys[COLORSPACE_UNMANAGED, :, :] = \
                chroma_xys[COLORSPACE_RIFT_S, :, :]
        else:
            # assume rec 709 if no color space provided (close to sRBG)
            chroma_xys[COLORSPACE_UNMANAGED, :, :] = \
                chroma_xys[COLORSPACE_REC_709, :, :]

        chroma_xys.flags.writeable = False

    cdef capi.ovrResult result = capi.ovr_SetClientColorDesc(
        _ptrSession, &desc)

    return <int>result
