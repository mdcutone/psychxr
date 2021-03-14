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


cdef class LibOVRHmdColorSpace(object):
    """Class for HMD color space data.

    """
    cdef capi.ovrHmdColorDesc c_data

    def __init__(self):
        """
        Attributes
        ----------
        colorSpace : int
        """
        pass

    @property
    def colorSpace(self):
        """The color space (`int`). A symbolic constant representing a color
        space.

        Valid values returned are ``COLORSPACE_UNKNOWN``,
        ``COLORSPACE_UNMANAGED``, ``COLORSPACE_RIFT_CV1``, ``COLORSPACE_RIFT_S``,
        ``COLORSPACE_QUEST``, ``COLORSPACE_REC_2020``, ``COLORSPACE_REC_709``,
        ``COLORSPACE_P3``, ``COLORSPACE_ADOBE_RGB``.

        """
        return <int>self.c_data.ColorSpace

    @colorSpace.setter
    def colorSpace(self, object value):
        self.c_data.ColorSpace = <capi.ovrColorSpace>value


def getHmdColorSpace():
    """Get HMD colorspace information.

    Upon starting a new session, the default colorspace used is

    Returns
    -------
    LibOVRHmdColorSpace
        HMD colorspace information.

    """
    global _ptrSession
    cdef capi.ovrHmdColorDesc desc
    desc = capi.ovr_GetHmdColorDesc(_ptrSession)

    cdef LibOVRHmdColorSpace toReturn = LibOVRHmdColorSpace()
    toReturn.c_data.ColorSpace = desc.ColorSpace

    return toReturn
