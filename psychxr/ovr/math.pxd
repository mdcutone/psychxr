#  math.pxd
#
#  Copyright 2018 Matthew Cutone <cutonem(a)yorku.ca> and Laurie M. Wilcox
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
"""This extension module exposes VR math primitives and functions from the
Oculus PC SDK.

This extension module makes use of the official Oculus PC SDK. A C/C++ interface
for tracking, rendering, and VR math for Oculus products. The Oculus PC SDK is
Copyright (c) Facebook Technologies, LLC and its affiliates. All rights
reserved. You must accept the 'EULA', 'Terms of Use' and 'Privacy Policy'
associated with the Oculus PC SDK to use this module in your software (which you
did when you downloaded the SDK to build this module, didn't ya?), if not see
https://www.oculus.com/legal/terms-of-service/ to access those documents.

"""
cimport ovr_math
cimport ovr_capi

cdef class ovrVector2i:
    cdef ovr_capi.ovrVector2i* c_data
    cdef ovr_capi.ovrVector2i  c_ovrVector2i

cdef class ovrSizei:
    cdef ovr_capi.ovrSizei* c_data
    cdef ovr_capi.ovrSizei  c_ovrSizei

cdef class ovrRecti:
    cdef ovr_capi.ovrRecti* c_data
    cdef ovr_capi.ovrRecti  c_ovrRecti

cdef class ovrVector3f:
    cdef ovr_math.Vector3f* c_data
    cdef ovr_math.Vector3f  c_Vector3f

cdef class ovrFovPort:
    cdef ovr_capi.ovrFovPort* c_data
    cdef ovr_capi.ovrFovPort  c_ovrFovPort

cdef class ovrQuatf:
    cdef ovr_math.Quatf* c_data
    cdef ovr_math.Quatf  c_Quatf

cdef class ovrPosef:
    cdef ovr_math.Posef* c_data
    cdef ovr_math.Posef  c_Posef

cdef class ovrMatrix4f:
    cdef ovr_math.Matrix4f* c_data
    cdef ovr_math.Matrix4f  c_Matrix4f