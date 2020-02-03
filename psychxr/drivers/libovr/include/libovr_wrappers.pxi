#  =============================================================================
#  libovr_wrappers.pxi - Wrappers for LibOVR data types
#  =============================================================================
#
#  libovr_wrappers.pxi
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

cdef np.npy_intp[1] VEC2_SHAPE = [2]
cdef np.npy_intp[1] VEC3_SHAPE = [3]
cdef np.npy_intp[1] FOVPORT_SHAPE = [4]
cdef np.npy_intp[1] QUAT_SHAPE = [4]
cdef np.npy_intp[2] MAT4_SHAPE = [4, 4]


cdef np.ndarray _wrap_ovrVector2f_as_ndarray(capi.ovrVector2f* prtVec):
    """Wrap an ovrVector2f object with a NumPy array."""
    return np.PyArray_SimpleNewFromData(
        1, VEC2_SHAPE, np.NPY_FLOAT32, <void*>prtVec)


cdef np.ndarray _wrap_ovrVector3f_as_ndarray(capi.ovrVector3f* prtVec):
    """Wrap an ovrVector3f object with a NumPy array."""
    return np.PyArray_SimpleNewFromData(
        1, VEC3_SHAPE, np.NPY_FLOAT32, <void*>prtVec)


cdef np.ndarray _wrap_ovrQuatf_as_ndarray(capi.ovrQuatf* prtVec):
    """Wrap an ovrQuatf object with a NumPy array."""
    return np.PyArray_SimpleNewFromData(
        1, QUAT_SHAPE, np.NPY_FLOAT32, <void*>prtVec)


cdef np.ndarray _wrap_ovrMatrix4f_as_ndarray(capi.ovrMatrix4f* prtVec):
    """Wrap an ovrMatrix4f object with a NumPy array."""
    return np.PyArray_SimpleNewFromData(
        2, MAT4_SHAPE, np.NPY_FLOAT32, <void*>prtVec.M)


cdef np.ndarray _wrap_Matrix4f_as_ndarray(libovr_math.Matrix4f* prtVec):
    """Wrap a Matrix4f object with a NumPy array."""
    return np.PyArray_SimpleNewFromData(
        2, MAT4_SHAPE, np.NPY_FLOAT32, <void*>prtVec.M)


cdef np.ndarray _wrap_ovrFovPort_as_ndarray(capi.ovrFovPort* prtVec):
    """Wrap an ovrFovPort object with a NumPy array."""
    return np.PyArray_SimpleNewFromData(
        1, FOVPORT_SHAPE, np.NPY_FLOAT32, <void*>prtVec)