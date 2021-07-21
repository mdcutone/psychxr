# distutils: language=c++
#  =============================================================================
#  xr_linear.pxd - Cython definitions for `xr_linear.h`
#  =============================================================================
#
#  Copyright 2021 Matthew Cutone <mcutone@opensciencetools.com>
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

cimport openxr

cdef extern from "xr_linear.h":
    cdef float MATH_PI
    cdef float DEFAULT_NEAR_Z
    cdef float INFINITE_FAR_Z

    cdef openxr.XrColor4f XrColorRed
    cdef openxr.XrColor4f XrColorGreen
    cdef openxr.XrColor4f XrColorBlue
    cdef openxr.XrColor4f XrColorYellow
    cdef openxr.XrColor4f XrColorPurple
    cdef openxr.XrColor4f XrColorCyan
    cdef openxr.XrColor4f XrColorLightGrey
    cdef openxr.XrColor4f XrColorDarkGrey

    cdef enum GraphicsAPI:
        GRAPHICS_VULKAN,
        GRAPHICS_OPENGL,
        GRAPHICS_OPENGL_ES,
        GRAPHICS_D3D

    ctypedef struct XrMatrix4x4f:
        float m[16]

    cdef float XrRcpSqrt(const float x)
    cdef void XrVector3f_Set(openxr.XrVector3f* v, float value)
    cdef void XrVector3f_Add(
            openxr.XrVector3f* result,
            openxr.XrVector3f* a,
            openxr.XrVector3f* b)
    cdef void XrVector3f_Sub(
            openxr.XrVector3f* result,
            openxr.XrVector3f* a,
            openxr.XrVector3f* b)
    cdef void XrVector3f_Min(
            openxr.XrVector3f* result,
            openxr.XrVector3f* a,
            openxr.XrVector3f* b)
    cdef void XrVector3f_Max(
            openxr.XrVector3f* result,
            openxr.XrVector3f* a,
            openxr.XrVector3f* b)
    cdef void XrVector3f_Decay(
            openxr.XrVector3f* result,
            openxr.XrVector3f* a,
            float value)
    cdef void XrVector3f_Lerp(
            openxr.XrVector3f* result,
            openxr.XrVector3f* a,
            openxr.XrVector3f* b,
            float fraction)
    cdef void XrVector3f_Scale(
            openxr.XrVector3f* result,
            openxr.XrVector3f* a,
            float scaleFactor)
    cdef float XrVector3f_Dot(
            openxr.XrVector3f* a,
            openxr.XrVector3f* b)
    cdef void XrVector3f_Cross(
            openxr.XrVector3f* result,
            openxr.XrVector3f* a,
            openxr.XrVector3f* b)
    cdef void XrVector3f_Normalize(openxr.XrVector3f* v)
    cdef float XrVector3f_Length(openxr.XrVector3f* v)
    cdef void XrQuaternionf_CreateFromAxisAngle(
            openxr.XrQuaternionf* result,
            openxr.XrVector3f* axis,
            float angleInRadians)
    cdef void XrQuaternionf_Lerp(
            openxr.XrQuaternionf* result,
            openxr.XrQuaternionf* a,
            openxr.XrQuaternionf* b,
            float fraction)
    cdef void XrQuaternionf_Multiply(
            openxr.XrQuaternionf* result,
            openxr.XrQuaternionf* a,
            openxr.XrQuaternionf* b)
    cdef void XrMatrix4x4f_Multiply(
            XrMatrix4x4f* result,
            XrMatrix4x4f* a,
            XrMatrix4x4f* b)
    cdef void XrMatrix4x4f_Transpose(
            XrMatrix4x4f* result,
            XrMatrix4x4f* src)
    cdef float XrMatrix4x4f_Minor(
            XrMatrix4x4f* matrix,
            int r0,
            int r1,
            int r2,
            int c0,
            int c1,
            int c2)
    cdef void XrMatrix4x4f_Invert(XrMatrix4x4f* result, XrMatrix4x4f* src)
    cdef void XrMatrix4x4f_InvertRigidBody(
            XrMatrix4x4f* result,
            XrMatrix4x4f* src)
    cdef void XrMatrix4x4f_CreateIdentity(XrMatrix4x4f* result)
    cdef void XrMatrix4x4f_CreateTranslation(
            XrMatrix4x4f* result, const float x, const float y, const float z)
    cdef void XrMatrix4x4f_CreateRotation(
            XrMatrix4x4f* result,
            float degreesX,
            float degreesY,
            float degreesZ)
    cdef void XrMatrix4x4f_CreateScale(
            XrMatrix4x4f* result, const float x, const float y, const float z)
    cdef void XrMatrix4x4f_CreateFromQuaternion(
            XrMatrix4x4f* result,
            openxr.XrQuaternionf* quat)
    cdef void XrMatrix4x4f_CreateTranslationRotationScale(
            XrMatrix4x4f* result,
            openxr.XrVector3f* translation,
            openxr.XrQuaternionf* rotation,
            openxr.XrVector3f* scale)
    cdef void XrMatrix4x4f_CreateProjection(
            XrMatrix4x4f* result,
            GraphicsAPI graphicsApi,
            float tanAngleLeft,
            float tanAngleRight,
            float tanAngleUp,
            float tanAngleDown,
            float nearZ,
            float farZ)
    cdef void XrMatrix4x4f_CreateProjectionFov(
            XrMatrix4x4f* result,
            GraphicsAPI graphicsApi,
            openxr.XrFovf fov,
            float nearZ,
            float farZ)
    cdef void XrMatrix4x4f_CreateOffsetScaleForBounds(
            XrMatrix4x4f* result,
            XrMatrix4x4f* matrix,
            openxr.XrVector3f* mins,
            openxr.XrVector3f* maxs)
    cdef bool XrMatrix4x4f_IsAffine(XrMatrix4x4f* matrix, float epsilon)
    cdef bool XrMatrix4x4f_IsOrthogonal(XrMatrix4x4f* matrix, float epsilon)
    cdef bool XrMatrix4x4f_IsOrthonormal(XrMatrix4x4f* matrix, float epsilon)
    cdef bool XrMatrix4x4f_IsRigidBody(XrMatrix4x4f* matrix, float epsilon)
    cdef void XrMatrix4x4f_GetTranslation(
            openxr.XrVector3f* result,
            XrMatrix4x4f* src)
    cdef void XrMatrix4x4f_GetRotation(
            openxr.XrQuaternionf* result,
            XrMatrix4x4f* src)
    cdef void XrMatrix4x4f_GetScale(
            openxr.XrVector3f* result,
            XrMatrix4x4f* src)
    cdef void XrMatrix4x4f_TransformVector3f(
            openxr.XrVector3f* result,
            XrMatrix4x4f* m,
            openxr.XrVector3f* v)
    cdef void XrMatrix4x4f_TransformVector4f(
            openxr.XrVector4f* result,
            XrMatrix4x4f* m,
            openxr.XrVector4f* v)
    cdef void XrMatrix4x4f_TransformBounds(
            openxr.XrVector3f* resultMins,
            openxr.XrVector3f* resultMaxs,
            XrMatrix4x4f* matrix,
            openxr.XrVector3f* mins,
            openxr.XrVector3f* maxs)
    cdef bool XrMatrix4x4f_CullBounds(
            XrMatrix4x4f* mvp,
            openxr.XrVector3f* mins,
            openxr.XrVector3f* maxs)


