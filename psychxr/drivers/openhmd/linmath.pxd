# distutils: language=c++
#  =============================================================================
#  linmath.pxd - Cython definitions for `linmath.h`
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

cdef extern from "linmath.h":

    ctypedef float[2] vec2  # length 2 vector type
    void vec2_add(vec2 r, vec2 a, vec2 b)
    void vec2_sub(vec2 r, vec2 a, vec2 b)
    void vec2_scale(vec2 r, vec2 a, vec2 b)
    float vec2_mul_inner(vec2 a, vec2 b)
    float vec2_len(vec2 v)
    void vec2_norm(vec2 r, vec2 n)
    void vec2_min(vec2 r, vec2 a, vec2 b)
    void vec2_max(vec2 r, vec2 a, vec2 b)
    void vec2_dup(vec2 r, vec2 src)

    ctypedef float[3] vec3  # length 3 vector type
    void vec3_add(vec3 r, vec3 a, vec3 b)
    void vec3_sub(vec3 r, vec3 a, vec3 b)
    void vec3_scale(vec3 r, vec3 a, vec3 b)
    float vec3_mul_inner(vec3 a, vec3 b)
    float vec3_len(vec3 v)
    void vec3_norm(vec3 r, vec3 n)
    void vec3_min(vec3 r, vec3 a, vec3 b)
    void vec3_max(vec3 r, vec3 a, vec3 b)
    void vec3_dup(vec3 r, vec3 src)
    void vec3_mul_cross(vec3 r, vec3 a, vec3 b)
    void vec3_reflect(vec3 r, vec3 v, vec3 n)

    ctypedef float[4] vec4  # length 4 vector type
    void vec4_add(vec4 r, vec4 a, vec4 b)
    void vec4_sub(vec4 r, vec4 a, vec4 b)
    void vec4_scale(vec4 r, vec4 a, vec4 b)
    float vec4_mul_inner(vec4 a, vec4 b)
    float vec4_len(vec4 v)
    void vec4_norm(vec4 r, vec4 n)
    void vec4_min(vec4 r, vec4 a, vec4 b)
    void vec4_max(vec4 r, vec4 a, vec4 b)
    void vec4_dup(vec4 r, vec4 src)
    void vec4_mul_cross(vec4 r, vec4 a, vec4 b)
    void vec4_reflect(vec4 r, vec4 v, vec4 n)

    ctypedef vec4[4] mat4x4  # 4x4 matrices
    void mat4x4_identity(mat4x4 M)
    void mat4x4_dup(mat4x4 M, mat4x4 N)
    void mat4x4_row(vec4 r, mat4x4 M, int i)
    void mat4x4_col(vec4 r, mat4x4 M, int i)
    void mat4x4_transpose(mat4x4 M, mat4x4 N)
    void mat4x4_add(mat4x4 M, mat4x4 a, mat4x4 b)
    void mat4x4_sub(mat4x4 M, mat4x4 a, mat4x4 b)
    void mat4x4_scale(mat4x4 M, mat4x4 a, k)
    void mat4x4_scale_aniso(mat4x4 M, mat4x4 a, float x, float y, float z)
    void mat4x4_mul(mat4x4 M, mat4x4 a, mat4x4 b)
    void mat4x4_mul_vec4(vec4 r, mat4x4 M, vec4 v)
    void mat4x4_translate(mat4x4 T, float x, float y, float z)
    void mat4x4_translate_in_place(mat4x4 M, float x, float y, float z)
    void mat4x4_from_vec3_mul_outer(mat4x4 M, vec3 a, vec3 b)
    void mat4x4_rotate(mat4x4 R, mat4x4 M, float x, float y, float z, float angle)
    void mat4x4_rotate_X(mat4x4 Q, mat4x4 M, float angle)
    void mat4x4_rotate_Y(mat4x4 Q, mat4x4 M, float angle)
    void mat4x4_rotate_Z(mat4x4 Q, mat4x4 M, float angle)
    void mat4x4_invert(mat4x4 T, mat4x4 M)
    void mat4x4_orthonormalize(mat4x4 R, mat4x4 M)
    void mat4x4_frustum(mat4x4 M, float l, float r, float b, float t, float n, float f)
    void mat4x4_ortho(mat4x4 M, float l, float r, float b, float t, float n, float f)
    void mat4x4_perspective(mat4x4 m, float y_fov, float aspect, float n, float f)
    void mat4x4_look_at(mat4x4 m, vec3 eye, vec3 center, vec3 up)

    ctypedef float[4] quat
    void quat_add(quat r, quat a, quat b)
    void quat_sub(quat r, quat a, quat b)
    void quat_norm(quat r, quat n)
    void quat_scale(quat r, quat a, quat b)
    float quat_mul_inner(quat a, quat b)
    void quat_identity(quat q)
    void quat_mul(quat r, quat p, quat q)
    void quat_conj(quat r, quat q)
    void quat_rotate(quat r, float angle, vec3 axis)
    void quat_mul_vec3(vec3 r, quat q, vec3 v)
    void mat4x4_from_quat(mat4x4 M, quat q)
    void mat4x4o_mul_quat(mat4x4 R, mat4x4 M, quat q)
    void quat_from_mat4x4(quat q, mat4x4 M)
    void mat4x4_arcball(mat4x4 R, mat4x4 M, vec2 _a, vec2 _b, s)