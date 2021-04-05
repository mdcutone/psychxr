# distutils: language=c++
#  =============================================================================
#  vrmath.pxd - Cython definitions for the VR math extension library
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

from libc.math cimport cos, sin


cdef extern from "linmath.h":
    ctypedef float[2] vec2  # length 2 vector type
    ctypedef float[3] vec3  # length 3 vector type
    ctypedef float[4] vec4  # length 4 vector type
    ctypedef vec4[4] mat4x4  # 4x4 matrices
    ctypedef vec4 quat  # quaternions

    void vec2_add(vec2 r, vec2 a, vec2 b)
    void vec2_sub(vec2 r, vec2 a, vec2 b)
    void vec2_scale(vec2 r, vec2 a, float s)
    float vec2_mul_inner(vec2 a, vec2 b)
    float vec2_len(vec2 v)
    void vec2_norm(vec2 r, vec2 n)
    void vec2_min(vec2 r, vec2 a, vec2 b)
    void vec2_max(vec2 r, vec2 a, vec2 b)
    void vec2_dup(vec2 r, vec2 src)

    void vec3_add(vec3 r, vec3 a, vec3 b)
    void vec3_sub(vec3 r, vec3 a, vec3 b)
    void vec3_scale(vec3 r, vec3 a, float s)
    float vec3_mul_inner(vec3 a, vec3 b)
    float vec3_len(vec3 v)
    void vec3_norm(vec3 r, vec3 n)
    void vec3_min(vec3 r, vec3 a, vec3 b)
    void vec3_max(vec3 r, vec3 a, vec3 b)
    void vec3_dup(vec3 r, vec3 src)
    void vec3_mul_cross(vec3 r, vec3 a, vec3 b)
    void vec3_reflect(vec3 r, vec3 v, vec3 n)

    void vec4_add(vec4 r, vec4 a, vec4 b)
    void vec4_sub(vec4 r, vec4 a, vec4 b)
    void vec4_scale(vec4 r, vec4 a, float s)
    float vec4_mul_inner(vec4 a, vec4 b)
    float vec4_len(vec4 v)
    void vec4_norm(vec4 r, vec4 n)
    void vec4_min(vec4 r, vec4 a, vec4 b)
    void vec4_max(vec4 r, vec4 a, vec4 b)
    void vec4_dup(vec4 r, vec4 src)
    void vec4_mul_cross(vec4 r, vec4 a, vec4 b)
    void vec4_reflect(vec4 r, vec4 v, vec4 n)

    void mat4x4_identity(mat4x4 M)
    void mat4x4_dup(mat4x4 M, mat4x4 N)
    void mat4x4_transpose(mat4x4 M, mat4x4 N)
    void mat4x4_add(mat4x4 M, mat4x4 a, mat4x4 b)
    void mat4x4_sub(mat4x4 M, mat4x4 a, mat4x4 b)
    void mat4x4_scale(mat4x4 M, mat4x4 a, float k)

    void quat_add(quat r, quat a, quat b)
    void quat_sub(quat r, quat a, quat b)
    void quat_norm(quat r, quat n)
    void quat_scale(quat r, quat a, float s)
    float quat_mul_inner(quat a, quat b)
    void quat_identity(quat q)
    void quat_mul(quat r, quat p, quat q)
    void quat_conj(quat r, quat q)
    void quat_rotate(quat r, float angle, vec3 axis)
    void quat_mul_vec3(vec3 r, quat q, vec3 v)


# ------------------------------------------------------------------------------
# Routines for working with vectors, matrices and quaternions.
#

cdef inline void vec3_set(vec3 r, float x, float y, float z):
    r[0] = x
    r[1] = y
    r[2] = z


cdef inline void vec3_zero(vec3 r):
    r[0] = r[1] = r[2] = 0.0


cdef inline float vec3_dist(vec3 a, vec3 b):
    cdef vec3 t
    vec3_sub(t, b, a)
    return vec3_len(t)

# Routines for working with matrices derived from functions in `linmath.h`.
# These have been modified to work with matrices whose values are stored in
# row-major order to avoid the additional transpose.
#

cdef inline void mat4x4_row(vec4 r, mat4x4 M, int i):
    cdef int k
    for k in range(4):
        r[k] = M[i][k]


cdef inline void mat4x4_col(vec4 r, mat4x4 M, int i):
    cdef int k
    for k in range(4):
        r[k] = M[k][i]


cdef inline void mat4x4_scale_aniso(mat4x4 M, mat4x4 a, float x, float y, float z):
    vec4_scale(M[0], a[0], x)
    vec4_scale(M[1], a[1], y)
    vec4_scale(M[2], a[2], z)
    vec4_dup(M[3], a[3])


cdef inline void mat4x4_mul(mat4x4 M, mat4x4 a, mat4x4 b):
    cdef mat4x4 temp
    cdef int r
    cdef int c
    cdef int k

    for r in range(4):
        for c in range(4):
            temp[r][c] = 0.
            for k in range(4):
                temp[r][c] += a[r][k] * b[k][c]

    mat4x4_dup(M, temp)


cdef inline void mat4x4_mul_vec4(vec4 r, mat4x4 M, vec4 v):
    cdef int i
    cdef int j

    for i in range(4):
        r[i] = 0.
        for j in range(4):
            r[j] += M[i][j] * v[i]

cdef inline void mat4x4_translate(mat4x4 T, float x, float y, float z):
    mat4x4_identity(T)
    T[0][3] = x
    T[1][3] = y
    T[2][3] = z

cdef inline void mat4x4_translate_in_place(mat4x4 M, float x, float y, float z):
    cdef vec4 t = [x, y, z, 0.]
    cdef vec4 r
    cdef int i

    for i in range(4):
        mat4x4_row(r, M, i)
        M[i][3] += vec4_mul_inner(r, t)


cdef inline void mat4x4_from_vec3_mul_outer(mat4x4 M, vec3 a, vec3 b):
    cdef int i
    cdef int j

    for i in range(4):
        for j in range(4):
            M[j][i] = a[i] * b[j] if i < 3 and j < 3 else <float>0.


cdef inline void mat4x4_rotate(mat4x4 R, mat4x4 M, float x, float y, float z,
                               float angle):
    cdef float s = <float>sin(angle)
    cdef float c = <float>cos(angle)
    cdef vec3 u = [x, y, z]
    cdef mat4x4 T
    cdef mat4x4 S
    cdef mat4x4 C

    if vec2_len(u) > 1e-4:
        vec3_norm(u, u)
        mat4x4_from_vec3_mul_outer(T, u, u)
        S[0] = [0., -u[2], u[1], 0.]
        S[1] = [u[2], 0, -u[0], 0.]
        S[2] = [-u[1], u[0], u[1], 0.]
        S[3] = [0., 0., 0., 0.]

        mat4x4_scale(S, S, s)
        mat4x4_identity(C)
        mat4x4_sub(C, C, T)
        mat4x4_scale(C, C, c)
        mat4x4_add(T, T, C)
        mat4x4_add(T, T, S)
        T[3][3] = <float>1.
        mat4x4_mul(R, M, T)
    else:
        mat4x4_dup(R, M)


cdef inline void mat4x4_invert(mat4x4 T, mat4x4 M):
    cdef float[6] s
    cdef float[6] c
    s[0] = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    s[1] = M[0][0] * M[2][1] - M[0][1] * M[2][0]
    s[2] = M[0][0] * M[3][1] - M[0][1] * M[3][0]
    s[3] = M[1][0] * M[2][1] - M[1][1] * M[2][0]
    s[4] = M[1][0] * M[3][1] - M[1][1] * M[3][0]
    s[5] = M[2][0] * M[3][1] - M[2][1] * M[3][0]

    c[0] = M[0][2] * M[1][3] - M[0][3] * M[1][2]
    c[1] = M[0][2] * M[2][3] - M[0][3] * M[2][2]
    c[2] = M[0][2] * M[3][3] - M[0][3] * M[3][2]
    c[3] = M[1][2] * M[2][3] - M[1][3] * M[2][2]
    c[4] = M[1][2] * M[3][3] - M[1][3] * M[3][2]
    c[5] = M[2][2] * M[3][3] - M[2][3] * M[3][2]

    cdef float idet = <float>1. / (s[0] * c[5] - s[1] * c[4] + s[2] * c[3] +
                                   s[3] * c[2] - s[4] * c[1] + s[5] * c[0])

    T[0][0] = (M[1][1] * c[5] - M[2][1] * c[4] + M[3][1] * c[3]) * idet
    T[0][1] = (-M[0][1] * c[5] + M[2][1] * c[2] - M[3][1] * c[1]) * idet
    T[0][2] = (M[0][1] * c[4] - M[1][1] * c[2] + M[3][1] * c[0]) * idet
    T[0][3] = (-M[0][1] * c[3] + M[1][1] * c[1] - M[2][1] * c[0]) * idet

    T[1][0] = (-M[1][0] * c[5] + M[2][0] * c[4] - M[3][0] * c[3]) * idet
    T[1][1] = (M[0][0] * c[5] - M[2][0] * c[2] + M[3][0] * c[1]) * idet
    T[1][2] = (-M[0][0] * c[4] + M[1][0] * c[2] - M[3][0] * c[0]) * idet
    T[1][3] = (M[0][0] * c[3] - M[1][0] * c[1] + M[2][0] * c[0]) * idet

    T[2][0] = (M[1][3] * s[5] - M[2][3] * s[4] + M[3][3] * s[3]) * idet
    T[2][1] = (-M[0][3] * s[5] + M[2][3] * s[2] - M[3][3] * s[1]) * idet
    T[2][2] = (M[0][3] * s[4] - M[1][3] * s[2] + M[3][3] * s[0]) * idet
    T[2][3] = (-M[0][3] * s[3] + M[1][3] * s[1] - M[2][3] * s[0]) * idet

    T[3][0] = (-M[1][2] * s[5] + M[2][2] * s[4] - M[3][2] * s[3]) * idet
    T[3][1] = (M[0][2] * s[5] - M[2][2] * s[2] + M[3][2] * s[1]) * idet
    T[3][2] = (-M[0][2] * s[4] + M[1][2] * s[2] - M[3][2] * s[0]) * idet
    T[3][3] = (M[0][2] * s[3] - M[1][2] * s[1] + M[2][2] * s[0]) * idet


cdef inline void mat4x4_orthonormalize(mat4x4 R, mat4x4 M):
    mat4x4_dup(R, M)
    cdef float s = 1.
    cdef vec3 h

    vec3_norm(R[2], R[2])

    s = vec3_mul_inner(R[1], R[2])
    vec3_scale(h, R[2], s)
    vec3_sub(R[1], R[1], h)
    vec3_norm(R[1], R[1])

    s = vec3_mul_inner(R[0], R[2])
    vec3_scale(h, R[2], s)
    vec3_sub(R[0], R[0], h)

    s = vec3_mul_inner(R[0], R[1])
    vec3_scale(h, R[1], s)
    vec3_sub(R[0], R[0], h)
    vec3_norm(R[0], R[0])


cdef inline void mat4x4_look_at(mat4x4 m, vec3 eye, vec3 center, vec3 up):
    cdef vec3 f
    vec3_sub(f, center, eye)
    vec3_norm(f, f)

    cdef vec3 s
    vec3_mul_cross(s, f, up)
    vec3_norm(s, s)

    cdef vec3 t
    vec3_mul_cross(t, s, f)

    m[0][0] = s[0]
    m[0][1] = s[1]
    m[0][2] = s[2]
    m[0][3] = 0.

    m[1][0] = t[0]
    m[1][1] = t[1]
    m[1][2] = t[2]
    m[1][3] = 0.

    m[2][0] = -f[0]
    m[2][1] = -f[1]
    m[2][2] = -f[2]
    m[2][3] = 0.

    m[3][0] = 0.
    m[3][1] = 0.
    m[3][2] = 0.
    m[3][3] = 1.

    mat4x4_translate_in_place(m, -eye[0], -eye[1], -eye[2])


cdef inline void mat4x4_from_quat(mat4x4 M, quat q):
    cdef:
        float a = q[3]
        float b = q[0]
        float c = q[1]
        float d = q[2]
        float a2 = a * a
        float b2 = b * b
        float c2 = c * c
        float d2 = d * d

    M[0][0] = a2 + b2 - c2 - d2
    M[1][0] = <float>2. * (b * c + a * d)
    M[2][0] = <float>2. * (b * d - a * c)
    M[3][0] = 0.

    M[0][1] = <float>2. * (b * c - a * d)
    M[1][1] = a2 - b2 + c2 - d2
    M[2][1] = <float>2. * (c * d + a * b)
    M[3][1] = 0.

    M[0][2] = <float>2. * (b * d + a * c)
    M[1][2] = <float>2. * (c * d - a * b)
    M[2][2] = a2 - b2 - c2 + d2
    M[3][2] = 0.

    M[0][3] = M[3][1] = M[3][2] = <float>0.
    M[3][3] = <float>1.


cdef inline void quat_imag(vec3 r, quat q):
    cdef int i
    for i in range(3):
        r[i] = q[i]


# ------------------------------------------------------------------------------
# PsychXR types compatible with those in `LibOVR`.
#

ctypedef struct pxrQuatf:
    float x
    float y
    float z
    float w


ctypedef struct pxrVector3f:
    float x
    float y
    float z


ctypedef struct pxrMatrix4f:
    float[4][4] M


ctypedef struct pxrPosef:
    pxrQuatf Orientation
    pxrVector3f Position


cdef struct pxrBounds3f:
    pxrVector3f[2] b