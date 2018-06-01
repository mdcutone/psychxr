#ifndef VRLINALG_H
#define VRLINALG_H

#include <math.h>

#ifdef _MSC_VER
#define inline __inline
//#define ALIGNMENT(n) __declspec(align(n))
#endif
// -----------------------------------------------------------------------------
//  >>> 3D MATH STRUCTURES AND FUNCTIONS <<<
//
//  Defined below are various data structures which describe 3D scene geometry,
//  and their transformation functions. These structures are intended to be
//  mostly compatible with similar representations specified in the Oculus(TM)
//  PC SDK.
//
//  Many of the functions below are derived from the 'linmath.h' header file
//  which ships with GLFW.
//

// -----------------------------------------------------------------------------
//  >>> vec2i <<<
//
//  A representation of a 2D point or vector, consisting of integer x and y
//  coordinates.
//
typedef struct {
    int x;
    int y;
} vec2i;

static inline void vec2i_add(vec2i* r, vec2i const a, vec2i const b)
{
    r->x = a.x + b.x;
    r->y = a.y + b.y;
}

static inline void vec2i_sub(vec2i* r, vec2i const a, vec2i const b)
{
    r->x = a.x - b.x;
    r->y = a.y - b.y;
}

static inline void vec2i_scale(vec2i* r, vec2i const v, int const s)
{
    r->x = v.x * s;
    r->y = v.y * s;
}

// -----------------------------------------------------------------------------
//  >>> vec2f <<<
//
//  A representation of a 2D point or vector, consisting of floating point x and
//  y coordinates.
//
typedef struct {
    float x;
    float y;
} vec2f;

static inline void vec2f_add(vec2f* r, vec2f const a, vec2f const b)
{
    r->x = a.x + b.x;
    r->y = a.y + b.y;
}

static inline void vec2f_sub(vec2f* r, vec2f const a, vec2f const b)
{
    r->x = a.x - b.x;
    r->y = a.y - b.y;
}

static inline void vec2f_scale(vec2f* r, vec2f const v, float const s)
{
    r->x = v.x * s;
    r->y = v.y * s;
}

static inline float vec2f_dot(vec2f const a, vec2f const b)
{
    return a.x * b.x + a.y * b.y;
}

static inline float vec2f_len_sqr(vec2f const v)
{
    return vec2f_dot(v, v);
}

static inline float vec2f_len(vec2f const v)
{
    return sqrt(vec2f_len_sqr(v));
}

static inline void vec2f_norm(vec2f* r, vec2f const v)
{
	float k = 1.0f / vec2f_len(v);
	vec2f_scale(r, v, k);
}

// *** vec3f ***
typedef struct vec3f_ {
    float x;
    float y;
    float z;
} vec3f;

static inline void vec3f_add(vec3f r, vec3f const a, vec3f const b)
{
    r.x = a.x + b.x;
    r.y = a.y + b.y;
    r.z = a.z + b.z;
}

static inline void vec3f_sub(vec3f r, vec3f const a, vec3f const b)
{
    r.x = a.x - b.x;
    r.y = a.y - b.y;
    r.z = a.z - b.z;
}

static inline void vec3f_scale(vec3f r, vec3f const v, float const s)
{
    r.x = v.x * s;
    r.y = v.y * s;
    r.z = v.z * s;
}

static inline float vec3f_dot(vec3f const a, vec3f const b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline float vec3f_len_sqr(vec3f const v)
{
    return vec3f_dot(v, v);
}

static inline float vec3f_len(vec3f const v)
{
    return sqrt(vec3f_len_sqr(v));
}

static inline void vec3f_norm(vec3f r, vec3f const v)
{
	float k = 1.0f / vec3f_len(v);
	vec3f_scale(r, v, k);
}

static inline void vec3f_mul_cross(vec3f r, vec3f const a, vec3f const b)
{
	r.x = a.y * b.z - a.z * b.y;
	r.y = a.z * b.x - a.x * b.z;
	r.z = a.x * b.y - a.y * b.x;
}

// *** vec4f ***
typedef struct vec4f_ {          // 4D float vector
    float x;
    float y;
    float z;
    float w;
} vec4f;

static inline void vec4f_add(vec4f r, vec4f const a, vec4f const b)
{
    r.x = a.x + b.x;
    r.y = a.y + b.y;
    r.z = a.z + b.z;
    r.w = a.w + b.w;
}

static inline void vec4f_sub(vec4f r, vec4f const a, vec4f const b)
{
    r.x = a.x - b.x;
    r.y = a.y - b.y;
    r.z = a.z - b.z;
    r.w = a.w - b.w;
}

static inline void vec4f_scale(vec4f r, vec4f const v, float const s)
{
    r.x = v.x * s;
    r.y = v.y * s;
    r.z = v.z * s;
    r.w = v.w * s;
}

static inline float vec4f_dot(vec4f const a, vec4f const b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

static inline float vec4f_len_sqr(vec4f const v)
{
    return vec4f_dot(v, v);
}

static inline float vec4f_len(vec4f const v)
{
    return sqrt(vec4f_len_sqr(v));
}

static inline void vec4f_norm(vec4f r, vec4f const v)
{
	float k = 1.0f / vec4f_len(v);
	vec4f_scale(r, v, k);
}

#endif