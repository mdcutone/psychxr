// Stub for required preprocessor defines, required until we figure out how to
// make Cython do this automatically.

#ifndef DEFINES_H_
#define DEFINES_H_ 1

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(XR_USE_PLATFORM_WIN32)
#define XR_USE_PLATFORM_WIN32 1
#endif
#if !defined(XR_USE_GRAPHICS_API_OPENGL)
#define XR_USE_GRAPHICS_API_OPENGL 1
#endif
#if !defined(XR_EXTENSION_PROTOTYPES)
#define XR_EXTENSION_PROTOTYPES 1
#endif
#ifdef __cplusplus
}
#endif
#endif
