# distutils: language=c++
#  =============================================================================
#  OVR OpenGL Specific (OVR_CAPI_GL.h) Cython Declaration File 
#  =============================================================================
#
#  ovr_capi_gl.pxd
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
"""This file exposes Oculus Rift(TM) C API types and functions, allowing Cython 
extensions to access them. The declarations in the file are contemporaneous
with version 1.26 (retrieved 09.25.2019) of the Oculus Rift(TM) PC SDK.

"""
from ovr_errorcode cimport ovrResult
from ovr_capi cimport ovrSession
from ovr_capi cimport ovrTextureSwapChainDesc, ovrTextureSwapChain
from ovr_capi cimport ovrMirrorTextureDesc, ovrMirrorTexture

cdef extern from "OVR_CAPI_GL.h":
    cdef ovrResult ovr_CreateTextureSwapChainGL(ovrSession session, const ovrTextureSwapChainDesc* desc, ovrTextureSwapChain* out_TextureSwapChain)
    cdef ovrResult ovr_GetTextureSwapChainBufferGL(ovrSession session, ovrTextureSwapChain chain, int index, unsigned int* out_TexId)
    cdef ovrResult ovr_CreateMirrorTextureWithOptionsGL(ovrSession session, const ovrMirrorTextureDesc* desc, ovrMirrorTexture* out_MirrorTexture)
    cdef ovrResult ovr_CreateMirrorTextureGL(ovrSession session, const ovrMirrorTextureDesc* desc, ovrMirrorTexture* out_MirrorTexture)
    cdef ovrResult ovr_GetMirrorTextureBufferGL(ovrSession session, ovrMirrorTexture mirrorTexture, unsigned int* out_TexId)
