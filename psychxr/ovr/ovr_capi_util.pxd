# distutils: language=c++
#  =============================================================================
#  OVR Misc. Utilities (OVR_CAPI_Util.h) Cython Declaration File 
#  =============================================================================
#
#  ovr_capi_util.pxd
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
from ovr_capi cimport ovrBool, ovrResult, ovrSession
from ovr_capi cimport ovrMatrix4f, ovrFovPort, ovrVector2f, ovrVector3f, ovrPosef
from ovr_capi cimport ovrTimewarpProjectionDesc

cdef extern from "OVR_CAPI_Util.h":
    ctypedef enum ovrProjectionModifier:
        ovrProjection_None = 0x00,
        ovrProjection_LeftHanded = 0x01,
        ovrProjection_FarLessThanNear = 0x02,
        ovrProjection_FarClipAtInfinity = 0x04,
        ovrProjection_ClipRangeOpenGL = 0x08

    ctypedef struct ovrDetectResult:
        ovrBool IsOculusServiceRunning
        ovrBool IsOculusHMDConnected

    ctypedef enum ovrHapticsGenMode:
        ovrHapticsGenMode_PointSample,
        ovrHapticsGenMode_Count

    ctypedef struct ovrAudioChannelData:
        const float* Samples
        int SamplesCount
        int Frequency

    ctypedef struct ovrHapticsClip:
        const void* Samples
        int SamplesCount

    cdef ovrDetectResult ovr_Detect(int timeoutMilliseconds)
    cdef ovrMatrix4f ovrMatrix4f_Projection(ovrFovPort fov, float znear, float zfar, unsigned int projectionModFlags)
    cdef ovrTimewarpProjectionDesc ovrTimewarpProjectionDesc_FromProjection(ovrMatrix4f projection, unsigned int projectionModFlags)
    cdef ovrMatrix4f ovrMatrix4f_OrthoSubProjection(ovrMatrix4f projection, ovrVector2f orthoScale, float orthoDistance, float HmdToEyeOffsetX)
    cdef void ovr_CalcEyePoses(ovrPosef headPose, const ovrVector3f hmdToEyeOffset[2], ovrPosef outEyePoses[2])
    cdef void ovr_CalcEyePoses2(ovrPosef headPose, const ovrPosef HmdToEyePose[2], ovrPosef outEyePoses[2])
    cdef void ovr_GetEyePoses(ovrSession session, long long frameIndex, ovrBool latencyMarker, const ovrVector3f hmdToEyeOffset[2], ovrPosef outEyePoses[2], double* outSensorSampleTime)
    cdef void ovr_GetEyePoses2(ovrSession session, long long frameIndex, ovrBool latencyMarker, const ovrPosef HmdToEyePose[2], ovrPosef outEyePoses[2], double* outSensorSampleTime)
    cdef void ovrPosef_FlipHandedness(const ovrPosef* inPose, ovrPosef* outPose)
    cdef ovrResult ovr_ReadWavFromBuffer(ovrAudioChannelData* outAudioChannel, const void* inputData, int dataSizeInBytes, int stereoChannelToUse)
    cdef ovrResult ovr_GenHapticsFromAudioData(ovrHapticsClip* outHapticsClip, const ovrAudioChannelData* audioChannel, ovrHapticsGenMode genMode)
    cdef void ovr_ReleaseAudioChannelData(ovrAudioChannelData* audioChannel)
    cdef void ovr_ReleaseHapticsClip(ovrHapticsClip* hapticsClip)
