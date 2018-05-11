#  =============================================================================
#  OVR Error Code (OVR_ErrorCode.h) Cython Declaration File 
#  =============================================================================
#
#  ovr_errorcode.pxd
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
with version 1.24 (retrieved 04.15.2018) of the Oculus Rift(TM) PC SDK. 

"""
from libc.stdint cimport int32_t

cdef extern from "OVR_ErrorCode.h":
    ctypedef int32_t ovrResult

    ctypedef enum ovrSuccessType:
        ovrSuccess = 0

    ctypedef enum ovrSuccessTypes:
        ovrSuccess_NotVisible = 1000,
        ovrSuccess_BoundaryInvalid = 1001,
        ovrSuccess_DeviceUnavailable = 1002

    ctypedef enum ovrErrorType:
        ovrError_MemoryAllocationFailure = -1000,
        ovrError_InvalidSession = -1002,
        ovrError_Timeout = -1003,
        ovrError_NotInitialized = -1004,
        ovrError_InvalidParameter = -1005,
        ovrError_ServiceError = -1006,
        ovrError_NoHmd = -1007,
        ovrError_Unsupported = -1009,
        ovrError_DeviceUnavailable = -1010,
        ovrError_InvalidHeadsetOrientation = -1011,
        ovrError_ClientSkippedDestroy = -1012,
        ovrError_ClientSkippedShutdown = -1013,
        ovrError_ServiceDeadlockDetected = -1014,
        ovrError_InvalidOperation = -1015,
        ovrError_InsufficientArraySize = -1016,
        ovrError_NoExternalCameraInfo = -1017,
        ovrError_LostTracking = -1018,
        ovrError_ExternalCameraInitializedFailed = -1019,
        ovrError_ExternalCameraCaptureFailed = -1020,
        ovrError_ExternalCameraNameListsBufferSize = -1021,
        ovrError_ExternalCameraNameListsMistmatch = -1022,
        ovrError_ExternalCameraNotCalibrated = -1023,
        ovrError_ExternalCameraNameWrongSize = -1024,
        ovrError_AudioDeviceNotFound = -2001,
        ovrError_AudioComError = -2002,
        ovrError_Initialize = -3000,
        ovrError_LibLoad = -3001,
        ovrError_LibVersion = -3002,
        ovrError_ServiceConnection = -3003,
        ovrError_ServiceVersion = -3004,
        ovrError_IncompatibleOS = -3005,
        ovrError_DisplayInit = -3006,
        ovrError_ServerStart = -3007,
        ovrError_Reinitialization = -3008,
        ovrError_MismatchedAdapters = -3009,
        ovrError_LeakingResources = -3010,
        ovrError_ClientVersion = -3011,
        ovrError_OutOfDateOS = -3012,
        ovrError_OutOfDateGfxDriver = -3013,
        ovrError_IncompatibleGPU = -3014,
        ovrError_NoValidVRDisplaySystem = -3015,
        ovrError_Obsolete = -3016,
        ovrError_DisabledOrDefaultAdapter = -3017,
        ovrError_HybridGraphicsNotSupported = -3018,
        ovrError_DisplayManagerInit = -3019,
        ovrError_TrackerDriverInit = -3020,
        ovrError_LibSignCheck = -3021,
        ovrError_LibPath = -3022,
        ovrError_LibSymbols = -3023,
        ovrError_RemoteSession = -3024,
        ovrError_InitializeVulkan = -3025,
        ovrError_BlacklistedGfxDriver = -3026,
        ovrError_DisplayLost = -6000,
        ovrError_TextureSwapChainFull = -6001,
        ovrError_TextureSwapChainInvalid = -6002,
        ovrError_GraphicsDeviceReset = -6003,
        ovrError_DisplayRemoved = -6004,
        ovrError_ContentProtectionNotAvailable = -6005,
        ovrError_ApplicationInvisible = -6006,
        ovrError_Disallowed = -6007,
        ovrError_DisplayPluggedIncorrectly = -6008,
        ovrError_DisplayLimitReached = -6009,
        ovrError_RuntimeException = -7000,
        ovrError_NoCalibration = -9000,
        ovrError_OldVersion = -9001,
        ovrError_MisformattedBlock = -9002

    ctypedef struct ovrErrorInfo:
        ovrResult Result
        char[512] ErrorString

cdef inline int OVR_SUCCESS(ovrResult result):
    return result >= ovrSuccessType.ovrSuccess

cdef inline int OVR_UNQUALIFIED_SUCCESS(ovrResult result):
    return result == ovrSuccessType.ovrSuccess

cdef inline int OVR_FAILURE(ovrResult result):
    return not OVR_SUCCESS(result)