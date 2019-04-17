============================================================
 :mod:`~psychxr.libovr` - Oculus Rift interface using LibOVR
============================================================

.. currentmodule:: psychxr.libovr

Classes and functions for using the Oculus Rift HMD and associated peripherals
via the official runtime/SDK (LibOVRRT). Currently, only OpenGL is supported for
rendering VR scenes.

.. note:: This extension module makes use of the official Oculus PC SDK. A C/C++ interface for tracking, rendering, and VR math for Oculus products. The Oculus PC SDK is Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved. You must accept the 'EULA', 'Terms of Use' and 'Privacy Policy' associated with the Oculus PC SDK to use this module in your software, see https://www.oculus.com/legal/terms-of-service/ to access those documents.

Classes
=======

:class:`~psychxr.libovr.LibOVRPose` - Combined position and orientation data
----------------------------------------------------------------------------

Poses are represented as a position vector/coordinate and orientation
quaternion.

Methods associated with this class perform various transformations with the
components of the pose (position and orientation) using routines found in
`OVR_MATH.h <https://developer.oculus.com/reference/libovr/1.32/o_v_r_math_8h/>`_, which is part of the Oculus PC SDK.

.. autoclass:: LibOVRPose
    :members:
    :undoc-members:
    :inherited-members:

:class:`~psychxr.libovr.LibOVRPoseState` - Rigid body pose state
---------------------------------------------------------------

Position and orientation of a tracked body reported by LibOVR. This includes
first and second derivatives for angular and linear motion.

.. autoclass:: LibOVRPoseState
    :members:
    :undoc-members:
    :inherited-members:

:class:`~psychxr.libovr.LibOVRHmdInfo` - HMD information
--------------------------------------------------------

.. autoclass:: LibOVRHmdInfo
    :members:
    :undoc-members:
    :inherited-members:

:class:`~psychxr.libovr.LibOVRTrackerInfo` - Tracker information
----------------------------------------------------------------

Class for storing tracker (sensor) information such as pose, status, and camera
frustum information. This object is returned by calling
:func:`~psychxr.libovr.getTrackerInfo`.

.. autoclass:: LibOVRTrackerInfo
    :members:
    :undoc-members:
    :inherited-members:

Functions
=========

.. autofunction:: success
.. autofunction:: unqualifedSuccess
.. autofunction:: failure
.. autofunction:: getBool
.. autofunction:: setBool
.. autofunction:: getInt
.. autofunction:: setInt
.. autofunction:: getFloat
.. autofunction:: setFloat
.. autofunction:: getFloatArray
.. autofunction:: setFloatArray
.. autofunction:: getString
.. autofunction:: setString
.. autofunction:: isOculusServiceRunning
.. autofunction:: isHmdConnected
.. autofunction:: getHmdInfo
.. autofunction:: getUserHeight
.. autofunction:: getEyeHeight
.. autofunction:: getNeckEyeDist
.. autofunction:: getEyeToNoseDist
.. autofunction:: initialize
.. autofunction:: create
.. autofunction:: destroyTextureSwapChain
.. autofunction:: destroyMirrorTexture
.. autofunction:: destroy
.. autofunction:: shutdown
.. autofunction:: getGraphicsLUID
.. autofunction:: setHighQuality
.. autofunction:: setHeadLocked
.. autofunction:: getPixelsPerTanAngleAtCenter
.. autofunction:: getDistortedViewport
.. autofunction:: getEyeRenderFov
.. autofunction:: setEyeRenderFov
.. autofunction:: calcEyeBufferSize
.. autofunction:: getTextureSwapChainLengthGL
.. autofunction:: getTextureSwapChainCurrentIndex
.. autofunction:: getTextureSwapChainBufferGL
.. autofunction:: createTextureSwapChainGL
.. autofunction:: setEyeColorTextureSwapChain
.. autofunction:: createMirrorTexture
.. autofunction:: getMirrorTexture
.. autofunction:: getTrackingState
.. autofunction:: getDevicePoses
.. autofunction:: calcEyePoses
.. autofunction:: getHmdToEyePose
.. autofunction:: setHmdToEyePose
.. autofunction:: getEyeRenderPose
.. autofunction:: setEyeRenderPose
.. autofunction:: getEyeProjectionMatrix
.. autofunction:: getEyeRenderViewport
.. autofunction:: setEyeRenderViewport
.. autofunction:: getEyeViewMatrix
.. autofunction:: getPredictedDisplayTime
.. autofunction:: timeInSeconds
.. autofunction:: perfHudMode
.. autofunction:: hidePerfHud
.. autofunction:: perfHudModes
.. autofunction:: waitToBeginFrame
.. autofunction:: beginFrame
.. autofunction:: commitTextureSwapChain
.. autofunction:: endFrame
.. autofunction:: resetFrameStats
.. autofunction:: getTrackingOriginType
.. autofunction:: setTrackingOriginType
.. autofunction:: recenterTrackingOrigin
.. autofunction:: getTrackerCount
.. autofunction:: getTrackerInfo
.. autofunction:: updatePerfStats
.. autofunction:: getAdaptiveGpuPerformanceScale
.. autofunction:: getFrameStatsCount
.. autofunction:: anyFrameStatsDropped
.. autofunction:: checkAswIsAvailable
.. autofunction:: getVisibleProcessId
.. autofunction:: checkAppLastFrameDropped
.. autofunction:: checkCompLastFrameDropped
.. autofunction:: getFrameStats
.. autofunction:: getLastErrorInfo
.. autofunction:: setBoundaryColor
.. autofunction:: resetBoundaryColor
.. autofunction:: getBoundaryVisible
.. autofunction:: showBoundary
.. autofunction:: hideBoundary
.. autofunction:: getBoundaryDimensions
.. autofunction:: updateInputState
.. autofunction:: getButton
.. autofunction:: getTouch
.. autofunction:: getThumbstickValues
.. autofunction:: getIndexTriggerValues
.. autofunction:: getHandTriggerValues
.. autofunction:: setControllerVibration
.. autofunction:: getSessionStatus