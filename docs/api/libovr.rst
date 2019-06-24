============================================================
 :mod:`~psychxr.libovr` - Oculus Rift interface using LibOVR
============================================================

.. currentmodule:: psychxr.libovr

Classes and functions for using the Oculus Rift (DK2, CV1, and S) HMD and
associated peripherals via the official runtime/SDK (LibOVRRT). Currently, only
OpenGL is supported for rendering VR scenes.

**This extension module makes use of the official Oculus PC SDK. A C/C++
interface for tracking, rendering, and VR math for Oculus products. The Oculus
PC SDK is Copyright (c) Facebook Technologies, LLC and its affiliates. All
rights reserved. You must accept the 'EULA', 'Terms of Use' and 'Privacy Policy'
associated with the Oculus PC SDK to use this module in your software,
see** `Legal Documents <https://www.oculus.com/legal/terms-of-service/>`_ **to
access those documents.**

Overview
========

Classes
~~~~~~~

.. autosummary::
    LibOVRPose
    LibOVRPoseState
    LibOVRHmdInfo
    LibOVRTrackerInfo

Functions
~~~~~~~~~

.. autosummary::
    success
    unqualifiedSuccess
    failure
    getBool
    setBool
    getInt
    setInt
    getFloat
    setFloat
    getFloatArray
    setFloatArray
    getString
    setString
    isOculusServiceRunning
    isHmdConnected
    getHmdInfo
    initialize
    create
    destroyTextureSwapChain
    destroyMirrorTexture
    destroy
    shutdown
    getGraphicsLUID
    setHighQuality
    setHeadLocked
    getPixelsPerTanAngleAtCenter
    getTanAngleToRenderTargetNDC
    getDistortedViewport
    getEyeRenderFov
    setEyeRenderFov
    getEyeAspectRatio
    getEyeHorizontalFovRadians
    getEyeVerticalFovRadians
    getEyeFocalLength
    calcEyeBufferSize
    getLayerEyeFovFlags
    setLayerEyeFovFlags
    getTextureSwapChainLengthGL
    getTextureSwapChainCurrentIndex
    getTextureSwapChainBufferGL
    createTextureSwapChainGL
    setEyeColorTextureSwapChain
    createMirrorTexture
    getMirrorTexture
    getTrackingState
    getDevicePoses
    calcEyePoses
    getHmdToEyePose
    setHmdToEyePose
    getEyeRenderPose
    setEyeRenderPose
    getEyeRenderViewport
    setEyeRenderViewport
    getEyeProjectionMatrix
    getEyeViewMatrix
    getPredictedDisplayTime
    timeInSeconds
    waitToBeginFrame
    beginFrame
    commitTextureSwapChain
    endFrame
    resetFrameStats
    getTrackingOriginType
    setTrackingOriginType
    recenterTrackingOrigin
    specifyTrackingOrigin
    clearShouldRecenterFlag
    getTrackerCount
    getTrackerInfo
    updatePerfStats
    getAdaptiveGpuPerformanceScale
    getFrameStatsCount
    anyFrameStatsDropped
    checkAswIsAvailable
    getVisibleProcessId
    checkAppLastFrameDropped
    checkCompLastFrameDropped
    getFrameStats
    getLastErrorInfo
    setBoundaryColor
    resetBoundaryColor
    getBoundaryVisible
    showBoundary
    hideBoundary
    getBoundaryDimensions
    getConnectedControllerTypes
    updateInputState
    getButton
    getTouch
    getThumbstickValues
    getIndexTriggerValues
    getHandTriggerValues
    setControllerVibration
    getSessionStatus

Details
=======

:class:`~psychxr.libovr.LibOVRPose` - Rigid body pose
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Poses of rigid bodies are represented as a position vector/coordinate and
orientation quaternion.

Methods associated with this class perform various transformations with the
components of the pose (position and orientation) using routines found in
`OVR_MATH.h <https://developer.oculus.com/reference/libovr/1.32/o_v_r_math_8h/>`_,
which is part of the Oculus PC SDK.

.. autoclass:: LibOVRPose
    :members:
    :undoc-members:
    :inherited-members:

:class:`~psychxr.libovr.LibOVRPoseState` - Rigid body pose state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pose, angular and linear motion derivatives of a tracked rigid body reported by
LibOVR.

.. autoclass:: LibOVRPoseState
    :members:
    :undoc-members:
    :inherited-members:

:class:`~psychxr.libovr.LibOVRHmdInfo` - HMD information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Class for general HMD information and capabilities.

.. autoclass:: LibOVRHmdInfo
    :members:
    :undoc-members:
    :inherited-members:

:class:`~psychxr.libovr.LibOVRTrackerInfo` - Tracker information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Class for storing tracker (sensor) information such as pose, status, and camera
frustum information. This object is returned by calling
:func:`~psychxr.libovr.getTrackerInfo`.

.. autoclass:: LibOVRTrackerInfo
    :members:
    :undoc-members:
    :inherited-members:

.. autofunction:: success
.. autofunction:: unqualifiedSuccess
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
.. autofunction:: getTanAngleToRenderTargetNDC
.. autofunction:: getDistortedViewport
.. autofunction:: getEyeRenderFov
.. autofunction:: setEyeRenderFov
.. autofunction:: getEyeAspectRatio
.. autofunction:: getEyeHorizontalFovRadians
.. autofunction:: getEyeVerticalFovRadians
.. autofunction:: getEyeFocalLength
.. autofunction:: calcEyeBufferSize
.. autofunction:: getLayerEyeFovFlags
.. autofunction:: setLayerEyeFovFlags
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
.. autofunction:: getEyeRenderViewport
.. autofunction:: setEyeRenderViewport
.. autofunction:: getEyeProjectionMatrix
.. autofunction:: getEyeViewMatrix
.. autofunction:: getPredictedDisplayTime
.. autofunction:: timeInSeconds
.. autofunction:: waitToBeginFrame
.. autofunction:: beginFrame
.. autofunction:: commitTextureSwapChain
.. autofunction:: endFrame
.. autofunction:: resetFrameStats
.. autofunction:: getTrackingOriginType
.. autofunction:: setTrackingOriginType
.. autofunction:: recenterTrackingOrigin
.. autofunction:: specifyTrackingOrigin
.. autofunction:: clearShouldRecenterFlag
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
.. autofunction:: getConnectedControllerTypes
.. autofunction:: updateInputState
.. autofunction:: getButton
.. autofunction:: getTouch
.. autofunction:: getThumbstickValues
.. autofunction:: getIndexTriggerValues
.. autofunction:: getHandTriggerValues
.. autofunction:: setControllerVibration
.. autofunction:: getSessionStatus