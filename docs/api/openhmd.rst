=====================================================================
 :mod:`~psychxr.drivers.openhmd` - Generic VR interface using OpenHMD
=====================================================================

.. currentmodule:: psychxr.drivers.openhmd

Classes and functions for using various VR devices (HMDs, controllers, and
generic spatial trackers) through the OpenHMD driver interface.

OpenHMD is a free and open-source software (FOSS) project which aims to provide
a driver interface for a variety of VR headsets, controllers and trackers. This
interface is under heavy development (both with PsychXR and OpenHMD itself) so
expect things to change without notice between releases of PsychXR.

This module isn't a simple wrapper around OpenHMD. The design of this OpenHMD
Python API is intended to be similar (to the extent that is feasible) to that of
`LibOVR`. Therefore, using the OpenHMD API in PsychXR is quite different than
using the C API. Where the C API uses getter and setter functions for nearly
everything, the PsychXR interface uses classes and functions analogous to those
in the `LibOVR` driver library. Unlike `LibOVR`, OpenHMD does not come with a
compositor. Therefore, the user must implement their own system to present
scenes to the display (ex. using Pyglet or GLFW). This requires considerably
more effort on behalf of the user than `LibOVR`, but may offer greater
flexibility for those that need it by removing the "black box" between the
application and the display.

Overview
========

Classes
~~~~~~~

.. autosummary::
    OHMDDeviceInfo
    OHMDPose
    OHMDDeviceInfo
    OHMDDisplayInfo
    OHMDControllerInfo

Functions
~~~~~~~~~

.. autosummary::
    success
    failure
    getVersion
    create
    destroy
    probe
    isContextProbed
    getDeviceCount
    getError
    getDevices
    getDisplayInfo
    openDevice
    closeDevice
    getDisplayInfo
    getDevicePose
    lastUpdateTimeElapsed
    update
    getString
    getListString
    getListInt

Details
=======

Classes
~~~~~~~

.. autoclass:: OHMDPose
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: OHMDDeviceInfo
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: OHMDDisplayInfo
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: OHMDControllerInfo
    :members:
    :undoc-members:
    :inherited-members:

Functions
~~~~~~~~~

.. autofunction:: success
.. autofunction:: failure
.. autofunction:: getVersion
.. autofunction:: create
.. autofunction:: destroy
.. autofunction:: probe
.. autofunction:: isContextProbed
.. autofunction:: getDeviceCount
.. autofunction:: getError
.. autofunction:: getDevices
.. autofunction:: getDisplayInfo
.. autofunction:: openDevice
.. autofunction:: closeDevice
.. autofunction:: getDisplayInfo
.. autofunction:: getDevicePose
.. autofunction:: lastUpdateTimeElapsed
.. autofunction:: update
.. autofunction:: getString
.. autofunction:: getListString
.. autofunction:: getListInt