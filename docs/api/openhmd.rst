====================================================================
 :mod:`~psychxr.drivers.openhmd` - OpenHMD interface for VR hardware
====================================================================

.. currentmodule:: psychxr.drivers.openhmd

Classes and functions for using various VR devices (HMDs, controllers, and
generic spatial trackers) through the OpenHMD driver interface.

OpenHMD is a free and open-source software (FOSS) project which aims to provide
a driver interface for a variety of VR headsets, controllers and trackers. This
interface is under heavy development (both with PsychXR and OpenHMD itself) so
expect things to change without notice between releases of PsychXR.

This module isn't a simple wrapper around OpenHMD. The design of OpenHMD Python
API is intended to be similar (to the extent that is feasible) to that of
`LibOVR`. Therefore, using the OpenHMD API in PsychXR is quite different than
using the C API. Where the C API uses getter and setter functions for
everything, the PsychXR interface uses classes and functions analogous to those
in the `LibOVR` driver library.

Overview
========

Classes
~~~~~~~

.. autosummary::
    OHMDDeviceInfo

Functions
~~~~~~~~~

.. autosummary::
    success
    failure
    create
    destroy
    probe
    getError
    update

Details
=======

Classes
~~~~~~~

.. autoclass:: OHMDDeviceInfo
    :members:
    :undoc-members:
    :inherited-members:

Functions
~~~~~~~~~

.. autofunction:: success
.. autofunction:: failure
.. autofunction:: create
.. autofunction:: destroy
.. autofunction:: getError
.. autofunction:: probe
.. autofunction:: update