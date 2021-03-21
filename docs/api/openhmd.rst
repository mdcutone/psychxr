====================================================================
 :mod:`~psychxr.drivers.openhmd` - VR device interface using OpenHMD
====================================================================

.. currentmodule:: psychxr.drivers.openhmd

Classes and functions for using various VR devices (HMDs, controllers, and
generic spatial trackers) through the OpenHMD driver interface.

OpenHMD is a free and open-source software (FOSS) project which aims to provide
a driver interface for a variety of VR headsets, controllers and trackers. This
interface is under heavy development (both in PsychXR and OpenHMD itself) so
expect things to change without notice between releases of PsychXR.

Overview
========

Classes
~~~~~~~

.. autosummary::
    OpenHMDDeviceInfo

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

.. autoclass:: OpenHMDDeviceInfo
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