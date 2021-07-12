=========================================================
 :mod:`~psychxr.tools.vrmath` - Tools for VR related math
=========================================================

.. currentmodule:: psychxr.tools.vrmath

Toolbox of classes and functions for performing VR related math. These are used
to describe and compute the spatial configuration of objects in a VR scene for
the purpose of rendering and interaction.

**Be aware that this module is currently in an early phase of development and
may be incomplete and buggy. Please test it out and report any bugs
encountered.**

Overview
========

Classes
~~~~~~~

.. autosummary::
    RigidBodyPose
    BoundingBox

Functions
~~~~~~~~~

.. autosummary::
    calcEyePoses

Details
=======

Classes
~~~~~~~

.. autoclass:: RigidBodyPose
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: BoundingBox
    :members:
    :undoc-members:
    :inherited-members:

Functions
~~~~~~~~~

.. autofunction:: calcEyePoses
