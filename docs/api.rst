API Reference
=============

These pages provide developers detailed documentation about functions and
classes within the latest release of the PsychXR library. Please submit an issue
to the tracker `here <https://github.com/mdcutone/psychxr/issues>`_ to report
any errata.

:mod:`psychxr.drivers` - VR hardware interfaces
-----------------------------------------------

These modules provide extension types and functions for working with a
particular VR driver API. Note that not all functionality provided by a driver
may be exposed by a given interface. Interfaces may differ greatly between
drivers in terms of available features, functionality, and performance. One must
chose which to use that best suits the requirements of the application.

.. toctree::
   :maxdepth: 1

   api/libovr
   api/openhmd

:mod:`psychxr.tools` - Tools for working with VR
------------------------------------------------

This module contains tools that are interoperable with all driver interfaces to
aid in composing virtual environments and enabling basic interaction within
virtual spaces. These tools combined with a graphics framework (e.g., GLFW,
ModernGL, etc.) and audio library (e.g. OpenAL), gives one a complete
environment for developing rich VR applications.

.. toctree::
   :maxdepth: 1

   api/vrmath
