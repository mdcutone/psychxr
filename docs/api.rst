API Reference
=============

These pages provide developers detailed documentation about functions and
classes within the latest release of the PsychXR library. Please submit an issue
to the issue tracker `here <https://github.com/mdcutone/psychxr/issues>`_ to
report any errata.

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
