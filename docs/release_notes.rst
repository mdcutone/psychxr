=============
Release Notes
=============

Version 0.2 Released - 2019-07-01
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PsychXR 0.2 is has numerous bug-fixes and enhancements based off user feedback.
Version 0.2 contains many API breaking features, however the API will be mostly
stable from this point forward.

The number of features added this release are too numerous to list individually,
so here is a summary of the biggest changes:

**General**

* NumPy is now required to build PsychXR. Matrices, vectors, and quaternions are
  now returned as NumPy arrays.
* Greatly improved documentation and examples for many functions.

**Oculus Rift Support (libovr)**

* The `libovr` extension module is now built with version 1.37 of the Oculus
  Rift SDK, fully supporting the new Oculus Rift S. The module now emits a
  warning if built with a different version.
* Installation automatically finds Oculus PC SDK header and library files. Only
  the location of the SDK needs to be specified. Hopefully this should improve
  the experience when building from source.
* Matrices, vectors, and quaternions are now exposed using NumPy arrays, math
  types like `ovrMatrix4f`, `ovrQuatf`, etc. have been dropped. However, many of
  these functions have been integrated as methods of the new `LibOVRPose` rigid
  body pose class.
* Lots of other changes.

`Click here to download PsychXR 0.2 <https://github.com/mdcutone/psychxr/releases>`_

PsychoPy is still using version 0.1.4 of PsychXR for Rift integration. Expect
support for 0.2.0 to be included in the coming weeks. To prevent breaking those
installations, PsychXR 0.2 will not be uploaded to PIP until PsychoPy has been
updated. However, you can download and install version 0.2.0 from the
`releases <https://github.com/mdcutone/psychxr/releases>`_ page.