=============
Release Notes
=============

Version 0.2.1 - 2019-07-18
~~~~~~~~~~~~~~~~~~~~~~~~~~

This release adds improved haptics support, bounding boxes, visibility culling,
and more features to `LibOVRPose`. There are a few minor API breaking changes,
however in the future API changes will raise deprecation warnings and be phased
out gradually over several releases.

**General**

* A bunch of documentation fixes and examples have been added, including a
  tutorial on rendering to the Rift using pure OpenGL.

**Oculus Rift Support (libovr)**

* Support for haptic buffers for use with Touch controllers. A haptics
  buffer contains an array of samples which specify Touch controller
  vibration amplitudes. Buffers can be passed to the haptics engine for
  playback, permitting custom vibration profiles.
* Added `mirrorOptions` to `createMirrorTexture` to customize how mirrors
  are presented (eg. pre-distortion, rectilinear, one eye only, etc.)
* Added `getViewMatrix` to `LibOVRPose` which creates view matrices, which
  transforms points into the space of the pose. This allows you to use rigid
  body poses to define eye locations for rendering.
* Added `getAzimuthElevation` and `getAngleTo` methods to `LibOVRPose`, for
  computing Euler angles of targets within the reference frame of a poses.
* Nearly all `LibOVRPose` transformation methods can write values to
  pre-allocated output arrays.
* Added an optional `originPose` to `calcEyePoses`.
* Added a bounding box attribute to poses. You can create an axis-aligned
  bounding box object (`LibOVRBounds`) and assign them to poses.
  `LibOVRBounds` has a `fit` method to compute boundaries for a 3D model if
  supplied a list of vertices.
* New `cullPose` function allows you to cull meshes associated with poses if
  they are not visible during rendering by testing if their bounding boxes
  fall outside of the view frustum. This reduces CPU/GPU workload when
  complex drawing scenes.
* Added logging callbacks. You can register a Python function as a callback
  for when LibOVR returns a message. Spits out lots of information, maybe
  you'll find some of it useful?
* `endFrame` returns the absolute system time it was called.
* ASW stats are also returned with `LibOVRPerfStatsPerCompositorFrame`.
* Fixed head-locking. Head-locking prevents compositor ASW from acting upon
  the layer. This fix restores the behaviour seen in the alpha releases
  of PsychXR.

`Click here to download PsychXR 0.2.1 <https://github.com/mdcutone/psychxr/releases>`_

Version 0.2.0 - 2019-07-01
~~~~~~~~~~~~~~~~~~~~~~~~~~

PsychXR 0.2.0 is has numerous bug-fixes and enhancements based off user feedback.
Version 0.2.0 contains many API breaking features, however the API will be mostly
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
installations, PsychXR 0.2.0 will not be uploaded to PIP until PsychoPy has been
updated. However, you can download and install version 0.2.0 from the
`releases <https://github.com/mdcutone/psychxr/releases>`_ page.