=============
Release Notes
=============

Version 0.2.4 - 2021-03-16
~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally another release of PsychXR after nearly a year! The 0.2.4 release of
PsychXR introduces new features and significant changes to how the project is
organized. The library itself has undergone considerable refactoring, intended
to make PsychXR more modular for a future plugin system to allow features (e.g.,
other drivers, tools, etc.) to augment the core installation.

The LibOVR driver interface has been updated to use version 23.0 (1.55) of the
Oculus PC SDK. This version of LibOVR introduces a new color management API to
help ensure content appears correct between headsets whose displays have
different color models. The :class:`~psychxr.drivers.libovr.LibOVRHmdColorSpace`
class has been added to the `libovr` module to work with this API. As a bonus,
the class provides chromaticity coordinates of the RGB primaries and the white
point for each display type specified. The color management stuff is pretty new,
if you plan on using this feature test it out and provide feedback if something
unexpected happens or the documentation is unclear.

Interaction is a very important aspect of VR, therefore having some basic
facility for that within PsychXR is appreciated. The
:class:`~psychxr.drivers.libovr.LibOVRPose` class has a new
:meth:`~psychxr.drivers.libovr.LibOVRPose.raycastPose` method for interacting
with the bounding boxes associated with poses. This allows for basic interaction
between the user and objects in the scene without needing to figure that stuff
out yourself.

See CHANGELOG for more information regarding changes this release.

**General**

* Major reorganization of the codebase. Driver interfaces have been moved to the
  :mod:`psychxr.drivers` sub-package. For backwards compatibility you can still
  access the `libovr` driver by invoking ``import psychxr.libovr as libovr``,
  but the proper way now is ``import psychxr.drivers.libovr as libovr``. The
  reason for this change is to facilitate a plugin system where new interface
  libraries will appear under :mod:`~psychxr.drivers`.

**Oculus Rift Support (libovr)**

* Version bump of the LibOVR PC SDK to 23.0 (internally 1.55 for some reason).
  This is the minimum version **REQUIRED** to build PsychXR 0.2.4 with LibOVR
  support (absolutely needed at this point).
* Added support for the Oculus color management API:
    - A new class called :class:`~psychxr.drivers.libovr.LibOVRHmdColorSpace`.
      Properties of this class can be used to get chromaticity coordinates for
      the RGB primaries and white point of the target color space. These values
      are provided by the manufacturer in their documentation but are made
      available for convenience.
    - New functions :func:`~psychxr.drivers.libovr.getHmdColorSpace` and
      :func:`~psychxr.drivers.libovr.setClientColorSpace`.
    - Symbolic constants added for ``COLORSPACE_UNKNOWN``,
      ``COLORSPACE_UNMANAGED``, ``COLORSPACE_RIFT_CV1``, ``COLORSPACE_RIFT_S``,
      ``COLORSPACE_QUEST``, ``COLORSPACE_REC_2020``, ``COLORSPACE_REC_709``,
      ``COLORSPACE_P3`` and ``COLORSPACE_ADOBE_RGB``.
* Added :meth:`~psychxr.drivers.libovr.LibOVRPose.raycastPose` for interaction
  with bounding boxes around other poses.
* Removed the ``BUILD_VERSION`` variable from the namespace of `libovr`.

`Click here to download PsychXR 0.2.4 <https://github.com/mdcutone/psychxr/releases>`_

Version 0.2.3 - 2019-12-10
~~~~~~~~~~~~~~~~~~~~~~~~~~

This release has some minor fixes and features such as improved memory use and
performance. There are some breaking changes in the release, see the CHANGELOG
file for more information.

**General**

* Added function `checkSessionStarted` to determine if there is an active VR
  session. This is helpful to determine if a session is active from another
  module or file within the same interpreter thread.
* Added `normalMatrix` and `getNormalMatrix` which retrieves a normal matrix
  for a mesh at a given pose. This matrix is commonly used by fragment
  shaders, and would usually need to be computed separately with the model
  matrix. That's no longer the case, you can now get a normal matrix along
  with your model matrix from a `LibOVRPose` instance.
* `LibOVRPose` matrices are now cached to improve performance and memory
  access. Returned `ndarray` matrices now reference data directly instead of
  being copied over to new arrays every time. Matrices are computed only
  after `pos` and `ori` are accessed/changed. Furthermore, they are computed
  only when any attribute or method of `LibOVRPose` which returns a matrix
  is invoked. If there are no changes to `pos` and `ori` between successive
  matrix related attribute or method calls, cached data will be returned
  immediately without additional computation. One caveat about this approach
  is that matrices are always recomputed when accessing values, even if
  attributes `pos` and `ori` were only read, since currently there is no way
  to determine if the referencing `ndarrays` modified their referenced data.
  So it's just always assumed that they did. There is also a `ctypes`
  attribute associated with the class which returns a dictionary of `ctypes`
  pointers to the underlying matrix data. This allows `pyglet`'s GL
  implementation to directly access the data contained in these matrices
  without needing to create pointers yourself from returned `ndarray`
  objects. See `Known Issues` for more information about possible problems
  associated with caching.
* Added `turn` method to `LibOVRPose` to rotate objects about an axis by
  some angle cumulatively.

`Click here to download PsychXR 0.2.3 <https://github.com/mdcutone/psychxr/releases>`_


Version 0.2.2 - 2019-10-16
~~~~~~~~~~~~~~~~~~~~~~~~~~

Bugfix release to address issues introduced in the 0.2+ codebase. This version
is being uploaded to PyPI for use with PsychoPy which has recently gotten
a new version of the Rift interface which supports PsychXR 0.2+. There should be
no breaking changes from the 0.2.1 release.

**Oculus Rift Support (libovr)**

* Fixed `LibOVRPerfStats` setting up incorrect pointers to
  `LibOVRPerfStatsPerCompositorFrame` objects, causing Cython to convert them
  to dictionaries instead of instances of `LibOVRPerfStatsPerCompositorFrame`.
* Fixed TOUCH_* module level constants not being exposed by __all__.

`Click here to download PsychXR 0.2.2 <https://github.com/mdcutone/psychxr/releases>`_

Version 0.2.1 - 2019-09-25
~~~~~~~~~~~~~~~~~~~~~~~~~~

This release adds improved haptics support, bounding boxes, visibility culling,
and more features to `LibOVRPose`. There are a few minor API breaking changes,
however in the future API changes will raise deprecation warnings and be phased
out gradually over several releases.

In the coming weeks, PsychoPy will be updated to support version 0.2.1 features.

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

Version 0.1.5 - 2019-09-25
~~~~~~~~~~~~~~~~~~~~~~~~~~

Hotfix release for version 0.1.4 which fixes an input bug where the range of
thumbstick values is improperly clipped.

**Oculus Rift Support (libovr)**

* Fixed thumbstick values being clipped to 1.0 regardless of the input.

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