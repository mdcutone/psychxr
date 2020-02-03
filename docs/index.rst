.. PsychXR documentation master file, created by
   sphinx-quickstart on Fri Dec 21 20:14:47 2018.

Welcome to PsychXR's Documentation
==================================

PsychXR is a collection of Python extension libraries for interacting with
eXtended Reality displays (HMDs), intended for neuroscience and psychology
research applications.

The goal of this project is to provide researchers in the vision science
community free and flexible tools to experiment and prototype ideas with virtual
reality hardware using the
`Python programming language <http://www.python.org/>`_. Allowing users to directly
leverage the vast ecosystem of scientific libraries associated with it (e.g.
`SciPy <https://www.scipy.org/>`_, `NumPy <https://www.numpy.org/>`_,
`PyTorch <https://pytorch.org/>`_, etc.)

PsychXR provides a lightweight interface to hardware drivers with extra tools
to simplify some aspects of creating virtual reality applications. However, this
library does not handle the actual rendering of VR scenes or audio. Knowledge of
`OpenGL <https://www.opengl.org/>`_ is required at this time to create images
and present them on the HMD. However, researchers may consider using
`PsychoPy <https://psychopy.org>`_ which uses PsychXR for HMD support, but also
provides a means of rendering 3D stimuli and developing psychophysics
experiments.

Contents
--------
.. toctree::
   :maxdepth: 1

   installing
   release_notes
   license
   api
   examples

Source Code
-----------

PsychXR is open-source, the source code is freely available for review and
modification. The source code repository is hosted on
`GitHub <https://github.com/mdcutone/psychxr>`_. There you can also submit bug
reports and pull requests.

PsychXR is mostly written in `Cython <https://cython.org/>`_ a superset of the
Python language.

Hardware Support
----------------

As of now, only the Oculus Rift series of HMDs are supported (DK2, CV1, S) via
the `~psychxr.drivers.libovr` driver interface which uses the official PC SDK.
However, additional drivers are being considered to be included in the future.

Related Projects
----------------

* `PsychoPy <https://psychopy.org>`_ is a software library for running
  neuroscience, psychology and psychophysics experiments which uses PsychXR for
  HMD support. In addition, PsychoPy handles rendering stimuli to the display,
  which PsychXR alone does not do.

* Octave/MATLAB users may consider using
  `PsychVRToolbox <http://psychtoolbox.org/docs/PsychOculusVR1>`_, part of the
  `PsychToolbox <http://psychtoolbox.org/>`_ package for similar functionality
  to PsychXR.

How to Cite PsychXR
-------------------

If you use PsychXR in one of your projects, please consider citing it using the
following:

   Cutone, M. D. & Wilcox, L. M. (2018). PsychXR (Version 0.2.0) [Software].
   Available from https://github.com/mdcutone/psychxr.

Site Index
----------

:ref:`genindex`


