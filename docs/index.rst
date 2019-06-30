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

PsychXR may be used as stand-alone library in conjunction with some OpenGL
framework like `pyGLFW <https://github.com/FlorianRhiem/pyGLFW>`_ or
`pyglet <https://pyglet.readthedocs.io/en/pyglet-1.3-maintenance/>`_. However,
the easiest solution for researchers is to use `PsychoPy <https://psychopy.org>`_
which uses PsychXR for HMD support, but also provides a framework for developing
psychophysics experiments.

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

Site Index
----------

:ref:`genindex`


