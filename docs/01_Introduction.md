# Introduction to PsychXR

PsychXR is a collection of Python extension libraries for interfacing with
head-mounted displays, intended for research applications in neuroscience and
psychology. While other solutions may exist, PsychXR is developed considering
the needs of the vision science community. Removing the "black-boxes"
proprietary game engines impose between your application and the HMD's API.

The libraries are written in Cython, providing high-performance API access,
leaving more headroom per-frame for your application code. PsychXR can be
used on its own to add HMD support to OpenGL applications. However, it's
considerably easier to develop experiments using PsychoPy which uses PsychXR
to provide HMD support.

Furthermore, PsychXR is released under the MIT license, which makes it
acceptable to distribute, inspect and modify the code as you see fit.

