# PsychXR
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/mdcutone/psychxr?include_prereleases)
![PyPI](https://img.shields.io/pypi/v/psychxr)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/psychxr)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/psychxr)

PsychXR is a collection of [Python](https://www.python.org/) extension libraries for interacting with eXtended Reality displays (HMDs), intended for neuroscience and psychology research applications. 

PsychXR is used by PsychoPy v3 (http://www.psychopy.org/) to provide HMD support.

## Supported Devices

* Oculus Rift DK2, CV1, and S

## Installation

See [Getting PsychXR](http://psychxr.org/installing.html) for installation instructions.

## Limitations

There are several limitations to the current version of PsychXR which may make it unsuitable for certain applications.

* Only Oculus VR HMDs which use the PC SDK (Rift S, CV1, and DK2) are supported, the CV1 and S are recommended (might work with the tethered Quest, haven't tested it out). There are currently no plans to support the mobile SDK.
* OpenGL is required for rendering, no support for other graphics APIs (i.e. Vulkan and DirectX) is available at this time. You must use some OpenGL framework such as Pyglet, GLFW ([example](https://github.com/mdcutone/psychxr/blob/master/demo/rift/libovr_headtracking.py)) or PyOpenGL with PsychXR to create visual stimuli.

## Support

If you encounter problems with PsychXR, please submit an issue to [PsychXR's issue tracker](https://github.com/mdcutone/psychxr/issues). *This software is not officially supported by any device vendor or manufacturer! Please do not direct PsychXR support requests to them.*

## Authors

* **Matthew D. Cutone** - The Centre for Vision Research, York University - (https://github.com/mdcutone)
* **Dr. Laurie M. Wilcox** - The Centre for Vision Research, York University

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## How To Cite PsychXR

If you use PsychXR for your research, please use the following citation:

```
Cutone, M. D. & Wilcox, L. M. (2019). PsychXR (Version 0.2.0) [Software]. Available from https://github.com/mdcutone/psychxr.
```

