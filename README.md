# PsychXR
[![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/mdcutone/psychxr?include_prereleases)](https://github.com/mdcutone/psychxr/releases)
[![PyPI](https://img.shields.io/pypi/v/psychxr)](https://pypi.org/project/psychxr/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/psychxr)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/psychxr)
[![Generic badge](https://img.shields.io/badge/supported-DK2%20|%20CV1%20|%20S%20|%20Quest-blue.svg)](https://shields.io/)

PsychXR is a collection of [Python](https://www.python.org/) extension libraries for interacting with eXtended Reality displays (HMDs), intended for neuroscience and psychology research applications.

PsychXR is used by PsychoPy v3 (http://www.psychopy.org/) to provide HMD support.

## Why PsychXR?

PsychXR is intended to provide direct access to driver APIs for HMDs using a thin and very fast interface. This can be useful in situations where other tools fail to expose or limit device capabilities.

PsychXR is free and open source, unlike many of the tools widely used by researchers for creating VR experiments. If you'd like to keep the future of VR research as  open as possible, consider using or supporting projects like [PsychXR](http://psychxr.org), [PsychoPy](https://www.psychopy.org/), [psychtoolbox](http://psychtoolbox.org/) and the [Godot Engine](https://godotengine.org/).  

## Supported Devices

* Oculus Rift DK2, CV1, S, Quest 1 & 2 (with thether)

## Installation

See [Getting PsychXR](http://psychxr.org/installing.html) for installation instructions. Pre-built packages are available for Python 3.6, however PsychXR will build on later versions of Python.

## Limitations

There are several limitations to the current version of PsychXR which may make it unsuitable for certain applications.

* Only Oculus VR HMDs which use the PC SDK (Rift S, CV1, DK2 and Quest 1+2) are supported, the CV1 and S are recommended. There are currently no plans to support the mobile SDK or distribution over app stores/
* OpenGL is required for rendering, no support for other graphics APIs (i.e. Vulkan and DirectX) is available at this time. You must use some OpenGL framework such as Pyglet, GLFW ([example](https://github.com/mdcutone/psychxr/blob/master/demo/rift/libovr_headtracking.py)) or PyOpenGL with PsychXR to create visual stimuli.

## Support

If you encounter problems with PsychXR, please submit an issue to [PsychXR's issue tracker](https://github.com/mdcutone/psychxr/issues). *This software is not officially supported by any device vendor or manufacturer! Please do not direct PsychXR support requests to them.*

## News

For updates on PsychXR's development, see [NEWS](https://github.com/mdcutone/psychxr/blob/master/NEWS.md).

## Contributing

If you would like to expand on PsychXR, feel free to submit pull requests. Help for doing the following:

* OpenXR or OpenVR support
* OpenHMD support

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

