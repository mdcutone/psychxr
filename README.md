# PsychXR

PsychXR is a collection of [Python](https://www.python.org/) extension libraries for interacting with eXtended Reality displays (HMDs), intended for neuroscience and psychology research applications. 

PsychXR is used by PsychoPy v3 (http://www.psychopy.org/) to provide HMD support.

## Supported Devices

* Oculus Rift DK2, CV1, and S

## Installation

See [Getting PsychXR](http://psychxr.org/installing.html) for installation instructions.

## Limitations

There are several limitations to the current version of PsychXR which may make it unsuitable for certain applications.

* Only Oculus VR HMDs which use the PC SDK (Rift S, CV1, and DK2) are supported, the CV1 and S are recommended. There are currently no plans to support the mobile SDK.
* OpenGL is required for rendering, no other graphics API (i.e. Vulkan and DirectX) is available at this time. You must use some OpenGL framework such as Pyglet, GLFW ([example](https://github.com/mdcutone/psychxr/blob/master/demo/rift/oculus_glfw.py)) or PyOpenGL with PsychXR to create visual stimuli.

## Support

If you encounter problems with PsychXR, please submit an issue to [PsychXR's issue tracker](https://github.com/mdcutone/psychxr/issues). *This software is not offically supported by any device vendor or manufacturer! Please do not direct PsychXR support requests to them.*

## Authors

* **Matthew D. Cutone** - The Centre for Vision Research, York University - (https://github.com/mdcutone)
* **Dr. Laurie M. Wilcox** - The Centre for Vision Research, York University

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## How To Cite PsychXR

If you use PsychXR for your research, please use the following citation:

```
Cutone, M. D. & Wilcox, L. M. (2018). PsychXR (Version 0.1.4) [Software]. Available from https://github.com/mdcutone/psychxr.
```

