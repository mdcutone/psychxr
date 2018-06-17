# PsychXR

Python extension library for interacting with eXtended Reality displays (HMDs), intended for research in neuroscience and psychology.

Device API wrappers are written in Cython, providing low latency and overhead. Only OpenGL based applications on Windows PCs are currently supported. Additional support for other platforms (Mac OS and Linux) is contingent on support from the device manufacturer.

## Supported Devices

* Oculus Rift DK2 and CV1

## Getting Started

The easiest way to get PsychXR is to get the [*.whl package](https://github.com/mdcutone/psychxr/releases) and install it using the 'pip install' command. The pre-compiled binaries are built on Windows 10 64-bit using the MSVC 15.0 compiler against Python 3.6 64-bit.

## Limitations

There are several limitations to the current version of PsychXR which may make it unsuitable for certain applications.

* Only Oculus VR HMDs which use the PC SDK (Rift CV2 and DK2) are supported, the CV1 is recommended. There are currently no plans to support the mobile SDK.
* You can only use one render layer.
* OpenGL is required for rendering, no other graphics API (i.e. Vulkan and DirectX) is available at this time.

## Authors

* **Matthew D. Cutone** - The Centre for Vision Research, York University - (https://github.com/mdcutone)
* **Dr. Laurie M. Wilcox** - The Centre for Vision Research, York University

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## How To Cite PsychXR

If you use PsychXR for your research, please use the following citation:

`Cutone, M. D. & Wilcox, L. M. (2018). PsychXR (Version X.X) [Software]. Available from https://github.com/mdcutone/psychxr.`

