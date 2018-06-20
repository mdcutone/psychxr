# PsychXR

PsychXR is a collection of [Python](https://www.python.org/) extension libraries for interacting with eXtended Reality displays (HMDs), intended for neuroscience and psychology research applications.

## Supported Devices

* Oculus Rift DK2 and CV1

## Installing

### Building from Source

If you choose to compile PsychXR from source, you must have the appropriate C++ compiler ([Microsoft Visual C++ Build Tools](https://www.microsoft.com/en-us/download/details.aspx?id=48159)) and SDKs installed on your computer. Since the Oculus Rift on Windows is the only supported HMD at this time, download the [Oculus SDK for Windows v1.25.0](https://developer.oculus.com/downloads/package/oculus-sdk-for-windows/1.25.0/) and extract the zip file somewhere accessable on your PC.

You can download the latest source distribution package for PsychXR from the releases page.

Once downloaded, open the "Visual C++ 2015 Native Build Tools Command Console" and change to the directory the source package is located. Now we need to configure the build using environment variables. The build script needs these values to know which SDK we're building extensions for and where the SDK files are located.

We tell the installer to build extensions for the Oculus SDK (LibOVR) by issuing the following command: 

```
set PSYCHXR_BUILD_LIBOVR=1
```

We need to tell the compiler where to find LibOVR's header and library files. To build LibOVR extensions, the installer needs to know where `..\OculusSDK\LibOVR\Include` and `\OculusSDK\LibOVR\Include\Extras` is located. Furthermore, we also need to provide a path to the location of `LibOVR.lib`. You specify these paths by entering the following commands:

```
set PSYCHXR_LIBOVR_INCLUDE=C:\OculusSDK\LibOVR\Include;C:\OculusSDK\LibOVR\Include\Extras
set PSYCHXR_LIBOVR_PATH=C:\OculusSDK\LibOVR\Lib\Windows\x64\Release\VS2015
```
The settings above depend on where you unpacked the Oculus SDK files, so set them appropriately. If you extracted the SDK package to `C:\`, the installer will use default values allowing you to skip setting the above variables. If you are using a newer version of Visual C++ Build Tools (eg. 2017), you must set the `PSYCHXR_LIBOVR_PATH` to the path where that version of the LibOVR library is found.

Now we can build the source package using the following command (obviously replacing `<version>` with the current version of the package):

```
python -m pip install psychxr-<version>.tar.gz
```

If everything goes well, PsychXR should be installed and ready to use. You can test it by issuing the following command into your Python interpreter:

```
>>> import psychxr.ovr as ovr
>>> ovr.capi.isHmdConnected()
True
```

### Installing Pre-Compiled Binaries

The easiest way to get PsychXR is by downloading the [Wheel package](https://github.com/mdcutone/psychxr/releases) and installing it with pip. The pre-compiled binaries are built on Windows 10 64-bit using the MSVC++ 15.0 against Python 3.6 64-bit. You can install the package with the following command (no environment variables or compilers needed):

```
python -m pip install psychxr-<version>.whl
```

## Limitations

There are several limitations to the current version of PsychXR which may make it unsuitable for certain applications.

* Only Oculus VR HMDs which use the PC SDK (Rift CV2 and DK2) are supported, the CV1 is recommended. There are currently no plans to support the mobile SDK.
* You can only use one render layer.
* OpenGL is required for rendering, no other graphics API (i.e. Vulkan and DirectX) is available at this time. You must use some OpenGL framework such as Pyglet, GLFW ([example](https://github.com/mdcutone/psychxr/blob/master/demo/rift/oculus_glfw.py)) or PyOpenGL with PsychXR to create visual stimuli.

## Support

If you encounter problems with PsychXR, please submit an issue to [PsychXR's issue tracker](https://github.com/mdcutone/psychxr/issues). 

## Authors

* **Matthew D. Cutone** - The Centre for Vision Research, York University - (https://github.com/mdcutone)
* **Dr. Laurie M. Wilcox** - The Centre for Vision Research, York University

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## How To Cite PsychXR

If you use PsychXR for your research, please use the following citation:

```
Cutone, M. D. & Wilcox, L. M. (2018). PsychXR (Version 0.2.1) [Software]. Available from https://github.com/mdcutone/psychxr.
```

