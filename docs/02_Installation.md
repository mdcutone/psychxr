# Installation

Follow the instructions below to get a copy of PsychXR installed on your
computer.

## Installing from PyPI

The latest source and binary packages are usually made available on
Python Package Index (PyPI). They can be pulled from PyPI  repository
automatically and installed by issuing the following command.

```
python -m pip install psychxr
```

If the binaries are not available for some reason (eg. your version of
Python is too new), `pip` will try to build the source distribution (it
will likely fail). In that case, you must set the environment variables
as shown below in "Building from Source" before running the above
command again.

## Installing Pre-Compiled Binaries

Pre-compiled binaries for PsychXR are available as a [Wheel package](https://github.com/mdcutone/psychxr/releases)
and which can be installed via `pip`. You can install the package with
the following command (no environment variables or compilers needed):

```
python -m pip install psychxr-<version>.whl
```

Note that the pre-compiled binaries are built on Windows 10 64-bit
using the MSVC++ 15.0 against Python 3.6 64-bit. If your configuration
differs, consider building from source.

## Building from Source

If you choose to compile PsychXR from source, you must have the
appropriate C++ compiler ([Microsoft Visual C++ Build Tools](https://www.microsoft.com/en-us/download/details.aspx?id=48159))
and SDKs installed on your computer. Since the Oculus Rift on Windows is
the only supported HMD at this time, this guide will only cover building
LibOVR extensions.

Download the [Oculus SDK for Windows v1.25.0](https://developer.oculus.com/downloads/package/oculus-sdk-for-windows/1.25.0/)
and extract the zip file somewhere accessable on your PC. You can
download the latest source distribution package for PsychXR from the
[releases page](https://github.com/mdcutone/psychxr/releases).

Once downloaded, open the "Visual C++ 2015 Native Build Tools Command
Console" and change to the directory the source package is located. Now
we need to configure the build using environment variables. The build
script needs these values to know which SDK we're building extensions
for and where the SDK files are located.

We tell the installer to build extensions for the Oculus SDK (LibOVR) by
issuing the following command:

```
set PSYCHXR_BUILD_LIBOVR=1
```

We need to tell the compiler where to find LibOVR's header and library
files. To build LibOVR extensions, the installer needs to know where
`..\OculusSDK\LibOVR\Include` and `..\OculusSDK\LibOVR\Include\Extras`
are located. Furthermore, we also need to provide a path to the location
of `LibOVR.lib` within the SDK directories. You specify these paths by
entering the following commands:

```
set PSYCHXR_LIBOVR_INCLUDE=C:\OculusSDK\LibOVR\Include;C:\OculusSDK\LibOVR\Include\Extras
set PSYCHXR_LIBOVR_PATH=C:\OculusSDK\LibOVR\Lib\Windows\x64\Release\VS2015
```
The settings above depend on where you unpacked the Oculus SDK files, so
set them appropriately. If you extracted the SDK package to `C:\`, the
installer will use default values allowing you to skip setting the above
variables. If you are using a newer version of Visual C++ Build Tools
(eg. 2017), you must set the `PSYCHXR_LIBOVR_PATH` to the path where
that version of the LibOVR library is found.

Now we can build the source package using the following command
(obviously replacing `<version>` with the current version of the
package):

```
python -m pip install psychxr-<version>.tar.gz
```

## Testing the Installation

If everything goes well, PsychXR should be installed and ready to use.
You can test it by issuing the following command into your Python
interpreter:

```
>>> import psychxr.ovr as ovr
>>> ovr.capi.isHmdConnected()
True
```