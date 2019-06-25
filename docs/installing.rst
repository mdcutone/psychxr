Getting PsychXR
===============

Installing from PyPI
--------------------

The latest source and binary packages are usually made available on Python
Package Index (PyPI). They can be pulled from PyPI repository automatically and
installed by issuing the following command::

    python -m pip install psychxr

If the binaries are not available for some reason (eg. your version of Python is
too new), pip will try to build the source distribution (it will likely fail).
In that case, you must set the environment variables as shown below in "Building
from Source" before running the above command again.

Installing Pre-Compiled Binaries
--------------------------------

Pre-compiled binaries for PsychXR are available as a Wheel package and which can
be installed via pip. You can install the package with the following command (no
environment variables or compilers needed)::

    python -m pip install psychxr-<version>.whl

Note that the pre-compiled binaries are built on Windows 10 64-bit using the
MSVC++ 15.0 against Python 3.6 64-bit. If your configuration differs, consider
building from source.

Building from Source
--------------------

If you choose to compile PsychXR from source, you must have the appropriate C++
compiler (Microsoft Visual C++ 2019 Build Tools) and SDKs installed on your
computer. Since the Oculus Rift on Windows is the only supported HMD at this
time, this guide will only cover building LibOVR extensions on that platform.

Download the `Oculus SDK for Windows
<https://developer.oculus.com/downloads/package/oculus-sdk-for-windows/>`_
(version v1.38.0 is recommended) and extract the zip file somewhere accessible
on your PC. You can download the latest source distribution package for PsychXR
from the `releases <https://github.com/mdcutone/psychxr/releases>`_ page.

Once downloaded, open the "Visual C++ 2017 Native Build Tools Command Prompt"
and change to the directory the source package is located. Now we need to
configure the build using environment variables. The build script needs these
values to know which SDK we're building extensions for and where the SDK files
are located.

We tell the installer to build extensions for the Oculus SDK (LibOVR) by issuing
the following command::

    set PSYCHXR_BUILD_LIBOVR=1


We need to tell the compiler where to find LibOVR's header and library files. To
build LibOVR extensions, the installer needs to know where the Oculus PC SDK
files are located. You specify the path to the SDK by entering the following
command::

    set PSYCHXR_LIBOVR_SDK_PATH=C:\OculusSDK


The settings above depend on where you unpacked the Oculus SDK files, so set
them appropriately. If you extracted the SDK package to ``C:\``, the installer
will use default values allowing you to skip setting the above variables.

Now we can build the source package using the following command (obviously
replacing <version> with the current version of the package)::

    python -m pip install psychxr-<version>.tar.gz


Testing the Installation
------------------------

If everything goes well, PsychXR should be installed and ready to use. You can
test it by issuing the following command into your Python interpreter::

    >>> import psychxr.libovr as libovr
    >>> libovr.isHmdConnected()
    True


