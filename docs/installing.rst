Getting PsychXR
===============

Preparing for Installation
--------------------------

PsychXR requires `Python 3.6
<https://www.python.org/downloads/release/python-366/>`_ 64-bit to run and was
extensively tested using version 3.6.6. It should be possible to build and run
PsychXR on newer versions of Python, but the 3.6 series is preferred at this
time. Since PsychXR will only build in 64-bit, ensure that the version of Python
you install is also 64-bit.

The PsychXR installer will automatically pull any dependencies, such as
`Cython <https://cython.org/>`_ and `NumPy <https://www.numpy.org/>`_, prior to
installation.

Installing from PyPI
--------------------

The latest source and binary packages are usually made available on Python
Package Index (PyPI). They can be pulled from PyPI repository automatically and
installed by issuing the following command::

    python -m pip install psychxr

If the binaries are not available for some reason (eg. your version of Python is
too new), pip will try to build the source distribution (it will likely fail).
In that case, see the "Building from Source" section for more information.

Installing Pre-Compiled Binaries
--------------------------------

Pre-compiled binaries for PsychXR are available as a Wheel package and which can
be installed via pip. You can install the package with the following command (no
environment variables or compilers needed)::

    python -m pip install psychxr-<version>.whl

Note that the pre-compiled binaries are built on Windows 10 64-bit using the
MSVC++ 19.0 against Python 3.6 64-bit. If your configuration differs, consider
building from source.

Building from Source
--------------------

If you choose to compile PsychXR from source, you must have the appropriate C++
compiler (`Microsoft Visual C++ 2019 Build Tools
<https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019>`_)
and SDKs installed on your computer. Since the Oculus Rift on Windows is the
only supported HMD at this time, this guide will only cover building LibOVR
extensions on that platform.

Download the `Oculus SDK for Windows
<https://developer.oculus.com/downloads/package/oculus-sdk-for-windows/>`_
(version v23.0 is required) and extract the zip file somewhere accessible on
your PC. You can download the latest source distribution package for PsychXR
from the `releases <https://github.com/mdcutone/psychxr/releases>`_ page.

.. note:: If you build PsychXR using a version of the Oculus PC SDK other than
          1.40, a warning will be raised upon importing the module. The software
          should work fine if the SDK version is newer and has no breaking
          changes. However, this may result in unexpected behaviour!

Once downloaded, open the "x64 Native Tools Command Prompt for VS 2019" and
change to the directory the source package is located. Now we need to configure
the build using environment variables. The build script needs these values to
know which SDK we're building extensions for and where the SDK files are
located.

We tell the installer to build extensions for the Oculus SDK (LibOVR) by issuing
the following command::

    set PSYCHXR_BUILD_LIBOVR=1

Issuing the above command is redundant at this time. Since LibOVR is the only
supported interface at this time, ``PSYCHXR_BUILD_LIBOVR`` defaults to ``1``
even without specifying the above command. In the future, other interfaces may
be installed selectively this way.

To build LibOVR extensions, the installer needs to know where the Oculus PC SDK
files are located. You specify the path to the SDK by entering the following
command::

    set PSYCHXR_LIBOVR_SDK_PATH=C:\path\to\OculusSDK


The settings above depend on where you unpacked the Oculus SDK files, so set
them appropriately. By default, the compiler will assume the SDK is located at
``C:\OculusSDK``, so you don't need to set the above environment variables if
you extracted it there.

Now we can build the source package using the following command (replacing
<version> with the current version of the package, which is **0.2.4**)::

    python -m pip install psychxr-<version>.tar.gz


Testing the Installation
------------------------

If everything goes well, PsychXR should be installed and ready to use. You can
test it by issuing the following command into your Python interpreter::

    >>> import psychxr.libovr as libovr
    >>> libovr.isHmdConnected()
    True


