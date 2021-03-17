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

Usually, you can simply download and install pre-built binaries which require no
additional steps like regular Python packages. However, if you plan on
developing *PsychXR*, or wish to run *PsychXR* on Python versions which do not
have official packages up on PyPI or GitHub yet, you will need to build
*PsychXR* from source. Doing so requires some preparation to setup a suitable
build environment on your computer. However, once the environment is setup the
process to actually build *PsychXR* is fairly straightforward.

*PsychXR* is (mostly) written in `Cython <https://cython.org/>`_, a superset of
the Python programming language, which is used to write the interface between
Python and HMD drivers. Unlike regular Python code, Cython libraries must be
'built' using a C++ compiler. Afterwards, they can be installed and used like
any other Python library.

.. note:: Since the Oculus Rift on Microsoft Windows is the only supported HMD
          at this time, this guide will only cover building LibOVR extension
          library on that platform.

Setting up the build environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you choose to build *PsychXR* from source, you must have the appropriate
environment setup on your computer. The software requires a C++ compiler to
build. Microsoft Windows does not usually ship with the tools to build C++
applications, therefore you must use the `Microsoft Visual Studio
<https://visualstudio.microsoft.com/downloads/>`_ installer to get those
features (the free "Community" edition will suffice). Upon downloading and
running the "Visual Studio Installer" application, you will be presented with a
pallet of optional features to install. Select "Desktop development with C++"
feature and start the installation (see below).

.. image:: ./_static/build_on_windows1.png
    :alt: Feature "Desktop development with C++" selected in the Visual Studio
          Installer.
    :align: center

Now create a folder somewhere accessible on your PC. For this example, we'll
create a folder on our desktop called `PsychXRBuild` (the full path being
``C:\Users\matth\Desktop\PsychXRBuild`` for this example). Download the `Oculus
SDK for Windows
<https://developer.oculus.com/downloads/package/oculus-sdk-for-windows/>`_
(version v23.0 is required) and extract the contents of the zip file to a folder
inside the `PsychXRBuild` called `OculusSDK`.

.. note:: Due to licensing restrictions the Oculus Rift PC SDK cannot be shipped
          with *PsychXR*.

Finally get a copy of the latest source distribution package for *PsychXR* which
can be downloaded from the `releases
<https://github.com/mdcutone/psychxr/releases>`_ page. Place that too into the
`PsychXRBuild` folder. When you open the `PsychXRBuild` folder in Explorer it
should look something like what's shown below.

.. image:: ./_static/psychxr_build_windows4.PNG
    :alt: Feature "Desktop development with C++" selected in the Visual Studio
          Installer.
    :align: center

.. note:: If you build PsychXR using a version of the Oculus PC SDK other than
          23.0 (1.55), a warning will be raised upon importing the module. The
          software should work fine if the SDK version is newer and has no
          breaking changes. However, this may result in unexpected behaviour!

Building from a source distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have our build environment setup, we can now build *PsychXR* from
its source distribution. To do so, open the "x64 Native Tools Command Prompt for
VS 2019" (to find it, check your Start menu under "Visual Studio 2019" as shown
below).

.. image:: ./_static/build_on_windows2.png
    :alt: Start menu item showing the "x64 Native Tools Command Prompt for VS
          2019" icon.
    :align: center

You should see the following a command prompt appear on-screen.

.. image:: ./_static/build_on_windows3.PNG
    :align: center

Within the prompt, change the directory to the `PsychXRBuild` folder created
earlier using the `cd` command, making it the present working directory::

    cd C:\Users\matth\Desktop\PsychXRBuild

Now we need to configure the build script by setting environment variables. The
build script needs these values to know which SDK we're building extensions for
and where any required files are located. We tell the installer to build
extensions for the Oculus SDK (LibOVR) by issuing the following command::

    set PSYCHXR_BUILD_LIBOVR=1

The above command is redundant at this time. Since LibOVR is the only supported
interface and *PsychXR* would be pretty useless without it,
``PSYCHXR_BUILD_LIBOVR`` defaults to ``1`` even without specifying the above
command. In the future, other interfaces may be installed selectively this way.

The installer needs to know where the Oculus PC SDK files are located to build
the LibOVR extension. The SDK contains files needed to compile the interface
which are provided by the manufacturer. It does this by looking at the value of
the environment variable ``PSYCHXR_LIBOVR_SDK_PATH``. Since we placed the SDK
files in a folder called `OculusSDK` located inside `PsychXRBuild`, we simply
need to enter the following command::

    set PSYCHXR_LIBOVR_SDK_PATH=C:\Users\matth\Desktop\PsychXRBuild\OculusSDK

Now we can build the source package by entering the following command (replacing
<version> with the current version of the package, which is **0.2.4**)::

    python -m pip install psychxr-<version>.tar.gz

Testing the Installation
------------------------

If everything goes well, PsychXR should be installed and ready to use. You can
test it by plugging in your HMD issuing the following command into your Python
interpreter::

    >>> import psychxr.libovr as libovr
    >>> libovr.isHmdConnected()
    True

