Frequently Asked Questions
==========================

This page is to address question frequently asked about PsychXR that come up
over email and at conferences. These questions and responses may be revised over
time.

**When can we expect OpenXR/OpenVR support?**

This is very often asked since there are many HMD platforms in use not supported
by `libovr`. The short answer is "soon" for OpenXR (Q4 2021 might see a preview
release). OpenVR is no longer being considered since the industry seems to be
moving towards OpenXR.

**Are there plans to support mobile HMDs?**

No. Only HMDs tethered (by wire or radio) to a PC that can run Python will be
supported for the foreseeable future.

**Can I use PsychXR for purposes other than research?**

Of course! Turns out PsychXR makes a good platform for general prototyping of VR
applications for those familiar with Python and OpenGL. In the future, PsychXR
might turn into a general purpose binding for VR drivers, but for now the focus
is the needs of the vision science community.

**Do I need to download the Oculus Rift PC SDK to build PsychXR from source?**

Due to stipulations in the Oculus Rift PC SDK license agreement, PsychXR cannot
ship with any libraries or source code from the SDK. Therefore, the user is
required to download the SDK if they wish build from source. It would be nice
for Facebook to make an exception for PsychXR someday, but the inclusion of
OpenXR in the future may not make that necessary.

**Where can I report issues with PsychXR?**

If you encounter some undefined behaviour or error when using PsychXR, submit a
report to the
`issue tracker on GitHub <https://github.com/mdcutone/psychxr/issues>`_. Provide
information about what you were doing at the time when the problem occurred, the
hardware in use, and a sample script which replicates it.

If you managed to fix the issue yourself, please consider creating a pull
request so the patch can be incorporated into the main branch.

**How can I get a feature I want added to PsychXR?**

There are several approaches you can take to have a feature added to PsychXR.

If you already have a fork of PsychXR and wish to contribute feature changes
you've made to the main branch, simply submit a pull request to the
`GitHub project page for PsychXR <https://github.com/mdcutone/psychxr>`_. If the
feature is deemed appropriate for the target audience of PsychXR, it may be
added at some point.

If you don't feel comfortable adding a feature yourself (or don't have the
time), you may post a feature request in the project's GitHub issue tracker. At
some point your request will be reviewed and may be considered to be added.

If you **really** need some feature added or would like to generally support
the development of PsychXR, paid commissions are available. Contact the project
lead directly (Matthew Cutone) for more information. Cost and project duration
varies depending on the complexity of the feature you would like added.

**Why is PsychXR written in Cython?**

In the very early days (circa 2017) PsychXR (internally known as PsychHMD) was
written entirely in C/C++. It became clear that this would not be sustainable in
the long term. Since PsychXR is an open-source project for vision scientists,
most are only users of high-level scripting languages (e.g., R, Python, and
Octave/MATLAB), having extension modules written in C/C++ would present a
barrier, making it harder for users examine the code and make contributions.
This would certainly result in the project dying off quickly, so another
solution was needed.

At one point a version of PsychXR was created using nothing but `ctypes`, but it
could not be made to work due to ABI issues. Finally,
`Cython <https://cython.org/>`_ was considered for PsychXR. Cython is superset
of the Python language which compiles to C/C++. A huge advantage of Cython is
the ability to seamlessly interface with C/C++ libraries and produce performant
code, all the while having syntax nearly identical to Python. This addressed
many of the concerns raised by the original C/C++ library. The library was now
easy to fix and extend by people who have only knowledge of Python (the target
audience). Proof of this was when a colleague of mine found and fixed a bug
themselves *in the extension code* without even knowing they were using Cython!
Another advantage of Cython which presented itself was maintainability. The
simple syntax of Cython made it very easy to refactor and keep on top of changes
being introduced to the Oculus Rift runtime (LibOVR). Features added to LibOVR
would get exposed by PsychXR sometimes within hours of a new SDK being released.
At one point, PsychXR supported features before popular game engines had.

As Cython is used to create many big data and scientific software packages used
by countless users, it's safe to bet the future of PsychXR on it.
