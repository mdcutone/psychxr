Frequently Asked Questions
==========================

This page is to address question frequently asked about PsychXR that come up
over email and at conferences. These questions may be revised over time to
provide better answers if need be. Responses are ordered from the most to least
asked.

**When can we expect OpenXR/OpenVR support?**

This is very often asked since there are many HMD platforms in use not supported
by `libovr`. The short answer is "soon" for OpenXR (Q4 2021 might see a preview
release). OpenVR is no longer being considered since OpenXR seems to be the
direction the VR industry is moving towards.

**Are there plans to support mobile HMDs?**

No. Only HMDs tethered (by wire or radio) to a PC that can run Python will be
supported for the foreseeable future.

**Can I use PsychXR for purposes other than research?**

Of course! Turns out PsychXR makes a good platform for general prototyping of VR
applications for those familiar with Python and OpenGL. In the future PsychXR
might turn into a general purpose binding for VR drivers, but for now the focus
is the vision science community.

**Do I need to download the Oculus Rift PC SDK to build PsychXR from source?**

Due to stipulations in the Oculus Rift PC SDK license agreement, PsychXR cannot
ship with any libraries or source code from the SDK. Therefore, the user is
required to download the SDK if they wish build from source. It would be nice
for Facebook to make an exception for PsychXR someday, but the inclusion of
OpenXR in the future may not make that necessary.

**How can I get a feature I want added to PsychXR?**

There are several approaches you can take to have a feature added to PsychXR.

If you already have a fork of PsychXR and wish to contribute feature changes
you've made to the main branch, simply submit a pull request on the GitHub
project page for PsychXR. If the feature is deemed appropriate for the target
audience of PsychXR, it may be added at some point.

If you don't feel comfortable adding a feature yourself (or don't have the
time), you may post a feature request in the project's GitHub issue tracker. At
some point your request will be reviewed and may be considered to be added.

If you **really** need some feature added or would like to generally support
the development of PsychXR, paid commissions are available. Contact the project
lead directly (Matthew Cutone) for more information. Cost and project duration
varies depending on the complexity of the feature you would like added.
