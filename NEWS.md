NEWS
----
Development updates and news about PsychXR.

May, 18 2020
============
You may have notcied a bit of a slow down in PsychXR's development in 
recent months. I've been busy with other projects and PsychXR has been 
"good enough" for tasks I've been running in the lab, so there isn't 
any incentive for me to spend time adding features to it which detracts 
from my other work. Particularly since testing new features requires 
lots of time to ensure things are working as advertised.

I don't track usage statistics directly, but I've recieved e-mails from 
users indicating to me that PsychXR has considerable use in multiple 
labs. Knowing this gives me motivation to continue developement for
quite some time, even after my own research ceases.

I plan to get back into development soon. There are a few things planned 
for upcoming releases I'll list here:

* Bump LIBOVR support to the latest version of the Oculus Rift SDK.
* Add the ability to reference LIBOVR timestamps to an external clock
  source. 
* Add a generic VR math library. This library can be built without the 
  Oculus SDK, providing similar features to `LibOVRPose` and 
  `LibOVRPoseState` that can be used with other VR systems.
* Split PsychXR drivers off into seperate packages which are loaded as
  plugins. This will make adding additonal drivers easier in the future,
  allowing PsychXR to build on platforms where a particular driver is 
  not supported.
  
I've also recieved numerous messages asking wheather other HMDs will 
eventually be supported. I've looked into possible drivers to support 
next (OpenVR, OpenHMD, OpenXR, etc.) but I haven't committed to anything
yet. It's looking likely that OpenXR will be the next driver to be added
which should support a large range of hardware, but there is no timeline
for when work will begin on it. If additional HMD support is something 
you need, I'm open to accepting PRs to expedaite the process.

I'm also looking into integrating OptiTrack support into PsychXR, 
possibly as a plugin. I've already written the library to do so that 
integrates well with PsychXR classes and have used it in experiments to
date. However, I haven't packaged it up nicely to be included yet. I 
feel the math library needs to be added beforehand since data coming off
NatNet needs to be formatted in a way that can be passed around between
drivers more easily.

Lastly, PsychoPy seems to be the most common route for using PsychXR for
many. One thing planned for PsychoPy is the introduction of extended 
off-screen rendering support. This will allow PsychoPy to better support 
stereoscopic rendering in general, but will greatly improve integration 
with VR hardware. Notibly, you will be able to run the same code across
different types of stereoscopic displays and HMDs without significant 
modification to your experiment's code. 3D stimulus classes are also
going to be added for GLTF2 models which will support PBR materials and 
HDR lighting. There are also tools currently in PsychoPy that can do 
VR specific things like lens correction (i.e. barrel distortion) which
permits labs to create their own HMD-like displays. The introduction of 
the generic VR math libary in PsychXR will aid in the doing so.
