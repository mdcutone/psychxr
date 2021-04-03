NEWS
====
Development updates and news about PsychXR between releases.

April 03, 2021
--------------

It's been a while since the last release of PsychXR, however version 0.2.4 is 
almost and introduces substantial changes such as OpenHMD support and color 
space management to `LibOVR`.

PsychXR has finally gotten a new driver interface, this one using `OpenHMD` 
which provides FOSS drivers for HMDs and VR peripherals. While this isn't the 
OpenXR interface everyone wanted (more on that later), it's something that work 
began on almost two years ago. The `OpenHMD` extension module isn't a simple 
wrapper around the library. Effort is being made to make it function similarly 
to `LibOVR` to make it easier to integrate into existing apps using the `LibOVR`
interface. 

`OpenXR` support is the next major development target. The specification seems 
mature enough now to include and has been seeing considerable adoption in 
industry. `OpenXR` is a big deal since it will provide a standardized interface 
for a wide variety of HMDs. `OpenXR` shares a lot in common with `LibOVR` (which
make sense considering how much Facebook contributed to that standard), so the 
resulting extension may closely resemble the existing one. The inclusion of 
`OpenXR` will mark the end of the `0.2.x` series of PsychXR. No timeline has 
been decided for when `OpenXR` will be added, but it's looking very likely to be 
included near the end of 2021.

May 18, 2020
------------
You may have noticed a bit of a slow down in PsychXR's development in recent 
months. I've been busy with other projects and PsychXR has been "good enough" 
for tasks I've been running in the lab, so there isn't any incentive for me to 
spend time adding features to it which detracts from my other work. Particularly 
since testing new features requires lots of time to ensure things are working as 
advertised.

I don't track usage statistics directly, but I've received e-mails from users
indicating to me that PsychXR has considerable use in multiple labs. Knowing 
this gives me motivation to continue development for quite some time, even after 
my own research ceases.

I plan to get back into development soon. There are a few things planned 
for upcoming releases I'll list here:

* Bump LIBOVR support to the latest version of the Oculus Rift SDK.
* Add the ability to reference LIBOVR timestamps to an external clock
  source. 
* Add a generic VR math library. This library can be built without the 
  Oculus SDK, providing similar features to `LibOVRPose` and 
  `LibOVRPoseState` that can be used with other VR systems.
* Split PsychXR drivers off into separate packages which are loaded as
  plugins. This will make adding additional drivers easier in the future,
  allowing PsychXR to build on platforms where a particular driver is 
  not supported.
  
I've also received numerous messages asking whether other HMDs will 
eventually be supported. I've looked into possible drivers to support 
next (OpenVR, OpenHMD, OpenXR, etc.) but I haven't committed to anything
yet. It's looking likely that OpenXR will be the next driver to be added
which should support a large range of hardware, but there is no timeline
for when work will begin on it. If additional HMD support is something 
you need, I'm open to accepting PRs to expedite the process.

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
with VR hardware. Notably, you will be able to run the same code across
different types of stereoscopic displays and HMDs without significant 
modification to your experiment's code. There are also tools currently 
in PsychoPy that can do VR specific things like lens correction (i.e. 
barrel distortion) which permits labs to create their own HMD-like 
displays. The introduction of the generic VR math library in PsychXR will 
aid in the doing so.
