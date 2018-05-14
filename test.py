# test application for PsychXR's OVR API

import pyglet
from pyglet.window import key
import psychxr.ovr.capi as capi

window = pyglet.window.Window(width=640, height=480, caption="Oculus Rift Test")
pyglet.app.run()

@window.event
def on_draw():
    window.clear()

@window.event
def on_key_press(symbol, modifiers):
    if symbol == key.ESCAPE:
        capi.ovr_Destroy(session)
        capi.ovr_Shutdown()
        pyglet.app.exit()

# initialize
init_pars = capi.ovrInitParams()
capi.ovr_Initialize(init_pars)

# create session
session = capi.ovrSession()
if capi.ovr_Create(session, capi.ovrGraphicsLuid()) < 0:
    print("Failed to create OVR session.")
    capi.ovr_Destroy(session)
    capi.ovr_Shutdown()
    pyglet.app.exit()

# get HMD descriptor
hmd_settings = capi.ovr_GetHmdDesc(session)

tm = capi.ovr_GetPredictedDisplayTime(session, 0)
tracker_state = capi.ovr_GetTrackingState(session, tm, False)
out_poses = []
pose = capi.ovr_GetDevicePoses(session,
                               [capi.ovrTrackedDevice_HMD],
                               1,
                               tm,
                               out_poses)

print(out_poses)

print(out_poses[0].ThePose.Position.x)


print(hmd_settings.Manufacturer)
print(hmd_settings.Resolution.w, hmd_settings.Resolution.h)
