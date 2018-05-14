# test application for PsychXR's OVR API

import pyglet
from pyglet.window import key
import pyglet.gl as GL
import psychxr.ovr.capi as capi
import math

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

# initialize OVR library
init_pars = capi.ovrInitParams()
init_pars.Flags = capi.ovrInit_Debug
capi.ovr_Initialize(init_pars)

# create OVR session
session = capi.ovrSession()
if capi.ovr_Create(session, capi.ovrGraphicsLuid()) < 0:
    print("Failed to create OVR session.")
    capi.ovr_Destroy(session)
    capi.ovr_Shutdown()
    pyglet.app.exit()

# get HMD descriptor
hmd_settings = capi.ovr_GetHmdDesc(session)

# get required texture size
recommendedTex0Size = capi.ovr_GetFovTextureSize(
    session, capi.ovrEye_Left, hmd_settings.DefaultEyeFov[0], 1.0)
recommendedTex1Size = capi.ovr_GetFovTextureSize(
    session, capi.ovrEye_Left, hmd_settings.DefaultEyeFov[1], 1.0)
bufferSize = capi.ovrSizei()
bufferSize.w = recommendedTex0Size.w + recommendedTex1Size.w
bufferSize.h = max(recommendedTex0Size.h, recommendedTex1Size.h)

# allocate and configure texture swap chain
textureSwapChain = capi.ovrTextureSwapChain()
desc = capi.ovrTextureSwapChainDesc()
desc.Type = capi.ovrTexture_2D
desc.ArraySize = 1
desc.Format = capi.OVR_FORMAT_R8G8B8A8_UNORM_SRGB
desc.Width = bufferSize.w
desc.Height = bufferSize.h
desc.MipLevels = 1
desc.SampleCount = 1
desc.StaticImage = False

# create texture swap chain
texId = 0
if capi.ovr_CreateTextureSwapChainGL(session, desc, textureSwapChain) == 0:
    capi.ovr_GetTextureSwapChainBufferGL(session, textureSwapChain, 0, texId)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texId)

