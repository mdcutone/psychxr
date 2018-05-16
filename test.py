# test application for PsychXR's OVR API

import pyglet
from pyglet.window import key
import pyglet.gl as GL
import psychxr.ovr.capi as capi
import math

capi.debug_mode = True

# initialize OVR library
init_pars = capi.ovrInitParams()
init_pars.Flags = capi.ovrInit_Debug
capi.ovr_Initialize(init_pars)

# create OVR session

if capi.ovr_Create() < 0:
    print("Failed to create OVR session.")
    capi.ovr_Destroy()
    capi.ovr_Shutdown()
    pyglet.app.exit()

# get HMD descriptor
hmd_settings = capi.ovr_GetHmdDesc()

# get required texture size
recommendedTex0Size = capi.ovr_GetFovTextureSize(
    capi.ovrEye_Left, hmd_settings.DefaultEyeFov[0], 1.0)
recommendedTex1Size = capi.ovr_GetFovTextureSize(
    capi.ovrEye_Left, hmd_settings.DefaultEyeFov[1], 1.0)
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

window = pyglet.window.Window(width=640, height=480, caption="Oculus Rift Test")

@window.event
def on_draw():
    result = capi.ovr_WaitToBeginFrame(0)
    absTime = capi.ovr_GetTimeInSeconds(0)
    ts = capi.ovr_GetTrackingState(absTime, 0)
    #if ts.StatusFlags & (capi.ovrStatus_OrientationTracked | capi.ovrStatus_PositionTracked):
    poses = []
    capi.ovr_GetDevicePoses([capi.ovrTrackedDevice_HMD], 1, absTime, poses)
    capi.ovr_BeginFrame(0)
    texture = capi.ovr_GetTextureSwapChainBufferGL(textureSwapChain, 0, 0)
    #e = capi.ovrErrorInfo()
    #capi.ovr_GetLastErrorInfo(e)
    #print(e.ErrorString)
    GL.glClearColor(0.5, 0.5, 0.5, 1.0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
    window.clear()
    capi.ovr_CommitTextureSwapChain(textureSwapChain)
    capi.ovr_EndFrame(0, None, [eyeLayer], 1)


@window.event
def on_key_press(symbol, modifiers):
    if symbol == key.ESCAPE:
        capi.ovr_Destroy()
        capi.ovr_Shutdown()
        pyglet.app.exit()


# create texture swap chain
texId = 0
if capi.ovr_CreateTextureSwapChainGL(desc, textureSwapChain) == 0:
    capi.ovr_GetTextureSwapChainBufferGL(textureSwapChain, 0, texId)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texId)

    eyeRenderDesc = [capi.ovrEyeRenderDesc(), capi.ovrEyeRenderDesc()]
    hmdToEyeViewPose = [capi.ovrPosef(), capi.ovrPosef()]

    eyeRenderDesc[0] = capi.ovr_GetRenderDesc(capi.ovrEye_Left,
                                              hmd_settings.DefaultEyeFov[0])

    eyeRenderDesc[1] = capi.ovr_GetRenderDesc(capi.ovrEye_Right,
                                              hmd_settings.DefaultEyeFov[1])

    hmdToEyeViewPose[0] = eyeRenderDesc[0].HmdToEyePose
    hmdToEyeViewPose[1] = eyeRenderDesc[1].HmdToEyePose

    eyeLayer = capi.ovrLayerEyeFov()
    eyeLayer.Header.Type = capi.ovrLayerType_EyeFov
    eyeLayer.Header.Flags = 0
    eyeLayer.ColorTexture = (textureSwapChain, textureSwapChain)
    eyeLayer.Fov = (eyeRenderDesc[0].Fov, eyeRenderDesc[1].Fov)
    eyeLayer.Viewport = (capi.ovrRecti(0, 0, bufferSize.w / 2, bufferSize.h),
                         capi.ovrRecti(bufferSize.w / 2, 0, bufferSize.w / 2, bufferSize.h))

def update(dt):
    pass
pyglet.clock.schedule_interval(update, 0.1)

pyglet.app.run()





