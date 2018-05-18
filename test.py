# test application for PsychXR's OVR API
import glfw
from OpenGL.GL import *

import psychxr.ovr.capi as capi
import ctypes

capi.debug_mode = True

def main():
    if not glfw.init():
        return -1
    capi.ovr_Shutdown()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    window = glfw.create_window(800, 600, "Oculus Test", None, None)

    if not window:
        glfw.terminate()
        return -1

    # initialize OVR library
    init_pars = capi.ovrInitParams()
    init_pars.Flags = capi.ovrInit_RequestVersion
    init_pars.RequestedMinorVersion = 24
    capi.ovr_Initialize(init_pars)

    if capi.ovr_Create() < 0:
        capi.ovr_Destroy()
        capi.ovr_Shutdown()
        glfw.terminate()
        return -1

    # get HMD descriptor
    hmd_desc = capi.ovr_GetHmdDesc()

    # get required texture size
    recommendedTex0Size = capi.ovr_GetFovTextureSize(
        capi.ovrEye_Left, hmd_desc.DefaultEyeFov[0], 1.0)
    recommendedTex1Size = capi.ovr_GetFovTextureSize(
        capi.ovrEye_Left, hmd_desc.DefaultEyeFov[1], 1.0)
    bufferSize = capi.ovrSizei()
    bufferSize.w = recommendedTex0Size.w + recommendedTex1Size.w
    bufferSize.h = max(recommendedTex0Size.h, recommendedTex1Size.h)

    # create a texture descriptor
    desc = capi.ovrTextureSwapChainDesc()
    desc.Type = capi.ovrTexture_2D
    desc.ArraySize = 1
    desc.Format = capi.OVR_FORMAT_R8G8B8A8_UNORM_SRGB
    desc.Width = bufferSize.w
    desc.Height = bufferSize.h
    desc.MipLevels = 1
    desc.SampleCount = 1
    desc.StaticImage = False
    desc.MiscFlags = capi.ovrTextureMisc_None
    desc.BindFlags = capi.ovrTextureBind_None

    glfw.make_context_current(window)

    # create FBO texture
    colorTex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, colorTex_id)
    glTexParameteri(
        GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(
        GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(
        GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(
        GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA, bufferSize.w,
        bufferSize.h, 0, GL_RGBA, GL_FLOAT, None)
    glBindTexture(GL_TEXTURE_2D, 0)

    # create framebuffers
    fbo_id = GLuint()
    glGenFramebuffers(1, ctypes.byref(fbo_id))
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_id)

    colorRb_id = GLuint()
    glGenRenderbuffers(1, ctypes.byref(colorRb_id))
    glBindRenderbuffer(GL_RENDERBUFFER, colorRb_id)
    glRenderbufferStorage(
        GL_RENDERBUFFER, GL_RGBA8, int(bufferSize.w),
        int(bufferSize.h))
    glFramebufferRenderbuffer(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER,
        colorRb_id)

    glBindRenderbuffer(GL_RENDERBUFFER, 0)

    depthRb_id = GLuint()
    glGenRenderbuffers(1, ctypes.byref(depthRb_id))
    glBindRenderbuffer(GL_RENDERBUFFER, depthRb_id)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
                          int(bufferSize.w), int(bufferSize.h))

    glFramebufferRenderbuffer(
        GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER,
        depthRb_id)

    glBindRenderbuffer(GL_RENDERBUFFER, 0)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    capi.ovr_CreateTextureSwapChainGL(desc, 0)
    # create VR structures
    eyeRenderDesc = [capi.ovrEyeRenderDesc(), capi.ovrEyeRenderDesc()]
    hmdToEyeViewPose = [capi.ovrPosef(), capi.ovrPosef()]

    eyeRenderDesc[0] = capi.ovr_GetRenderDesc(capi.ovrEye_Left,
                                              hmd_desc.DefaultEyeFov[0])

    eyeRenderDesc[1] = capi.ovr_GetRenderDesc(capi.ovrEye_Right,
                                              hmd_desc.DefaultEyeFov[1])

    print(eyeRenderDesc[0].Fov)

    hmdToEyeViewPose[0] = eyeRenderDesc[0].HmdToEyePose
    hmdToEyeViewPose[1] = eyeRenderDesc[1].HmdToEyePose

    eyeLayer = capi.ovrLayerEyeFov()
    eyeLayer.Header.Type = capi.ovrLayerType_EyeFov
    eyeLayer.Header.Flags = capi.ovrLayerFlag_TextureOriginAtBottomLeft
    # eyeLayer.ColorTexture = (swap_chain, swap_chain)
    eyeLayer.Fov = (eyeRenderDesc[0].Fov, eyeRenderDesc[1].Fov)
    eyeLayer.Viewport = (capi.ovrRecti(0, 0, bufferSize.w / 2, bufferSize.h),
                         capi.ovrRecti(bufferSize.w / 2, 0, bufferSize.w / 2,
                                       bufferSize.h))

    while not glfw.window_should_close(window):
        frame = 0
        absTime = capi.ovr_GetPredictedDisplayTime(0)
        hmdState = capi.ovr_GetTrackingState(absTime, 0)
        poses = [capi.ovrPoseStatef()]
        capi.ovr_CalcEyePoses(hmdState.HeadPose.ThePose, hmdToEyeViewPose, eyeLayer.RenderPose)
        capi.ovr_SetInt("PerfHudMode", capi.ovrPerfHud_AppRenderTiming)
        this_idx = capi.ovr_GetTextureSwapChainCurrentIndex(0, 0)
        texture_id = capi.ovr_GetTextureSwapChainBufferGL(0, frame, 0)
        #print(texture_id)
        capi.ovr_WaitToBeginFrame(frame)
        capi.ovr_BeginFrame(frame)

        glBindFramebuffer(GL_FRAMEBUFFER, fbo_id)
        #glBindTexture(GL_TEXTURE_2D, texture_id)
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
            texture_id, 0)

        #glBindTexture(GL_TEXTURE_2D, 0)

        glViewport(
            0,
            0,
            1024,
            1024)
        glScissor(
            0,
            0,
            1024,
            1024)

        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        #glBindFramebuffer(
        #    GL_READ_FRAMEBUFFER, fbo_id)
        #glBindFramebuffer(
        #    GL_DRAW_FRAMEBUFFER, 0)

        #glBlitFramebuffer(
        #    0, 0,
        #    bufferSize.w, bufferSize.h,
        #    0, 0,
        #    800, 600,
        #    GL_COLOR_BUFFER_BIT, GL_NEAREST)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        capi.ovr_CommitTextureSwapChain(0)

        #glBindTexture(GL_TEXTURE_2D, 0)

        capi.ovr_EndFrame(frame, None, eyeLayer)

        glfw.swap_buffers(window)
        glfw.poll_events()

        #frame += 1


    capi.ovr_Destroy()
    capi.ovr_Shutdown()
    glfw.terminate()
    return 0

if __name__ == "__main__":
    main()