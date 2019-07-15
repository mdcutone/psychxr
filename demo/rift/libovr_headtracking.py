# PsychXR Oculus Rift head tracking example. This file is public domain. See
# See http://psychxr.org/examples/libovr_rendering.html for more information.
#
import OpenGL.GL as GL
import ctypes
import glfw
from psychxr.libovr import *
import sys


def main():
    if not glfw.init():
        return -1

    if not glfw.init():
        return -1

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

    window = glfw.create_window(800, 600, "Oculus Test", None, None)

    if not window:
        glfw.terminate()

    glfw.make_context_current(window)

    glfw.swap_interval(0)

    if failure(initialize()):
        return -1

    if failure(create()):
        shutdown()
        return -1

    hmdInfo = getHmdInfo()

    for eye, fov in enumerate(hmdInfo.defaultEyeFov):
        setEyeRenderFov(eye, fov)

    texSizeLeft = calcEyeBufferSize(EYE_LEFT)
    texSizeRight = calcEyeBufferSize(EYE_RIGHT)

    bufferW = texSizeLeft[0] + texSizeRight[0]
    bufferH = max(texSizeLeft[1], texSizeRight[1])

    createTextureSwapChainGL(TEXTURE_SWAP_CHAIN0, bufferW, bufferH)

    for eye in range(EYE_COUNT):
        setEyeColorTextureSwapChain(eye, TEXTURE_SWAP_CHAIN0)

    eye_w = int(bufferW / 2)
    eye_h = bufferH
    viewports = ((0, 0, eye_w, eye_h), (eye_w, 0, eye_w, eye_h))
    for eye, vp in enumerate(viewports):
        setEyeRenderViewport(eye, vp)

    setHighQuality(True)

    fboId = GL.GLuint()
    GL.glGenFramebuffers(1, ctypes.byref(fboId))
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fboId)
    depthRb_id = GL.GLuint()
    GL.glGenRenderbuffers(1, ctypes.byref(depthRb_id))
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depthRb_id)
    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH24_STENCIL8,
        int(bufferW), int(bufferH))  # buffer size used here!
    GL.glFramebufferRenderbuffer(
        GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER,
        depthRb_id)
    GL.glFramebufferRenderbuffer(
        GL.GL_FRAMEBUFFER, GL.GL_STENCIL_ATTACHMENT, GL.GL_RENDERBUFFER,
        depthRb_id)
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, 0)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    mirrorFbo = GL.GLuint()
    GL.glGenFramebuffers(1, ctypes.byref(mirrorFbo))
    createMirrorTexture(800, 600, mirrorOptions=MIRROR_OPTION_DEFAULT)

    frame_index = 0

    projectionMatrix = []
    for eye in range(EYE_COUNT):
         projectionMatrix.append(getEyeProjectionMatrix(eye))

    planeMatrix = LibOVRPose((0., 0., -2.)).asMatrix()

    # begin application loop
    while not glfw.window_should_close(window):
        waitToBeginFrame(frame_index)

        abs_time = getPredictedDisplayTime(frame_index)

        tracking_state, calibrated_origin = getTrackingState(abs_time, True)

        headPose, state = tracking_state[TRACKED_DEVICE_TYPE_HMD]
        calcEyePoses(headPose.pose)

        beginFrame(frame_index)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fboId)

        _, swapIdx = getTextureSwapChainCurrentIndex(TEXTURE_SWAP_CHAIN0)
        _, tex_id = getTextureSwapChainBufferGL(TEXTURE_SWAP_CHAIN0, swapIdx)

        GL.glFramebufferTexture2D(
            GL.GL_DRAW_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D, tex_id, 0)

        for eye in range(EYE_COUNT):
            vp = getEyeRenderViewport(eye)
            GL.glViewport(*vp)
            GL.glScissor(*vp)

            P = projectionMatrix[eye]
            MV = getEyeViewMatrix(eye)

            GL.glEnable(GL.GL_SCISSOR_TEST)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadTransposeMatrixf(P)

            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadTransposeMatrixf(MV)

            GL.glClearColor(0.0, 0.0, 0.0, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            GL.glPushMatrix()
            GL.glMultTransposeMatrixf(planeMatrix)
            GL.glBegin(GL.GL_QUADS)
            GL.glColor3f(1.0, 0.0, 0.0)
            GL.glVertex3f(-1.0, -1.0, 0.0)
            GL.glColor3f(0.0, 1.0, 0.0)
            GL.glVertex3f(-1.0, 1.0, 0.0)
            GL.glColor3f(0.0, 0.0, 1.0)
            GL.glVertex3f(1.0, 1.0, 0.0)
            GL.glColor3f(1.0, 1.0, 1.0)
            GL.glVertex3f(1.0, -1.0, 0.0)
            GL.glEnd()
            GL.glPopMatrix()

        GL.glDisable(GL.GL_DEPTH_TEST)

        commitTextureSwapChain(TEXTURE_SWAP_CHAIN0)

        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)

        endFrame(frame_index)

        frame_index += 1

        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, mirrorFbo)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)

        _, mirrorId = getMirrorTexture()

        GL.glFramebufferTexture2D(
            GL.GL_READ_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D, mirrorId, 0)

        GL.glViewport(0, 0, 800, 600)
        GL.glScissor(0, 0, 800, 600)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glBlitFramebuffer(0, 0, 800, 600,
                             0, 600, 800, 0,
                             GL.GL_COLOR_BUFFER_BIT,
                             GL.GL_NEAREST)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        glfw.swap_buffers(window)

        updateInputState(CONTROLLER_TYPE_TOUCH)
        A = getButton(CONTROLLER_TYPE_TOUCH, BUTTON_A, 'falling')
        B = getButton(CONTROLLER_TYPE_TOUCH, BUTTON_B, 'falling')

        if A[0]:
            recenterTrackingOrigin()
        elif B[0]:
            break

        glfw.poll_events()

        _, sessionStatus = getSessionStatus()
        if sessionStatus.shouldQuit:
            break

    destroyMirrorTexture()
    destroyTextureSwapChain(TEXTURE_SWAP_CHAIN0)

    glfw.terminate()

    destroy()
    shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())

