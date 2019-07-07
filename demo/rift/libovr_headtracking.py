# PsychXR Oculus Rift head tracking example. This file is public domain.
#
import OpenGL.GL as GL
import ctypes
import glfw
import numpy as np
from psychxr.libovr import *
import sys


def main():
    # start GLFW
    if not glfw.init():
        return -1

    # setup GLFW window options
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

    # open the window
    window = glfw.create_window(800, 600, "Oculus Test", None, None)

    if not window:
        glfw.terminate()

    # always call this before setting up render layers
    glfw.make_context_current(window)

    # disable v-sync, we are syncing to the v-trace of head-set, leaving this on
    # will cause the HMD to lock to the frequency/phase of the display.
    glfw.swap_interval(0)

    # --------------------------------------------------------------------------
    # Configure Rendering

    # initialize the runtime
    if failure(initialize()):
        _, msg = getLastErrorInfo()
        raise RuntimeError(msg)

    # create a new session
    if failure(create()):
        _, msg = getLastErrorInfo()
        raise RuntimeError(msg)

    # get general information about the HMD
    hmdInfo = getHmdInfo()

    # specify the eye render FOV for each render layer
    for eye, fov in enumerate(hmdInfo.defaultEyeFov):
        setEyeRenderFov(eye, fov)

    # get the optimal buffer dimensions for each eye
    texSizeLeft = calcEyeBufferSize(EYE_LEFT)
    texSizeRight = calcEyeBufferSize(EYE_RIGHT)

    # We are using a shared texture, so we need to combine dimensions.
    bufferW = texSizeLeft[0] + texSizeRight[0]
    bufferH = max(texSizeLeft[1], texSizeRight[1])

    # initialize texture swap chain
    createTextureSwapChainGL(TEXTURE_SWAP_CHAIN0, bufferW, bufferH)

    # set the same swap chain for both eyes since we are using a shared buffer
    for eye in range(EYE_COUNT):
        setEyeColorTextureSwapChain(eye, TEXTURE_SWAP_CHAIN0)

    # determine the viewports for each eye's image on the buffer
    eye_w = int(bufferW / 2)
    eye_h = bufferH
    setInt(PERF_HUD_MODE, PERF_HUD_OFF)
    # set the viewports
    viewports = ((0, 0, eye_w, eye_h), (eye_w, 0, eye_w, eye_h))
    for eye, vp in enumerate(viewports):
        setEyeRenderViewport(eye, vp)

    # enable high quality mode
    setHighQuality(True)

    # create a frame buffer object as a render target for the HMD textures
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

    # mirror texture FBO
    mirrorFbo = GL.GLuint()
    GL.glGenFramebuffers(1, ctypes.byref(mirrorFbo))

    # setup a mirror texture, same size as the window
    createMirrorTexture(800, 600)

    # frame index, increment this every frame
    frame_index = 0

    # rigid body pose of the rectangle
    rectMatrix = LibOVRPose((0., 0., -2.)).asMatrix()

    # begin application loop
    while not glfw.window_should_close(window):
        # wait for the buffer to be freed by the compositor, this is like
        # waiting for v-sync.
        waitToBeginFrame(frame_index)

        # predicted mid-frame time
        abs_time = getPredictedDisplayTime(frame_index)

        # get the current tracking state
        tracking_state, calibrated_origin = getTrackingState(abs_time, True)

        # calculate eye poses, this needs to be called every frame
        headPose, state = tracking_state[TRACKED_DEVICE_TYPE_HMD]
        calcEyePoses(headPose.pose)

        # start frame rendering
        beginFrame(frame_index)

        # bind the render FBO
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fboId)

        # get the current texture handle for this eye view, these are queued
        # in the swap chain  and released when free. Making draw calls to
        # any other texture in the swap chain not returned here will report
        # and error.
        _, swapIdx = getTextureSwapChainCurrentIndex(TEXTURE_SWAP_CHAIN0)
        _, tex_id = getTextureSwapChainBufferGL(TEXTURE_SWAP_CHAIN0, swapIdx)

        # bind the returned texture ID to the frame buffer's texture slot
        GL.glFramebufferTexture2D(
            GL.GL_DRAW_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D, tex_id, 0)

        # for each eye, do some rendering
        for eye in range(EYE_COUNT):

            # Set the viewport as what was configured for the render layer. We
            # also need to enable scissor testings with the same rect as the
            # viewport. This constrains rendering operations to one partition of
            # of the buffer since we are using a 'packed' layout.
            vp = getEyeRenderViewport(eye)
            GL.glViewport(*vp)
            GL.glScissor(*vp)

            # Get view and projection matrices
            P = getEyeProjectionMatrix(eye)
            MV = getEyeViewMatrix(eye)
            # Note - you don't need to get eye projection matrices each frame,
            # they are computed only when the eye FOVs are updated. You can
            # compute the eye projection matrices once before entering your
            # render loop if you don't plan on changing them during a session.
            #
            # However, the view matrices should be computed every frame!

            GL.glEnable(GL.GL_SCISSOR_TEST)  # enable scissor test
            GL.glEnable(GL.GL_DEPTH_TEST)

            # Set the projection matrix.
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadTransposeMatrixf(P)

            # Set the view matrix. This contains the translation for the head in
            # the virtual space computed by the API.
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadTransposeMatrixf(MV)
            # Note - We are not using shaders here to keep things simple.
            # However, you can pass computed transforms to a shader program
            # if you like.

            # Okay, let's begin drawing stuff. Clear the background first.
            GL.glClearColor(0.0, 0.0, 0.0, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # Draw a white 2x2 meter square positioned 5 meters in front of the
            # virtual space's origin.
            GL.glPushMatrix()
            GL.glMultTransposeMatrixf(rectMatrix)  # set the position of rect
            GL.glBegin(GL.GL_QUADS)  # start drawing it
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

        # commit the texture when were done drawing to it
        commitTextureSwapChain(TEXTURE_SWAP_CHAIN0)

        # unbind the frame buffer, we're done with it
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)

        # end frame rendering, submitting the eye layer to the compositor
        endFrame(frame_index)

        # increment frame index
        frame_index += 1

        # blit mirror texture
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, mirrorFbo)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)

        _, mirrorId = getMirrorTexture()

        # bind the rift's mirror texture to the framebuffer
        GL.glFramebufferTexture2D(
            GL.GL_READ_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D, mirrorId, 0)

        # render the mirror texture to the on-screen window's back buffer
        GL.glViewport(0, 0, 800, 600)
        GL.glScissor(0, 0, 800, 600)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glBlitFramebuffer(0, 0, 800, 600,
                             0, 600, 800, 0,  # this flips the texture
                             GL.GL_COLOR_BUFFER_BIT,
                             GL.GL_NEAREST)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        # if button 'A' is released on the touch controller, recenter the
        # viewer in the scene. If 'B' was pressed, exit the loop.
        updateInputState(CONTROLLER_TYPE_TOUCH)
        A = getButton(CONTROLLER_TYPE_TOUCH, BUTTON_A, 'falling')
        B = getButton(CONTROLLER_TYPE_TOUCH, BUTTON_B, 'falling')

        if A[0]:  # first value is the state, second is the polling time
            recenterTrackingOrigin()
        elif B[0]:
            # exit if button 'B' is pressed
            break

        # flip the GLFW window and poll events
        glfw.poll_events()

        # check session status
        _, sessionStatus = getSessionStatus()
        if sessionStatus.shouldQuit:
            break

    # free resources
    destroyMirrorTexture()
    destroyTextureSwapChain(TEXTURE_SWAP_CHAIN0)

    # close the GLFW application
    glfw.terminate()

    # end the rift session cleanly
    destroy()
    shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())

