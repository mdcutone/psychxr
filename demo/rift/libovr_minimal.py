# PsychXR Oculus Rift minimal example.
#
import OpenGL.GL as GL
import ctypes
import glfw
import numpy as np
from psychxr.libovr import *
import sys

HEAD_TRACKING = True

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

    # configure the internal render descriptors for each eye
    for eye, fov in enumerate(hmdInfo.defaultEyeFov):
        setEyeRenderFov(eye, fov)

    # get the optimal buffer dimensions for each eye
    texSizeLeft = calcEyeBufferSize(LIBOVR_EYE_LEFT)
    texSizeRight = calcEyeBufferSize(LIBOVR_EYE_RIGHT)

    # We are using a shared texture, so we need to combine dimensions.
    bufferW = texSizeLeft[0] + texSizeRight[0]
    bufferH = max(texSizeLeft[1], texSizeRight[1])

    # initialize texture swap chain
    createTextureSwapChainGL(LIBOVR_TEXTURE_SWAP_CHAIN0, bufferW, bufferH)

    # set the same swap chain for both eyes since we are using a shared buffer
    for eye in range(LIBOVR_EYE_COUNT):
        setEyeColorTextureSwapChain(eye, LIBOVR_TEXTURE_SWAP_CHAIN0)

    # determine the viewports for each eye's image on the buffer
    eye_w = int(bufferW / 2)
    eye_h = bufferH

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

    # setup a mirror texture
    createMirrorTexture(800, 600)

    # frame index, increment this every frame
    frame_index = 0

    # compute projection matrices
    proj_left = getEyeProjectionMatrix(LIBOVR_EYE_LEFT)
    proj_right = getEyeProjectionMatrix(LIBOVR_EYE_RIGHT)

    # begin application loop
    while not glfw.window_should_close(window):
        # wait for the buffer to be freed by the compositor, this is like
        # waiting for v-sync.
        waitToBeginFrame(frame_index)

        # predicted mid-frame time
        abs_time = getPredictedDisplayTime(frame_index)

        # get the current tracking state
        tracking_state = getTrackingState(abs_time, True)

        # calculate eye poses, this needs to be called every frame
        calcEyePoses(tracking_state.headPose.pose)

        # get the view matrix from the HMD after calculating the pose
        view_left = getEyeViewMatrix(LIBOVR_EYE_LEFT)
        view_right = getEyeViewMatrix(LIBOVR_EYE_RIGHT)

        # start frame rendering
        beginFrame(frame_index)

        # bind the render FBO
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fboId)

        # get the current texture handle for this eye view, these are queued
        # in the swap chain  and released when free. Making draw calls to
        # any other texture in the swap chain not returned here will report
        # and error.
        _, swapIdx = getTextureSwapChainCurrentIndex(LIBOVR_TEXTURE_SWAP_CHAIN0)
        _, tex_id = getTextureSwapChainBufferGL(LIBOVR_TEXTURE_SWAP_CHAIN0, swapIdx)

        # bind the returned texture ID to the frame buffer's texture slot
        GL.glFramebufferTexture2D(
            GL.GL_DRAW_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D, tex_id, 0)

        # for each eye, do some rendering
        for eye in range(LIBOVR_EYE_COUNT):
            # OpenGL draw commands here
            pass

        # commit the texture when were done drawing to it
        commitTextureSwapChain(LIBOVR_TEXTURE_SWAP_CHAIN0)

        # unbind the frame buffer, we're done with it
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)

        # end frame rendering, submitting the eye layer to the compositor
        endFrame(frame_index)

        # increment frame index
        frame_index += 1

        # update session status
        #session_status = capi.getSessionStatus()

        # blit mirror texture
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, mirrorFbo)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)

        _, mirrorId = getMirrorTexture()

        # bind the rift's texture to the framebuffer
        GL.glFramebufferTexture2D(
            GL.GL_READ_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D, mirrorId, 0)

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
        updateInputState(LIBOVR_CONTROLLER_TYPE_TOUCH)
        A = getButton(LIBOVR_CONTROLLER_TYPE_TOUCH, LIBOVR_BUTTON_A, 'falling')
        B = getButton(LIBOVR_CONTROLLER_TYPE_TOUCH, LIBOVR_BUTTON_B, 'falling')

        if A[0]:  # first value is the state, second is the polling time
            recenterTrackingOrigin()
        elif B[0]:
            # exit if button 'B' is pressed
            break

        # flip the GLFW window and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()

    # switch off the performance summary
    perfHudMode("Off")

    # free resources
    destroyMirrorTexture()
    destroyTextureSwapChain(LIBOVR_TEXTURE_SWAP_CHAIN0)
    destroy()

    # end the rift session cleanly
    shutdown()

    # close the GLFW application
    glfw.terminate()

    return 0


if __name__ == "__main__":
    sys.exit(main())
