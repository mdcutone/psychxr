# PsychXR Oculus Rift minimal example.
#
import OpenGL.GL as GL
import ctypes
import glfw

import psychxr.ovr.capi as capi
import psychxr.ovr.math as vrmath
capi.debug_mode = True
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

    # start an Oculus session
    capi.startSession()

    # get general information about the HMD
    hmd_desc = capi.getHmdDesc()

    # set the perf hud on
    capi.perfHudMode("PerfSummary")

    # configure the internal render descriptors for each eye
    for eye in range(capi.ovrEye_Count):
        capi.configEyeRenderDesc(eye, hmd_desc.DefaultEyeFov[eye])

    # Get the buffer dimensions specified by the Rift SDK, we need them to
    # setup OpenGL frame buffers.
    tex_size_left = capi.getFovTextureSize(
        capi.ovrEye_Left, hmd_desc.DefaultEyeFov[0])
    tex_size_right = capi.getFovTextureSize(
        capi.ovrEye_Right, hmd_desc.DefaultEyeFov[1])

    # We are using a shared texture, so we need to combine dimensions.
    buffer_w = tex_size_left.w + tex_size_right.w
    buffer_h = max(tex_size_left.h, tex_size_right.h)

    # Allocate a swap chain for render buffer textures, the handle used is an
    # integer. You can allocated up to 32 swap chains, however you will likely
    # run out of video memory by then.

    # configure the swap chain
    swap_config = capi.ovrTextureSwapChainDesc()
    swap_config.Format = capi.OVR_FORMAT_R8G8B8A8_UNORM_SRGB
    swap_config.Type = capi.ovrTexture_2D
    swap_config.Width = buffer_w
    swap_config.Height = buffer_h

    # Initialize texture swap chain
    swap_chain = capi.createTextureSwapChainGL(swap_config)

    # Since we are using a shared texture, each eye's viewport is half the width
    # of the allocated buffer texture.
    eye_w = int(buffer_w / 2)
    eye_h = buffer_h

    # setup the render layer
    viewports = (vrmath.ovrRecti(0, 0, eye_w, eye_h),
                 vrmath.ovrRecti(eye_w, 0, eye_w, eye_h))

    for eye in range(capi.ovrEye_Count):
        capi.setRenderViewport(eye, viewports[eye])

    capi.setRenderSwapChain(0, swap_chain)  # set the swap chain
    #rift.setRenderSwapChain(1, None)

    # create a frame buffer object as a render target for the HMD textures
    fboId = GL.GLuint()
    GL.glGenFramebuffers(1, ctypes.byref(fboId))
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fboId)
    depthRb_id = GL.GLuint()
    GL.glGenRenderbuffers(1, ctypes.byref(depthRb_id))
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depthRb_id)
    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH24_STENCIL8,
        int(buffer_w), int(buffer_h))  # buffer size used here!
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
    mirror_config = capi.ovrMirrorTextureDesc()
    mirror_config.Width = 800
    mirror_config.Height = 600

    capi.setupMirrorTexture(mirror_config)  # same size as window

    # frame index, increment this every frame
    frame_index = 0

    # compute projection matrices
    proj_left = capi.getEyeProjectionMatrix(capi.ovrEye_Left)
    proj_right = capi.getEyeProjectionMatrix(capi.ovrEye_Right)

    # get the session status
    session_status = capi.getSessionStatus()

    # begin application loop
    while not glfw.window_should_close(window):
        # wait for the buffer to be freed by the compositor, this is like
        # waiting for v-sync.
        capi.waitToBeginFrame(frame_index)
        #print(proj_left.M)
        # get current display time + predicted mid-frame time
        abs_time = capi.getDisplayTime(frame_index)

        # get the current tracking state
        tracking_state = capi.getTrackingState(abs_time)

        # Calculate eye poses, this needs to be called every frame, do this
        # after calling 'wait_to_begin_frame' to minimize the motion-to-photon
        # latency.
        left_eye_pose, right_eye_pose = capi.calcEyePoses(
            tracking_state)

        # get the view matrix from the HMD after calculating the pose
        view_left = capi.getEyeViewMatrix(left_eye_pose)
        view_right = capi.getEyeViewMatrix(right_eye_pose)

        # start frame rendering
        capi.beginFrame(frame_index)

        # bind the render FBO
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fboId)

        # get the current texture handle for this eye view, these are queued
        # in the swap chain  and released when free. Making draw calls to
        # any other texture in the swap chain not returned here will report
        # and error.
        tex_id = capi.getTextureSwapChainBufferGL(swap_chain)

        # bind the returned texture ID to the frame buffer's texture slot
        GL.glFramebufferTexture2D(
            GL.GL_DRAW_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D, tex_id, 0)

        # for each eye, do some rendering
        for eye in range(capi.ovrEye_Count):
            # Set the viewport as what was configured for the render layer. We
            # also need to enable scissor testings with the same rect as the
            # viewport. This constrains rendering operations to one partition of
            # of the buffer since we are using a 'packed' layout.
            vp = capi.getRenderViewport(eye)
            GL.glViewport(*vp.asTuple())
            GL.glScissor(*vp.asTuple())

            GL.glEnable(GL.GL_SCISSOR_TEST)  # enable scissor test
            GL.glEnable(GL.GL_DEPTH_TEST)

            # Here we can make whatever OpenGL we wish to draw our images. As an
            # example, I'm going to clear the eye buffer texture all some color,
            # with the colour determined by the active eye buffer.
            if eye == capi.ovrEye_Left:
                GL.glMatrixMode(GL.GL_PROJECTION)
                GL.glLoadIdentity()
                GL.glMultMatrixf(proj_left.ctypes)
                GL.glMatrixMode(GL.GL_MODELVIEW)
                GL.glLoadIdentity()
                if HEAD_TRACKING:
                    GL.glMultMatrixf(view_left.ctypes)

                GL.glClearColor(0.5, 0.5, 0.5, 1.0)
                GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

                GL.glColor3f(1.0, 1.0, 1.0)
                GL.glPushMatrix()
                GL.glBegin(GL.GL_QUADS)
                GL.glVertex3f(-1.0, -1.0, -5.0)
                GL.glVertex3f(-1.0, 1.0, -5.0)
                GL.glVertex3f(1.0, 1.0, -5.0)
                GL.glVertex3f(1.0, -1.0, -5.0)
                GL.glEnd()
                GL.glPopMatrix()

            elif eye == capi.ovrEye_Right:
                GL.glMatrixMode(GL.GL_PROJECTION)
                GL.glLoadIdentity()
                GL.glMultMatrixf(proj_right.ctypes)
                GL.glMatrixMode(GL.GL_MODELVIEW)
                GL.glLoadIdentity()
                if HEAD_TRACKING:
                    GL.glMultMatrixf(view_right.ctypes)

                GL.glClearColor(0.5, 0.5, 0.5, 1.0)
                GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

                GL.glColor3f(1.0, 1.0, 1.0)
                GL.glPushMatrix()
                GL.glBegin(GL.GL_QUADS)
                GL.glVertex3f(-1.0, -1.0, -5.0)
                GL.glVertex3f(-1.0, 1.0, -5.0)
                GL.glVertex3f(1.0, 1.0, -5.0)
                GL.glVertex3f(1.0, -1.0, -5.0)
                GL.glEnd()
                GL.glPopMatrix()

        GL.glDisable(GL.GL_DEPTH_TEST)

        # commit the texture when were done drawing to it
        capi.commitSwapChain(swap_chain)

        # unbind the frame buffer, we're done with it
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)

        # end frame rendering, submitting the eye layer to the compositor
        capi.endFrame(frame_index)

        # increment frame index
        frame_index += 1

        # update session status
        session_status = capi.getSessionStatus()

        # blit mirror texture
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, mirrorFbo)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)

        # bind the rift's texture to the framebuffer
        GL.glFramebufferTexture2D(
            GL.GL_READ_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D, capi.getMirrorTexture(), 0)

        GL.glViewport(0, 0, 800, 600)
        GL.glScissor(0, 0, 800, 600)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glBlitFramebuffer(0, 0, 800, 600,
                             0, 600, 800, 0,  # this flips the texture
                             GL.GL_COLOR_BUFFER_BIT,
                             GL.GL_NEAREST)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        capi.pollController('touch')  # update touch controller state

        # if button 'A' is released on the touch controller, recenter the
        # viewer in the scene.
        if capi.getButtons('touch', 'A', 'falling'):
            capi.recenterTrackingOrigin()
        elif capi.getButtons('touch', 'B', 'falling'):
            # exit if button 'B' is pressed
            break

        # flip the GLFW window and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()

    # switch off the performance summary
    capi.perfHudMode("Off")

    # end the rift session cleanly, all swap chains are destroyed here
    capi.endSession()

    # close the GLFW application
    glfw.terminate()

    return 0


if __name__ == "__main__":
    sys.exit(main())

