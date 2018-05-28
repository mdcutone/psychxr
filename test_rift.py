# PsychXR Oculus Rift minimal example.
#
import OpenGL.GL as GL
import ctypes
import glfw

import psychxr.ovr.rift as rift
rift.debug_mode = True
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

    # disable v-sync, we are syncing to the v-trace of head-set
    glfw.swap_interval(0)

    # start a Oculus session
    rift.start_session()

    # get the buffer size
    buffer_size = rift.get_buffer_size()

    # create a frambuffer object complete with render buffer as a render target
    fboId = GL.GLuint()
    GL.glGenFramebuffers(1, ctypes.byref(fboId))
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fboId)
    depthRb_id = GL.GLuint()
    GL.glGenRenderbuffers(1, ctypes.byref(depthRb_id))
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depthRb_id)
    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH24_STENCIL8,
                          int(buffer_size[0]), int(buffer_size[1]))
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

    # setup the render layer after the GL context is current
    rift.setup_render_layer(buffer_size)

    # setup a mirror texture
    rift.setup_mirror_texture(800, 600)  # same size as window

    # frame index, increment this every frame
    frame_index = 0

    # begin application loop
    while not glfw.window_should_close(window):
        # wait for the buffer to be freed by the compositor, this is like
        # waiting for v-sync.
        rift.wait_to_begin_frame(frame_index)

        # get current display time + predicted mid-frame time
        abs_time = rift.get_display_time(frame_index)

        # Calculate eye poses, this needs to be called every frame, do this
        # after calling 'wait_to_begin_frame' to minimize the motion-to-photon
        # latency.
        rift.calc_eye_poses(abs_time)

        # start frame rendering
        rift.begin_frame(frame_index)

        # bind the render target
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fboId)

        # bind the rift's texture to the framebuffer
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D, rift.get_texture_swap_buffer(), 0)

        # get viewport sizes
        vp_width = int(buffer_size[0] / 2)
        vp_height = int(buffer_size[1])

        # enable scissor test
        GL.glEnable(GL.GL_SCISSOR_TEST)

        # render to each eye
        for eye in range(2):
            # set the viewport
            if eye == 0:  # left eye
                GL.glViewport(0, 0, vp_width, vp_height)
                GL.glScissor(0, 0, vp_width, vp_height)

                # DRAW LEFT EYE STUFF HERE
                # clear left eye to red
                GL.glClearColor(1.0, 0.2, 0.2, 1.0)

            elif eye == 1:  # right eye
                GL.glViewport(vp_width, 0, vp_width, vp_height)
                GL.glScissor(vp_width, 0, vp_width, vp_height)

                # DRAW RIGHT EYE STUFF HERE
                # clear right eye to blue
                GL.glClearColor(0.2, 0.2, 1.0, 1.0)

            # clear the screen
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        # unbind the frame buffer, we're done with it
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        # end frame rendering, submitting the texture to the compositior
        rift.end_frame(frame_index)

        # increment frame index
        frame_index += 1

        # blit mirror texture
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, mirrorFbo)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)

        # bind the rift's texture to the framebuffer
        GL.glFramebufferTexture2D(
            GL.GL_READ_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D, rift.get_mirror_texture(), 0)

        GL.glViewport(0, 0, 800, 600)
        GL.glScissor(0, 0, 800, 600)
        GL.glBlitFramebuffer(0, 0, 800, 600,
                             0, 600, 800, 0,  # this flips the texture
                             GL.GL_COLOR_BUFFER_BIT,
                             GL.GL_NEAREST)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        # flip the GLFW window and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()

    # end the rift session cleanly
    rift.end_session()

    # close the GLFW application
    glfw.terminate()

    return 0


if __name__ == "__main__":
    sys.exit(main())

