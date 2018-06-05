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

    # disable v-sync, we are syncing to the v-trace of head-set, leaving this on
    # will cause the HMD to lock to the frequency/phase of the display.
    glfw.swap_interval(0)

    # start an Oculus session
    rift.start_session()

    # get general information about the HMD
    print(rift.get_hmd_info())

    # get the buffer dimensions specified by the Rift SDK, we need them to
    # setup OpenGL frame buffers.
    buffer_size = rift.get_buffer_size()

    # Allocate a swap chain for render buffer textures, the handle used is an
    # integer. You can allocated up to 32 swap chains, however you will likely
    # run out of video memory by then.
    swap_chain = rift.alloc_swap_chain(*buffer_size)

    # setup a the render layer
    rift.setup_render_layer(buffer_size[0], buffer_size[1], swap_chain)

    # create a frame buffer object as a render target for the HMD textures
    fboId = GL.GLuint()
    GL.glGenFramebuffers(1, ctypes.byref(fboId))
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fboId)
    depthRb_id = GL.GLuint()
    GL.glGenRenderbuffers(1, ctypes.byref(depthRb_id))
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depthRb_id)
    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH24_STENCIL8,
        int(buffer_size[0]), int(buffer_size[1]))  # buffer size used here!
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

        # bind the render FBO
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fboId)

        # get the current texture handle for this eye view, these are queued
        # in the swap chain  and released when free. Making draw calls to
        # any other texture in the swap chain not returned here will report
        # and error.
        tex_id = rift.get_swap_chain_buffer(swap_chain)

        # bind the returned texture ID to the frame buffer's texture slot
        GL.glFramebufferTexture2D(
            GL.GL_DRAW_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D, tex_id, 0)

        # for each eye, do some rendering
        for eye in ('left', 'right'):
            # Set the viewport as what was configured for the render layer. We
            # also need to enable scissor testings with the same rect as the
            # viewport. This constrains rendering operations to one partition of
            # of the buffer since we are using a 'packed' layout.
            x, y, w, h = rift.get_render_layer_viewport(eye)
            GL.glViewport(x, y, w, h)
            GL.glScissor(x, y, w, h)
            GL.glEnable(GL.GL_SCISSOR_TEST)  # enable scissor test

            # Here we can make whatever OpenGL we wish to draw our image. As an
            # example, I'm going to clear the eye buffer texture all some color,
            # with the colour determined by the active eye buffer.
            if eye == 'left':
                GL.glClearColor(1.0, 0.2, 0.2, 1.0)  # red
            elif eye == 'right':
                GL.glClearColor(0.2, 0.2, 1.0, 1.0)  # blue

            view, proj = rift.get_eye_view_matrix(eye)
            print(view.get_translation().as_tuple())

            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        # commit the texture when were done drawing to it
        rift.commit_swap_chain(swap_chain)

        # unbind the frame buffer, we're done with it
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)

        # end frame rendering, submitting the eye layer to the compositor
        rift.end_frame(frame_index)

        # increment frame index
        frame_index += 1

        # update session status
        rift.update_session_status()

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

    # end the rift session cleanly, all swap chains are destroyed here
    rift.end_session()

    # close the GLFW application
    glfw.terminate()

    return 0


if __name__ == "__main__":
    sys.exit(main())

