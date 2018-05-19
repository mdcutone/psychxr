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
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 2)
    window = glfw.create_window(800, 600, "Oculus Test", None, None)

    if not window:
        glfw.terminate()
        return -1

    # initialize OVR library
    capi.initialize()

    if capi.create() < 0:
        glfw.terminate()
        return -1

    glfw.make_context_current(window)

    capi.setup_gl()
    buffer_w, buffer_h = capi.get_buffer_size()

    # create framebuffers
    fboId = GLuint()
    glGenFramebuffers(1, fboId)
    glBindFramebuffer(GL_FRAMEBUFFER, fboId)

    colorRb_id = GLuint()
    glGenRenderbuffers(1, ctypes.byref(colorRb_id))
    glBindRenderbuffer(GL_RENDERBUFFER, colorRb_id)
    glFramebufferRenderbuffer(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER,
        colorRb_id)
    glBindRenderbuffer(GL_RENDERBUFFER, 0)

    depthRb_id = GLuint()
    glGenRenderbuffers(1, ctypes.byref(depthRb_id))
    glBindRenderbuffer(GL_RENDERBUFFER, depthRb_id)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8,
                          int(buffer_w), int(buffer_h))
    glFramebufferRenderbuffer(
        GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER,
        depthRb_id)
    glFramebufferRenderbuffer(
        GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_RENDERBUFFER,
        depthRb_id)
    glBindRenderbuffer(GL_RENDERBUFFER, 0)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    while not glfw.window_should_close(window):
        capi.begin_frame()
        capi.ovr_SetInt(b"PerfHudMode", capi.ovrPerfHud_CompRenderTiming)

        texture_id = capi.get_swap_chain_buffer_gl()
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fboId)
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D,
                               texture_id, 0)

        glViewport(0, 0, int(buffer_w/2), buffer_h)
        glScissor(
            0,
            0,
            int(buffer_w / 2),
            buffer_h)

        #glMatrixMode(GL_PROJECTION)
        #glLoadIdentity()
        #glOrtho(-1, 1, -1, 1, -1, 1)
        #glMatrixMode(GL_MODELVIEW)
        #glLoadIdentity()

        glClearColor(1.0, 0.5, 0.5, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glFinish()
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)

        #glBindTexture(GL_TEXTURE_2D, 0)
        capi.end_frame()

        #
        glBindFramebuffer(
            GL_READ_FRAMEBUFFER, fboId)
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D,
                               capi.get_mirror_texture(), 0)
        glBindFramebuffer(
            GL_DRAW_FRAMEBUFFER, 0)

        glBlitFramebuffer(
            0, 0,
            buffer_w, buffer_h,
            800, 600,
            0, 0,
            GL_COLOR_BUFFER_BIT, GL_NEAREST)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glfw.swap_buffers(window)
        glfw.poll_events()

    capi.destroy()
    capi.shutdown()
    glfw.terminate()
    return 0

if __name__ == "__main__":
    main()