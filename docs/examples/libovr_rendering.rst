============================
LibOVR Rendering to the Rift
============================

This tutorial covers how to use PsychXR to render graphics to the Oculus Rift
HMD using pure OpenGL and the :mod:`libovr` extension.

Setting up your application for rendering involves the following steps:

    1. Initialize your OpenGL context.
    2. Create a new VR session.
    3. Setup render buffers using data from `LibOVR` and configure the render layer.
    4. Create a mirror texture.
    5. Begin rendering frames in your application loop.

Create an OpenGL Context
------------------------

We need to setup a window with a OpenGL context. The window will be used to
display our HMD mirror. First we need to import `pyGLFW`::

    import glfw

We create a window using GLFW using the following code::

    if not glfw.init():
        return -1

    # for this example, we are using OpenGL 2.1 to keep things simple
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

    window = glfw.create_window(800, 600, "Oculus Test", None, None)

    if not window:
        glfw.terminate()

    glfw.make_context_current(window)

    glfw.swap_interval(0)

We need to set ``glfw.swap_interval(0)`` to prevent the application from syncing
rendering with the monitor.

Create a VR Session
-------------------

First, import the `libovr` extension module to gain access to the `LibOVR` API::

    from psychxr.libovr import *

Initialize `LibOVR` and create a new VR session using the following::

    # initialize the runtime
    if failure(initialize()):
        return -1

    # create a new session
    if failure(create()):
        shutdown()
        return -1

It is good practice to check if API commands raised errors by using ``failure()``
on the returned values. If an error is raised after ``initialize()`` succeeds,
you should call ``shutdown()``.

If both ``initialize()`` and ``create()`` return no errors, the session has been
successfully created. At this point, the Oculus application will start if not
already running in the background.

Setup Rendering
---------------

We need to access data from `LibOVR` to instruct how to setup OpenGL for
rendering. First, we need to get information about the particular model of HMD
were using by calling::

    hmdInfo = getHmdInfo()

The returned object contains FOV information we need to use to setup HMD
rendering. The FOV for each eye to use is specified by calling::

    for eye, fov in enumerate(hmdInfo.defaultEyeFov):
        setEyeRenderFov(eye, fov)

Now that we have our FOVs set, we can compute the eye buffer sizes by calling::

    texSizeLeft = calcEyeBufferSize(EYE_LEFT)
    texSizeRight = calcEyeBufferSize(EYE_RIGHT)

For this example, were are going to use a single buffer for both eyes, arranged
side-by-side. We compute the dimensions of this buffer by combining the
horizontal sizes together::

    bufferW = texSizeLeft[0] + texSizeRight[0]
    bufferH = max(texSizeLeft[1], texSizeRight[1])

Now that we know our buffer sizes, we need to tell `LibOVR` which sub-region of
the buffer is allocated to each eye by specifying viewports. We compute the
viewports and set them by doing the following::

    eye_w = int(bufferW / 2)
    eye_h = bufferH

    viewports = ((0, 0, eye_w, eye_h), (eye_w, 0, eye_w, eye_h))
    for eye, vp in enumerate(viewports):
        setEyeRenderViewport(eye, vp)

At this point, we can create a swap chain which is used to pass buffers to the
`LibOVR` compositor for display on the HMD. We use the handle
``TEXTURE_SWAP_CHAIN0`` to access the swap chain. We can create the swap chain by
doing the following::

    createTextureSwapChainGL(TEXTURE_SWAP_CHAIN0, bufferW, bufferH)

    for eye in range(EYE_COUNT):
        setEyeColorTextureSwapChain(eye, TEXTURE_SWAP_CHAIN0)

Since we are using a single texture for both eyes, we set them to use the same
handle. If two buffers are used, one for each eye, you need to call
``createTextureSwapChainGL`` twice using different handles (eg.
``TEXTURE_SWAP_CHAIN0`` for the left eye and ``TEXTURE_SWAP_CHAIN1`` for the
right.)

You can tell the compositor to enable high-quality mode, which applies 4x
anisotropic filtering during distortion to reduce sampling artifacts by
calling::

    setHighQuality(True)

Finally, we create an OpenGL framebuffer which will serve as a render target for
image buffers pulled from the swap chain. You must use the computed buffer sizes
above to configure render buffers::

    fboId = GL.GLuint()
    GL.glGenFramebuffers(1, ctypes.byref(fboId))
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fboId)

    depthRb_id = GL.GLuint()
    GL.glGenRenderbuffers(1, ctypes.byref(depthRb_id))
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depthRb_id)
    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH24_STENCIL8,
        int(bufferW), int(bufferH))  # <<< buffer dimensions computed earlier
    GL.glFramebufferRenderbuffer(
        GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER,
        depthRb_id)
    GL.glFramebufferRenderbuffer(
        GL.GL_FRAMEBUFFER, GL.GL_STENCIL_ATTACHMENT, GL.GL_RENDERBUFFER,
        depthRb_id)

    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, 0)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

