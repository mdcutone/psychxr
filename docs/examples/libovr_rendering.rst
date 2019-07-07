=====================
Rendering with OpenGL
=====================

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

Setup Eye Render Buffers
------------------------

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

We now start calling OpenGL commands to build our framebuffer. You can use
`pyglet` or `PyOpenGL` to do this. Here we use `PyOpenGL` for OpenGL commands by
importing::

    import OpenGL.GL as GL
    import ctypes  # needed for some OpenGL commands

We now create an OpenGL framebuffer which will serve as a render target for
image buffers pulled from the swap chain. You must use the computed buffer sizes
above to configure associated render buffers::

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


Finally we create a mirror texture using the size used when creating the GLFW
window and create a framebuffer for it::

    createMirrorTexture(800, 600)
    mirrorFbo = GL.GLuint()
    GL.glGenFramebuffers(1, ctypes.byref(mirrorFbo))

Rendering to the HMD
--------------------

Now that we setup our swap chains and buffers, we can begin rendering graphics
to the HMD. Each frame we increment the frame index, get tracking state
information about the HMD, use that data to draw the scene, and finally poll any
input devices. This process repeats until the user exits the application.

First we create a variable to store the frame index and initialize it to 0::

    frame_index = 0

Before we enter our main application loop, we request the projection matrices
for each eye. These are computed based on the FOV settings that were specified
earlier. Since these values don't usually change, we can call the following
once::

    projectionMatrix = []
    for eye in range(EYE_COUNT):
         projectionMatrix.append(getEyeProjectionMatrix(eye))

To demonstrate using `LibOVRPose` objects to define rigid body transformations,
we'll create one to position an object the scene. Here we create a `LibOVRPose`
instance and set its Z position to -2 meters (recall -Z is forward in OpenGL).
Then we convert the pose to a 4x4 transformation matrix by calling the
`asMatrix` method::

    planeMatrix = LibOVRPose((0., 0., -2.)).asMatrix()

We create our main loop using a `while` statement, since the loop should run
until the user exits. Here we make the loop conditional on the whether the user
closes the on-screen mirror window.

Upon entering the loop, we call `waitToBeginFrame` to hold the application until
`LibOVR` is ready to start accepting frames. Once the function returns, we get
the HMD head pose at the predicted time the frame will appear on the display,
then use that data to calculate eye poses with `calcEyePoses`::

    while not glfw.window_should_close(window):

        # predicted mid-frame time
        abs_time = getPredictedDisplayTime(frame_index)

        # get the current tracking state
        tracking_state, calibrated_origin = getTrackingState(abs_time, True)

        # calculate eye poses, this needs to be called every frame
        headPose, state = tracking_state[TRACKED_DEVICE_TYPE_HMD]
        calcEyePoses(headPose.pose)

Now we can begin rendering to the eye buffers. First, we tell `LibOVR` that
frame rendering will commence by calling `beginFrame`. Afterwards, we get the
current swap chain buffer and set that texture as the OpenGL framebuffer draw
target::

    # while not glfw.window_should_close(window):
    # ...
        # start frame rendering
        beginFrame(frame_index)

        # bind the render FBO
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fboId)

        # get the current swap chain buffer index and OpenGL texture
        _, swapIdx = getTextureSwapChainCurrentIndex(TEXTURE_SWAP_CHAIN0)
        _, tex_id = getTextureSwapChainBufferGL(TEXTURE_SWAP_CHAIN0, swapIdx)

        # bind the returned texture ID to the frame buffer's texture slot
        GL.glFramebufferTexture2D(
            GL.GL_DRAW_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D, tex_id, 0)

We create `for` loop to render images to each eye. Here we render a
multi-colored plane transformed by `planeMatrix`::

    # while not glfw.window_should_close(window):
    # ...
        for eye in range(EYE_COUNT):

            # Set the viewport as what was configured for the render layer. We
            # also need to enable scissor testings with the same rect as the
            # viewport. This constrains rendering operations to one partition of
            # of the buffer since we are using a 'side-by-side' layout.
            vp = getEyeRenderViewport(eye)
            GL.glViewport(*vp)
            GL.glScissor(*vp)

            # Get view and projection matrices
            P = projectionMatrix[eye]
            MV = getEyeViewMatrix(eye)

            GL.glEnable(GL.GL_SCISSOR_TEST)  # enable scissor test
            GL.glEnable(GL.GL_DEPTH_TEST)

            # Set the projection matrix.
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadTransposeMatrixf(P)

            # Set the view matrix. This contains the translation for the head in
            # the virtual space computed by the API.
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadTransposeMatrixf(MV)

            # Okay, let's begin drawing stuff. Clear the background first.
            GL.glClearColor(0.0, 0.0, 0.0, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # Draw a multicolored 2x2 meter square positioned 5 meters in front
            # of the virtual space's origin.
            GL.glPushMatrix()
            GL.glMultTransposeMatrixf(planeMatrix)  # set the position of plane in the scene
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

        # unbind the frame buffer, we're done with it
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)


After rendering the eye buffer images, we commit the texture to the swap chain.
At this point, we can no longer modify the contents of the texture. Then we call
`endFrame` to submit the texture for display on the HMD and increment the frame
index::

    # while not glfw.window_should_close(window):
    # ...
        # commit the texture when were done drawing to it
        commitTextureSwapChain(TEXTURE_SWAP_CHAIN0)

        # end frame rendering, submitting the eye layer to the compositor
        endFrame(frame_index)

        frame_index += 1  # increment frame index

Now we draw the mirror texture to the display. This will present the distorted
image on the window we created. This involves binding the mirror framebuffer,
getting the mirror texture buffer ID, and blitting the texture to the window's
back buffer::

    # while not glfw.window_should_close(window):
    # ...
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

        glfw.swap_buffers(window)  # put the mirror on-screen

Getting Input
-------------

We can get input from `LibOVR` managed input devices, or use keyboard and mouse
input via GLFW. Here we get the 'A' and 'B' button states of the paired `Touch`
controllers. If 'A' is released the tracking origin is re-centered to the
current head position, if 'B' is released, the application will exit by breaking
out of the `while` loop::

    # while not glfw.window_should_close(window):
    # ...
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

        # flip the GLFW window and poll events, needs to be called
        glfw.poll_events()

Accessing Session Status
------------------------
We can use the current session status to determine if the user requests the
application exit via the system UI. If the `shouldQuit` flag is `True`, we can
break out of the rendering loop. This can be implemented using the following::

    # while not glfw.window_should_close(window):
    # ...

        _, sessionStatus = getSessionStatus()  # get current session status
        if sessionStatus.shouldQuit:
            break


Exiting the Application
-----------------------

If the application breaks out of the rendering loop, we need to free up
resources we created earlier and shutdown the VR session. This is done by
calling the following commands::

    # free resources
    destroyMirrorTexture()
    destroyTextureSwapChain(TEXTURE_SWAP_CHAIN0)

    # close the GLFW application
    glfw.terminate()

    # end the rift session cleanly
    destroy()
    shutdown()
