#  =============================================================================
#  libovr_gl.pxi - OpenGL related functions
#  =============================================================================
#
#  libovr_gl.pxi
#
#  Copyright 2020 Matthew Cutone <cutonem(a)yorku.ca> and Laurie M. Wilcox
#  <lmwilcox(a)yorku.ca>; The Centre For Vision Research, York University,
#  Toronto, Canada
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#
# texture formats, color and depth
FORMAT_R8G8B8A8_UNORM = capi.OVR_FORMAT_R8G8B8A8_UNORM
FORMAT_R8G8B8A8_UNORM_SRGB = capi.OVR_FORMAT_R8G8B8A8_UNORM_SRGB
FORMAT_R16G16B16A16_FLOAT =  capi.OVR_FORMAT_R16G16B16A16_FLOAT
FORMAT_R11G11B10_FLOAT = capi.OVR_FORMAT_R11G11B10_FLOAT
FORMAT_D16_UNORM = capi.OVR_FORMAT_D16_UNORM
FORMAT_D24_UNORM_S8_UINT = capi.OVR_FORMAT_D24_UNORM_S8_UINT
FORMAT_D32_FLOAT = capi.OVR_FORMAT_D32_FLOAT

# swapchain handles, more than enough for now
TEXTURE_SWAP_CHAIN0 = 0
TEXTURE_SWAP_CHAIN1 = 1
TEXTURE_SWAP_CHAIN2 = 2
TEXTURE_SWAP_CHAIN3 = 3
TEXTURE_SWAP_CHAIN4 = 4
TEXTURE_SWAP_CHAIN5 = 5
TEXTURE_SWAP_CHAIN6 = 6
TEXTURE_SWAP_CHAIN7 = 7
TEXTURE_SWAP_CHAIN8 = 8
TEXTURE_SWAP_CHAIN9 = 9
TEXTURE_SWAP_CHAIN10 = 10
TEXTURE_SWAP_CHAIN11 = 11
TEXTURE_SWAP_CHAIN12 = 12
TEXTURE_SWAP_CHAIN13 = 13
TEXTURE_SWAP_CHAIN14 = 14
TEXTURE_SWAP_CHAIN15 = 15
TEXTURE_SWAP_CHAIN_COUNT = 16

# mirror texture options
MIRROR_OPTION_DEFAULT = capi.ovrMirrorOption_Default
MIRROR_OPTION_POST_DISTORTION = capi.ovrMirrorOption_PostDistortion
MIRROR_OPTION_LEFT_EYE_ONLY = capi.ovrMirrorOption_LeftEyeOnly
MIRROR_OPTION_RIGHT_EYE_ONLY = capi.ovrMirrorOption_RightEyeOnly
MIRROR_OPTION_INCLUDE_GUARDIAN = capi.ovrMirrorOption_IncludeGuardian
MIRROR_OPTION_INCLUDE_NOTIFICATIONS = capi.ovrMirrorOption_IncludeNotifications
MIRROR_OPTION_INCLUDE_SYSTEM_GUI = capi.ovrMirrorOption_IncludeSystemGui
MIRROR_OPTION_FORCE_SYMMETRIC_FOV = capi.ovrMirrorOption_ForceSymmetricFov

cdef capi.ovrTextureSwapChain[16] _swapChains
cdef capi.ovrMirrorTexture _mirrorTexture


def calcEyeBufferSize(int eye, float texelsPerPixel=1.0):
    """Get the recommended buffer (texture) sizes for eye buffers.

    Should be called after :func:`setEyeRenderFov`. Returns buffer resolutions in
    pixels (w, h). The values can be used when configuring a framebuffer or swap
    chain for rendering.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.
    texelsPerPixel : float, optional
        Display pixels per texture pixels at the center of the display. Use a
        value less than 1.0 to improve performance at the cost of resolution.
        Specifying a larger texture is possible, but not recommended by the
        manufacturer.

    Returns
    -------
    tuple
        Buffer widths and heights (w, h) for each eye.

    Examples
    --------

    Getting the buffer size for the swap chain::

        # get HMD info
        hmdInfo = getHmdInfo()

        # eye FOVs must be set first!
        leftFov, rightFov = hmdInfo.defaultEyeFov
        setEyeRenderFov(EYE_LEFT, leftFov)
        setEyeRenderFov(EYE_RIGHT, rightFov)

        leftBufferSize, rightBufferSize = calcEyeBufferSize()
        leftW leftH = leftBufferSize
        rightW, rightH = rightBufferSize
        # combined size if using a single texture buffer for both eyes
        bufferW, bufferH = leftW + rightW, max(leftH, rightH)

        # create a swap chain
        createTextureSwapChainGL(TEXTURE_SWAP_CHAIN0, bufferW, bufferH)

    Notes
    -----
    * This function returns the recommended texture resolution for a specified
      eye. If you are using a single buffer for both eyes, that buffer should be
      as wide as the combined width of both eye's values.

    """
    global _ptrSession
    global _eyeRenderDesc

    cdef capi.ovrSizei bufferSize = capi.ovr_GetFovTextureSize(
        _ptrSession,
        <capi.ovrEyeType>0,
        _eyeRenderDesc[0].Fov,
        <float>texelsPerPixel)

    return bufferSize.w, bufferSize.h


def createTextureSwapChainGL(int swapChain, int width, int height, int textureFormat=FORMAT_R8G8B8A8_UNORM_SRGB, int levels=1):
    """Create a texture swap chain for eye image buffers.

    Parameters
    ----------
    swapChain : int
        Swap chain handle to initialize, usually :data:`SWAP_CHAIN*`.
    width : int
        Width of texture in pixels.
    height : int
        Height of texture in pixels.
    textureFormat : int
        Texture format to use. Valid color texture formats are:

        * ``FORMAT_R8G8B8A8_UNORM``
        * ``FORMAT_R8G8B8A8_UNORM_SRGB``
        * ``FORMAT_R16G16B16A16_FLOAT``
        * ``FORMAT_R11G11B10_FLOAT``

        Depth texture formats:

        * ``FORMAT_D16_UNORM``
        * ``FORMAT_D24_UNORM_S8_UINT``
        * ``FORMAT_D32_FLOAT``

    Other Parameters
    ----------------
    levels : int
        Mip levels to use, default is 1.

    Returns
    -------
    int
        The result of the ``OVR::ovr_CreateTextureSwapChainGL`` API call.

    Examples
    --------

    Create a texture swap chain::

        result = createTextureSwapChainGL(TEXTURE_SWAP_CHAIN0,
            texWidth, texHeight, FORMAT_R8G8B8A8_UNORM)
        # set the swap chain for each eye buffer
        for eye in range(EYE_COUNT):
            setEyeColorTextureSwapChain(eye, TEXTURE_SWAP_CHAIN0)

    """
    global _swapChains
    global _ptrSession

    if _swapChains[swapChain] != NULL:
        raise ValueError("Swap chain TEXTURE_SWAP_CHAIN{} already "
                         "initialized!".format(swapChain))

    # configure the texture
    cdef capi.ovrTextureSwapChainDesc swapConfig
    swapConfig.Type = capi.ovrTexture_2D
    swapConfig.Format = <capi.ovrTextureFormat>textureFormat
    swapConfig.ArraySize = 1
    swapConfig.Width = <int>width
    swapConfig.Height = <int>height
    swapConfig.MipLevels = <int>levels
    swapConfig.SampleCount = 1
    swapConfig.StaticImage = capi.ovrFalse  # always buffered
    swapConfig.MiscFlags = capi.ovrTextureMisc_None
    swapConfig.BindFlags = capi.ovrTextureBind_None

    # create the swap chain
    cdef capi.ovrResult result = \
        capi.ovr_CreateTextureSwapChainGL(
            _ptrSession,
            &swapConfig,
            &_swapChains[swapChain])

    #_eyeLayer.ColorTexture[swapChain] = _swapChains[swapChain]

    return result


def getTextureSwapChainLengthGL(int swapChain):
    """Get the length of a specified swap chain.

    Parameters
    ----------
    swapChain : int
        Swap chain handle to query. Must be a swap chain initialized by a
        previous call to :func:`createTextureSwapChainGL`.

    Returns
    -------
    tuple of int
        Result of the ``ovr_GetTextureSwapChainLength`` API call and the
        length of that swap chain.

    See Also
    --------
    getTextureSwapChainCurrentIndex : Get the current swap chain index.
    getTextureSwapChainBufferGL : Get the current OpenGL swap chain buffer.

    Examples
    --------

    Get the swap chain length for the previously created
    ``TEXTURE_SWAP_CHAIN0``::

        result, length = getTextureSwapChainLengthGL(TEXTURE_SWAP_CHAIN0)

    """
    cdef int outLength
    cdef capi.ovrResult result = 0
    global _swapChains
    global _ptrSession
    global _eyeLayer

    # check if there is a swap chain in the slot
    if _eyeLayer.ColorTexture[swapChain] == NULL:
        raise RuntimeError(
            "Cannot get swap chain length, NULL eye buffer texture.")

    # get the current texture index within the swap chain
    result = capi.ovr_GetTextureSwapChainLength(
        _ptrSession, _swapChains[swapChain], &outLength)

    return result, outLength


def getTextureSwapChainCurrentIndex(int swapChain):
    """Get the current buffer index within the swap chain.

    Parameters
    ----------
    swapChain : int
        Swap chain handle to query. Must be a swap chain initialized by a
        previous call to :func:`createTextureSwapChainGL`.

    Returns
    -------
    tuple of int
        Result of the ``OVR::ovr_GetTextureSwapChainCurrentIndex`` API call and
        the index of the buffer.

    See Also
    --------
    getTextureSwapChainLengthGL : Get the length of a swap chain.
    getTextureSwapChainBufferGL : Get the current OpenGL swap chain buffer.

    """
    cdef int current_idx = 0
    global _swapChains
    global _eyeLayer
    global _ptrSession

    # check if there is a swap chain in the slot
    if _eyeLayer.ColorTexture[swapChain] == NULL:
        raise RuntimeError(
            "Cannot get buffer ID, NULL eye buffer texture.")

    # get the current texture index within the swap chain
    cdef capi.ovrResult result = capi.ovr_GetTextureSwapChainCurrentIndex(
        _ptrSession, _swapChains[swapChain], &current_idx)

    return result, current_idx


def getTextureSwapChainBufferGL(int swapChain, int index):
    """Get the texture buffer as an OpenGL name at a specific index in the
    swap chain for a given swapChain.

    Parameters
    ----------
    swapChain : int
        Swap chain handle to query. Must be a swap chain initialized by a
        previous call to :func:`createTextureSwapChainGL`.
    index : int
        Index within the swap chain to retrieve its OpenGL texture name.

    Returns
    -------
    tuple (int, int)
        Result of the ``OVR::ovr_GetTextureSwapChainBufferGL`` API call and the
        OpenGL texture buffer name. A OpenGL buffer name is invalid when 0,
        check the returned API call result for an error condition.

    Examples
    --------

    Get the OpenGL texture buffer name associated with the swap chain index::

        # get the current available index
        swapChain = TEXTURE_SWAP_CHAIN0
        result, currentIdx = getSwapChainCurrentIndex(swapChain)

        # get the OpenGL buffer name
        result, texId = getTextureSwapChainBufferGL(swapChain, currentIdx)

        # bind the texture
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D, texId, 0)

    """
    cdef unsigned int tex_id = 0  # OpenGL texture handle
    global _swapChains
    global _eyeLayer
    global _ptrSession

    # get the next available texture ID from the swap chain
    cdef capi.ovrResult result = capi.ovr_GetTextureSwapChainBufferGL(
        _ptrSession, _swapChains[swapChain], index, &tex_id)

    return result, tex_id


def setEyeColorTextureSwapChain(int eye, int swapChain):
    """Set the color texture swap chain for a given eye.

    Should be called after a successful :func:`createTextureSwapChainGL` call
    but before any rendering is done.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.
    swapChain : int
        Swap chain handle to query. Must be a swap chain initialized by a
        previous call to :func:`createTextureSwapChainGL`.

    See Also
    --------
    createTextureSwapChainGL : Create a OpenGL buffer swap chain.

    Examples
    --------

    Associate the swap chain with both eyes (single buffer for stereo views)::

        setEyeColorTextureSwapChain(EYE_LEFT, TEXTURE_SWAP_CHAIN0)
        setEyeColorTextureSwapChain(EYE_RIGHT, TEXTURE_SWAP_CHAIN0)

        # same as above but with a loop
        for eye in range(EYE_COUNT):
            setEyeColorTextureSwapChain(eye, TEXTURE_SWAP_CHAIN0)

    Associate a swap chain with each eye (separate buffer for stereo views)::

        setEyeColorTextureSwapChain(EYE_LEFT, TEXTURE_SWAP_CHAIN0)
        setEyeColorTextureSwapChain(EYE_RIGHT, TEXTURE_SWAP_CHAIN1)

        # with a loop ...
        for eye in range(EYE_COUNT):
            setEyeColorTextureSwapChain(eye, TEXTURE_SWAP_CHAIN0 + eye)

    """
    global _swapChains
    global _eyeLayer

    _eyeLayer.ColorTexture[eye] = _swapChains[swapChain]


def createMirrorTexture(int width, int height, int textureFormat=FORMAT_R8G8B8A8_UNORM_SRGB, int mirrorOptions=MIRROR_OPTION_DEFAULT):
    """Create a mirror texture.

    This displays the content of the rendered images being presented on the
    HMD. The image is automatically refreshed to reflect the current content
    on the display. This displays the post-distortion texture.

    Parameters
    ----------
    width : int
        Width of texture in pixels.
    height : int
        Height of texture in pixels.
    textureFormat : int
        Color texture format to use, valid texture formats are:

        * ``FORMAT_R8G8B8A8_UNORM``
        * ``FORMAT_R8G8B8A8_UNORM_SRGB``
        * ``FORMAT_R16G16B16A16_FLOAT``
        * ``FORMAT_R11G11B10_FLOAT``

    mirrorOptions : int, optional
        Mirror texture options. Specifies how to display the rendered content.
        By default, ``MIRROR_OPTION_DEFAULT`` is used which displays the
        post-distortion image of both eye buffers side-by-side. Other options
        are available by specifying the following flags:

        * ``MIRROR_OPTION_POST_DISTORTION`` - Barrel distorted eye buffer.
        * ``MIRROR_OPTION_LEFT_EYE_ONLY`` and ``MIRROR_OPTION_RIGHT_EYE_ONLY`` -
          show rectilinear images of either the left of right eye. These values
          are mutually exclusive.
        * ``MIRROR_OPTION_INCLUDE_GUARDIAN`` - Show guardian boundary system in
          mirror texture.
        * ``MIRROR_OPTION_INCLUDE_NOTIFICATIONS`` - Show notifications received
          on the mirror texture.
        * ``MIRROR_OPTION_INCLUDE_SYSTEM_GUI`` - Show the system menu when
          accessed via the home button on the controller.
        * ``MIRROR_OPTION_FORCE_SYMMETRIC_FOV`` - Force mirror output to use
          symmetric FOVs. Only valid when ``MIRROR_OPTION_POST_DISTORTION`` is
          not specified.

        Multiple option flags can be combined by using the ``|`` operator and
        passed to `mirrorOptions`. However, some options cannot be used in
        conjunction with each other, if so, this function may return
        ``ERROR_INVALID_PARAMETER``.

    Returns
    -------
    int
        Result of API call ``OVR::ovr_CreateMirrorTextureWithOptionsGL``.

    """
    # create the descriptor
    cdef capi.ovrMirrorTextureDesc mirrorDesc
    global _ptrSession
    global _mirrorTexture

    mirrorDesc.Format = <capi.ovrTextureFormat>textureFormat
    mirrorDesc.Width = <int>width
    mirrorDesc.Height = <int>height
    mirrorDesc.MiscFlags = capi.ovrTextureMisc_None
    mirrorDesc.MirrorOptions = <capi.ovrMirrorOptions>mirrorOptions

    cdef capi.ovrResult result = capi.ovr_CreateMirrorTextureWithOptionsGL(
        _ptrSession, &mirrorDesc, &_mirrorTexture)

    return <int>result


def getMirrorTexture():
    """Mirror texture ID.

    Returns
    -------
    tuple (int, int)
        Result of API call ``OVR::ovr_GetMirrorTextureBufferGL`` and the mirror
        texture ID. A mirror texture ID == 0 is invalid.

    Examples
    --------

    Getting the mirror texture for use::

        # get the mirror texture
        result, mirrorTexId = getMirrorTexture()
        if failure(result):
            # raise error ...

        # bind the mirror texture texture to the framebuffer
        glFramebufferTexture2D(
            GL_READ_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D, mirrorTexId, 0)

    """
    cdef unsigned int mirror_id

    global _ptrSession
    global _mirrorTexture

    if _mirrorTexture == NULL:  # no texture created
        return None

    cdef capi.ovrResult result = \
        capi.ovr_GetMirrorTextureBufferGL(
            _ptrSession,
            _mirrorTexture,
            &mirror_id)

    return <int>result, <unsigned int>mirror_id