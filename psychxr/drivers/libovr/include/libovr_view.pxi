#  =============================================================================
#  libovr_view.pxi - Rendering and compositor related types and functions
#  =============================================================================
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
# layer header flags
LAYER_FLAG_HIGH_QUALITY = capi.ovrLayerFlag_HighQuality
LAYER_FLAG_TEXTURE_ORIGIN_AT_BOTTOM_LEFT = \
    capi.ovrLayerFlag_TextureOriginAtBottomLeft
LAYER_FLAG_HEAD_LOCKED = capi.ovrLayerFlag_HeadLocked

# VR related data persistent across frames
cdef capi.ovrLayerEyeFov _eyeLayer
cdef capi.ovrPosef[2] _eyeRenderPoses
cdef capi.ovrEyeRenderDesc[2] _eyeRenderDesc
# cdef capi.ovrViewScaleDesc _viewScale

# near and far clipping planes
cdef float[2] _nearClip
cdef float[2] _farClip

# prepare the render layer
_eyeLayer.Header.Type = capi.ovrLayerType_EyeFov
_eyeLayer.Header.Flags = \
    capi.ovrLayerFlag_TextureOriginAtBottomLeft | \
    capi.ovrLayerFlag_HighQuality
_eyeLayer.ColorTexture[0] = _eyeLayer.ColorTexture[1] = NULL

# geometric data
cdef libovr_math.Matrix4f[2] _eyeProjectionMatrix
cdef libovr_math.Matrix4f[2] _eyeViewMatrix
cdef libovr_math.Matrix4f[2] _eyeViewProjectionMatrix


def setHeadLocked(bint enable):
    """Set the render layer state for head locking.

    Head-locking prevents the compositor from applying asynchronous time warp
    (ASW) which compensates for rendering latency. Under normal circumstances
    where head pose data is retrieved from `LibOVR` using
    :func:`getTrackingState` or :func:`getDevicePoses` calls, it
    should be enabled to prevent juddering and improve visual stability.

    However, when using custom head poses (eg. fixed, or from a motion tracker)
    this system may cause the render layer to slip around, as internal IMU data
    will be incongruous with externally supplied head posture data. If you plan
    on passing custom poses to :func:`calcEyePoses`, ensure that head locking is
    enabled.

    Head locking is disabled by default when a session is started.

    Parameters
    ----------
    enable : bool
        Enable head-locking when rendering to the eye render layer.

    """
    global _eyeLayer
    if enable:
        _eyeLayer.Header.Flags |= capi.ovrLayerFlag_HeadLocked
    else:
        _eyeLayer.Header.Flags &= ~capi.ovrLayerFlag_HeadLocked


def isHeadLocked():
    """Check if head locking is enabled.

    Returns
    -------
    bool
        ``True`` if head-locking is enabled.

    See Also
    --------
    setHeadLocked

    """
    return (_eyeLayer.Header.Flags & capi.ovrLayerFlag_HeadLocked) == \
           capi.ovrLayerFlag_HeadLocked


def setHighQuality(bint enable):
    """Enable high quality mode.

    This enables 4x anisotropic sampling by the compositor to reduce the
    appearance of high-frequency artifacts in the visual periphery due to
    distortion.

    Parameters
    ----------
    enable : bool
        Enable high-quality mode.

    """
    global _eyeLayer
    if enable:
        _eyeLayer.Header.Flags |= capi.ovrLayerFlag_HighQuality
    else:
        _eyeLayer.Header.Flags &= ~capi.ovrLayerFlag_HighQuality


def getPixelsPerTanAngleAtCenter(int eye):
    """Get pixels per tan angle (=1) at the center of the display.

    Values reflect the FOVs set by the last call to :func:`setEyeRenderFov` (or
    else the default FOVs will be used.)

    Parameters
    ----------
    eye : int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.

    Returns
    -------
    tuple
        Pixels per tan angle at the center of the screen.

    """
    global _eyeRenderDesc

    cdef capi.ovrVector2f toReturn = \
        _eyeRenderDesc[eye].PixelsPerTanAngleAtCenter

    return toReturn.x, toReturn.y


def getTanAngleToRenderTargetNDC(int eye, object tanAngle):
    """Convert FOV tan angle to normalized device coordinates (NDC).

    Parameters
    ----------
    eye : int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.
    tanAngle : tuple, list of float or ndarray
        Horizontal and vertical tan angles [X, Y] from display center.

    Returns
    -------
    tuple
        NDC coordinates X, Y [-1, 1].

    """
    global _eyeRenderDesc

    cdef libovr_math.Vector2f vecIn
    vecIn.x = tanAngle[0]
    vecIn.y = tanAngle[1]

    cdef libovr_math.Vector2f toReturn = \
        (<libovr_math.FovPort>_eyeRenderDesc[eye].Fov).TanAngleToRendertargetNDC(
            vecIn)

    return toReturn.x, toReturn.y


def getPixelsPerDegree(int eye):
    """Get pixels per degree at the center of the display.

    Values reflect the FOVs set by the last call to :func:`setEyeRenderFov` (or
    else the default FOVs will be used.)

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.

    Returns
    -------
    tuple
        Pixels per degree at the center of the screen (h, v).

    """
    global _eyeRenderDesc

    cdef capi.ovrVector2f pixelsPerTanAngle = \
        _eyeRenderDesc[eye].PixelsPerTanAngleAtCenter

    # tan(angle)=1 -> 45 deg
    cdef float horzPixelPerDeg = <float>pixelsPerTanAngle.x / <float>45.0
    cdef float vertPixelPerDeg = <float>pixelsPerTanAngle.y / <float>45.0

    return horzPixelPerDeg, vertPixelPerDeg


def getDistortedViewport(int eye):
    """Get the distorted viewport.

    You must call :func:`setEyeRenderFov` first for values to be valid.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.

    """
    cdef capi.ovrRecti distVp = _eyeRenderDesc[eye].DistortedViewport

    cdef np.ndarray toReturn = np.asarray([
        distVp.Pos.x,
        distVp.Pos.x,
        distVp.Size.w,
        distVp.Size.h],
        dtype=np.int)

    return toReturn


def getEyeRenderFov(int eye):
    """Get the field-of-view to use for rendering.

    The FOV for a given eye are defined as a tuple of tangent angles (Up,
    Down, Left, Right). By default, this function will return the default
    (recommended) FOVs after :func:`create` is called.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.

    Returns
    -------
    tuple
        Eye FOV tangent angles [UpTan, DownTan, LeftTan, RightTan], distance to
        near and far clipping planes in meters.

    Examples
    --------
    Getting the tangent angles::

        leftFov, nearClip, farClip = getEyeRenderFOV(EYE_LEFT)
        # left FOV tangent angles, do the same for the right
        upTan, downTan, leftTan, rightTan =  leftFov

    """
    global _eyeRenderDesc
    global _nearClip
    global _farClip

    cdef np.ndarray to_return = np.asarray([
        _eyeRenderDesc[eye].Fov.UpTan,
        _eyeRenderDesc[eye].Fov.DownTan,
        _eyeRenderDesc[eye].Fov.LeftTan,
        _eyeRenderDesc[eye].Fov.RightTan],
        dtype=np.float32)

    return to_return, _nearClip[eye], _farClip[eye]


def setEyeRenderFov(int eye, object fov, float nearClip=0.01, float farClip=1000.):
    """Set the field-of-view of a given eye. This is used to compute the
    projection matrix.

    By default, this function will return the default FOVs after :func:`create`
    is called (see :py:attr:`LibOVRHmdInfo.defaultEyeFov`). You can override
    these values using :py:attr:`LibOVRHmdInfo.maxEyeFov` and
    :py:attr:`LibOVRHmdInfo.symmetricEyeFov`, or with custom values (see
    Examples below).

    Parameters
    ----------
    eye : int
        Eye index. Values are ``EYE_LEFT`` and ``EYE_RIGHT``.
    fov : array_like
        Eye FOV tangent angles [UpTan, DownTan, LeftTan, RightTan].
    nearClip, farClip : float
        Near and far clipping planes in meters. Used when computing the
        projection matrix.

    Examples
    --------

    Setting eye render FOVs to symmetric (needed for mono rendering)::

        leftFov, rightFov = getSymmetricEyeFOVs()
        setEyeRenderFOV(EYE_LEFT, leftFov)
        setEyeRenderFOV(EYE_RIGHT, rightFov)

    Using custom values::

        # Up, Down, Left, Right tan angles
        setEyeRenderFOV(EYE_LEFT, [1.0, -1.0, -1.0, 1.0])

    """
    global _ptrSession
    global _eyeRenderDesc
    global _eyeLayer
    global _nearClip
    global _farClip
    global _eyeProjectionMatrix

    cdef capi.ovrFovPort fov_in
    fov_in.UpTan = <float>fov[0]
    fov_in.DownTan = <float>fov[1]
    fov_in.LeftTan = <float>fov[2]
    fov_in.RightTan = <float>fov[3]

    _eyeRenderDesc[<int>eye] = capi.ovr_GetRenderDesc(
        _ptrSession,
        <capi.ovrEyeType>eye,
        fov_in)

    # set in eye layer too
    _eyeLayer.Fov[eye] = _eyeRenderDesc[eye].Fov

    # set clipping planes
    _nearClip[<int>eye] = nearClip
    _farClip[<int>eye] = farClip

    # compute the projection matrix
    _eyeProjectionMatrix[eye] = \
        <libovr_math.Matrix4f>capi.ovrMatrix4f_Projection(
            _eyeRenderDesc[eye].Fov,
            _nearClip[eye],
            _farClip[eye],
            capi.ovrProjection_ClipRangeOpenGL)


def getEyeAspectRatio(int eye):
    """Get the aspect ratio of an eye.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.

    Returns
    -------
    float
        Aspect ratio of the eye's FOV (width / height).

    """
    cdef libovr_math.FovPort fovPort = \
        <libovr_math.FovPort>_eyeRenderDesc[eye].Fov

    return (fovPort.LeftTan + fovPort.RightTan) / \
           (fovPort.UpTan + fovPort.DownTan)


def getEyeHorizontalFovRadians(int eye):
    """Get the angle of the horizontal field-of-view (FOV) for a given eye.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.

    Returns
    -------
    float
        Horizontal FOV of a given eye in radians.

    """
    cdef libovr_math.FovPort fovPort = \
        <libovr_math.FovPort>_eyeRenderDesc[eye].Fov

    return fovPort.GetHorizontalFovRadians()


def getEyeVerticalFovRadians(int eye):
    """Get the angle of the vertical field-of-view (FOV) for a given eye.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.

    Returns
    -------
    float
        Vertical FOV of a given eye in radians.

    """
    cdef libovr_math.FovPort fovPort = \
        <libovr_math.FovPort>_eyeRenderDesc[eye].Fov

    return fovPort.GetVerticalFovRadians()


def getEyeFocalLength(int eye):
    """Get the focal length of the eye's frustum.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.

    Returns
    -------
    float
        Focal length in meters.

    Notes
    -----
    * This does not reflect the optical focal length of the HMD.

    """
    return 1.0 / tan(getEyeHorizontalFovRadians(eye) / 2.0)


def getLayerEyeFovFlags():
    """Get header flags for the render layer.

    Returns
    -------
    unsigned int
        Flags from ``OVR::ovrLayerEyeFov.Header.Flags``.

    See Also
    --------
    setLayerEyeFovFlags : Set layer flags.

    Examples
    --------
    Check if a flag is set::

        layerFlags = getLayerEyeFovFlags()
        if (layerFlags & LAYER_FLAG_HIGH_QUALITY) == LAYER_FLAG_HIGH_QUALITY:
            print('high quality enabled!')

    """
    global _eyeLayer
    return <unsigned int>_eyeLayer.Header.Flags


def setLayerEyeFovFlags(unsigned int flags):
    """Set header flags for the render layer.

    Parameters
    ----------
    flags : int
        Flags to set. Flags can be ORed together to apply multiple settings.
        Valid values for flags are:

        * ``LAYER_FLAG_HIGH_QUALITY`` : Enable high quality mode which tells the
          compositor to use 4x anisotropic filtering when sampling.
        * ``LAYER_FLAG_TEXTURE_ORIGIN_AT_BOTTOM_LEFT`` : Tell the compositor the
          texture origin is at the bottom left, required for using OpenGL
          textures.
        * ``LAYER_FLAG_HEAD_LOCKED`` : Enable head locking, which forces the
          render layer transformations to be head referenced.

    See Also
    --------
    getLayerEyeFovFlags : Get layer flags.

    Notes
    -----
    * ``LAYER_FLAG_HIGH_QUALITY`` and
      ``LAYER_FLAG_TEXTURE_ORIGIN_AT_BOTTOM_LEFT`` are recommended settings and
      are enabled by default.

    Examples
    --------
    Enable head-locked mode::

        layerFlags = getLayerEyeFovFlags()  # get current flags
        layerFlags |= LAYER_FLAG_HEAD_LOCKED  # set head-locking
        setLayerEyeFovFlags(layerFlags)  # set the flags again

    """
    global _eyeLayer
    _eyeLayer.Header.Flags = <capi.ovrLayerFlags>flags


include "libovr_gl.pxi"


def calcEyePoses(LibOVRPose headPose, object originPose=None):
    """Calculate eye poses using a given head pose.

    Eye poses are derived from the specified head pose, relative eye poses, and
    the scene tracking origin.

    Calculated eye poses are stored and passed to the compositor when
    :func:`endFrame` is called unless ``LAYER_FLAG_HEAD_LOCKED`` is set. You can
    access the computed poses via the :func:`getEyeRenderPose` function. If
    using custom head poses, ensure :func:`setHeadLocked` is ``True`` or the
    ``LAYER_FLAG_HEAD_LOCKED`` render layer flag is set.

    Parameters
    ----------
    headPose : :py:class:`LibOVRPose`
        Head pose.
    originPose : :py:class:`LibOVRPose`, optional
        Optional world origin pose to transform head pose. You can apply
        transformations to this pose to simulate movement through a scene.

    Examples
    --------

    Compute the eye poses from tracker data::

        abs_time = getPredictedDisplayTime()
        tracking_state, calibrated_origin = getTrackingState(abs_time, True)
        headPoseState, status = tracking_state[TRACKED_DEVICE_TYPE_HMD]

        # calculate head pose
        hmd.calcEyePoses(headPoseState.pose)

        # computed render poses appear here
        renderPoseLeft, renderPoseRight = hmd.getEyeRenderPoses()

    Using external data to set the head pose from a motion capture system::

        # rigid body in the scene defining the scene origin
        rbHead = LibOVRPose(*headRb.posOri)
        calcEyePoses(rbHead)

    Note that the external tracker latency might be larger than builtin
    tracking. To get around this, enable forward prediction in your mocap
    software to equal roughly to average `getPredictedDisplayTime() -
    mocapMidExposureTime`, or time integrate poses to mid-frame time.

    """
    global _ptrSession
    global _eyeLayer
    global _eyeRenderPoses
    global _eyeRenderDesc
    global _eyeViewMatrix
    global _eyeProjectionMatrix
    global _eyeViewProjectionMatrix

    cdef capi.ovrPosef[2] hmdToEyePoses
    hmdToEyePoses[0] = _eyeRenderDesc[0].HmdToEyePose
    hmdToEyePoses[1] = _eyeRenderDesc[1].HmdToEyePose

    # calculate the eye poses
    capi.ovr_CalcEyePoses2(headPose.c_data[0], hmdToEyePoses, _eyeRenderPoses)

    # compute the eye transformation matrices from poses
    cdef libovr_math.Vector3f pos, originPos
    cdef libovr_math.Quatf ori, originOri
    cdef libovr_math.Vector3f up
    cdef libovr_math.Vector3f forward
    cdef libovr_math.Matrix4f rm

    # get origin pose components
    if originPose is not None:
        originPos = <libovr_math.Vector3f>(<LibOVRPose>originPose).c_data.Position
        originOri = <libovr_math.Quatf>(<LibOVRPose>originPose).c_data.Orientation
        if not originOri.IsNormalized():  # make sure orientation is normalized
            originOri.Normalize()

    cdef int eye = 0
    for eye in range(capi.ovrEye_Count):
        if originPose is not None:
            pos = originPos + <libovr_math.Vector3f>_eyeRenderPoses[eye].Position
            ori = originOri * <libovr_math.Quatf>_eyeRenderPoses[eye].Orientation
        else:
            pos = <libovr_math.Vector3f>_eyeRenderPoses[eye].Position
            ori = <libovr_math.Quatf>_eyeRenderPoses[eye].Orientation

        if not ori.IsNormalized():  # make sure orientation is normalized
            ori.Normalize()

        rm = libovr_math.Matrix4f(ori)
        up = rm.Transform(libovr_math.Vector3f(0., 1., 0.))
        forward = rm.Transform(libovr_math.Vector3f(0., 0., -1.))
        _eyeViewMatrix[eye] = \
            libovr_math.Matrix4f.LookAtRH(pos, pos + forward, up)
        _eyeViewProjectionMatrix[eye] = \
            _eyeProjectionMatrix[eye] * _eyeViewMatrix[eye]


def getHmdToEyePose(int eye):
    """HMD to eye pose.

    These are the prototype eye poses specified by LibOVR, defined only
    after :func:`create` is called. These poses are referenced to the HMD
    origin. Poses are transformed by calling :func:`calcEyePoses`, updating the
    values returned by :func:`getEyeRenderPose`.

    The horizontal (x-axis) separation of the eye poses are determined by the
    configured lens spacing (slider adjustment). This spacing is supposed to
    correspond to the actual inter-ocular distance (IOD) of the user. You can
    get the IOD used for rendering by adding up the absolute values of the
    x-components of the eye poses, or by multiplying the value of
    :func:`getEyeToNoseDist` by two. Furthermore, the IOD values can be altered,
    prior to calling :func`calcEyePoses`, to override the values specified by
    LibOVR.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.

    Returns
    -------
    tuple (LibOVRPose, LibOVRPose)
        Copy of the HMD to eye pose.

    See Also
    --------
    setHmdToEyePose : Set the HMD to eye pose.

    Examples
    --------
    Get the HMD to eye poses::

        leftPose = getHmdToEyePose(EYE_LEFT)
        rightPose = getHmdToEyePose(EYE_RIGHT)

    """
    global _eyeRenderDesc
    return LibOVRPose.fromPtr(&_eyeRenderDesc[eye].HmdToEyePose)


def setHmdToEyePose(int eye, LibOVRPose eyePose):
    """Set the HMD eye poses.

    This overwrites the values returned by LibOVR and will be used in successive
    calls of :func:`calcEyePoses` to compute eye render poses. Note that the
    poses store the view space translations, not the relative position in the
    scene.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.

    See Also
    --------
    getHmdToEyePose : Get the current HMD to eye pose.

    Examples
    --------
    Set both HMD to eye poses::

        eyePoses = [LibOVRPose((0.035, 0.0, 0.0)), LibOVRPose((-0.035, 0.0, 0.0))]
        for eye in enumerate(eyePoses):
            setHmdToEyePose(eye, eyePoses[eye])

    """
    global _eyeRenderDesc
    _eyeRenderDesc[0].HmdToEyePose = eyePose.c_data[0]


def getEyeRenderPose(int eye):
    """Get eye render poses.

    Pose are those computed by the last :func:`calcEyePoses` call. Returned
    objects are copies of the data stored internally by the session
    instance. These poses are used to derive the view matrix when rendering
    for each eye, and used for visibility culling.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.

    Returns
    -------
    tuple (LibOVRPose, LibOVRPose)
        Copies of the HMD to eye poses for the left and right eye.

    See Also
    --------
    setEyeRenderPose : Set an eye's render pose.

    Examples
    --------

    Get the eye render poses::

        leftPose = getHmdToEyePose(EYE_LEFT)
        rightPose = getHmdToEyePose(EYE_RIGHT)

    Get the left and right view matrices::

        eyeViewMatrices = []
        for eye in enumerate(EYE_COUNT):
            eyeViewMatrices.append(getHmdToEyePose(eye).asMatrix())

    Same as above, but overwrites existing view matrices::

        # identity 4x4 matrices
        eyeViewMatrices = [
            numpy.identity(4, dtype=numpy.float32),
            numpy.identity(4, dtype=numpy.float32)]
        for eye in range(EYE_COUNT):
            getHmdToEyePose(eye).asMatrix(eyeViewMatrices[eye])

    """
    global _eyeRenderPoses
    return LibOVRPose.fromPtr(&_eyeRenderPoses[eye])


def setEyeRenderPose(int eye, LibOVRPose eyePose):
    """Set eye render pose.

    Setting the eye render pose will update the values returned by
    :func:`getEyeRenderPose`.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.

    See Also
    --------
    getEyeRenderPose : Get an eye's render pose.

    """
    global _eyeRenderPoses
    global _eyeViewMatrix
    global _eyeProjectionMatrix
    global _eyeViewProjectionMatrix

    _eyeRenderPoses[eye] = eyePose.c_data[0]

    # re-compute the eye transformation matrices from poses
    cdef libovr_math.Vector3f pos
    cdef libovr_math.Quatf ori
    cdef libovr_math.Vector3f up
    cdef libovr_math.Vector3f forward
    cdef libovr_math.Matrix4f rm

    pos = <libovr_math.Vector3f>_eyeRenderPoses[eye].Position
    ori = <libovr_math.Quatf>_eyeRenderPoses[eye].Orientation

    if not ori.IsNormalized():  # make sure orientation is normalized
        ori.Normalize()

    rm = libovr_math.Matrix4f(ori)
    up = rm.Transform(libovr_math.Vector3f(0., 1., 0.))
    forward = rm.Transform(libovr_math.Vector3f(0., 0., -1.))
    _eyeViewMatrix[eye] = \
        libovr_math.Matrix4f.LookAtRH(pos, pos + forward, up)
    # VP matrix
    _eyeViewProjectionMatrix[eye] = \
        _eyeProjectionMatrix[eye] * _eyeViewMatrix[eye]


def getEyeProjectionMatrix(int eye, np.ndarray[np.float32_t, ndim=2] out=None):
    """Compute the projection matrix.

    The projection matrix is computed by the runtime using the eye FOV
    parameters set with :py:attr:`libovr.LibOVRSession.setEyeRenderFov` calls.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.
    out : `ndarray` or `None`, optional
        Alternative matrix to write values to instead of returning a new one.

    Returns
    -------
    ndarray
        4x4 projection matrix.

    Examples
    --------

    Get the left and right projection matrices as a list::

        eyeProjectionMatrices = []
        for eye in range(EYE_COUNT):
            eyeProjectionMatrices.append(getEyeProjectionMatrix(eye))

    Same as above, but overwrites existing view matrices::

        # identity 4x4 matrices
        eyeProjectionMatrices = [
            numpy.identity(4, dtype=numpy.float32),
            numpy.identity(4, dtype=numpy.float32)]

        # for eye in range(EYE_COUNT) also works
        for eye in enumerate(eyeProjectionMatrices):
            getEyeProjectionMatrix(eye, out=eyeProjectionMatrices[eye])

    Using eye projection matrices with PyOpenGL (fixed-function)::

        P = getEyeProjectionMatrix(eye)
        glMatrixMode(GL.GL_PROJECTION)
        glLoadTransposeMatrixf(P)

    For `Pyglet` (which is the stardard GL interface for `PsychoPy`), you need
    to convert the matrix to a C-types pointer before passing it to
    `glLoadTransposeMatrixf`::

        P = getEyeProjectionMatrix(eye)
        P = P.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        glMatrixMode(GL.GL_PROJECTION)
        glLoadTransposeMatrixf(P)

    If using fragment shaders, the matrix can be passed on to them as such::

        P = getEyeProjectionMatrix(eye)
        P = P.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # after the program was installed in the current rendering state via
        # `glUseProgram` ...
        loc = glGetUniformLocation(program, b"m_Projection")
        glUniformMatrix4fv(loc, 1, GL_TRUE, P)  # `transpose` must be `True`

    """
    global _eyeProjectionMatrix

    cdef np.ndarray[np.float32_t, ndim=2] to_return

    if out is None:
        to_return = np.zeros((4, 4), dtype=np.float32)
    else:
        to_return = out

    # fast copy matrix to numpy array
    cdef float [:, :] mv = to_return
    cdef Py_ssize_t i, j
    cdef Py_ssize_t N = 4
    i = j = 0
    for i in range(N):
        for j in range(N):
            mv[i, j] = _eyeProjectionMatrix[eye].M[i][j]

    return to_return


def getEyeRenderViewport(int eye, np.ndarray[np.int_t, ndim=1] out=None):
    """Get the eye render viewport.

    The viewport defines the region on the swap texture a given eye's image is
    drawn to.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.
    out : `ndarray`, optional
        Optional NumPy array to place values. If None, this function will return
        a new array. Must be dtype=int and length 4.

    Returns
    -------
    ndarray
        Viewport rectangle [x, y, w, h].

    """
    global _eyeLayer
    cdef np.ndarray[np.int_t, ndim=1] to_return

    if out is None:
        to_return = np.zeros((4,), dtype=np.int)
    else:
        to_return = out

    to_return[0] = _eyeLayer.Viewport[eye].Pos.x
    to_return[1] = _eyeLayer.Viewport[eye].Pos.y
    to_return[2] = _eyeLayer.Viewport[eye].Size.w
    to_return[3] = _eyeLayer.Viewport[eye].Size.h

    return to_return


def setEyeRenderViewport(int eye, object values):
    """Set the eye render viewport.

    The viewport defines the region on the swap texture a given eye's image is
    drawn to.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.
    values : array_like
        Viewport rectangle [x, y, w, h].

    Examples
    --------

    Setting the viewports for both eyes on a single swap chain buffer::

        # Calculate the optimal eye buffer sizes for the FOVs, these will define
        # the dimensions of the render target.
        leftBufferSize, rightBufferSize = calcEyeBufferSizes()

        # Define the viewports, which specifies the region on the render target a
        # eye's image will be drawn to and accessed from. Viewports are rectangles
        # defined like [x, y, w, h]. The x-position of the rightViewport is offset
        # by the width of the left viewport.
        leftViewport = [0, 0, leftBufferSize[0], leftBufferSize[1]]
        rightViewport = [leftBufferSize[0], 0,
                         rightBufferSize[0], rightBufferSize[1]]

        # set both viewports
        setEyeRenderViewport(EYE_LEFT, leftViewport)
        setEyeRenderViewport(EYE_RIGHT, rightViewport)

    """
    global _eyeLayer
    _eyeLayer.Viewport[eye].Pos.x = <int>values[0]
    _eyeLayer.Viewport[eye].Pos.y = <int>values[1]
    _eyeLayer.Viewport[eye].Size.w = <int>values[2]
    _eyeLayer.Viewport[eye].Size.h = <int>values[3]


def getEyeViewMatrix(int eye, np.ndarray[np.float32_t, ndim=2] out=None):
    """Compute a view matrix for a specified eye.

    View matrices are derived from the eye render poses calculated by the
    last :func:`calcEyePoses` call or update by :func:`setEyeRenderPose`.

    Parameters
    ----------
    eye: int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.
    out : ndarray or None, optional
        Optional array to write to. Must have ndim=2, dtype=np.float32, and
        shape == (4,4).

    Returns
    -------
    ndarray
        4x4 view matrix. Object `out` will be returned if specified.

    """
    global _eyeViewMatrix
    cdef np.ndarray[np.float32_t, ndim=2] to_return

    if out is None:
        to_return = np.zeros((4, 4), dtype=np.float32)
    else:
        to_return = out

    cdef Py_ssize_t i, j, N
    i = j = 0
    N = 4
    for i in range(N):
        for j in range(N):
            to_return[i, j] = _eyeViewMatrix[eye].M[i][j]

    return to_return
