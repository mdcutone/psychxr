#  =============================================================================
#  libovr_extras.pxi - Misc. types and functions for use with LibOVR
#  =============================================================================
#
#  libovr_extras.pxi
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

def cullPose(int eye, LibOVRPose pose):
    """Test if a pose's bounding box or position falls outside of an eye's view
    frustum.

    Poses can be assigned bounding boxes which enclose any 3D models associated
    with them. A model is not visible if all the corners of the bounding box
    fall outside the viewing frustum. Therefore any primitives (i.e. triangles)
    associated with the pose can be culled during rendering to reduce CPU/GPU
    workload.

    If `pose` does not have a valid bounding box (:py:class:`LibOVRBounds`)
    assigned to its :py:attr:`~LibOVRPose.bounds` attribute, this function will
    test is if the position of `pose` is outside the view frustum.

    Parameters
    ----------
    eye : int
        Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.
    pose : LibOVRPose
        Pose to test.

    Returns
    -------
    bool
        ``True`` if the pose's bounding box is not visible to the given `eye`
        and should be culled during rendering.

    Examples
    --------
    Check if a pose should be culled (needs to be done for each eye)::

        cullModel = cullPose(eye, pose)
        if not cullModel:
            # ... OpenGL calls to draw the model here ...

    Notes
    -----
    * Frustums used for testing are defined by the current render FOV for the
      eye (see: :func:`getEyeRenderFov` and :func:`getEyeSetFov`).
    * This function does not test if an object is occluded by another within the
      frustum. If an object is completely occluded, it will still be fully
      rendered, and nearer object will be drawn on-top of it. A trick to
      improve performance in this case is to use ``glDepthFunc(GL_LEQUAL)`` with
      ``glEnable(GL_DEPTH_TEST)`` and render objects from nearest to farthest
      from the head pose. This will reject fragment color calculations for
      occluded locations.

    """
    # This is based on OpenXR's function `XrMatrix4x4f_CullBounds` found in
    # `xr_linear.h`
    global _eyeViewProjectionMatrix

    cdef libovr_math.Bounds3f* bbox
    cdef libovr_math.Vector4f test_point
    cdef libovr_math.Vector4f[8] corners
    cdef Py_ssize_t i

    # compute the MVP matrix to transform poses into HCS
    cdef libovr_math.Matrix4f mvp = \
        _eyeViewProjectionMatrix[eye] * \
        libovr_math.Matrix4f(<libovr_math.Posef>pose.c_data[0])

    if pose.bounds is not None:
        # has a bounding box
        bbox = pose._bbox.c_data

        # bounding box is cleared/not valid, don't cull
        if bbox.b[1].x <= bbox.b[0].x and \
                bbox.b[1].y <= bbox.b[0].y and \
                bbox.b[1].z <= bbox.b[0].z:
            return False

        # compute the corners of the bounding box
        for i in range(8):
            test_point = libovr_math.Vector4f(
                bbox.b[1].x if (i & 1) else bbox.b[0].x,
                bbox.b[1].y if (i & 2) else bbox.b[0].y,
                bbox.b[1].z if (i & 4) else bbox.b[0].z,
                1.0)
            corners[i] = mvp.Transform(test_point)

        # If any of these loops exit normally, the bounding box is completely
        # off to one side of the viewing frustum
        for i in range(8):
            if corners[i].x > -corners[i].w:
                break
        else:
            return True

        for i in range(8):
            if corners[i].x < corners[i].w:
                break
        else:
            return True

        for i in range(8):
            if corners[i].y > -corners[i].w:
                break
        else:
            return True

        for i in range(8):
            if corners[i].y < corners[i].w:
                break
        else:
            return True

        for i in range(8):
            if corners[i].z > -corners[i].w:
                break
        else:
            return True

        for i in range(8):
            if corners[i].z < corners[i].w:
                break
        else:
            return True
    else:
        # no bounding box, cull position of the pose
        test_point = mvp.Transform(
            libovr_math.Vector4f(
                pose.c_data.Position.x,
                pose.c_data.Position.y,
                pose.c_data.Position.z,
                1.0))

        if test_point.x <= -test_point.w:
            return True
        elif test_point.x >= test_point.w:
            return True
        elif test_point.y <= -test_point.w:
            return True
        elif test_point.y >= test_point.w:
            return True
        elif test_point.z <= -test_point.w:
            return True
        elif test_point.z >= test_point.w:
            return True

    return False