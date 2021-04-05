#  =============================================================================
#  libovr_pose.pxi - Wrapper extensions for rigid body poses
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
import ctypes


cdef class LibOVRPose(object):
    """Class for representing rigid body poses.

    This class is an abstract representation of a rigid body pose, where the
    position of the body in a scene is represented by a vector/coordinate and
    the orientation with a quaternion. LibOVR uses this format for poses to
    represent the posture of tracked devices (e.g. HMD, touch controllers, etc.)
    and other objects in a VR scene. There are many class methods and properties
    provided to handle accessing, manipulating, and interacting with poses.
    Rigid body poses assume a right-handed coordinate system (-Z is forward and
    +Y is up).

    Poses can be manipulated using operators such as ``*``, ``~``, and ``*=``.
    One pose can be transformed by another by multiplying them using the
    ``*`` operator::

        newPose = pose1 * pose2

    The above code returns `pose2` transformed by `pose1`, putting `pose2` into
    the reference frame of `pose1`. Using the inplace multiplication operator
    ``*=``, you can transform a pose into another reference frame without making
    a copy. One can get the inverse of a pose by using the ``~`` operator::

        poseInv = ~pose

    Poses can be converted to 4x4 transformation matrices with `getModelMatrix`,
    `getViewMatrix`, and `getNormalMatrix`. One can use these matrices when
    rendering to transform the vertices and normals of a model associated with
    the pose by passing the matrices to OpenGL. The `ctypes` property eliminates
    the need to copy data by providing pointers to data stored by instances of
    this class. This is useful for some Python OpenGL libraries which require
    matrices to be provided as pointers.

    Bounding boxes can be given to poses by assigning a :class:`LibOVRBounds`
    instance to the `bounds` attribute. Bounding boxes are used for visibility
    culling, to determine if a mesh associated with a pose is visible to the
    viewer and whether it should be drawn or not. This aids in reducing
    workload for the application by only rendering objects that are visible from
    a given eye's view.

    Parameters
    ----------
    pos : array_like
        Initial position vector (x, y, z).
    ori : array_like
        Initial orientation quaternion (x, y, z, w).

    """
    cdef capi.ovrPosef* c_data
    cdef bint ptr_owner

    cdef np.ndarray _pos
    cdef np.ndarray _ori

    cdef libovr_math.Matrix4f _modelMatrix
    cdef libovr_math.Matrix4f _invModelMatrix
    cdef libovr_math.Matrix4f _normalMatrix
    cdef libovr_math.Matrix4f _viewMatrix
    cdef libovr_math.Matrix4f _invViewMatrix

    cdef np.ndarray _modelMatrixArr
    cdef np.ndarray _invModelMatrixArr
    cdef np.ndarray _normalMatrixArr
    cdef np.ndarray _viewMatrixArr
    cdef np.ndarray _invViewMatrixArr
    cdef dict _ptrMatrices

    cdef bint _matrixNeedsUpdate

    cdef LibOVRBounds _bbox

    def __init__(self, pos=(0., 0., 0.), ori=(0., 0., 0., 1.)):
        self._new_struct(pos, ori)

    def __cinit__(self, *args, **kwargs):
        self.ptr_owner = False

        # make sure we have proxy objects
        self._modelMatrixArr = _wrap_Matrix4f_as_ndarray(&self._modelMatrix)
        self._invModelMatrixArr = _wrap_Matrix4f_as_ndarray(&self._invModelMatrix)
        self._normalMatrixArr = _wrap_Matrix4f_as_ndarray(&self._normalMatrix)
        self._viewMatrixArr = _wrap_Matrix4f_as_ndarray(&self._viewMatrix)
        self._invViewMatrixArr = _wrap_Matrix4f_as_ndarray(&self._invViewMatrix)

        # ctypes pointers to matrices
        self._ptrMatrices = {
            'modelMatrix': self._modelMatrixArr.ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)),
            'inverseModelMatrix': self._invModelMatrixArr.ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)),
            'viewMatrix': self._viewMatrixArr.ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)),
            'inverseViewMatrix': self._invViewMatrixArr.ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)),
            'normalMatrix': self._normalMatrixArr.ctypes.data_as(
                ctypes.POINTER(ctypes.c_float))}

    @staticmethod
    cdef LibOVRPose fromPtr(capi.ovrPosef* ptr, bint owner=False):
        cdef LibOVRPose wrapper = LibOVRPose.__new__(LibOVRPose)
        wrapper.c_data = ptr
        wrapper.ptr_owner = owner

        wrapper._pos = _wrap_ovrVector3f_as_ndarray(&ptr.Position)
        wrapper._ori = _wrap_ovrQuatf_as_ndarray(&ptr.Orientation)
        wrapper._bbox = None
        wrapper._matrixNeedsUpdate = True

        return wrapper

    cdef void _new_struct(self, object pos, object ori):
        if self.c_data is not NULL:
            return

        cdef capi.ovrPosef* ptr = <capi.ovrPosef*>PyMem_Malloc(sizeof(capi.ovrPosef))

        if ptr is NULL:
            raise MemoryError

        # clear memory
        ptr.Position.x = <float>pos[0]
        ptr.Position.y = <float>pos[1]
        ptr.Position.z = <float>pos[2]
        ptr.Orientation.x = <float>ori[0]
        ptr.Orientation.y = <float>ori[1]
        ptr.Orientation.z = <float>ori[2]
        ptr.Orientation.w = <float>ori[3]

        self.c_data = ptr
        self.ptr_owner = True

        self._pos = _wrap_ovrVector3f_as_ndarray(&ptr.Position)
        self._ori = _wrap_ovrQuatf_as_ndarray(&ptr.Orientation)
        self._bbox = None
        self._matrixNeedsUpdate = True

    def __dealloc__(self):
        # don't do anything crazy like set c_data=NULL without deallocating!
        if self.c_data is not NULL:
            if self.ptr_owner is True:
                PyMem_Free(self.c_data)
                self.c_data = NULL

    def __repr__(self):
        return f'LibOVRPose(pos={repr(self.pos)}, ori={repr(self.ori)})'

    def __mul__(LibOVRPose a, LibOVRPose b):
        """Multiplication operator (*) to combine poses.
        """
        cdef capi.ovrPosef* ptr = <capi.ovrPosef*>PyMem_Malloc(sizeof(capi.ovrPosef))

        if ptr is NULL:
            raise MemoryError

        cdef libovr_math.Posef pose_r = \
            <libovr_math.Posef>a.c_data[0] * <libovr_math.Posef>b.c_data[0]

        # copy into
        ptr[0] = <capi.ovrPosef>pose_r
        return LibOVRPose.fromPtr(ptr, True)

    def __imul__(self, LibOVRPose other):
        """Multiplication operator (*=) to combine poses.
        """
        cdef libovr_math.Posef this_pose = <libovr_math.Posef>self.c_data[0]
        self.c_data[0] = <capi.ovrPosef>(
                <libovr_math.Posef>other.c_data[0] * this_pose)

        self._matrixNeedsUpdate = True

        return self

    def __invert__(self):
        """Invert operator (~) to invert a pose."""
        return self.inverted()

    def __eq__(self, LibOVRPose other):
        """Equality operator (==) for two poses.

        The tolerance of the comparison is defined by the Oculus SDK as 1e-5.

        """
        return (<libovr_math.Posef>self.c_data[0]).IsEqual(
            <libovr_math.Posef>other.c_data[0], <float>1e-5)

    def __ne__(self, LibOVRPose other):
        """Inequality operator (!=) for two poses.

        The tolerance of the comparison is defined by the Oculus SDK as 1e-5.

        """
        return not (<libovr_math.Posef>self.c_data[0]).IsEqual(
            <libovr_math.Posef>other.c_data[0], <float>1e-5)

    def __deepcopy__(self, memo=None):
        # create a new object with a copy of the data stored in c_data
        # allocate new struct
        cdef capi.ovrPosef* ptr = <capi.ovrPosef*>PyMem_Malloc(sizeof(capi.ovrPosef))

        if ptr is NULL:
            raise MemoryError

        cdef LibOVRPose to_return = LibOVRPose.fromPtr(ptr, owner=True)

        # copy over data
        to_return.c_data[0] = self.c_data[0]
        if memo is not None:
            memo[id(self)] = to_return

        return to_return

    def copy(self):
        """Create an independent copy of this object.

        Returns
        -------
        RigidBodyPose
            Copy of this pose.

        """
        cdef LibOVRPose toReturn = LibOVRPose()
        (<LibOVRPose>toReturn).c_data[0] = self.c_data[0]

        return toReturn

    @property
    def bounds(self):
        """Bounding object associated with this pose."""
        return self._bbox

    @bounds.setter
    def bounds(self, LibOVRBounds value):
        self._bbox = value

    def isEqual(self, LibOVRPose pose, float tolerance=1e-5):
        """Check if poses are close to equal in position and orientation.

        Same as using the equality operator (==) on poses, but you can specify
        and arbitrary value for `tolerance`.

        Parameters
        ----------
        pose : LibOVRPose
            The other pose.
        tolerance : float, optional
            Tolerance for the comparison, default is 1e-5 as defined in
            `OVR_MATH.h`.

        Returns
        -------
        bool
            True if pose components are within `tolerance` from this pose.

        """
        return (<libovr_math.Posef>self.c_data[0]).IsEqual(
            <libovr_math.Posef>pose.c_data[0], tolerance)

    def duplicate(self):
        """Create a deep copy of this object.

        Same as calling `copy.deepcopy` on an instance.

        Returns
        -------
        LibOVRPose
            An independent copy of this object.

        """
        return self.__deepcopy__()

    def __repr__(self):
        return \
            "LibOVRPose(pos=({px}, {py}, {pz}), ori=({rx}, {ry}, {rz}, {rw}))".format(
                px=self.c_data[0].Position.x,
                py=self.c_data[0].Position.y,
                pz=self.c_data[0].Position.z,
                rx=self.c_data[0].Orientation.x,
                ry=self.c_data[0].Orientation.y,
                rz=self.c_data[0].Orientation.z,
                rw=self.c_data[0].Orientation.w)

    def __str__(self):
        return \
            "LibOVRPose(pos=({px}, {py}, {pz}), ori=({rx}, {ry}, {rz}, {rw}))".format(
                px=self.c_data[0].Position.x,
                py=self.c_data[0].Position.y,
                pz=self.c_data[0].Position.z,
                rx=self.c_data[0].Orientation.x,
                ry=self.c_data[0].Orientation.y,
                rz=self.c_data[0].Orientation.z,
                rw=self.c_data[0].Orientation.w)

    def setIdentity(self):
        """Clear this pose's translation and orientation."""
        self.c_data[0].Position.x = 0.0
        self.c_data[0].Position.y = 0.0
        self.c_data[0].Position.z = 0.0

        self.c_data[0].Orientation.x = 0.0
        self.c_data[0].Orientation.y = 0.0
        self.c_data[0].Orientation.z = 0.0
        self.c_data[0].Orientation.w = 1.0

    def _updateMatrices(self):
        """Update model, inverse, and normal matrices due to an attribute change.
        """
        if not self._matrixNeedsUpdate:
            return

        self._modelMatrix = libovr_math.Matrix4f(<libovr_math.Posef>self.c_data[0])
        self._invModelMatrix = self._modelMatrix.InvertedHomogeneousTransform()
        self._normalMatrix = self._invModelMatrix.Transposed()

        # update the view matrix
        cdef libovr_math.Vector3f pos = <libovr_math.Vector3f>self.c_data.Position
        cdef libovr_math.Quatf ori = <libovr_math.Quatf>self.c_data.Orientation

        if not ori.IsNormalized():  # make sure orientation is normalized
            ori.Normalize()

        cdef libovr_math.Matrix4f rm = libovr_math.Matrix4f(ori)
        cdef libovr_math.Vector3f up = rm.Transform(
            libovr_math.Vector3f(0., 1., 0.))
        cdef libovr_math.Vector3f forward = rm.Transform(
            libovr_math.Vector3f(0., 0., -1.))
        self._viewMatrix = libovr_math.Matrix4f.LookAtRH(pos, pos + forward, up)
        self._invViewMatrix = self._viewMatrix.InvertedHomogeneousTransform()

        self._matrixNeedsUpdate = False

    @property
    def pos(self):
        """ndarray : Position vector [X, Y, Z].

        Examples
        --------

        Set the position of the pose::

            myPose.pos = [0., 0., -1.5]

        Get the x, y, and z coordinates of a pose::

            x, y, z = myPose.pos

        The `ndarray` returned by `pos` directly references the position field
        data in the pose data structure (`ovrPosef`). Updating values will
        directly edit the values in the structure. For instance, you can specify
        a component of a pose's position::

            myPose.pos[2] = -10.0  # z = -10.0

        Assigning `pos` a name will create a reference to that `ndarray` which
        can edit values in the structure::

            p = myPose.pos
            p[1] = 1.5  # sets the Y position of 'myPose' to 1.5

        """
        self._matrixNeedsUpdate = True
        return self._pos

    @pos.setter
    def pos(self, object value):
        self._matrixNeedsUpdate = True
        self._pos[:] = value

    def getPos(self, np.ndarray[np.float32_t, ndim=1] out=None):
        """Position vector X, Y, Z.

        The returned object is a NumPy array which contains a copy of the data
        stored in an internal structure (ovrPosef). The array is conformal with
        the internal data's type (float32) and size (length 3).

        Parameters
        ----------
        out : ndarray or None
            Optional array to write values to. Must have a float32 data type.

        Returns
        -------
        ndarray
            Position coordinate of this pose.

        Examples
        --------

        Get the position coordinates::

            x, y, z = myPose.getPos()  # Python float literals
            # ... or ...
            pos = myPose.getPos()  # NumPy array shape=(3,) and dtype=float32

        Write the position to an existing array by specifying `out`::

            position = numpy.zeros((3,), dtype=numpy.float32)  # mind the dtype!
            myPose.getPos(position)  # position now contains myPose.pos

        You can also pass a view/slice to `out`::

            coords = numpy.zeros((100,3,), dtype=numpy.float32)  # big array
            myPose.getPos(coords[42,:])  # row 42

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = out

        toReturn[0] = self.c_data[0].Position.x
        toReturn[1] = self.c_data[0].Position.y
        toReturn[2] = self.c_data[0].Position.z

        self._matrixNeedsUpdate = True

        return toReturn

    def setPos(self, object pos):
        """Set the position of the pose in a scene.

        Parameters
        ----------
        pos : array_like
            Position vector [X, Y, Z].

        """
        self.c_data[0].Position.x = <float>pos[0]
        self.c_data[0].Position.y = <float>pos[1]
        self.c_data[0].Position.z = <float>pos[2]
        self._matrixNeedsUpdate = True

    @property
    def ori(self):
        """ndarray : Orientation quaternion [X, Y, Z, W].
        """
        self._matrixNeedsUpdate = True
        return self._ori

    @ori.setter
    def ori(self, object value):
        self._matrixNeedsUpdate = True
        self._ori[:] = value

    def getOri(self, np.ndarray[np.float32_t, ndim=1] out=None):
        """Orientation quaternion X, Y, Z, W. Components X, Y, Z are imaginary
        and W is real.

        The returned object is a NumPy array which references data stored in an
        internal structure (ovrPosef). The array is conformal with the internal
        data's type (float32) and size (length 4).

        Parameters
        ----------
        out : ndarray  or None
            Optional array to write values to. Must have a float32 data type.

        Returns
        -------
        ndarray
            Orientation quaternion of this pose.

        Notes
        -----

        * The orientation quaternion should be normalized.

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((4,), dtype=np.float32)
        else:
            toReturn = out

        toReturn[0] = self.c_data[0].Orientation.x
        toReturn[1] = self.c_data[0].Orientation.y
        toReturn[2] = self.c_data[0].Orientation.z
        toReturn[3] = self.c_data[0].Orientation.w

        self._matrixNeedsUpdate = True

        return toReturn

    def setOri(self, object ori):
        """Set the orientation of the pose in a scene.

        Parameters
        ----------
        ori : array_like
            Orientation quaternion [X, Y, Z, W].

        """
        self.c_data[0].Orientation.x = <float>ori[0]
        self.c_data[0].Orientation.y = <float>ori[1]
        self.c_data[0].Orientation.z = <float>ori[2]
        self.c_data[0].Orientation.w = <float>ori[3]

        self._matrixNeedsUpdate = True

    @property
    def posOri(self):
        """tuple (ndarray, ndarray) : Position vector and orientation
        quaternion.
        """
        self._matrixNeedsUpdate = True
        return self.pos, self.ori

    @posOri.setter
    def posOri(self, object value):
        self._matrixNeedsUpdate = True
        self.pos = value[0]
        self.ori = value[1]

    @property
    def at(self):
        """ndarray : Forward vector of this pose (-Z is forward)
        (read-only).
        """
        return self.getAt()

    def getAt(self, np.ndarray[np.float32_t, ndim=1] out=None):
        """Get the `at` vector for this pose.

        Parameters
        ----------
        out : ndarray or None
            Optional array to write values to. Must have shape (3,) and a float32
            data type.

        Returns
        -------
        ndarray
            The vector for `at`.

        Examples
        --------
        Setting the listener orientation for 3D positional audio (PyOpenAL)::

            myListener.set_orientation((*myPose.getAt(), *myPose.getUp()))

        See Also
        --------
        getUp : Get the `up` vector.

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = out

        cdef libovr_math.Vector3f at = \
            (<libovr_math.Posef>self.c_data[0]).TransformNormal(
                libovr_math.Vector3f(0.0, 0.0, -1.0))

        toReturn[0] = at.x
        toReturn[1] = at.y
        toReturn[2] = at.z

        return toReturn

    @property
    def up(self):
        """ndarray : Up vector of this pose (+Y is up) (read-only)."""
        return self.getUp()

    def getUp(self, np.ndarray[np.float32_t, ndim=1] out=None):
        """Get the 'up' vector for this pose.

        Parameters
        ----------
        out : ndarray, optional
            Optional array to write values to. Must have shape (3,) and a float32
            data type.

        Returns
        -------
        ndarray
            The vector for up.

        Examples
        --------
        Using the `up` vector with gluLookAt::

            up = myPose.getUp()  # myPose.up also works
            center = myPose.pos
            target = targetPose.pos  # some target pose
            gluLookAt(*(*up, *center, *target))

        See Also
        --------
        getAt : Get the `up` vector.

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = out

        cdef libovr_math.Vector3f up = \
            (<libovr_math.Posef>self.c_data[0]).TransformNormal(
                libovr_math.Vector3f(0.0, 1.0, 0.0))

        toReturn[0] = up.x
        toReturn[1] = up.y
        toReturn[2] = up.z

        return toReturn

    def getOriAxisAngle(self, degrees=True):
        """The axis and angle of rotation for this pose's orientation.

        Parameters
        ----------
        degrees : bool, optional
            Return angle in degrees. Default is ``True``.

        Returns
        -------
        tuple (ndarray, float)
            Axis and angle.

        """
        cdef libovr_math.Vector3f axis
        cdef float angle
        cdef np.ndarray[np.float32_t, ndim=1] ret_axis = \
            np.zeros((3,), dtype=np.float32)

        (<libovr_math.Quatf>self.c_data.Orientation).GetAxisAngle(&axis, &angle)
        ret_axis[0] = axis.x
        ret_axis[1] = axis.y
        ret_axis[2] = axis.z

        if degrees:
            angle *= RAD_TO_DEGF

        return angle, ret_axis

    def setOriAxisAngle(self, object axis, float angle, bint degrees=True):
        """Set the orientation of this pose using an axis and angle.

        Parameters
        ----------
        axis : array_like
            Axis of rotation [rx, ry, rz].
        angle : float
            Angle of rotation.
        degrees : bool, optional
            Specify ``True`` if `angle` is in degrees, or else it will be
            treated as radians. Default is ``True``.

        """
        cdef libovr_math.Vector3f axis3f = \
            libovr_math.Vector3f(<float>axis[0], <float>axis[1], <float>axis[2])

        if degrees:
            angle *= DEG_TO_RADF

        self.c_data.Orientation = \
            <capi.ovrQuatf>libovr_math.Quatf(axis3f, angle)

        self._matrixNeedsUpdate = True

    def turn(self, object axis, float angle, bint degrees=True):
        """Turn (or rotate) this pose about an axis. Successive calls of `turn`
        are cumulative.

        Parameters
        ----------
        axis : array_like
            Axis of rotation [rx, ry, rz].
        angle : float
            Angle of rotation.
        degrees : bool, optional
            Specify ``True`` if `angle` is in degrees, or else it will be
            treated as radians. Default is ``True``.

        """
        cdef libovr_math.Vector3f axis3f = \
            libovr_math.Vector3f(<float>axis[0], <float>axis[1], <float>axis[2])

        if degrees:
            angle *= DEG_TO_RADF

        self.c_data.Orientation = <capi.ovrQuatf>(
                <libovr_math.Quatf>self.c_data.Orientation *
                libovr_math.Quatf(axis3f, angle))

        self._matrixNeedsUpdate = True

    def getYawPitchRoll(self, LibOVRPose refPose=None, bint degrees=True, np.ndarray[np.float32_t] out=None):
        """Get the yaw, pitch, and roll of the orientation quaternion.

        Parameters
        ----------
        refPose : LibOVRPose, optional
            Reference pose to compute angles relative to. If `None` is
            specified, computed values are referenced relative to the world
            axes.
        degrees : bool, optional
            Return angle in degrees. Default is ``True``.
        out : ndarray
            Alternative place to write yaw, pitch, and roll values. Must have
            shape (3,) and a float32 data type.

        Returns
        -------
        ndarray
            Yaw, pitch, and roll of the pose in degrees.

        Notes
        -----

        * Uses ``OVR::Quatf.GetYawPitchRoll`` which is part of the Oculus PC
          SDK.

        """
        cdef float yaw, pitch, roll
        cdef libovr_math.Posef inPose = <libovr_math.Posef>self.c_data[0]
        cdef libovr_math.Posef invRef

        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = out

        if refPose is not None:
            invRef = (<libovr_math.Posef>refPose.c_data[0]).Inverted()
            inPose = invRef * inPose

        inPose.Rotation.GetYawPitchRoll(&yaw, &pitch, &roll)

        toReturn[0] = yaw
        toReturn[1] = pitch
        toReturn[2] = roll

        return np.degrees(toReturn) if degrees else toReturn

    def getOriAngle(self, bint degrees=True):
        """Get the angle of this pose's orientation.

        Parameters
        ----------
        degrees : bool, optional
            Return angle in degrees. Default is ``True``.

        Returns
        -------
        float
            Angle of quaternion `ori`.

        """
        cdef float to_return = \
            (<libovr_math.Quatf>self.c_data.Orientation).Angle()

        return to_return * RAD_TO_DEGF if degrees else to_return

    def alignTo(self, object alignTo):
        """Align this pose to another point or pose.

        This sets the orientation of this pose to one which orients the forward
        axis towards `alignTo`.

        Parameters
        ----------
        alignTo : array_like or LibOVRPose
            Position vector [x, y, z] or pose to align to.

        """
        cdef libovr_math.Vector3f targ
        cdef libovr_math.Posef* pose = <libovr_math.Posef*>self.c_data

        if isinstance(alignTo, LibOVRPose):
            targ = <libovr_math.Vector3f>(<LibOVRPose>alignTo).c_data[0].Position
        else:
            targ = libovr_math.Vector3f(
                <float>alignTo[0], <float>alignTo[1], <float>alignTo[2])

        cdef libovr_math.Vector3f fwd = libovr_math.Vector3f(0, 0, -1)
        targ = pose.InverseTransform(targ)
        targ.Normalize()

        self.c_data.Orientation = \
            <capi.ovrQuatf>(pose.Rotation * libovr_math.Quatf.Align(targ, fwd))

        self._matrixNeedsUpdate = True

    def getSwingTwist(self, object twistAxis):
        """Swing and twist decomposition of this pose's rotation quaternion.

        Where twist is a quaternion which rotates about `twistAxis` and swing is
        perpendicular to that axis. When multiplied, the quaternions return the
        original quaternion at `ori`.

        Parameters
        ----------
        twistAxis : array_like
            World referenced twist axis [ax, ay, az].

        Returns
        -------
        tuple
            Swing and twist quaternions [x, y, z, w].

        Examples
        --------
        Get the swing and twist quaternions about the `up` direction of this
        pose::

            swing, twist = myPose.getSwingTwist(myPose.up)

        """
        cdef libovr_math.Vector3f axis = libovr_math.Vector3f(
            <float>twistAxis[0], <float>twistAxis[1], <float>twistAxis[2])

        cdef libovr_math.Quatf qtwist
        cdef libovr_math.Quatf qswing = \
            (<libovr_math.Quatf>self.c_data.Orientation).GetSwingTwist(
                axis, &qtwist)

        cdef np.ndarray[np.float32_t, ndim=1] rswing = \
            np.zeros((4,), dtype=np.float32)

        rswing[0] = qswing.x
        rswing[1] = qswing.y
        rswing[2] = qswing.z
        rswing[3] = qswing.w

        cdef np.ndarray[np.float32_t, ndim=1] rtwist = \
            np.zeros((4,), dtype=np.float32)

        rtwist[0] = qtwist.x
        rtwist[1] = qtwist.y
        rtwist[2] = qtwist.z
        rtwist[3] = qtwist.w

        return rswing, rtwist

    def getAngleTo(self, object target, object dir=(0., 0., -1.), bint degrees=True):
        """Get the relative angle to a point in world space from the `dir`
        vector in the local coordinate system of this pose.

        Parameters
        ----------
        target : LibOVRPose or array_like
            Pose or point [x, y, z].
        dir : array_like
            Direction vector [x, y, z] within the reference frame of the pose to
            compute angle from. Default is forward along the -Z axis
            (0., 0., -1.).
        degrees : bool, optional
            Return angle in degrees if ``True``, else radians. Default is
            ``True``.

        Returns
        -------
        float
            Angle between the forward vector of this pose and the target. Values
            are always positive.

        Examples
        --------
        Get the angle in degrees between the pose's -Z axis (default) and a
        point::

            point = [2, 0, -4]
            angle = myPose.getAngleTo(point)

        Get the angle from the `up` direction to the point in radians::

            upAxis = (0., 1., 0.)
            angle = myPose.getAngleTo(point, dir=upAxis, degrees=False)

        """
        cdef libovr_math.Vector3f targ

        if isinstance(target, LibOVRPose):
            targ = <libovr_math.Vector3f>(<LibOVRPose>target).c_data[0].Position
        else:
            targ = libovr_math.Vector3f(
                <float>target[0], <float>target[1], <float>target[2])

        cdef libovr_math.Vector3f direction = libovr_math.Vector3f(
            <float>dir[0], <float>dir[1], <float>dir[2])

        targ = (<libovr_math.Posef>self.c_data[0]).InverseTransform(targ)
        cdef float angle = direction.Angle(targ)

        return angle * RAD_TO_DEGF if degrees else angle

    def getAzimuthElevation(self, object target, bint degrees=True):
        """Get the azimuth and elevation angles of a point relative to this
        pose's forward direction (0., 0., -1.).

        Parameters
        ----------
        target : LibOVRPose or array_like
            Pose or point [x, y, z].
        degrees : bool, optional
            Return angles in degrees if ``True``, else radians. Default is
            ``True``.

        Returns
        -------
        tuple (float, float)
            Azimuth and elevation angles of the target point. Values are
            signed.

        """
        cdef libovr_math.Vector3f targ = libovr_math.Vector3f()

        if isinstance(target, LibOVRPose):
            targ = <libovr_math.Vector3f>(<LibOVRPose>target).c_data[0].Position
        else:
            targ = libovr_math.Vector3f(
                <float>target[0], <float>target[1], <float>target[2])

        # put point into reference frame of pose
        targ = (<libovr_math.Posef>self.c_data[0]).InverseTransform(targ)

        cdef float az = atan2(targ.x, -targ.z)
        cdef float el = atan2(targ.y, -targ.z)

        az = az * RAD_TO_DEGF if degrees else az
        el = el * RAD_TO_DEGF if degrees else el

        return az, el

    @property
    def modelMatrix(self):
        """Pose as a 4x4 homogeneous transformation matrix."""
        self._updateMatrices()

        return self._modelMatrixArr

    @property
    def inverseModelMatrix(self):
        """Pose as a 4x4 homogeneous inverse transformation matrix."""
        self._updateMatrices()

        return self._invModelMatrixArr

    @property
    def normalMatrix(self):
        """Normal matrix for transforming normals of meshes associated with
        poses."""
        self._updateMatrices()

        return self._normalMatrixArr

    def getModelMatrix(self, bint inverse=False, np.ndarray[np.float32_t, ndim=2] out=None):
        """Get this pose as a 4x4 transformation matrix.

        Parameters
        ----------
        inverse : bool
            If ``True``, return the inverse of the matrix.
        out : ndarray, optional
            Alternative place to write the matrix to values. Must be a `ndarray`
            of shape (4, 4,) and have a data type of float32. Values are written
            assuming row-major order.

        Returns
        -------
        ndarray
            4x4 transformation matrix.

        Notes
        -----
        * This function create a new `ndarray` with data copied from cache. Use
          the `modelMatrix` or `inverseModelMatrix` attributes for direct cache
          memory access.

        Examples
        --------
        Using view matrices with PyOpenGL (fixed-function)::

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glMultTransposeMatrixf(myPose.getModelMatrix())
            # run draw commands ...
            glPopMatrix()

        For `Pyglet` (which is the standard GL interface for `PsychoPy`), you
        need to convert the matrix to a C-types pointer before passing it to
        `glLoadTransposeMatrixf`::

            M = myPose.getModelMatrix().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glMultTransposeMatrixf(M)
            # run draw commands ...
            glPopMatrix()

        If using fragment shaders, the matrix can be passed on to them as such::

            M = myPose.getModelMatrix().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            M = M.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            # after the program was installed in the current rendering state via
            # `glUseProgram` ...
            loc = glGetUniformLocation(program, b"m_Model")
            glUniformMatrix4fv(loc, 1, GL_TRUE, P)  # `transpose` must be `True`

        """
        self._updateMatrices()

        if out is None:
            if not inverse:
                return self._modelMatrixArr.copy()
            else:
                return self._invModelMatrixArr.copy()

        cdef libovr_math.Matrix4f* m = NULL

        if not inverse:
            m = &self._modelMatrix
        else:
            m = &self._invModelMatrix

        cdef Py_ssize_t i, j, N
        i = j = 0
        N = 4
        for i in range(N):
            for j in range(N):
                out[i, j] = m.M[i][j]

        return out

    @property
    def viewMatrix(self):
        """View matrix."""
        self._updateMatrices()

        return self._viewMatrixArr

    @property
    def inverseViewMatrix(self):
        """View matrix inverse."""
        self._updateMatrices()

        return self._viewMatrixArr

    def getViewMatrix(self, bint inverse=False, np.ndarray[np.float32_t, ndim=2] out=None):
        """Convert this pose into a view matrix.

        Creates a view matrix which transforms points into eye space using the
        current pose as the eye position in the scene. Furthermore, you can use
        view matrices for rendering shadows if light positions are defined
        as `LibOVRPose` objects. Using :func:`calcEyePoses` and
        :func:`getEyeViewMatrix` are preferred when rendering VR scenes since
        features like visibility culling are not available otherwise.

        Parameters
        ----------
        inverse : bool, optional
            Return the inverse of the view matrix. Default it ``False``.
        out : ndarray, optional
            Alternative place to write the matrix to values. Must be a `ndarray`
            of shape (4, 4,) and have a data type of float32. Values are written
            assuming row-major order.

        Returns
        -------
        ndarray
            4x4 view matrix derived from the pose.

        Notes
        -----
        * This function create a new `ndarray` with data copied from cache. Use
          the `viewMatrix` attribute for direct cache memory access.

        Examples
        --------
        Compute eye poses from a head pose and compute view matrices::

            iod = 0.062  # 63 mm
            headPose = LibOVRPose((0., 1.5, 0.))  # 1.5 meters up from origin
            leftEyePose = LibOVRPose((-(iod / 2.), 0., 0.))
            rightEyePose = LibOVRPose((iod / 2., 0., 0.))

            # transform eye poses relative to head poses
            leftEyeRenderPose = headPose * leftEyePose
            rightEyeRenderPose = headPose * rightEyePose

            # compute view matrices
            eyeViewMatrix = [leftEyeRenderPose.getViewMatrix(),
                             rightEyeRenderPose.getViewMatrix()]

        """
        self._updateMatrices()

        if out is None:
            if not inverse:
                return self._viewMatrixArr.copy()
            else:
                return self._invViewMatrixArr.copy()

        cdef libovr_math.Matrix4f* m = NULL

        if not inverse:
            m = &self._viewMatrix
        else:
            m = &self._invViewMatrix

        cdef Py_ssize_t i, j, N
        i = j = 0
        N = 4
        for i in range(N):
            for j in range(N):
                out[i, j] = m.M[i][j]

        return out

    def getNormalMatrix(self, np.ndarray[np.float32_t, ndim=2] out=None):
        """Get a normal matrix used to transform normals within a fragment
        shader.

        Parameters
        ----------
        out : ndarray, optional
            Alternative place to write the matrix to values. Must be a `ndarray`
            of shape (4, 4,) and have a data type of float32. Values are written
            assuming row-major order.

        Returns
        -------
        ndarray
            4x4 normal matrix.

        Notes
        -----
        * This function create a new `ndarray` with data copied from cache. Use
          the `normalMatrix` attribute for direct cache memory access.

        """
        self._updateMatrices()

        if out is None:
            return self._normalMatrixArr.copy()

        cdef Py_ssize_t i, j, N
        i = j = 0
        N = 4
        for i in range(N):
            for j in range(N):
                out[i, j] = self._normalMatrix.M[i][j]

        return out

    @property
    def ctypes(self):
        """Pointers to matrix data.

        This attribute provides a dictionary of pointers to cached matrix data
        to simplify passing data to OpenGL. This is particularly useful when
        using `pyglet` which accepts matrices as pointers. Dictionary keys are
        strings sharing the same name as the attributes whose data they point
        to.

        Examples
        --------
        Setting the model matrix::

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glMultTransposeMatrixf(myPose.ctypes['modelMatrix'])
            # run draw commands ...
            glPopMatrix()

        If using fragment shaders, the matrix can be passed on to them as such::

            # after the program was installed in the current rendering state via
            # `glUseProgram` ...
            loc = glGetUniformLocation(program, b"m_Model")
            # `transpose` must be `True`
            glUniformMatrix4fv(loc, 1, GL_TRUE, myPose.ctypes['modelMatrix'])

        """
        self._updateMatrices()

        return self._ptrMatrices

    def normalize(self):
        """Normalize this pose.

        Notes
        -----
        Uses ``OVR::Posef.Normalize`` which is part of the Oculus PC SDK.

        """
        (<libovr_math.Posef>self.c_data[0]).Normalize()
        self._matrixNeedsUpdate = True

        return self

    def invert(self):
        """Invert this pose.

        Notes
        -----
        * Uses ``OVR::Posef.Inverted`` which is part of the Oculus PC SDK.

        """
        self.c_data[0] = \
            <capi.ovrPosef>((<libovr_math.Posef>self.c_data[0]).Inverted())

        self._matrixNeedsUpdate = True

        return self

    def inverted(self):
        """Get the inverse of the pose.

        Returns
        -------
        LibOVRPose
            Inverted pose.

        Notes
        -----
        * Uses ``OVR::Posef.Inverted`` which is part of the Oculus PC SDK.

        """
        cdef capi.ovrPosef* ptr = <capi.ovrPosef*>PyMem_Malloc(
            sizeof(capi.ovrPosef))

        if ptr is NULL:
            raise MemoryError

        cdef libovr_math.Posef* pose = <libovr_math.Posef*>self.c_data
        ptr[0] = <capi.ovrPosef>(pose.Inverted())

        return LibOVRPose.fromPtr(ptr, True)

    def rotate(self, object v, np.ndarray[np.float32_t, ndim=1] out=None):
        """Rotate a position vector.

        Parameters
        ----------
        v : array_like
            Vector to rotate.
        out : ndarray, optional
            Optional output array. Must have `dtype=float32` and `shape=(3,)`.

        Returns
        -------
        ndarray
            Vector rotated by the pose's orientation.

        Notes
        -----
        * Uses ``OVR::Posef.Rotate`` which is part of the Oculus PC SDK.

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = out

        cdef libovr_math.Posef* pose = <libovr_math.Posef*>self.c_data
        cdef libovr_math.Vector3f pos_in = libovr_math.Vector3f(
            <float>v[0], <float>v[1], <float>v[2])
        cdef libovr_math.Vector3f rotated_pos = pose.Rotate(pos_in)

        toReturn[0] = rotated_pos.x
        toReturn[1] = rotated_pos.y
        toReturn[2] = rotated_pos.z

        return toReturn

    def inverseRotate(self, object v, np.ndarray[np.float32_t, ndim=1] out=None):
        """Inverse rotate a position vector.

        Parameters
        ----------
        v : array_like
            Vector to inverse rotate (x, y, z).
        out : ndarray, optional
            Optional output array. Must have `dtype=float32` and `shape=(3,)`.

        Returns
        -------
        ndarray
            Vector rotated by the pose's inverse orientation.

        Notes
        -----
        * Uses ``OVR::Vector3f.InverseRotate`` which is part of the Oculus PC
          SDK.

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = out

        cdef libovr_math.Posef* pose = <libovr_math.Posef*>self.c_data
        cdef libovr_math.Vector3f pos_in = libovr_math.Vector3f(
            <float>v[0], <float>v[1], <float>v[2])
        cdef libovr_math.Vector3f invRotatedPos = pose.InverseRotate(pos_in)

        toReturn[0] = invRotatedPos.x
        toReturn[1] = invRotatedPos.y
        toReturn[2] = invRotatedPos.z

        return toReturn

    def translate(self, object v, np.ndarray[np.float32_t, ndim=1] out=None):
        """Translate a position vector.

        Parameters
        ----------
        v : array_like
            Vector to translate [x, y, z].
        out : ndarray, optional
            Optional output array. Must have `dtype=float32` and `shape=(3,)`.

        Returns
        -------
        ndarray
            Vector translated by the pose's position.

        Notes
        -----
        * Uses ``OVR::Vector3f.Translate`` which is part of the Oculus PC SDK.

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = out

        cdef libovr_math.Posef* pose = <libovr_math.Posef*>self.c_data
        cdef libovr_math.Vector3f pos_in = libovr_math.Vector3f(
            <float>v[0], <float>v[1], <float>v[2])
        cdef libovr_math.Vector3f translated_pos = pose.Translate(pos_in)

        toReturn[0] = translated_pos.x
        toReturn[1] = translated_pos.y
        toReturn[2] = translated_pos.z

        return toReturn

    def transform(self, object v, np.ndarray[np.float32_t, ndim=1] out=None):
        """Transform a position vector.

        Parameters
        ----------
        v : array_like
            Vector to transform [x, y, z].
        out : ndarray, optional
            Optional output array. Must have `dtype=float32` and `shape=(3,)`.

        Returns
        -------
        ndarray
            Vector transformed by the pose's position and orientation.

        Notes
        -----
        * Uses ``OVR::Vector3f.Transform`` which is part of the Oculus PC SDK.

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = out

        cdef libovr_math.Posef* pose = <libovr_math.Posef*>self.c_data
        cdef libovr_math.Vector3f pos_in = libovr_math.Vector3f(
            <float>v[0], <float>v[1], <float>v[2])
        cdef libovr_math.Vector3f transformed_pos = pose.Transform(pos_in)

        toReturn[0] = transformed_pos.x
        toReturn[1] = transformed_pos.y
        toReturn[2] = transformed_pos.z

        return toReturn

    def inverseTransform(self, object v, np.ndarray[np.float32_t, ndim=1] out=None):
        """Inverse transform a position vector.

        Parameters
        ----------
        v : array_like
            Vector to transform (x, y, z).
        out : ndarray, optional
            Optional output array. Must have `dtype=float32` and `shape=(3,)`.

        Returns
        -------
        ndarray
            Vector transformed by the inverse of the pose's position and
            orientation.

        Notes
        -----
        * Uses ``OVR::Vector3f.InverseTransform`` which is part of the Oculus PC
          SDK.

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = out

        cdef libovr_math.Posef* pose = <libovr_math.Posef*>self.c_data
        cdef libovr_math.Vector3f pos_in = libovr_math.Vector3f(
            <float>v[0], <float>v[1], <float>v[2])
        cdef libovr_math.Vector3f transformed_pos = pose.InverseTransform(pos_in)

        toReturn[0] = transformed_pos.x
        toReturn[1] = transformed_pos.y
        toReturn[2] = transformed_pos.z

        return toReturn

    def transformNormal(self, object v, np.ndarray[np.float32_t, ndim=1] out=None):
        """Transform a normal vector.

        Parameters
        ----------
        v : array_like
            Vector to transform (x, y, z).
        out : ndarray, optional
            Optional output array. Must have `dtype=float32` and `shape=(3,)`.

        Returns
        -------
        ndarray
            Normal vector transformed by the pose's position and orientation.

        Notes
        -----
        * Uses ``OVR::Vector3f.TransformNormal`` which is part of the Oculus PC
          SDK.

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = out

        cdef libovr_math.Posef* pose = <libovr_math.Posef*>self.c_data
        cdef libovr_math.Vector3f pos_in = libovr_math.Vector3f(
            <float>v[0], <float>v[1], <float>v[2])
        cdef libovr_math.Vector3f transformed_pos = pose.TransformNormal(pos_in)

        toReturn[0] = transformed_pos.x
        toReturn[1] = transformed_pos.y
        toReturn[2] = transformed_pos.z

        return toReturn

    def inverseTransformNormal(self, object v, np.ndarray[np.float32_t, ndim=1] out=None):
        """Inverse transform a normal vector.

        Parameters
        ----------
        v : array_like
            Vector to transform (x, y, z).
        out : ndarray, optional
            Optional output array. Must have `dtype=float32` and `shape=(3,)`.

        Returns
        -------
        ndarray
            Normal vector transformed by the inverse of the pose's position and
            orientation.

        Notes
        -----
        * Uses ``OVR::Vector3f.InverseTransformNormal`` which is part of the
          Oculus PC SDK.

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = out

        cdef libovr_math.Posef* pose = <libovr_math.Posef*>self.c_data
        cdef libovr_math.Vector3f pos_in = libovr_math.Vector3f(
            <float>v[0], <float>v[1], <float>v[2])
        cdef libovr_math.Vector3f transformed_pos = \
            pose.InverseTransformNormal(pos_in)

        toReturn[0] = transformed_pos.x
        toReturn[1] = transformed_pos.y
        toReturn[2] = transformed_pos.z

        return toReturn

    def apply(self, object v, np.ndarray[np.float32_t, ndim=1] out=None):
        """Apply a transform to a position vector.

        Parameters
        ----------
        v : array_like
            Vector to transform (x, y, z).
        out : ndarray, optional
            Optional output array. Must have `dtype=float32` and `shape=(3,)`.

        Returns
        -------
        ndarray
            Vector transformed by the pose's position and orientation.

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = out

        cdef libovr_math.Vector3f pos_in = libovr_math.Vector3f(
            <float>v[0], <float>v[1], <float>v[2])
        cdef libovr_math.Vector3f transformed_pos = \
            (<libovr_math.Posef>self.c_data[0]).Apply(pos_in)

        toReturn[0] = transformed_pos.x
        toReturn[1] = transformed_pos.y
        toReturn[2] = transformed_pos.z

        return toReturn

    def distanceTo(self, object v):
        """Distance to a point or pose from this pose.

        Parameters
        ----------
        v : array_like
            Vector to transform (x, y, z).

        Returns
        -------
        float
            Distance to a point or LibOVRPose.

        Examples
        --------

        Get the distance between poses::

            distance = thisPose.distanceTo(otherPose)

        Get the distance to a point coordinate::

            distance = thisPose.distanceTo([0.0, 0.0, 5.0])

        Do something if the tracked right hand pose is within 0.5 meters of some
        object::

            # use 'getTrackingState' instead for hand poses, just an example
            handPose = getDevicePoses(TRACKED_DEVICE_TYPE_RTOUCH,
                                      absTime, latencyMarker=False)
            # object pose
            objPose = LibOVRPose((0.0, 1.0, -0.5))

            if handPose.distanceTo(objPose) < 0.5:
                # do something here ...

        Vary the touch controller's vibration amplitude as a function of
        distance to some pose. As the hand gets closer to the point, the
        amplitude of the vibration increases::

            dist = handPose.distanceTo(objPose)
            vibrationRadius = 0.5

            if dist < vibrationRadius:  # inside vibration radius
                amplitude = 1.0 - dist / vibrationRadius
                setControllerVibration(CONTROLLER_TYPE_RTOUCH,
                    'low', amplitude)
            else:
                # turn off vibration
                setControllerVibration(CONTROLLER_TYPE_RTOUCH, 'off')

        """
        cdef libovr_math.Vector3f pos_in
        cdef libovr_math.Posef* pose = <libovr_math.Posef*>self.c_data

        if isinstance(v, LibOVRPose):
            pos_in = <libovr_math.Vector3f>((<LibOVRPose>v).c_data[0]).Position
        else:
            pos_in = libovr_math.Vector3f(<float>v[0], <float>v[1], <float>v[2])

        cdef float to_return = pose.Translation.Distance(pos_in)

        return to_return

    def raycastSphere(self, object targetPose, float radius=0.5, object rayDir=(0., 0., -1.), float maxRange=0.0):
        """Raycast to a sphere.

        Project an invisible ray of finite or infinite length from this pose in
        `rayDir` and check if it intersects with the `targetPose` bounding
        sphere.

        This method allows for very basic interaction between objects
        represented by poses in a scene, including tracked devices.

        Specifying `maxRange` as >0.0 casts a ray of finite length in world
        units. The distance between the target and ray origin position are
        checked prior to casting the ray; automatically failing if the ray can
        never reach the edge of the bounding sphere centered about `targetPose`.
        This avoids having to do the costly transformations required for
        picking.

        This raycast implementation can only determine if contact is being made
        with the object's bounding sphere, not where on the object the ray
        intersects. This method might not work for irregular or elongated
        objects since bounding spheres may not approximate those shapes well. In
        such cases, one may use multiple spheres at different locations and
        radii to pick the same object.

        Parameters
        ----------
        targetPose : array_like
            Coordinates of the center of the target sphere (x, y, z).
        radius : float, optional
            The radius of the target.
        rayDir : array_like, optional
            Vector indicating the direction for the ray (default is -Z).
        maxRange : float, optional
            The maximum range of the ray. Ray testing will fail automatically if
            the target is out of range. Ray is infinite if maxRange=0.0.

        Returns
        -------
        bool
            True if the ray intersects anywhere on the bounding sphere, False in
            every other condition.

        Examples
        --------

        Basic example to check if the HMD is aligned to some target::

            targetPose = LibOVRPose((0.0, 1.5, -5.0))
            targetRadius = 0.5  # 2.5 cm
            isAligned = hmdPose.raycastSphere(targetPose.pos,
                                              radius=targetRadius)

        Check if someone is touching a target with their finger when making a
        pointing gesture while providing haptic feedback::

            targetPose = LibOVRPose((0.0, 1.5, -0.25))
            targetRadius = 0.025  # 2.5 cm
            fingerLength = 0.1  # 10 cm

            # check if making a pointing gesture with their right hand
            isPointing = getTouch(CONTROLLER_TYPE_RTOUCH,
                TOUCH_RINDEXPOINTING)

            if isPointing:
                # do raycasting operation
                isTouching = handPose.raycastSphere(
                    targetPose.pos, radius=targetRadius, maxRange=fingerLength)
                if isTouching:
                    # do something here, like make the controller vibrate
                    setControllerVibration(
                        CONTROLLER_TYPE_RTOUCH, 'low', 0.5)
                else:
                    # stop vibration if no longer touching
                    setControllerVibration(CONTROLLER_TYPE_RTOUCH, 'off')

        """
        cdef libovr_math.Vector3f targetPos = libovr_math.Vector3f(
            <float>targetPose[0], <float>targetPose[1], <float>targetPose[2])
        cdef libovr_math.Vector3f _rayDir = libovr_math.Vector3f(
            <float>rayDir[0], <float>rayDir[1], <float>rayDir[2])
        cdef libovr_math.Posef originPos = <libovr_math.Posef>self.c_data[0]

        # if the ray is finite, does it ever touch the edge of the sphere?
        cdef float targetDist
        if maxRange != 0.0:
            targetDist = targetPos.Distance(originPos.Translation) - radius
            if targetDist > maxRange:
                return False

        # put the target in the ray caster's local coordinate system
        cdef libovr_math.Vector3f offset = -originPos.InverseTransform(targetPos)

        # find the discriminant, this is based on the method described here:
        # http://antongerdelan.net/opengl/raycasting.html
        cdef float desc = <float>pow(_rayDir.Dot(offset), 2.0) - \
               (offset.Dot(offset) - <float>pow(radius, 2.0))

        # one or more roots? if so we are touching the sphere
        return desc >= 0.0

    def raycastPose(self, LibOVRPose targetPose, object rayDir=(0., 0., -1.), float maxRange=0.0):
        """Raycast a pose's bounding box.

        This function tests if and where a ray projected from the position of
        this pose in the direction of `rayDir` intersects the bounding box
        of another :class:`LibOVRPose`. The bounding box of the target object
        will be oriented by the pose it's associated with.

        Parameters
        ----------
        targetPose : LibOVRPose
            Target pose with bounding box.
        rayDir : array_like
            Vector specifying the direction the ray should be projected. This
            direction is in the reference of the pose.
        maxRange : float
            Length of the ray. If 0.0, the ray will be assumed to have infinite
            length.

        Returns
        -------
        ndarray
            Position in scene coordinates the ray intersects the bounding box
            nearest to this pose. Returns `None` if there is no intersect or
            the target class does not have a valid bounding box.

        Examples
        --------
        Test where a ray intersects another pose's bounding box and create a
        pose object there::

            intercept = thisPose.raycastPose(targetPose)

            if intercept is not None:
                interceptPose = LibOVRPose(intercept)

        Check if a user is touching a bounding box with their right index
        finger::

            fingerLength = 0.1  # 10 cm
            # check if making a pointing gesture with their right hand
            if getTouch(CONTROLLER_TYPE_RTOUCH, TOUCH_RINDEXPOINTING):
                isTouching = handPose.raycastPose(targetPose, maxRange=fingerLength)
                if isTouching is not None:
                    #  run some code here for when touching ...
                else:
                    #  run some code here for when not touching ...

        """
        # check if there is a bounding box
        if targetPose.bounds is None:
            return None

        # based off algorithm:
        # http://www.opengl-tutorial.org/miscellaneous/clicking-on-objects/
        # picking-with-custom-ray-obb-function/
        cdef libovr_math.Vector3f rayOrig = \
            <libovr_math.Vector3f>self.c_data[0].Position
        cdef libovr_math.Vector3f _rayDir = libovr_math.Vector3f(
            <float>rayDir[0], <float>rayDir[1], <float>rayDir[2])
        cdef libovr_math.Matrix4f modelMatrix = targetPose._modelMatrix
        cdef libovr_math.Vector3f boundsOffset = \
            <libovr_math.Vector3f>targetPose.c_data[0].Position
        cdef libovr_math.Vector3f[2] bounds = targetPose._bbox.c_data.b
        cdef libovr_math.Vector3f axis

        # rotate `rayDir` by this pose
        _rayDir = (<libovr_math.Posef>self.c_data[0]).TransformNormal(_rayDir)

        cdef float e, f
        cdef float tmin = 0.0
        cdef float tmax = 1.8446742974197924e19  # from OVR_MATH.h
        cdef libovr_math.Vector3f d = boundsOffset - rayOrig

        # solve intersects for each pair of planes along each axis
        cdef int i, N
        N = 3
        for i in range(N):
            axis.x = modelMatrix.M[0][i]
            axis.y = modelMatrix.M[1][i]
            axis.z = modelMatrix.M[2][i]
            e = axis.Dot(d)
            f = _rayDir.Dot(axis)

            if np.fabs(f) > 1e-5:
                t1 = (e + bounds[0][i]) / f
                t2 = (e + bounds[1][i]) / f

                if t1 > t2:
                    temp = t1
                    t1 = t2
                    t2 = temp

                if t2 < tmax:
                    tmax = t2

                if t1 > tmin:
                    tmin = t1

                if tmin > tmax:
                    return None

            else:
                # very close to parallel with the face
                if -e + bounds[0][i] > 0.0 or -e + bounds[1][i] < 0.0:
                    return None

        # return if intercept was too far
        if maxRange != 0.0 and tmin > (<float>maxRange):
            return None

        # if we made it here, there was an intercept
        cdef libovr_math.Vector3f result = (_rayDir * tmin) + rayOrig

        # output to numpy array
        cdef np.ndarray[np.float32_t, ndim=1] toReturn = \
            np.array((result.x, result.y, result.z), dtype=np.float32)

        return toReturn

    def interp(self, LibOVRPose end, float s, bint fast=False):
        """Interpolate between poses.

        Linear interpolation is used on position (Lerp) while the orientation
        has spherical linear interpolation (Slerp) applied.

        Parameters
        ----------
        end : LibOVRPose
            End pose.
        s : float
            Interpolation factor between interval 0.0 and 1.0.
        fast : bool, optional
            If True, use fast interpolation which is quicker but less accurate
            over larger distances.

        Returns
        -------
        LibOVRPose
            Interpolated pose at `s`.

        Notes
        -----
        * Uses ``OVR::Posef.Lerp`` and ``OVR::Posef.FastLerp`` which is part of
          the Oculus PC SDK.

        """
        if 0.0 > s > 1.0:
            raise ValueError("Interpolation factor must be between 0.0 and 1.0.")

        cdef libovr_math.Posef toPose = <libovr_math.Posef>end.c_data[0]
        cdef capi.ovrPosef* ptr = <capi.ovrPosef*>PyMem_Malloc(sizeof(capi.ovrPosef))

        if ptr is NULL:
            raise MemoryError

        if not fast:
            ptr[0] = <capi.ovrPosef>(
                (<libovr_math.Posef>self.c_data[0]).Lerp(toPose, s))
        else:
            ptr[0] = <capi.ovrPosef>(
                (<libovr_math.Posef>self.c_data[0]).FastLerp(toPose, s))

        return LibOVRPose.fromPtr(ptr, True)

    def isVisible(self, int eye):
        """Check if this pose if visible to a given eye.

        Visibility testing is done using the current eye render pose for `eye`.
        This pose must have a valid bounding box assigned to `bounds`. If not,
        this method will always return ``True``.

        See :func:`cullPose` for more information about the implementation of
        visibility culling. Note this function only works if there is an active
        VR session.

        Parameters
        ----------
        eye : int
            Eye index. Use either ``EYE_LEFT`` or ``EYE_RIGHT``.

        Returns
        -------
        bool
            ``True`` if this pose's bounding box intersects the FOV of the
            specified `eye`. Returns ``False`` if the pose's bounding box does
            not intersect the viewing frustum for `eye` or if a VR session has
            not been started.

        Examples
        --------
        Check if a pose should be culled (needs to be done for each eye)::

            if cullModel.isVisible():
                # ... OpenGL calls to draw the model here ...

        """
        global _ptrSession
        if _ptrSession != NULL:
            return not cullPose(eye, self)
        else:
            return False


cdef class LibOVRBounds(object):
    """Class for constructing and representing 3D axis-aligned bounding boxes.

    A bounding box is a construct which represents a 3D rectangular volume
    about some pose, defined by its minimum and maximum extents in the reference
    frame of the pose. The axes of the bounding box are aligned to the axes of
    the world or the associated pose.

    Bounding boxes are primarily used for visibility testing; to determine if
    the extents of an object associated with a pose (eg. the vertices of a
    model) falls completely outside of the viewing frustum. If so, the model can
    be culled during rendering to avoid wasting CPU/GPU resources on objects not
    visible to the viewer. See :func:`cullPose` for more information.

    Parameters
    ----------
    extents : tuple, optional
        Minimum and maximum extents of the bounding box (`mins`, `maxs`) where
        `mins` and `maxs` specified as coordinates [x, y, z]. If no extents are
        specified, the bounding box will be invalid until defined.

    Examples
    --------
    Create a bounding box and add it to a pose::

        # minumum and maximum extents of the bounding box
        mins = (-.5, -.5, -.5)
        maxs = (.5, .5, .5)
        bounds = (mins, maxs)
        # create the bounding box and add it to a pose
        bbox = LibOVRBounds(bounds)
        modelPose = LibOVRPose()
        modelPose.boundingBox = bbox

    """
    cdef libovr_math.Bounds3f* c_data
    cdef bint ptr_owner

    cdef np.ndarray _mins
    cdef np.ndarray _maxs

    def __init__(self, object extents=None):
        """
        Attributes
        ----------
        isValid : bool
        extents : tuple
        mins : ndarray
        maxs : ndarray
        """
        self._new_struct(extents)

    def __cinit__(self, *args, **kwargs):
        self.ptr_owner = False

    @staticmethod
    cdef LibOVRBounds fromPtr(libovr_math.Bounds3f* ptr, bint owner=False):
        cdef LibOVRBounds wrapper = LibOVRBounds.__new__(LibOVRBounds)
        wrapper.c_data = ptr
        wrapper.ptr_owner = owner

        wrapper._mins = _wrap_ovrVector3f_as_ndarray(<capi.ovrVector3f*>&ptr.b[0])
        wrapper._maxs = _wrap_ovrVector3f_as_ndarray(<capi.ovrVector3f*>&ptr.b[1])

        return wrapper

    cdef void _new_struct(self, object extents):
        if self.c_data is not NULL:
            return

        cdef libovr_math.Bounds3f* ptr = \
            <libovr_math.Bounds3f*>PyMem_Malloc(sizeof(libovr_math.Bounds3f))

        if ptr is NULL:
            raise MemoryError

        if extents is not None:
            ptr.b[0].x = <float>extents[0][0]
            ptr.b[0].y = <float>extents[0][1]
            ptr.b[0].z = <float>extents[0][2]
            ptr.b[1].x = <float>extents[1][0]
            ptr.b[1].y = <float>extents[1][1]
            ptr.b[1].z = <float>extents[1][2]
        else:
            ptr.Clear()

        self.c_data = ptr
        self.ptr_owner = True

        self._mins = _wrap_ovrVector3f_as_ndarray(<capi.ovrVector3f*>&ptr.b[0])
        self._maxs = _wrap_ovrVector3f_as_ndarray(<capi.ovrVector3f*>&ptr.b[1])

    def __dealloc__(self):
        if self.c_data is not NULL:
            if self.ptr_owner is True:
                PyMem_Free(self.c_data)
                self.c_data = NULL

    def clear(self):
        """Clear the bounding box."""
        self.c_data.Clear()

    @property
    def isValid(self):
        """``True`` if a bounding box is valid. Bounding boxes are valid if all
        dimensions of `mins` are less than each of `maxs` which is the case
        after :py:meth:`~LibOVRBounds.clear` is called.

        If a bounding box is invalid, :func:`cullPose` will always return
        ``True``.

        """
        if self.c_data.b[1].x <= self.c_data.b[0].x and \
                self.c_data.b[1].y <= self.c_data.b[0].y and \
                self.c_data.b[1].z <= self.c_data.b[0].z:
            return False

        return True

    def fit(self, object points, bint clear=True):
        """Fit an axis aligned bounding box to enclose specified points. The
        resulting bounding box is guaranteed to enclose all points, however
        volume is not necessarily minimized or optimal.

        Parameters
        ----------
        points : array_like
            2D array of points [x, y, z] to fit, can be a list of vertices from
            a 3D model associated with the bounding box.
        clear : bool, optional
            Clear the bounding box prior to fitting. If ``False`` the current
            bounding box will be re-sized to fit new points.

        Examples
        --------
        Create a bounding box around vertices specified in a list::

            # model vertices
            vertices = [[-1.0, -1.0, 0.0],
                        [-1.0, 1.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [1.0, -1.0, 0.0]]

            # create an empty bounding box
            bbox = LibOVRBounds()
            bbox.fit(vertices)

            # associate the bounding box to a pose
            modelPose = LibOVRPose()
            modelPose.bounds = bbox

        """
        cdef np.ndarray[np.float32_t, ndim=2] points_in = np.asarray(
            points, dtype=np.float32)
        cdef libovr_math.Vector3f new_point = libovr_math.Vector3f()

        if clear:
            self.c_data.Clear()

        cdef Py_ssize_t i, N
        cdef float[:, :] mv_points = points_in  # memory view
        N = <Py_ssize_t>points_in.shape[0]
        for i in range(N):
            new_point.x = mv_points[i, 0]
            new_point.y = mv_points[i, 1]
            new_point.z = mv_points[i, 2]
            self.c_data.AddPoint(new_point)

    def addPoint(self, object point):
        """Resize the bounding box to encompass a given point. Calling this
        function for each vertex of a model will create an optimal bounding box
        for it.

        Parameters
        ----------
        point : array_like
            Vector/coordinate to add [x, y, z].

        See Also
        --------
        fit : Fit a bounding box to enclose a list of points.

        """
        cdef libovr_math.Vector3f new_point = libovr_math.Vector3f(
            <float>point[0], <float>point[1], <float>point[2])

        self.c_data.AddPoint(new_point)

    @property
    def extents(self):
        """The extents of the bounding box (`mins`, `maxs`)."""
        return self._mins, self._maxs

    @extents.setter
    def extents(self, object value):
        self._mins[:] = value[0]
        self._maxs[:] = value[1]

    @property
    def mins(self):
        """Point defining the minimum extent of the bounding box."""
        return self._mins

    @mins.setter
    def mins(self, object value):
        self._mins[:] = value

    @property
    def maxs(self):
        """Point defining the maximum extent of the bounding box."""
        return self._maxs

    @maxs.setter
    def maxs(self, object value):
        self._mins[:] = value


cdef class LibOVRPoseState(object):
    """Class for representing rigid body poses with additional state
    information.

    Pose states contain the pose of the tracked body, but also angular and
    linear motion derivatives experienced by the pose. The pose within a state
    can be accessed via the :py:attr:`~psychxr.libovr.LibOVRPoseState.thePose`
    attribute.

    Velocity and acceleration for linear and angular motion can be used to
    compute forces applied to rigid bodies and predict the future positions of
    objects (see :py:meth:`~psychxr.libovr.LibOVRPoseState.timeIntegrate`). You
    can create `LibOVRPoseState` objects using data from other sources, such as
    nDOF IMUs for use with VR environments.

    Parameters
    ----------
    thePose : LibOVRPose, list, tuple or None
        Rigid body pose this state refers to. Can be a `LibOVRPose` pose
        instance or a tuple/list of a position coordinate (x, y, z) and
        orientation quaternion (x, y, z, w). If ``None`` the pose will be
        initialized as an identity pose.
    linearVelocity : array_like
        Linear acceleration vector [vx, vy, vz] in meters/sec.
    angularVelocity : array_like
        Angular velocity vector [vx, vy, vz] in radians/sec.
    linearAcceleration : array_like
        Linear acceleration vector [ax, ay, az] in meters/sec^2.
    angularAcceleration : array_like
        Angular acceleration vector [ax, ay, az] in radians/sec^2.
    timeInSeconds : float
        Time in seconds this state refers to.

    """
    cdef capi.ovrPoseStatef* c_data
    cdef bint ptr_owner  # owns the data

    # these will hold references until this object is de-allocated
    cdef LibOVRPose _thePose
    cdef np.ndarray _linearVelocity
    cdef np.ndarray _angularVelocity
    cdef np.ndarray _linearAcceleration
    cdef np.ndarray _angularAcceleration

    def __init__(self,
                 object thePose=None,
                 object linearVelocity=(0., 0., 0.),
                 object angularVelocity=(0., 0., 0.),
                 object linearAcceleration=(0., 0. ,0.),
                 object angularAcceleration=(0., 0., 0.),
                 double timeInSeconds=0.0):
        self._new_struct(
            thePose,
            linearVelocity,
            angularVelocity,
            linearAcceleration,
            angularAcceleration,
            timeInSeconds)

    def __cinit__(self, *args, **kwargs):
        self.ptr_owner = False

    @staticmethod
    cdef LibOVRPoseState fromPtr(capi.ovrPoseStatef* ptr, bint owner=False):
        # bypass __init__ if wrapping a pointer
        cdef LibOVRPoseState wrapper = LibOVRPoseState.__new__(LibOVRPoseState)
        wrapper.c_data = ptr
        wrapper.ptr_owner = owner

        wrapper._thePose = LibOVRPose.fromPtr(&wrapper.c_data.ThePose)
        wrapper._linearVelocity = _wrap_ovrVector3f_as_ndarray(
                &wrapper.c_data.LinearVelocity)
        wrapper._linearAcceleration = _wrap_ovrVector3f_as_ndarray(
                &wrapper.c_data.LinearAcceleration)
        wrapper._angularVelocity = _wrap_ovrVector3f_as_ndarray(
                &wrapper.c_data.AngularVelocity)
        wrapper._angularAcceleration = _wrap_ovrVector3f_as_ndarray(
                &wrapper.c_data.AngularAcceleration)

        return wrapper

    cdef void _new_struct(
            self,
            object pose,
            object linearVelocity,
            object angularVelocity,
            object linearAcceleration,
            object angularAcceleration,
            double timeInSeconds):

        if self.c_data is not NULL:  # already allocated, __init__ called twice?
            return

        cdef capi.ovrPoseStatef* _ptr = \
            <capi.ovrPoseStatef*>PyMem_Malloc(
                sizeof(capi.ovrPoseStatef))

        if _ptr is NULL:
            raise MemoryError

        self.c_data = _ptr
        self.ptr_owner = True

        # setup property wrappers
        self._thePose = LibOVRPose.fromPtr(&self.c_data.ThePose)
        self._linearVelocity = _wrap_ovrVector3f_as_ndarray(
            &self.c_data.LinearVelocity)
        self._linearAcceleration = _wrap_ovrVector3f_as_ndarray(
            &self.c_data.LinearAcceleration)
        self._angularVelocity = _wrap_ovrVector3f_as_ndarray(
            &self.c_data.AngularVelocity)
        self._angularAcceleration = _wrap_ovrVector3f_as_ndarray(
            &self.c_data.AngularAcceleration)

        # set values
        if pose is None:
            _ptr.ThePose.Position = [0., 0., 0.]
            _ptr.ThePose.Orientation = [0., 0., 0., 1.]
        elif isinstance(pose, LibOVRPose):
            _ptr.ThePose.Position = (<LibOVRPose>pose).c_data.Position
            _ptr.ThePose.Orientation = (<LibOVRPose>pose).c_data.Orientation
        elif isinstance(pose, (tuple, list,)):
            self._thePose.posOri = pose
        else:
            raise TypeError('Invalid value for `pose`, must be `LibOVRPose`'
                            ', `list` or `tuple`.')

        self._angularVelocity[:] = angularVelocity
        self._linearVelocity[:] = linearVelocity
        self._angularAcceleration[:] = angularAcceleration
        self._linearAcceleration[:] = linearAcceleration

        _ptr.TimeInSeconds = 0.0

    def __dealloc__(self):
        # don't do anything crazy like set c_data=NULL without deallocating!
        if self.c_data is not NULL:
            if self.ptr_owner is True:
                PyMem_Free(self.c_data)

    def __deepcopy__(self, memo=None):
        """Deep copy returned by :py:func:`copy.deepcopy`.

        New :py:class:`LibOVRPoseState` instance with a copy of the data in a
        separate memory location. Does not increase the reference count of the
        object being copied.

        Examples
        --------

        Deep copy::

            import copy
            a = LibOVRPoseState()
            b = copy.deepcopy(a)  # create independent copy of 'a'

        """
        cdef capi.ovrPoseStatef* ptr = \
            <capi.ovrPoseStatef*>PyMem_Malloc(sizeof(capi.ovrPoseStatef))

        if ptr is NULL:
            raise MemoryError

        cdef LibOVRPoseState to_return = LibOVRPoseState.fromPtr(ptr, True)

        # copy over data
        to_return.c_data[0] = self.c_data[0]

        if memo is not None:
            memo[id(self)] = to_return

        return to_return

    def duplicate(self):
        """Create a deep copy of this object.

        Same as calling `copy.deepcopy` on an instance.

        Returns
        -------
        LibOVRPoseState
            An independent copy of this object.

        """
        return self.__deepcopy__()

    @property
    def thePose(self):
        """Rigid body pose."""
        return self._thePose

    @thePose.setter
    def thePose(self, LibOVRPose value):
        self.c_data.ThePose = value.c_data[0]  # copy into

    @property
    def angularVelocity(self):
        """Angular velocity vector in radians/sec."""
        return self._angularVelocity

    @angularVelocity.setter
    def angularVelocity(self, object value):
        self._angularVelocity[:] = value

    @property
    def linearVelocity(self):
        """Linear velocity vector in meters/sec."""
        return self._linearVelocity

    @linearVelocity.setter
    def linearVelocity(self, object value):
        self._linearVelocity[:] = value

    @property
    def angularAcceleration(self):
        """Angular acceleration vector in radians/s^2."""
        return self._angularAcceleration

    @angularAcceleration.setter
    def angularAcceleration(self, object value):
        self._angularAcceleration[:] = value

    @property
    def linearAcceleration(self):
        """Linear acceleration vector in meters/s^2."""
        return self._linearAcceleration

    @linearAcceleration.setter
    def linearAcceleration(self, object value):
        self._linearAcceleration[:] = value

    @property
    def timeInSeconds(self):
        """Absolute time this data refers to in seconds."""
        return <double>self.c_data[0].TimeInSeconds

    @timeInSeconds.setter
    def timeInSeconds(self, double value):
        self.c_data[0].TimeInSeconds = value

    def timeIntegrate(self, float dt):
        """Time integrate rigid body motion derivatives referenced by the
        current pose.

        Parameters
        ----------
        dt : float
            Time delta in seconds.

        Returns
        -------
        LibOVRPose
            Pose at `dt`.

        Examples
        --------

        Time integrate a pose for 20 milliseconds (note the returned object is a
        :py:mod:`LibOVRPose`, not another :py:class:`LibOVRPoseState`)::

            newPose = oldPose.timeIntegrate(0.02)
            pos, ori = newPose.posOri  # extract components

        Time integration can be used to predict the pose of an object at HMD
        V-Sync if velocity and acceleration are known. Usually we would pass the
        predicted time to `getDevicePoses` or `getTrackingState` for a more
        robust estimate of HMD pose at predicted display time. However, in most
        cases the following will yield the same position and orientation as
        `LibOVR` within a few decimal places::

            tsec = timeInSeconds()
            ptime = getPredictedDisplayTime(frame_index)

            _, headPoseState = getDevicePoses(
                [TRACKED_DEVICE_TYPE_HMD],
                absTime=tsec,  # not the predicted time!
                latencyMarker=True)

            dt = ptime - tsec  # time difference from now and v-sync
            headPoseAtVsync = headPose.timeIntegrate(dt)
            calcEyePoses(headPoseAtVsync)

        """
        cdef libovr_math.Posef res = \
            (<libovr_math.Posef>self.c_data[0].ThePose).TimeIntegrate(
                <libovr_math.Vector3f>self.c_data[0].LinearVelocity,
                <libovr_math.Vector3f>self.c_data[0].AngularVelocity,
                <libovr_math.Vector3f>self.c_data[0].LinearAcceleration,
                <libovr_math.Vector3f>self.c_data[0].AngularAcceleration,
                dt)

        cdef capi.ovrPosef* ptr = \
            <capi.ovrPosef*>PyMem_Malloc(sizeof(capi.ovrPosef))

        if ptr is NULL:
            raise MemoryError(
                "Failed to allocate 'ovrPosef' in 'timeIntegrate'.")

        cdef LibOVRPose to_return = LibOVRPose.fromPtr(ptr, True)

        # copy over data
        to_return.c_data[0] = <capi.ovrPosef>res

        return to_return