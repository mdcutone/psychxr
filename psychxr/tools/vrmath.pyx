# distutils: language=c++
#  =============================================================================
#  vrmath.pyx - Toolbox of VR math classes and functions
#  =============================================================================
#
#  Copyright 2021 Matthew Cutone <mcutone@opensciencetools.com>
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
"""Extension module for VR math.

This module contains many useful tools for various linear math operations
typically used in VR applications.

"""
# ------------------------------------------------------------------------------
# Module information
#
__author__ = "Matthew D. Cutone"
__credits__ = []
__copyright__ = "Copyright 2021 Matthew D. Cutone"
__license__ = "MIT"
__version__ = '0.2.4rc2'
__status__ = "Stable"
__maintainer__ = "Matthew D. Cutone"
__email__ = "mcutone@opensciencetools.com"

# ------------------------------------------------------------------------------
# Imports and constants
#
__all__ = ['RigidBodyPose', 'BoundingBox', 'calcEyePoses']

import ctypes

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.math cimport pow, sin, cos, M_PI, sqrt, fabs, acos
from libc.float cimport FLT_MAX

import numpy as np
cimport numpy as np
np.import_array()

# Helper functions and data, used internally by PsychXR and not exposed to the
# public Python API.
#
RAD_TO_DEGF = <float>180.0 / M_PI
DEG_TO_RADF = M_PI / <float>180.0


cdef float maxf(float a, float b):
    return a if a >= b else b


cdef np.npy_intp[1] VEC3_SHAPE = [3]
cdef np.npy_intp[1] QUAT_SHAPE = [4]
cdef np.npy_intp[2] MAT4_SHAPE = [4, 4]


cdef np.ndarray _wrap_pxrVector3f_as_ndarray(pxrVector3f* prtVec):
    """Wrap an pxrVector3f object with a NumPy array."""
    return np.PyArray_SimpleNewFromData(
        1, VEC3_SHAPE, np.NPY_FLOAT32, <void*>prtVec)


cdef np.ndarray _wrap_pxrQuatf_as_ndarray(pxrQuatf* prtVec):
    """Wrap an pxrQuatf object with a NumPy array."""
    return np.PyArray_SimpleNewFromData(
        1, QUAT_SHAPE, np.NPY_FLOAT32, <void*>prtVec)


cdef np.ndarray _wrap_pxrMatrix4f_as_ndarray(pxrMatrix4f* prtVec):
    """Wrap an pxrMatrix4f object with a NumPy array."""
    return np.PyArray_SimpleNewFromData(
        2, MAT4_SHAPE, np.NPY_FLOAT32, <void*>prtVec.M)


# ------------------------------------------------------------------------------
# Classes and functions for VRMATH
#

cdef class RigidBodyPose(object):
    """Class representing a rigid-body pose.

    This class is an abstract representation of a rigid body pose, where the
    position of the body in a scene is represented by a vector/coordinate and
    the orientation with a quaternion. There are many class methods and
    properties provided to handle accessing, manipulating, and interacting with
    poses. Rigid body poses assume a right-handed coordinate system (-Z is
    forward and +Y is up).

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

    Parameters
    ----------
    pos : array_like
        Initial position vector (x, y, z).
    ori : array_like
        Initial orientation quaternion (x, y, z, w).

    Notes
    -----
    * This class is intended to be a drop in replacement for the
      :class:`~psychxr.drivers.libovr.LibOVRPose` class, sharing much of the
      same attributes and methods. However, this class does not require the
      LibOVR SDK to use it making it suitable to work with other VR drivers.

    """
    def __init__(self, pos=(0., 0., 0.), ori=(0., 0., 0., 1.)):
        self._new_struct(pos, ori)
        self._updateMatrices()

    def __cinit__(self, *args, **kwargs):
        self.ptr_owner = False

        # make sure we have proxy objects
        self._modelMatrixArr = _wrap_pxrMatrix4f_as_ndarray(&self._modelMatrix)
        self._invModelMatrixArr = _wrap_pxrMatrix4f_as_ndarray(&self._invModelMatrix)
        self._normalMatrixArr = _wrap_pxrMatrix4f_as_ndarray(&self._normalMatrix)
        self._viewMatrixArr = _wrap_pxrMatrix4f_as_ndarray(&self._viewMatrix)
        self._invViewMatrixArr = _wrap_pxrMatrix4f_as_ndarray(&self._invViewMatrix)

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

        self._matrixNeedsUpdate = True

    @staticmethod
    cdef RigidBodyPose fromPtr(pxrPosef* ptr, bint owner):
        cdef RigidBodyPose wrapper = RigidBodyPose.__new__(RigidBodyPose)
        wrapper.c_data = ptr
        wrapper.ptr_owner = owner

        wrapper._pos = _wrap_pxrVector3f_as_ndarray(&ptr.Position)
        wrapper._ori = _wrap_pxrQuatf_as_ndarray(&ptr.Orientation)
        wrapper._bbox = None
        wrapper._matrixNeedsUpdate = True

        return wrapper

    cdef void _new_struct(self, object pos, object ori):
        if self.c_data is not NULL:
            return

        cdef pxrPosef* ptr = \
            <pxrPosef*>PyMem_Malloc(sizeof(pxrPosef))

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

        self._pos = _wrap_pxrVector3f_as_ndarray(&ptr.Position)
        self._ori = _wrap_pxrQuatf_as_ndarray(&ptr.Orientation)
        self._bbox = None
        self._matrixNeedsUpdate = True

    def __dealloc__(self):
        # don't do anything crazy like set c_data=NULL without deallocating!
        if self.c_data is not NULL:
            if self.ptr_owner is True:
                PyMem_Free(self.c_data)
                self.c_data = NULL

    def __repr__(self):
        return f'RigidBodyPose(pos={repr(self.pos)}, ori={repr(self.ori)})'

    def __mul__(RigidBodyPose a, RigidBodyPose b):
        """Multiplication operator (*) to combine poses.
        """
        cdef pxrPosef* ptr = <pxrPosef*>PyMem_Malloc(
            sizeof(pxrPosef))

        if ptr is NULL:
            raise MemoryError

        # multiply the rotations
        quat_mul(
            &ptr.Orientation.x,
            &a.c_data.Orientation.x,
            &b.c_data.Orientation.x)

        # apply the transformation
        quat_mul_vec3(
            &ptr.Position.x,
            &a.c_data.Orientation.x,
            &b.c_data.Position.x)
        vec3_add(
            &ptr.Position.x,
            &ptr.Position.x,
            &a.c_data.Position.x)

        return RigidBodyPose.fromPtr(ptr, True)

    def __imul__(self, RigidBodyPose other):
        """Multiplication operator (*=) to combine poses.
        """
        cdef pxrQuatf new_ori
        cdef pxrVector3f new_pos
        cdef pxrQuatf q_temp = self.c_data.Orientation

        # multiply the rotations
        quat_mul(
            &new_ori.x,
            &self.c_data.Orientation.x,
            &other.c_data.Orientation.x)

        # apply the transformation
        quat_mul_vec3(
            &new_pos.x,
            &q_temp.x,
            &other.c_data.Position.x)
        vec3_add(
            &new_pos.x,
            &new_pos.x,
            &self.c_data.Position.x)

        self.c_data.Orientation = new_ori
        self.c_data.Position = new_pos

        self._matrixNeedsUpdate = True

        return self

    def __invert__(self):
        """Invert operator (~) to invert a pose."""
        return self.inverted()

    def __eq__(self, RigidBodyPose other):
        """Equality operator (==) for two poses.

        The tolerance of the comparison is 1e-5.

        """
        return self.isEqual(other)

    def __ne__(self, RigidBodyPose other):
        """Inequality operator (!=) for two poses.

        The tolerance of the comparison is 1e-5.

        """
        return not self.isEqual(other)

    def __deepcopy__(self, memo=None):
        # create a new object with a copy of the data stored in c_data
        # allocate new struct
        cdef pxrPosef* ptr = \
            <pxrPosef*>PyMem_Malloc(sizeof(pxrPosef))

        if ptr is NULL:
            raise MemoryError

        cdef RigidBodyPose to_return = RigidBodyPose.fromPtr(ptr, owner=True)

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
        cdef RigidBodyPose toReturn = RigidBodyPose()
        (<RigidBodyPose>toReturn).c_data[0] = self.c_data[0]

        return toReturn

    @property
    def bounds(self):
        """Bounding object associated with this pose."""
        return self._bbox

    @bounds.setter
    def bounds(self, BoundingBox value):
        self._bbox = value

    def isEqual(self, RigidBodyPose pose, float tolerance=1e-5):
        """Check if poses are close to equal in position and orientation.

        Same as using the equality operator (==) on poses, but you can specify
        and arbitrary value for `tolerance`.

        Parameters
        ----------
        pose : OHMDPose
            The other pose.
        tolerance : float, optional
            Tolerance for the comparison, default is 1e-5.

        Returns
        -------
        bool
            ``True`` if pose components are within `tolerance` from this pose.

        """
        cdef pxrVector3f* pos = &pose.c_data.Position
        cdef pxrQuatf* ori = &pose.c_data.Orientation
        cdef bint to_return = (
            <float>fabs(pos.x - self.c_data.Position.x) < tolerance and
            <float>fabs(pos.y - self.c_data.Position.y) < tolerance and
            <float>fabs(pos.z - self.c_data.Position.z) < tolerance and
            <float>fabs(ori.x - self.c_data.Orientation.x) < tolerance and
            <float>fabs(ori.y - self.c_data.Orientation.y) < tolerance and
            <float>fabs(ori.z - self.c_data.Orientation.z) < tolerance and
            <float>fabs(ori.w - self.c_data.Orientation.w) < tolerance)

        return to_return

    def setIdentity(self):
        """Clear this pose's translation and orientation."""
        self.c_data[0].Position.x = 0.0
        self.c_data[0].Position.y = 0.0
        self.c_data[0].Position.z = 0.0

        self.c_data[0].Orientation.x = 0.0
        self.c_data[0].Orientation.y = 0.0
        self.c_data[0].Orientation.z = 0.0
        self.c_data[0].Orientation.w = 1.0

        self._matrixNeedsUpdate = True

    def _updateMatrices(self):
        """Update model, inverse, and normal matrices due to an attribute change.
        """
        if not self._matrixNeedsUpdate:
            return

        cdef mat4x4 m_rotate
        cdef mat4x4 m_translate

        # compute model matrix
        mat4x4_from_quat(m_rotate, &self.c_data.Orientation.x)
        mat4x4_translate(
            m_translate,
            self.c_data.Position.x,
            self.c_data.Position.y,
            self.c_data.Position.z)
        mat4x4_mul(self._modelMatrix.M, m_translate, m_rotate)

        # get its inverse
        mat4x4_invert_fast(self._invModelMatrix.M, self._modelMatrix.M)

        # normal matrix
        mat4x4_transpose(self._normalMatrix.M, self._invModelMatrix.M)

        cdef vec3 center
        cdef vec3 up = [0., 1., 0.]
        cdef vec3 forward = [0., 0., -1.]
        quat_mul_vec3(
            &self._vecUp.x, &self.c_data.Orientation.x, up)
        quat_mul_vec3(
            &self._vecForward.x, &self.c_data.Orientation.x, forward)
        vec3_add(center, &self.c_data.Position.x, &self._vecForward.x)
        mat4x4_look_at(
            self._viewMatrix.M, &self.c_data.Position.x, center, &self._vecUp.x)
        mat4x4_invert_fast(self._invViewMatrix.M, self._viewMatrix.M)

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
        data in the pose data structure (`pxrPosef`). Updating values will
        directly edit the values in the structure. For instance, you can specify
        a component of a pose's position::

            myPose.pos[2] = -10.0  # z = -10.0

        Assigning `pos` a name will create a reference to that `ndarray` which
        can edit values in the structure::

            p = myPose.pos
            p[1] = 1.5  # sets the Y position of 'myPose' to 1.5

        Do not do this since the intermediate object returned by the
        multiplication operator will be garbage collected and `pos` will end up
        being filled with invalid values::

            pos = (myPose * myPose2).pos  # BAD!

            # do this instead ...
            myPoseCombined = myPose * myPose2  # keep intermediate alive
            pos = myPoseCombined.pos  # get the pos

        """
        self._matrixNeedsUpdate = True
        return self._pos

    @pos.setter
    def pos(self, object value):
        self._pos[:] = value
        self._matrixNeedsUpdate = True

    def getPos(self, np.ndarray[np.float32_t, ndim=1] out=None):
        """Position vector X, Y, Z.

        The returned object is a NumPy array which contains a copy of the data
        stored in an internal structure (pxrPosef). The array is conformal with
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
        self._ori[:] = value
        self._matrixNeedsUpdate = True

    def getOri(self, np.ndarray[np.float32_t, ndim=1] out=None):
        """Orientation quaternion X, Y, Z, W. Components X, Y, Z are imaginary
        and W is real.

        The returned object is a NumPy array which references data stored in an
        internal structure (pxrPosef). The array is conformal with the internal
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
        self.pos = value[0]
        self.ori = value[1]

        self._matrixNeedsUpdate = True

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

        self._updateMatrices()

        toReturn[0] = self._vecUp.x
        toReturn[1] = self._vecUp.y
        toReturn[2] = self._vecUp.z

        return toReturn

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
            Optional array to write values to. Must have shape (3,) and a
            float32 data type.

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

        self._updateMatrices()

        toReturn[0] = self._vecForward.x
        toReturn[1] = self._vecForward.y
        toReturn[2] = self._vecForward.z

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
        cdef pxrQuatf q_in
        cdef pxrVector3f axis
        cdef float angle = 0.0
        cdef np.ndarray[np.float32_t, ndim=1] ret_axis = \
            np.zeros((3,), dtype=np.float32)

        # imaginary components of the quaternion
        q_in.x = self.c_data.Orientation.x
        q_in.y = self.c_data.Orientation.y
        q_in.z = self.c_data.Orientation.z
        q_in.w = self.c_data.Orientation.w
        quat_norm(&q_in.x, &q_in.x)

        cdef double sp = sqrt(
            q_in.x * q_in.x + q_in.y * q_in.y + q_in.z * q_in.z)

        cdef float v
        cdef bint non_zero = (
            fabs(q_in.x) > 1e-5 or
            fabs(q_in.y) > 1e-5 or
            fabs(q_in.z) > 1e-5)

        if non_zero:   # has a rotation
            v = <float>1. / <float>sp
            vec3_scale(&axis.x, &q_in.x, v)
            angle = <float>2.0 * <float>acos(q_in.w)
        else:
            axis.x = <float>1.0
            axis.y = axis.z = 0.0

        ret_axis[:] = (axis.x, axis.y, axis.z)

        if degrees:
            angle *= RAD_TO_DEGF

        return ret_axis, angle

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
        cdef pxrVector3f vec_axis
        vec3_set(
            &vec_axis.x,
            <float>axis[0],
            <float>axis[1],
            <float>axis[2])

        cdef float half_rad
        if degrees:
            half_rad = (DEG_TO_RADF * <float>angle) / <float>2.0
        else:
            half_rad = <float>angle / <float>2.0

        vec3_norm(&vec_axis.x, &vec_axis.x)
        cdef bint all_zeros = (
            fabs(vec_axis.x) < 1e-5 and
            fabs(vec_axis.y) < 1e-5 and
            fabs(vec_axis.z) < 1e-5)

        if all_zeros:
            raise ValueError("Value for parameter `axis` is zero-length.")

        vec3_scale(&vec_axis.x, &vec_axis.x, <float>sin(half_rad))
        self.c_data.Orientation.x = vec_axis.x
        self.c_data.Orientation.y = vec_axis.y
        self.c_data.Orientation.z = vec_axis.z
        self.c_data.Orientation.w = <float>cos(half_rad)

        self._matrixNeedsUpdate = True

    def normalize(self):
        """Normalize this pose.
        """
        quat_norm(
            &self.c_data.Orientation.x, &self.c_data.Orientation.x)

        return self

    def normalized(self):
        """Get a normalized version of this pose.

        Returns
        -------
        RigidBodyPose
            Normalized pose.

        """
        cdef pxrPosef* ptr = <pxrPosef*>PyMem_Malloc(
            sizeof(pxrPosef))

        if ptr is NULL:
            raise MemoryError

        quat_norm(
            &ptr.Orientation.x, &self.c_data.Orientation.x)

        ptr.Position = self.c_data.Position

        cdef RigidBodyPose to_return = RigidBodyPose.fromPtr(ptr, owner=True)

        return to_return

    def invert(self):
        """Invert this pose.
        """
        # inverse the rotation
        quat_conj(
            &self.c_data.Orientation.x,
            &self.c_data.Orientation.x)

        # inverse the translation
        vec3_scale(
            &self.c_data.Position.x,
            &self.c_data.Position.x,
            <float>-1.)

        # apply the rotation
        quat_mul_vec3(
            &self.c_data.Position.x,
            &self.c_data.Orientation.x,
            &self.c_data.Position.x)

        self._matrixNeedsUpdate = True

        return self

    def inverted(self):
        """Get the inverse of the pose.

        Returns
        -------
        RigidBodyPose
            Inverted pose.

        """
        cdef pxrPosef* ptr = <pxrPosef*>PyMem_Malloc(
            sizeof(pxrPosef))

        if ptr is NULL:
            raise MemoryError

        # inverse the rotation
        quat_conj(
            &ptr.Orientation.x,
            &self.c_data.Orientation.x)

        # inverse the translation
        vec3_scale(
            &ptr.Position.x,
            &self.c_data.Position.x,
            <float>-1.)

        # apply the rotation
        quat_mul_vec3(
            &ptr.Position.x,
            &ptr.Orientation.x,
            &ptr.Position.x)

        return RigidBodyPose.fromPtr(ptr, True)

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

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = out

        cdef pxrPosef* pose = <pxrPosef*>self.c_data
        cdef pxrVector3f pos_in
        # cdef pxrVector3f pos_rotated

        pos_in.x = <float>v[0]
        pos_in.y = <float>v[1]
        pos_in.z = <float>v[2]

        quat_mul_vec3(
            <float*>toReturn.data,
            &pose.Orientation.x,
            &pos_in.x)

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

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = out

        cdef pxrPosef* pose = <pxrPosef*>self.c_data
        cdef pxrQuatf ori_inv
        cdef pxrVector3f temp

        temp.x = <float>v[0]
        temp.y = <float>v[1]
        temp.z = <float>v[2]

        # inverse the rotation
        quat_conj(&ori_inv.x, &pose.Orientation.x)

        # apply it
        quat_mul_vec3(
            <float*>toReturn.data,
            &ori_inv.x,
            &temp.x)

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

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = out

        cdef pxrPosef* pose = <pxrPosef*>self.c_data
        cdef pxrVector3f temp
        cdef pxrVector3f translated_pos

        temp.x = <float>v[0]
        temp.y = <float>v[1]
        temp.z = <float>v[2]

        vec3_add(<float*>toReturn.data, &temp.x, &pose.Position.x)

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
            Vector transformed by the poses position and orientation.

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = out

        cdef pxrPosef* pose = <pxrPosef*>self.c_data
        cdef pxrVector3f temp

        temp.x = <float>v[0]
        temp.y = <float>v[1]
        temp.z = <float>v[2]

        quat_mul_vec3(&temp.x, &pose.Orientation.x, &temp.x)
        vec3_add(<float*>toReturn.data, &temp.x, &pose.Position.x)

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

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = out

        cdef pxrPosef* pose = <pxrPosef*>self.c_data
        cdef pxrVector3f temp
        cdef pxrQuatf q_inv

        temp.x = <float>v[0]
        temp.y = <float>v[1]
        temp.z = <float>v[2]

        # inverse the rotation and transformation
        quat_conj(&q_inv.x, &pose.Orientation.x)
        vec3_sub(&temp.x, &pose.Position.x, &temp.x)
        quat_mul_vec3(<float*>toReturn.data, &q_inv.x, &temp.x)

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
            Vector transformed by the pose's position and orientation.

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = out

        cdef pxrPosef* pose = <pxrPosef*>self.c_data
        cdef pxrVector3f temp

        temp.x = <float>v[0]
        temp.y = <float>v[1]
        temp.z = <float>v[2]

        # inverse the rotation and transformation
        quat_mul_vec3(<float*>toReturn.data, &pose.Orientation.x, &temp.x)

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
            Normal vector transformed by the inverse pose's position and
            orientation.

        """
        cdef np.ndarray[np.float32_t, ndim=1] toReturn
        if out is None:
            toReturn = np.zeros((3,), dtype=np.float32)
        else:
            toReturn = out

        cdef pxrPosef* pose = <pxrPosef*>self.c_data
        cdef pxrQuatf ori_inv
        cdef pxrVector3f temp

        temp.x = <float>v[0]
        temp.y = <float>v[1]
        temp.z = <float>v[2]

        # inverse the rotation and transformation
        quat_conj(&ori_inv.x, &pose.Orientation.x)
        quat_mul_vec3(<float*>toReturn.data, &ori_inv.x, &temp.x)

        return toReturn

    def apply(self, object v, np.ndarray[np.float32_t, ndim=1] out=None):
        """Apply a transform to a position vector. This is similar to
        `transform`.

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

        """
        return self.transform(v, out)

    def distanceTo(self, object v):
        """Distance to a point or pose from this pose.

        Parameters
        ----------
        v : array_like
            Vector to transform (x, y, z).

        Returns
        -------
        float
            Distance to a point or RigidBodyPose.

        Examples
        --------

        Get the distance between poses::

            distance = thisPose.distanceTo(otherPose)

        Get the distance to a point coordinate::

            distance = thisPose.distanceTo([0.0, 0.0, 5.0])

        """
        cdef pxrVector3f temp
        cdef pxrPosef* pose = <pxrPosef*>self.c_data

        if isinstance(v, RigidBodyPose):
            temp = <pxrVector3f>((<RigidBodyPose>v).c_data[0]).Position
        else:
            temp.x = <float>v[0]
            temp.y = <float>v[1]
            temp.z = <float>v[2]

        temp.x -= pose.Position.x
        temp.y -= pose.Position.y
        temp.z -= pose.Position.z

        cdef float to_return = vec3_len(&temp.x)

        return to_return

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

            M = myPose.getModelMatrix().ctypes.data_as(
                ctypes.POINTER(ctypes.c_float))
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glMultTransposeMatrixf(M)
            # run draw commands ...
            glPopMatrix()

        If using fragment shaders, the matrix can be passed on to them as such::

            M = myPose.getModelMatrix().ctypes.data_as(
                ctypes.POINTER(ctypes.c_float))
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

        cdef pxrMatrix4f* m = NULL

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
    def normalMatrix(self):
        """Normal matrix for transforming normals of meshes associated with
        poses."""
        self._updateMatrices()

        return self._normalMatrixArr

    @property
    def viewMatrix(self):
        """View matrix derived from the current pose."""
        self._updateMatrices()

        return self._viewMatrixArr

    @property
    def inverseViewMatrix(self):
        """View matrix inverse."""
        self._updateMatrices()

        return self._invViewMatrixArr

    def getViewMatrix(self, bint inverse=False, np.ndarray[np.float32_t, ndim=2] out=None):
        """Convert this pose into a view matrix.

        Creates a view matrix which transforms points into eye space using the
        current pose as the eye position in the scene. Furthermore, you can use
        view matrices for rendering shadows if light positions are defined
        as `RigidBodyPose` objects. Using :func:`calcEyePoses` and
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

            iod = 0.062  # 62 mm
            headPose = RigidBodyPose((0., 1.5, 0.))  # 1.5 meters up from origin
            leftEyePose = RigidBodyPose((-(iod / 2.), 0., 0.))
            rightEyePose = RigidBodyPose((iod / 2., 0., 0.))

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

        cdef pxrMatrix4f* m = NULL

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

    def raycastSphere(self, object targetPose, float radius=0.5,
                      object rayDir=(0., 0., -1.), float maxRange=0.0):
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

            targetPose = RigidBodyPose((0.0, 1.5, -5.0))
            targetRadius = 0.5  # 2.5 cm
            isAligned = hmdPose.raycastSphere(targetPose.pos,
                                              radius=targetRadius)

        """
        cdef pxrVector3f targetPos
        cdef pxrVector3f _rayDir

        vec3_set(
            &targetPos.x,
            <float>targetPose[0],
            <float>targetPose[1],
            <float>targetPose[2])
        vec3_set(
            &_rayDir.x,
            <float>rayDir[0],
            <float>rayDir[1],
            <float>rayDir[2])

        # If the ray is finite, does it ever touch the edge of the sphere? If
        # not, exit the routine to avoid wasting time calculating the intercept.
        cdef float targetDist
        if maxRange != 0.0:
            targetDist = \
                vec3_dist(&targetPos.x, &self.c_data.Position.x) - radius
            if targetDist > maxRange:
                return False

        # put the target in the ray caster's local coordinate system
        cdef pxrQuatf ori_inv
        cdef pxrVector3f offset

        # inverse the rotation and transformation
        quat_conj(&ori_inv.x, &self.c_data.Orientation.x)
        vec3_sub(&offset.x, &offset.x, &self.c_data.Position.x)
        quat_mul_vec3(&offset.x, &ori_inv.x, &offset.x)
        vec3_scale(&offset.x, &offset.x, <float>-1.)

        # find the discriminant, this is based on the method described here:
        # http://antongerdelan.net/opengl/raycasting.html
        cdef float u = vec3_mul_inner(&_rayDir.x, &offset.x)
        cdef float v = vec3_mul_inner(&offset.x, &offset.x)
        cdef float desc = <float>pow(u, 2.0) - (v - <float>pow(radius, 2.0))

        # one or more roots? if so we are touching the sphere
        return desc >= 0.0

    # def toBytes(self):
    #     """Get the position and orientation struct as bytes.
    #
    #     This can be used to serialize data about this pose into a format that
    #     can be more readily sent over a network or written to a binary file.
    #
    #     Returns
    #     -------
    #     bytes
    #
    #     """
    #     pass


cdef class BoundingBox(object):
    """Class for constructing and representing 3D bounding boxes.

    A bounding box is a construct which represents a 3D rectangular volume
    about some pose, defined by its minimum and maximum extents in the reference
    frame of the pose or world coordinate system. The axes of the bounding box
    are aligned to the axes of the world or the associated pose.

    Bounding boxes are primarily used for visibility testing; to determine if
    the extents of an object associated with a pose (eg. the vertices of a
    model) falls completely outside of the viewing frustum. If so, the model can
    be culled during rendering to avoid wasting CPU/GPU resources on objects not
    visible to the viewer. See :func:`cullPose` for more information.

    Bounding boxes on their own are axis-aligned. When associated with a
    :class:`RigidBodyPose` class, they become object-aligned and are transformed
    by their parent pose during visibility checks.

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
        bbox = BoundingBox(bounds)
        modelPose = BoundingBox()
        modelPose.boundingBox = bbox

    """
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
    cdef BoundingBox fromPtr(pxrBounds3f* ptr, bint owner):
        cdef BoundingBox wrapper = BoundingBox.__new__(BoundingBox)
        wrapper.c_data = ptr
        wrapper.ptr_owner = owner

        wrapper._mins = _wrap_pxrVector3f_as_ndarray(<pxrVector3f*>&ptr.b[0])
        wrapper._maxs = _wrap_pxrVector3f_as_ndarray(<pxrVector3f*>&ptr.b[1])

        return wrapper

    cdef void _new_struct(self, object extents):
        if self.c_data is not NULL:
            return

        cdef pxrBounds3f* ptr = \
            <pxrBounds3f*>PyMem_Malloc(sizeof(pxrBounds3f))

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
            # just clear
            ptr.b[0].x = ptr.b[0].y = ptr.b[0].z = FLT_MAX
            ptr.b[1].x = ptr.b[1].y = ptr.b[1].z = -FLT_MAX

        self.c_data = ptr
        self.ptr_owner = True

        self._mins = _wrap_pxrVector3f_as_ndarray(<pxrVector3f*>&ptr.b[0])
        self._maxs = _wrap_pxrVector3f_as_ndarray(<pxrVector3f*>&ptr.b[1])

    def __dealloc__(self):
        if self.c_data is not NULL:
            if self.ptr_owner is True:
                PyMem_Free(self.c_data)
                self.c_data = NULL

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

    def clear(self):
        """Clear the bounding box.

        After calling, the `isValid` property will be `False` until either
        `mins` and `maxs` is specified or new points are provided.

        """
        self.c_data.b[0].x = self.c_data.b[0].y = self.c_data.b[0].z = FLT_MAX
        self.c_data.b[1].x = self.c_data.b[1].y = self.c_data.b[1].z = -FLT_MAX

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
            bbox = BoundingBox()
            bbox.fit(vertices)

            # associate the bounding box to a pose
            modelPose = RigidBodyPose()
            modelPose.bounds = bbox

        """
        cdef np.ndarray[np.float32_t, ndim=2] points_in = np.asarray(
            points, dtype=np.float32)

        if clear:
            self.clear()

        cdef Py_ssize_t i, N
        cdef float px, py, pz
        cdef float[:, :] mv_points = points_in  # memory view
        N = <Py_ssize_t>points_in.shape[0]
        for i in range(N):
            px = mv_points[i, 0]
            py = mv_points[i, 1]
            pz = mv_points[i, 2]

            # expand the bounding box if the point falls outside it
            self.c_data.b[0].x = \
                self.c_data.b[0].x if self.c_data.b[0].x < px else px
            self.c_data.b[0].y = \
                self.c_data.b[0].y if self.c_data.b[0].y < py else py
            self.c_data.b[0].z = \
                self.c_data.b[0].z if self.c_data.b[0].z < pz else pz
            self.c_data.b[1].x = \
                self.c_data.b[1].x if px < self.c_data.b[1].x else px
            self.c_data.b[1].y = \
                self.c_data.b[1].y if py < self.c_data.b[1].y else py
            self.c_data.b[1].z = \
                self.c_data.b[1].z if pz < self.c_data.b[1].z else pz

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


def calcEyePoses(RigidBodyPose headPose, float iod):
    """Compute the poses of the viewer's eyes given the tracked head position.

    Parameters
    ----------
    headPose : RigidBodyPose
        Object representing the pose of the head. This should be transformed so
        that the position is located between the viewer's eyes.
    iod : float
        Interocular (or lens) separation of the viewer in meters (m).

    Returns
    -------
    tuple
        Left and right eye poses as `RigidBodyPose` objects.

    Examples
    --------
    Calculate the poses of the user's eyes given the tracked head position and
    get the view matrices for rendering::

        leftEyePose, rightEyePose = calcEyePoses(headPose, iod=0.062)

        leftViewMatrix = leftEyePose.viewMatrix
        rightViewMatrix = rightViewMatrix.viewMatrix

    """
    cdef float[2] eyeOffset
    cdef float halfIOD = <float>iod / <float>2.0
    eyeOffset[0] = -halfIOD  # left eye
    eyeOffset[1] = halfIOD  # right eye

    cdef Py_ssize_t eye = 0
    cdef Py_ssize_t eye_count = 2
    cdef pxrPosef* eyePoses[2]
    cdef pxrPosef* this_pose = NULL
    for eye in range(eye_count):
        # allocate new pose object
        this_pose = <pxrPosef*>PyMem_Malloc(sizeof(pxrPosef))
        if this_pose is NULL:
            raise MemoryError

        # eyes have the same orientation as the head, facing forward
        this_pose.Orientation = headPose.c_data.Orientation

        # clear the position vector
        vec3_zero(&this_pose.Position.x)

        # apply the transformation
        this_pose.Position.x = eyeOffset[eye]
        quat_mul_vec3(
            &this_pose.Position.x,
            &this_pose.Orientation.x,
            &this_pose.Position.x)
        vec3_add(
            &this_pose.Position.x,
            &headPose.c_data.Position.x,
            &this_pose.Position.x)

        eyePoses[eye] = this_pose

    return RigidBodyPose.fromPtr(eyePoses[0], True), \
           RigidBodyPose.fromPtr(eyePoses[1], True)


def cullPose(np.ndarray[np.float32_t, ndim=2] eyeViewMatrix,
             np.ndarray[np.float32_t, ndim=2] eyeProjectionMatrix,
             RigidBodyPose pose):
    """Test if a pose's bounding box or position falls outside of an eye's view
    frustum.

    Poses can be assigned bounding boxes which enclose any 3D models associated
    with them. A model is not visible if all the corners of the bounding box
    fall outside the viewing frustum. Therefore any primitives (i.e. triangles)
    associated with the pose can be culled during rendering to reduce CPU/GPU
    workload.

    If `pose` does not have a valid bounding box (:py:class:`BoundingBox`)
    assigned to its :py:attr:`~RigidBodyPose.bounds` attribute, this function
    will test is if the position of `pose` is outside the view frustum.

    Parameters
    ----------
    eyeViewMatrix : ndarray
        View matrix of the eye viewing the scene.
    eyeProjectionMatrix : ndarray
        Projection matrix of the eye viewing the scene.
    pose : RigidBodyPose
        Pose to test.

    Returns
    -------
    bool
        ``True`` if the pose's bounding box is not visible to the given `eye`
        and should be culled during rendering.

    Examples
    --------
    Check if a pose should be culled (needs to be done for each eye)::

        cullModel = cullPose(eyePose.viewMatrix, projectionMatrix, pose)
        if not cullModel:
            # ... OpenGL calls to draw the model here ...

    Notes
    -----
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
    cdef pxrBounds3f* bbox
    cdef pxrVector4f test_point
    cdef pxrVector4f[8] corners
    cdef Py_ssize_t i

    # compute the MVP matrix to transform poses into HCS
    cdef pxrMatrix4f mvp
    mat4x4_mul(&mvp.M[0], <vec4*>eyeProjectionMatrix.data, <vec4*>eyeViewMatrix.data)
    mat4x4_mul(&mvp.M[0], <vec4*>eyeProjectionMatrix.data, &mvp.M[0])

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
            test_point.x = bbox.b[1].x if (i & 1) else bbox.b[0].x
            test_point.y = bbox.b[1].y if (i & 2) else bbox.b[0].y
            test_point.z = bbox.b[1].z if (i & 4) else bbox.b[0].z
            test_point.w = 1.0

            mat4x4_mul_vec4(&corners[i].x, &mvp.M[0], &test_point.x)

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
        test_point.x = pose.c_data.Position.x
        test_point.y = pose.c_data.Position.y
        test_point.z = pose.c_data.Position.z
        test_point.w = 1.0

        mat4x4_mul_vec4(&test_point.x, &mvp.M[0], &test_point.x)

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


cdef void mult_quat(float* a, float* b, float* out):
    """Multiply two quaternions.
    """
    out[0] = (a[1] * b[2] - a[2] * b[1]) + a[0] * b[3] + b[0] * a[3]
    out[1] = (a[2] * b[0] - a[0] * b[2]) + a[1] * b[3] + b[1] * a[3]
    out[2] = (a[0] * b[1] - a[1] * b[0]) + a[2] * b[3] + b[2] * a[3]
    out[3] = a[3] * b[3] - (b[0] * a[0] + b[1] * a[1] + b[2] * a[2])


def multQuat(object q1, object q2, np.ndarray[np.float32_t, ndim=1] out=None):
    """Multiply two quaternions, combining their rotations.

    Parameters
    ----------
    q1, q2 : ArrayLike
        Quaternions (x, y, z, w) to multiply.
    out : ndarray
        Optional output array for values. Must be length 4 and have a dtype of
        np.float32.

    Returns
    -------
    ndarray
        Quaternion (x, y, z, w).

    """
    cdef np.ndarray[np.float32_t, ndim=1] a = np.asarray(q1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] b = np.asarray(q2, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] toReturn

    if out is None:
        toReturn = np.empty((4,), dtype=np.float32)
    else:
        assert len(out) == 4
        toReturn = out

    mult_quat(<float*>a.data, <float*>b.data, <float*>toReturn.data)

    return toReturn


if __name__ == "__main__":
    pass