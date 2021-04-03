# distutils: language=c++
#  =============================================================================
#  _openhmd.pyx - Python Interface Module for OpenHMD
#  =============================================================================
#
#  Copyright 2021 Matthew Cutone <mcutone@opensciencetools.com> and Laurie M.
#  Wilcox <lmwilcox(a)yorku.ca>; The Centre For Vision Research, York
#  University, Toronto, Canada
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
__version__ = "0.2.4"
__status__ = "Stable"
__maintainer__ = "Matthew D. Cutone"
__email__ = "mcutone@opensciencetools.com"

# ------------------------------------------------------------------------------
# Module information
#
__all__ = ['RigidBodyPose']

import ctypes
from . cimport vrmath
from . cimport linmath

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.math cimport pow, tan, M_PI, atan2, sqrt, fabs

import numpy as np
cimport numpy as np
np.import_array()


cdef np.npy_intp[1] VEC2_SHAPE = [2]
cdef np.npy_intp[1] VEC3_SHAPE = [3]
cdef np.npy_intp[1] FOVPORT_SHAPE = [4]
cdef np.npy_intp[1] QUAT_SHAPE = [4]
cdef np.npy_intp[2] MAT4_SHAPE = [4, 4]


cdef np.ndarray _wrap_pxrVector3f_as_ndarray(vrmath.pxrVector3f* prtVec):
    """Wrap an pxrVector3f object with a NumPy array."""
    return np.PyArray_SimpleNewFromData(
        1, VEC3_SHAPE, np.NPY_FLOAT32, <void*>prtVec)


cdef np.ndarray _wrap_pxrQuatf_as_ndarray(vrmath.pxrQuatf* prtVec):
    """Wrap an pxrQuatf object with a NumPy array."""
    return np.PyArray_SimpleNewFromData(
        1, QUAT_SHAPE, np.NPY_FLOAT32, <void*>prtVec)


cdef np.ndarray _wrap_pxrMatrix4f_as_ndarray(vrmath.pxrMatrix4f* prtVec):
    """Wrap an pxrMatrix4f object with a NumPy array."""
    return np.PyArray_SimpleNewFromData(
        2, MAT4_SHAPE, np.NPY_FLOAT32, <void*>prtVec.M)


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
    cdef vrmath.pxrPosef* c_data
    cdef bint ptr_owner

    cdef np.ndarray _pos
    cdef np.ndarray _ori

    cdef vrmath.pxrMatrix4f _modelMatrix
    cdef vrmath.pxrMatrix4f _invModelMatrix
    cdef vrmath.pxrMatrix4f _normalMatrix
    cdef vrmath.pxrMatrix4f _viewMatrix
    cdef vrmath.pxrMatrix4f _invViewMatrix

    cdef vrmath.pxrVector3f _vecUp
    cdef vrmath.pxrVector3f _vecForward

    cdef np.ndarray _modelMatrixArr
    cdef np.ndarray _invModelMatrixArr
    cdef np.ndarray _normalMatrixArr
    cdef np.ndarray _viewMatrixArr
    cdef np.ndarray _invViewMatrixArr
    cdef dict _ptrMatrices

    cdef bint _matrixNeedsUpdate

    def __init__(self, pos=(0., 0., 0.), ori=(0., 0., 0., 1.)):
        self._new_struct(pos, ori)

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

    @staticmethod
    cdef RigidBodyPose fromPtr(vrmath.pxrPosef* ptr, bint owner=False):
        cdef RigidBodyPose wrapper = RigidBodyPose.__new__(RigidBodyPose)
        wrapper.c_data = ptr
        wrapper.ptr_owner = owner

        wrapper._pos = _wrap_pxrVector3f_as_ndarray(&ptr.Position)
        wrapper._ori = _wrap_pxrQuatf_as_ndarray(&ptr.Orientation)
        # wrapper._bbox = None
        wrapper._matrixNeedsUpdate = True

        return wrapper

    cdef void _new_struct(self, object pos, object ori):
        if self.c_data is not NULL:
            return

        cdef vrmath.pxrPosef* ptr = \
            <vrmath.pxrPosef*>PyMem_Malloc(sizeof(vrmath.pxrPosef))

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
        self._matrixNeedsUpdate = True

    def __dealloc__(self):
        # don't do anything crazy like set c_data=NULL without deallocating!
        if self.c_data is not NULL:
            if self.ptr_owner is True:
                PyMem_Free(self.c_data)
                self.c_data = NULL

    def __repr__(self):
        return f'RigidBodyPose(pos={repr(self.pos)}, ori={repr(self.ori)})'

    def __imul__(self, RigidBodyPose other):
        """Multiplication operator (*=) to combine poses.
        """

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
        cdef vrmath.pxrPosef* ptr = \
            <vrmath.pxrPosef*>PyMem_Malloc(sizeof(vrmath.pxrPosef))

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
        cdef vrmath.pxrVector3f* pos = &pose.c_data.Position
        cdef vrmath.pxrQuatf* ori = &pose.c_data.Orientation
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

    def _updateMatrices(self):
        """Update model, inverse, and normal matrices due to an attribute change.
        """
        if not self._matrixNeedsUpdate:
            return

        cdef linmath.mat4x4 m_rotate
        cdef linmath.mat4x4 m_translate

        # compute model matrix
        linmath.mat4x4_from_quat(m_rotate, &self.c_data.Orientation.x)
        linmath.mat4x4_translate(
            m_translate,
            self.c_data.Position.x,
            self.c_data.Position.y,
            self.c_data.Position.z)
        linmath.mat4x4_mul(self._modelMatrix.M, m_translate, m_rotate)

        # get its inverse
        linmath.mat4x4_invert(self._invModelMatrix.M, self._modelMatrix.M)

        # normal matrix
        linmath.mat4x4_transpose(self._normalMatrix.M, self._invModelMatrix.M)

        cdef linmath.vec3 center
        cdef linmath.vec3 up = [0., 1., 0.]
        cdef linmath.vec3 forward = [0., 0., -1.]
        linmath.quat_mul_vec3(
            &self._vecUp.x, &self.c_data.Orientation.x, up)
        linmath.quat_mul_vec3(
            &self._vecForward.x, &self.c_data.Orientation.x, forward)
        linmath.vec3_add(center, &self.c_data.Position.x, &self._vecForward.x)
        linmath.mat4x4_look_at(
            self._viewMatrix.M, &self.c_data.Position.x, center, &self._vecUp.x)
        linmath.mat4x4_invert(self._invViewMatrix.M, self._viewMatrix.M)

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
        self._matrixNeedsUpdate = True

        self._ori[:] = value

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

    def normalize(self):
        """Normalize this pose.
        """
        linmath.quat_norm(
            &self.c_data.Orientation.x, &self.c_data.Orientation.x)

        return self

    def normalized(self):
        """Get a normalized version of this pose.

        Returns
        -------
        RigidBodyPose
            Normalized pose.

        """
        cdef vrmath.pxrPosef* ptr = <vrmath.pxrPosef*>PyMem_Malloc(
            sizeof(vrmath.pxrPosef))

        if ptr is NULL:
            raise MemoryError

        linmath.quat_norm(
            &ptr.Orientation.x, &self.c_data.Orientation.x)

        ptr.Position = self.c_data.Position

        cdef RigidBodyPose to_return = RigidBodyPose.fromPtr(ptr, owner=True)

        return to_return

    def invert(self):
        """Invert this pose.
        """
        # inverse the rotation
        linmath.quat_conj(
            &self.c_data.Orientation.x,
            &self.c_data.Orientation.x)

        # inverse the translation
        linmath.vec3_scale(
            &self.c_data.Position.x,
            &self.c_data.Position.x,
            <float>-1.)

        # apply the rotation
        linmath.quat_mul_vec3(
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
        cdef vrmath.pxrPosef* ptr = <vrmath.pxrPosef*>PyMem_Malloc(
            sizeof(vrmath.pxrPosef))

        if ptr is NULL:
            raise MemoryError

        # inverse the rotation
        linmath.quat_conj(
            &ptr.Orientation.x,
            &self.c_data.Orientation.x)

        # inverse the translation
        linmath.vec3_scale(
            &ptr.Position.x,
            &self.c_data.Position.x,
            <float>-1.)

        # apply the rotation
        linmath.quat_mul_vec3(
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

        cdef vrmath.pxrPosef* pose = <vrmath.pxrPosef*>self.c_data
        cdef vrmath.pxrVector3f pos_in
        cdef vrmath.pxrVector3f pos_rotated

        pos_in.x = <float>v[0]
        pos_in.y = <float>v[1]
        pos_in.z = <float>v[2]

        linmath.quat_mul_vec3(
            &pos_rotated.x,
            &pose.Orientation.x,
            &pos_in.x)

        toReturn[0] = pos_rotated.x
        toReturn[1] = pos_rotated.y
        toReturn[2] = pos_rotated.z

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

        cdef vrmath.pxrPosef* pose = <vrmath.pxrPosef*>self.c_data
        cdef vrmath.pxrQuatf ori_inv
        cdef vrmath.pxrVector3f temp

        temp.x = <float>v[0]
        temp.y = <float>v[1]
        temp.z = <float>v[2]

        # inverse the rotation
        linmath.quat_conj(&ori_inv.x, &pose.Orientation.x)

        # apply it
        linmath.quat_mul_vec3(
            &temp.x,
            &ori_inv.x,
            &temp.x)

        toReturn[0] = temp.x
        toReturn[1] = temp.y
        toReturn[2] = temp.z

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

        cdef vrmath.pxrPosef* pose = <vrmath.pxrPosef*>self.c_data
        cdef vrmath.pxrVector3f temp
        cdef vrmath.pxrVector3f translated_pos

        temp.x = <float>v[0]
        temp.y = <float>v[1]
        temp.z = <float>v[2]

        linmath.vec3_add(&temp.x, &temp.x, &pose.Position.x)

        toReturn[0] = temp.x
        toReturn[1] = temp.y
        toReturn[2] = temp.z

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

        cdef vrmath.pxrPosef* pose = <vrmath.pxrPosef*>self.c_data
        cdef vrmath.pxrVector3f temp

        temp.x = <float>v[0]
        temp.y = <float>v[1]
        temp.z = <float>v[2]

        linmath.quat_mul_vec3(&temp.x, &pose.Orientation.x, &temp.x)
        linmath.vec3_add(&temp.x, &temp.x, &pose.Position.x)

        toReturn[0] = temp.x
        toReturn[1] = temp.y
        toReturn[2] = temp.z

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

        cdef vrmath.pxrPosef* pose = <vrmath.pxrPosef*>self.c_data
        cdef vrmath.pxrQuatf ori_inv
        cdef vrmath.pxrVector3f temp

        temp.x = <float>v[0]
        temp.y = <float>v[1]
        temp.z = <float>v[2]

        # inverse the rotation and transformation
        linmath.quat_conj(&ori_inv.x, &pose.Orientation.x)
        linmath.vec3_sub(&temp.x, &temp.x, &pose.Position.x)
        linmath.quat_mul_vec3(&temp.x, &ori_inv.x, &temp.x)

        toReturn[0] = temp.x
        toReturn[1] = temp.y
        toReturn[2] = temp.z

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

        cdef vrmath.pxrPosef* pose = <vrmath.pxrPosef*>self.c_data
        cdef vrmath.pxrVector3f temp

        temp.x = <float>v[0]
        temp.y = <float>v[1]
        temp.z = <float>v[2]

        # inverse the rotation and transformation
        linmath.quat_mul_vec3(&temp.x, &pose.Orientation.x, &temp.x)

        toReturn[0] = temp.x
        toReturn[1] = temp.y
        toReturn[2] = temp.z

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

        cdef vrmath.pxrPosef* pose = <vrmath.pxrPosef*>self.c_data
        cdef vrmath.pxrQuatf ori_inv
        cdef vrmath.pxrVector3f temp

        temp.x = <float>v[0]
        temp.y = <float>v[1]
        temp.z = <float>v[2]

        # inverse the rotation and transformation
        linmath.quat_conj(&ori_inv.x, &pose.Orientation.x)
        linmath.quat_mul_vec3(&temp.x, &ori_inv.x, &temp.x)

        toReturn[0] = temp.x
        toReturn[1] = temp.y
        toReturn[2] = temp.z

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
            Distance to a point or RigidBodyPose.

        Examples
        --------

        Get the distance between poses::

            distance = thisPose.distanceTo(otherPose)

        Get the distance to a point coordinate::

            distance = thisPose.distanceTo([0.0, 0.0, 5.0])

        """
        cdef vrmath.pxrVector3f temp
        cdef vrmath.pxrPosef* pose = <vrmath.pxrPosef*>self.c_data

        if isinstance(v, RigidBodyPose):
            temp = <vrmath.pxrVector3f>((<RigidBodyPose>v).c_data[0]).Position
        else:
            temp.x = <float>v[0]
            temp.y = <float>v[1]
            temp.z = <float>v[2]

        temp.x -= pose.Position.x
        temp.y -= pose.Position.y
        temp.z -= pose.Position.z

        cdef float to_return = linmath.vec3_len(&temp.x)

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

        cdef vrmath.pxrMatrix4f* m = NULL

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

            iod = 0.062  # 63 mm
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

        cdef vrmath.pxrMatrix4f* m = NULL

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
    """Class representing a bounding box."""
    pass

