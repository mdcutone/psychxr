#  =============================================================================
#  Test routines for the `vrmath` module
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

import numpy as np
from psychxr.tools.vrmath import *
from psychxr.drivers.libovr import LibOVRPose


def test_get_and_set():
    """Test for various getter and setters for `vrmath` classes.
    """
    # create a new rigid body pose object
    rbp1 = RigidBodyPose()

    # check if pose information has been initialized correctly
    assert np.allclose(rbp1.pos, [0., 0., 0.])
    assert np.allclose(rbp1.ori, [0., 0., 0., 1.])

    # write a value to `pos` using a list, check if it did so
    new_pos = np.array([1., 1., -1.])
    rbp1.pos[:] = new_pos
    assert np.allclose(rbp1.pos, new_pos)

    # do the same for `ori`
    new_ori = np.array([1., 1., -1., 1.])
    rbp1.ori[:] = new_ori
    assert np.allclose(rbp1.ori, new_ori)

    # clear the pose and check
    rbp1.setIdentity()
    assert np.allclose(rbp1.pos, [0., 0., 0.])
    assert np.allclose(rbp1.ori, [0., 0., 0., 1.])

    # check `posOri` attribute setter
    rbp1.posOri = (new_pos, new_ori)
    assert np.allclose(rbp1.pos, new_pos) and np.allclose(rbp1.ori, new_ori)

    # check `posOri` getter
    pos, ori = rbp1.posOri
    assert np.allclose(pos, new_pos) and np.allclose(ori, new_ori)


def test_pose_operators():
    """Test the `RigidBodPose` multiply and invert operators.
    """
    # test rotation only
    rbp1 = RigidBodyPose()
    rbp2 = RigidBodyPose()

    rbp1.setOriAxisAngle((0, 1, 0), 90.0, degrees=True)
    rbp2.setOriAxisAngle((0, 1, 0), 90.0, degrees=True)

    rbp3 = rbp1 * rbp2  # combine

    axis, angle = rbp3.getOriAxisAngle()
    assert np.allclose(axis, (0, 1, 0)) and np.allclose(angle, 180.0)

    # check if the inplace operator works too
    rbp1 *= rbp2
    assert rbp1 == rbp3

    # check if computed matrices are inverses
    ident = np.identity(4)  # expected value
    assert np.allclose(rbp3.modelMatrix @ rbp3.inverseModelMatrix, ident)
    assert np.allclose(rbp3.viewMatrix @ rbp3.inverseViewMatrix, ident)

    # test the inverse operator
    rbp3Inv = ~rbp3
    inverseModelMatrix = rbp3Inv.modelMatrix
    assert np.allclose(rbp3.modelMatrix @ inverseModelMatrix, ident)


def test_compare_rigid_body_types():
    """Tests to ensure the LibOVR and VRTools versions of the rigid body classes
    are equivalent in function.
    """
    # create test instances for rigid body types
    pose_libovr = LibOVRPose()
    pose_vrmath = RigidBodyPose()

    # test `setOriAxisAngle` and other properties
    np.random.seed(12345)
    N = 1000
    angles = np.random.uniform(low=-360., high=360., size=(N,))
    axes = np.random.uniform(low=-1., high=1., size=(N, 3))
    pos = np.random.uniform(low=-10., high=10., size=(N, 3))

    for i in range(N):
        pose_libovr.setPos(pos[i])
        pose_vrmath.setPos(pos[i])
        pose_libovr.setOriAxisAngle(axes[i], angles[i], degrees=True)
        pose_vrmath.setOriAxisAngle(axes[i], angles[i], degrees=True)

        # test properties and getter/setter methods (e.g., `pos`, `ori`, etc.)
        assert np.allclose(pose_vrmath.pos, pose_libovr.pos)
        assert np.allclose(pose_vrmath.getPos(), pose_libovr.getPos())
        assert np.allclose(pose_vrmath.ori, pose_libovr.ori)
        assert np.allclose(pose_vrmath.getOri(), pose_libovr.getOri())
        assert np.allclose(pose_vrmath.at, pose_libovr.at, atol=1e-5)
        assert np.allclose(pose_vrmath.getAt(), pose_libovr.getAt(), atol=1e-5)
        assert np.allclose(pose_vrmath.up, pose_libovr.up, atol=1e-5)
        assert np.allclose(pose_vrmath.getUp(), pose_libovr.getUp(), atol=1e-5)

        # test `inverted`
        assert np.allclose(
            pose_vrmath.inverted().pos, pose_libovr.inverted().pos, atol=1e-5)
        assert np.allclose(
            pose_vrmath.inverted().ori, pose_libovr.inverted().ori, atol=1e-5)
        assert np.allclose(  # model matrix
            pose_vrmath.modelMatrix,
            pose_libovr.modelMatrix,
            atol=1e-5)
        assert np.allclose(  # inverse model matrix
            pose_vrmath.inverseModelMatrix,
            pose_libovr.inverseModelMatrix,
            atol=1e-5)
        assert np.allclose(  # view matrix
            pose_vrmath.viewMatrix,
            pose_libovr.viewMatrix,
            atol=1e-5)
        assert np.allclose(  # inverse view matrix
            pose_vrmath.inverseViewMatrix,
            pose_libovr.inverseViewMatrix,
            atol=1e-5)
        assert np.allclose(  # normal matrix
            pose_vrmath.normalMatrix,
            pose_libovr.normalMatrix,
            atol=1e-5)

    # reset
    pose_libovr.setIdentity()
    pose_vrmath.setIdentity()

    assert np.allclose(pose_vrmath.pos, pose_libovr.pos)
    assert np.allclose(pose_vrmath.ori, pose_libovr.ori)


if __name__ == "__main__":
    test_get_and_set()
    test_pose_operators()
    test_compare_rigid_body_types()
