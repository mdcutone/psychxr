#  =============================================================================
#  Oculus(TM) Rift SDK Python Interface Module
#  =============================================================================
#
#  rift.pxy
#
#  Copyright 2018 Matthew Cutone <cutonem(a)yorku.ca> and Laurie M. Wilcox
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
"""This file exposes LibOVR functions to Python.

"""
cimport ovr_capi, ovr_capi_gl, ovr_errorcode, ovr_capi_util
cimport ovr_math
cimport libc.math as cmath
import OpenGL.GL as GL

# -----------------
# Initialize module
# -----------------
#
cdef ovr_capi.ovrInitParams _init_params_  # initialization parameters

# HMD descriptor storing information about the HMD being used.
cdef ovr_capi.ovrHmdDesc _hmd_desc_

# Since we are only using one session per module instance, so we are going to
# create our session pointer here and use it module-wide.
#
cdef ovr_capi.ovrSession _ptr_session_
cdef ovr_capi.ovrGraphicsLuid _ptr_luid_

# Frame index
#
cdef long long _frame_index_ = 0

# create an array of texture swap chains
#
cdef ovr_capi.ovrTextureSwapChain _swap_chain_[32]

# mirror texture swap chain, we only create one here
#
cdef ovr_capi.ovrMirrorTexture _mirror_texture_ = NULL

# Persistent VR related structures to store head pose and other data used across
# frames.
#
cdef ovr_capi.ovrEyeRenderDesc[2] _eye_render_desc_
cdef ovr_capi.ovrPosef[2] _hmd_to_eye_view_pose_

# Render layer
#
cdef ovr_capi.ovrLayerEyeFov _eye_layer_

# Arrays to store device poses.
#
cdef ovr_capi.ovrTrackedDeviceType[9] _device_types_
cdef ovr_capi.ovrPoseStatef[9] _device_poses_

# Session status
#
cdef ovr_capi.ovrSessionStatus _session_status_

# Function to check for errors returned by OVRLib functions
#
cdef ovr_errorcode.ovrErrorInfo _last_error_info_  # store our last error here
def check_result(result):
    if ovr_errorcode.OVR_FAILURE(result):
        ovr_capi.ovr_GetLastErrorInfo(&_last_error_info_)
        raise RuntimeError(
            str(result) + ": " + _last_error_info_.ErrorString.decode("utf-8"))

# Enable error checking on OVRLib functions by setting 'debug_mode=True'. All
# LibOVR functions that return a 'ovrResult' type will be checked. A
# RuntimeError will be raised if the returned value indicates failure with the
# associated message passed from LibOVR.
#
debug_mode = False

# Controller indices in controller state array.
#
ctypedef enum LibOVRControllers:
    xbox = 0
    remote = 1
    touch = 2
    left_touch = 3
    right_touch = 4
    count = 5

# Store controller states.
#
cdef ovr_capi.ovrInputState _ctrl_states_[5]
cdef ovr_capi.ovrInputState _ctrl_states_prev_[5]  # previous controller states

# Controller indices look-up table.
#
cdef dict ctrl_index_lut = {
    "xbox": LibOVRControllers.xbox,
    "remote": LibOVRControllers.remote,
    "touch": LibOVRControllers.touch,
    "left_touch": LibOVRControllers.left_touch,
    "right_touch": LibOVRControllers.right_touch
}

# Look-up table of button values to test which are pressed.
#
cdef dict ctrl_button_lut = {
    "A": ovr_capi.ovrButton_A,
    "B": ovr_capi.ovrButton_B,
    "RThumb" : ovr_capi.ovrButton_RThumb,
    "RShoulder": ovr_capi.ovrButton_RShoulder,
    "X": ovr_capi.ovrButton_X,
    "Y": ovr_capi.ovrButton_Y,
    "LThumb": ovr_capi.ovrButton_LThumb,
    "LShoulder": ovr_capi.ovrButton_LThumb,
    "Up": ovr_capi.ovrButton_Up,
    "Down": ovr_capi.ovrButton_Down,
    "Left": ovr_capi.ovrButton_Left,
    "Right": ovr_capi.ovrButton_Right,
    "Enter": ovr_capi.ovrButton_Enter,
    "Back": ovr_capi.ovrButton_Back,
    "VolUp": ovr_capi.ovrButton_VolUp,
    "VolDown": ovr_capi.ovrButton_VolDown,
    "Home": ovr_capi.ovrButton_Home,
    "Private": ovr_capi.ovrButton_Private,
    "RMask": ovr_capi.ovrButton_RMask,
    "LMask": ovr_capi.ovrButton_LMask}

# Python accessible list of valid button names.
button_names = list(ctrl_button_lut.keys())

# Look-up table of controller touches.
#
cdef dict ctrl_touch_lut = {
    "A": ovr_capi.ovrTouch_A,
    "B": ovr_capi.ovrTouch_B,
    "RThumb" : ovr_capi.ovrTouch_RThumb,
    "RThumbRest": ovr_capi.ovrTouch_RThumbRest,
    "RIndexTrigger" : ovr_capi.ovrTouch_RThumb,
    "X": ovr_capi.ovrTouch_X,
    "Y": ovr_capi.ovrTouch_Y,
    "LThumb": ovr_capi.ovrTouch_LThumb,
    "LThumbRest": ovr_capi.ovrTouch_LThumbRest,
    "LIndexTrigger" : ovr_capi.ovrTouch_LIndexTrigger,
    "RIndexPointing": ovr_capi.ovrTouch_RIndexPointing,
    "RThumbUp": ovr_capi.ovrTouch_RThumbUp,
    "LIndexPointing": ovr_capi.ovrTouch_LIndexPointing,
    "LThumbUp": ovr_capi.ovrTouch_LThumbUp}

# Python accessible list of valid touch names.
touch_names = list(ctrl_touch_lut.keys())

# Performance information for profiling.
#
cdef ovr_capi.ovrPerfStats _perf_stats_


cdef class ovrColorf:
    cdef ovr_capi.ovrColorf* c_data
    cdef ovr_capi.ovrColorf  c_ovrColorf

    def __cinit__(self, float r=0.0, float g=0.0, float b=0.0, float a=0.0):
        self.c_data = &self.c_ovrColorf

        self.c_data.r = r
        self.c_data.g = g
        self.c_data.b = b
        self.c_data.a = a

    @property
    def r(self):
        return self.c_data.r

    @r.setter
    def r(self, float value):
        self.c_data.r = value

    @property
    def g(self):
        return self.c_data.g

    @g.setter
    def g(self, float value):
        self.c_data.g = value

    @property
    def b(self):
        return self.c_data.b

    @b.setter
    def b(self, float value):
        self.c_data.b = value

    @property
    def a(self):
        return self.c_data.a

    @a.setter
    def a(self, float value):
        self.c_data.a = value

    def as_tuple(self):
        return self.c_data.r, self.c_data.g, self.c_data.b, self.c_data.a

# ---------------------
# Oculus SDK Math Types
# ---------------------
#
cdef class ovrVector2i:
    cdef ovr_capi.ovrVector2i* c_data
    cdef ovr_capi.ovrVector2i  c_ovrVector2i

    def __cinit__(self, int x=0, int y=0):
        self.c_data = &self.c_ovrVector2i

        self.c_data.x = x
        self.c_data.y = y

    @property
    def x(self):
        return self.c_data.x

    @x.setter
    def x(self, int value):
        self.c_data.x = value

    @property
    def y(self):
        return self.c_data.y

    @y.setter
    def y(self, int value):
        self.c_data.y = value

    def as_tuple(self):
        return self.c_data.x, self.c_data.y

    def as_list(self):
        return list(self.c_data.x, self.c_data.y)

    def __eq__(self, ovrVector2i b):
        return self.c_data.x == b.c_data.x and self.c_data.y == b.c_data.y

    def __ne__(self, ovrVector2i b):
        return self.c_data.x != b.c_data.x or self.c_data.y != b.c_data.y

    def __add__(ovrVector2i a, ovrVector2i b):
        cdef ovrVector2i to_return = ovrVector2i(
            a.c_data.x + b.c_data.x,
            a.c_data.y + b.c_data.y)

        return to_return

    def __iadd__(self, ovrVector2i b):
        self.c_data.x += b.c_data.x
        self.c_data.y += b.c_data.y

        return self

    def __sub__(ovrVector2i a, ovrVector2i b):
        cdef ovrVector2i to_return = ovrVector2i(
            a.c_data.x - b.c_data.x,
            a.c_data.y - b.c_data.y)

        return to_return

    def __isub__(self, ovrVector2i b):
        self.c_data.x -= b.c_data.x
        self.c_data.y -= b.c_data.y

        return self

    def __neg__(self):
        cdef ovrVector2i to_return = ovrVector2i(-self.c_data.x, -self.c_data.y)

        return to_return

    def __mul__(ovrVector2i a, object b):
        cdef ovrVector2i to_return
        if isinstance(b, ovrVector2i):
            to_return = ovrVector2i(a.c_data.x * b.c_data.x,
                                    a.c_data.y * b.c_data.y)
        elif isinstance(b, (int, float)):
            to_return = ovrVector2i(a.c_data.x * <int>b,
                                    a.c_data.y * <int>b)

        return to_return

    def __imul__(self, object b):
        cdef ovrVector2i to_return
        if isinstance(b, ovrVector2i):
            self.c_data.x *= b.c_data.x
            self.c_data.y *= b.c_data.y
        elif isinstance(b, (int, float)):
            self.c_data.x *= <int>b
            self.c_data.y *= <int>b

        return self

    def __truediv__(ovrVector2i a, object b):
        cdef int rcp = <int>1 / <int>b
        cdef ovrVector2i to_return = ovrVector2i(
            a.c_data.x * rcp,
            a.c_data.y * rcp)

        return to_return

    def __itruediv__(self, object b):
        cdef int rcp = <int>1 / <int>b
        self.c_data.x *= rcp
        self.c_data.y *= rcp

        return self

    @staticmethod
    def min(ovrVector2i a, ovrVector2i b):
        cdef ovrVector2i to_return = ovrVector2i(
            a.c_data.x if a.c_data.x < b.c_data.x else b.c_data.x,
            a.c_data.y if a.c_data.y < b.c_data.y else b.c_data.y)

        return to_return

    @staticmethod
    def max(ovrVector2i a, ovrVector2i b):
        cdef ovrVector2i to_return = ovrVector2i(
            a.c_data.x if a.c_data.x > b.c_data.x else b.c_data.x,
            a.c_data.y if a.c_data.y > b.c_data.y else b.c_data.y)

        return to_return

    def clamped(self, int max_mag):
        cdef int mag_squared = self.length_sq()
        if mag_squared > max_mag * max_mag:
            return self * (max_mag / cmath.sqrt(mag_squared))

        return self

    def is_equal(self, ovrVector2i b, int tolerance = 0):
        return cmath.fabs(b.c_data.x - self.c_data.x) <= tolerance and \
            cmath.fabs(b.c_data.y - self.c_data.y) <= tolerance

    def compare(self, ovrVector2i b, int tolerance = 0):
        return self.is_equal(b, tolerance)

    def __getitem__(self, int idx):
        assert 0 <= idx < 2
        cdef int* ptr_val = &self.c_data.x + idx

        return <int>ptr_val[0]

    def __setitem__(self, int idx, int val):
        assert 0 <= idx < 2
        cdef int* ptr_val = &self.c_data.x + idx
        ptr_val[0] = val

    def entrywise_multiply(self, ovrVector2i b):
        cdef ovrVector2i to_return = ovrVector2i(
            self.c_data.x * b.c_data.x,
            self.c_data.y * b.c_data.y)

        return to_return

    def dot(self, ovrVector2i b):
        cdef int dot_prod = \
            self.c_data.x * b.c_data.x + self.c_data.y * b.c_data.y

        return <int>dot_prod

    def angle(self, ovrVector2i b):
        cdef int div = self.length_sq() * b.length_sq()
        assert div != <int>0
        cdef int to_return = self.dot(b) / cmath.sqrt(div)

        return to_return

    def length_sq(self):
        return \
            <int>(self.c_data.x * self.c_data.x + self.c_data.y * self.c_data.y)

    def length(self):
        return <int>cmath.sqrt(self.length_sq())

    def distance_sq(self, ovrVector2i b):
        return (self - b).length_sq()

    def distance(self, ovrVector2i b):
        return (self - b).length()

    def is_normalized(self):
        return cmath.fabs(self.length_sq() - <int>1) < 0

    def normalize(self):
        cdef int s = self.length()
        if s != <int>0:
            s = <int>1 / s

        self *= s

    def normalized(self):
        cdef int s = self.length()
        if s != <int>0:
            s = <int>1 / s

        return self * s

    def lerp(self, ovrVector2i b, int f):
        return self * (<int>1 - f) + b * f

    def project_to(self, ovrVector2i b):
        cdef int l2 = self.length_sq()
        assert l2 != <int>0

        return b * (self.dot(b) / l2)

    def is_clockwise(self, ovrVector2i b):
        return (self.c_data.x * b.c_data.y - self.c_data.y * b.c_data.x) < 0


cdef class ovrSizei:
    cdef ovr_capi.ovrSizei* c_data
    cdef ovr_capi.ovrSizei  c_ovrSizei

    def __cinit__(self, int w=0, int h=0):
        self.c_data = &self.c_ovrSizei

        self.c_data.w = w
        self.c_data.h = h

    @property
    def w(self):
        return self.c_data.w

    @w.setter
    def w(self, int value):
        self.c_data.w = value

    @property
    def h(self):
        return self.c_data.h

    @h.setter
    def h(self, int value):
        self.c_data.h = value

    def as_tuple(self):
        return self.c_data.w, self.c_data.h

    def as_list(self):
        return [self.c_data.w, self.c_data.h]

    def __eq__(self, ovrSizei b):
        return self.c_data.w == b.c_data.w and self.c_data.h == b.c_data.h

    def __ne__(self, ovrSizei b):
        return self.c_data.w != b.c_data.w or self.c_data.h != b.c_data.h

    def __add__(ovrSizei a, ovrSizei b):
        cdef ovrSizei to_return = ovrSizei(
            a.c_data.w + b.c_data.w,
            a.c_data.h + b.c_data.h)

        return to_return

    def __iadd__(self, ovrSizei b):
        self.c_data.w += b.c_data.w
        self.c_data.h += b.c_data.h

        return self

    def __sub__(ovrSizei a, ovrSizei b):
        cdef ovrSizei to_return = ovrSizei(
            a.c_data.w - b.c_data.w,
            a.c_data.h - b.c_data.h)

        return to_return

    def __isub__(self, ovrSizei b):
        self.c_data.w -= b.c_data.w
        self.c_data.h -= b.c_data.h

        return self

    def __neg__(self):
        cdef ovrSizei to_return = ovrSizei(-self.c_data.w, -self.c_data.h)

        return to_return

    def __mul__(ovrSizei a, object b):
        cdef ovrSizei to_return
        if isinstance(b, ovrSizei):
            to_return = ovrSizei(a.c_data.w * b.c_data.w,
                                 a.c_data.h * b.c_data.h)
        elif isinstance(b, (int, float)):
            to_return = ovrSizei(a.c_data.w * <int>b,
                                 a.c_data.h * <int>b)

        return to_return

    def __imul__(self, object b):
        cdef ovrSizei to_return
        if isinstance(b, ovrSizei):
            self.c_data.w *= b.c_data.w
            self.c_data.h *= b.c_data.h
        elif isinstance(b, (int, float)):
            self.c_data.w *= <int>b
            self.c_data.h *= <int>b

        return self

    def __truediv__(ovrSizei a, object b):
        cdef float rcp = <float>1 / <float>b
        cdef ovrSizei to_return = ovrSizei(
            <int>(<float>a.c_data.w * rcp),
            <int>(<float>a.c_data.h * rcp))

        return to_return

    def __itruediv__(self, object b):
        cdef float rcp = <float>1 / <float>b
        self.c_data.w = <int>(<float>self.c_data.w * rcp)
        self.c_data.h = <int>(<float>self.c_data.h * rcp)

        return self

    @staticmethod
    def min(ovrSizei a, ovrSizei b):
        cdef ovrSizei to_return = ovrSizei(
            a.c_data.w if a.c_data.w < b.c_data.w else b.c_data.w,
            a.c_data.h if a.c_data.h < b.c_data.h else b.c_data.h)

        return to_return

    @staticmethod
    def max(ovrSizei a, ovrSizei b):
        cdef ovrSizei to_return = ovrSizei(
            a.c_data.w if a.c_data.w > b.c_data.w else b.c_data.w,
            a.c_data.h if a.c_data.h > b.c_data.h else b.c_data.h)

        return to_return

    def area(self):
        return self.c_data.w * self.c_data.h

    def to_vector(self):
        cdef ovrVector2i to_return = ovrVector2i(self.c_data.w, self.c_data.h)

        return to_return

    def as_vector(self):
        return self.to_vector()


cdef class ovrRecti:
    cdef ovr_capi.ovrRecti* c_data
    cdef ovr_capi.ovrRecti  c_ovrRecti

    # nested field objects
    cdef ovrVector2i obj_pos
    cdef ovrSizei obj_size

    def __init__(self, *args, **kwargs):
        pass

    def __cinit__(self, *args, **kwargs):
        self.c_data = &self.c_ovrRecti

        cdef int nargin = <int>len(args)  # get number of arguments
        if nargin == 0:
            self.c_data.Pos.x = 0
            self.c_data.Pos.y = 0
            self.c_data.Size.w = 0
            self.c_data.Size.h = 0
        elif nargin == 4:
            self.c_data.Pos.x = args[0]
            self.c_data.Pos.y = args[1]
            self.c_data.Size.w = args[2]
            self.c_data.Size.h = args[3]
        elif nargin == 2 and \
                isinstance(args[0], ovrVector2i) and \
                isinstance(args[1], ovrSizei):
            self.c_data.Pos.x = (<ovrVector2i>args[0]).c_data.x
            self.c_data.Pos.y = (<ovrVector2i>args[0]).c_data.y
            self.c_data.Size.w = (<ovrSizei>args[1]).c_data.w
            self.c_data.Size.h = (<ovrSizei>args[1]).c_data.h

    @property
    def x(self):
        return self.c_data.Pos.x

    @x.setter
    def x(self, int value):
        self.c_data.Pos.x = value

    @property
    def y(self):
        return self.c_data.Pos.y

    @y.setter
    def y(self, int value):
        self.c_data.Pos.y = value

    @property
    def w(self):
        return self.c_data.Size.w

    @w.setter
    def w(self, int value):
        self.c_data.Size.w = value

    @property
    def h(self):
        return self.c_data.Size.h

    @h.setter
    def h(self, int value):
        self.c_data.Size.h = value

    def as_tuple(self):
        return self.c_data.Pos.x, self.c_data.Pos.y, \
               self.c_data.Size.w, self.c_data.Size.h

    def as_list(self):
        return [self.c_data.Pos.x, self.c_data.Pos.y,
                self.c_data.Size.w, self.c_data.Size.h]

    def __len__(self):
        return 4

    def get_pos(self):
        cdef ovrVector2i to_return = ovrVector2i(self.c_data.Pos.x,
                                                 self.c_data.Pos.y)

        return to_return

    def get_size(self):
        cdef ovrSizei to_return = ovrSizei(self.c_data.Size.w,
                                           self.c_data.Size.h)

        return to_return

    def set_pos(self, ovrVector2i pos):
        self.c_data.Pos.x = pos.c_data.x
        self.c_data.Pos.y = pos.c_data.y

    def set_size(self, ovrSizei size):
        self.c_data.Size.w = size.c_data.w
        self.c_data.Size.h = size.c_data.h

    def __eq__(self, ovrRecti b):
        return self.c_data.Pos.x == b.c_data.Pos.x and \
               self.c_data.Pos.y == b.c_data.Pos.y and \
               self.c_data.Size.w == b.c_data.Size.w and \
               self.c_data.Size.h == b.c_data.Size.h

    def __ne__(self, ovrRecti b):
        return not self.__eq__(b)


cdef class ovrVector3f(object):
    cdef ovr_math.Vector3f* c_data
    cdef ovr_math.Vector3f  c_Vector3f

    def __init__(self, *args, **kwargs):
        pass

    def __cinit__(self, *args, **kwargs):
        self.c_data = &self.c_Vector3f

        cdef int nargin = <int>len(args)  # get number of arguments
        if nargin == 0:
            self.c_data[0] = ovr_math.Vector3f.Zero()
        elif nargin == 3:
            self.c_data[0] = ovr_math.Vector3f(
                <float>args[0], <float>args[1], <float>args[2])

    @property
    def x(self):
        return self.c_data.x

    @x.setter
    def x(self, float value):
        self.c_data.x = value

    @property
    def y(self):
        return self.c_data.y

    @y.setter
    def y(self, float value):
        self.c_data.y = value

    @property
    def z(self):
        return self.c_data.z

    @z.setter
    def z(self, float value):
        self.c_data.z = value

    def as_tuple(self):
        return self.c_data.x, self.c_data.y, self.c_data.z

    @property
    def ctypes(self):
        return (GL.GLfloat * 3)(
            self.c_data.x,
            self.c_data.y,
            self.c_data.z)

    def __len__(self):
        return 3

    @staticmethod
    def zero():
        cdef ovrVector3f to_return = ovrVector3f()
        return to_return

    def __eq__(self, ovrVector3f b):
        return (<ovrVector3f>self).c_data[0] == b.c_data[0]

    def __ne__(self, ovrVector3f b):
        return (<ovrVector3f>self).c_data[0] != b.c_data[0]

    def __add__(ovrVector3f a, ovrVector3f b):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = a.c_data[0] + b.c_data[0]

        return to_return

    def __iadd__(self, ovrVector3f b):
        (<ovrVector3f>self).c_data[0] += b.c_data[0]

        return self

    def __sub__(ovrVector3f a, ovrVector3f b):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = a.c_data[0] - b.c_data[0]

        return to_return

    def __isub__(self, ovrVector3f b):
        (<ovrVector3f>self).c_data[0] -= b.c_data[0]

        return self

    def __neg__(self):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = -(<ovrVector3f>self).c_data[0]

        return to_return

    def __mul__(ovrVector3f a, object b):
        cdef ovrVector3f to_return = ovrVector3f()
        if isinstance(b, ovrVector3f):
            (<ovrVector3f>to_return).c_data[0] = \
                (<ovrVector3f>a).c_data[0] * (<ovrVector3f>b).c_data[0]
        elif isinstance(b, (int, float)):
            (<ovrVector3f>to_return).c_data[0] = \
                (<ovrVector3f>a).c_data[0] * <float>b

        return to_return

    def __imul__(self, object b):
        if isinstance(b, (int, float)):
            (<ovrVector3f>self).c_data[0] = \
                (<ovrVector3f>self).c_data[0] * <float>b

        return self

    def __truediv__(ovrVector3f a, object b):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = \
            (<ovrVector3f>a).c_data[0] / (<ovrVector3f>b).c_data[0]

        return to_return

    def __itruediv__(self, object b):
        if isinstance(b, (int, float)):
            (<ovrVector3f>self).c_data[0] = \
                (<ovrVector3f>self).c_data[0] / <float>b

        return self

    @staticmethod
    def min(ovrVector3f a, ovrVector3f b):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = \
            ovr_math.Vector3f.Min(a.c_data[0], b.c_data[0])

        return to_return

    @staticmethod
    def max(ovrVector3f a, ovrVector3f b):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = \
            ovr_math.Vector3f.Max(a.c_data[0], b.c_data[0])

        return to_return

    def clamped(self, float max_mag):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = \
            (<ovrVector3f>self).c_data[0].Clamped(max_mag)

        return to_return

    def is_equal(self, ovrVector3f b, float tolerance = 0.0):
        return (<ovrVector3f>self).c_data[0].IsEqual(b.c_data[0], tolerance)

    def compare(self, ovrVector3f b, float tolerance = 0.0):
        return self.is_equal(b, tolerance)

    def __getitem__(self, int idx):
        assert 0 <= idx < 3
        cdef float* ptr_val = &self.c_data.x + idx

        return <float>ptr_val[0]

    def __setitem__(self, int idx, float val):
        assert 0 <= idx < 3
        cdef float* ptr_val = &self.c_data.x + idx
        ptr_val[0] = val

    def entrywise_multiply(self, ovrVector3f b):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = \
            (<ovrVector3f>self).c_data[0].EntrywiseMultiply(
            b.c_data[0])

        return to_return

    def dot(self, ovrVector3f b):
        return <float>(<ovrVector3f>self).c_data[0].Dot(b.c_data[0])

    def cross(self, ovrVector3f b):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = \
            (<ovrVector3f>self).c_data[0].Cross(b.c_data[0])

        return to_return

    def angle(self, ovrVector3f b):
        return <float>(<ovrVector3f>self).c_data[0].Angle(b.c_data[0])

    def length_sq(self):
        return <float>(<ovrVector3f>self).c_data[0].LengthSq()

    def length(self):
        return <float>(<ovrVector3f>self).c_data[0].Length()

    def distance_sq(self, ovrVector3f b):
        return <float>(<ovrVector3f>self).c_data[0].DistanceSq(b.c_data[0])

    def distance(self, ovrVector3f b):
        return <float>(<ovrVector3f>self).c_data[0].Distance(b.c_data[0])

    def is_normalized(self):
        return <bint>(<ovrVector3f>self).c_data[0].IsNormalized()

    def normalize(self):
        (<ovrVector3f>self).c_data[0].Normalize()

    def normalized(self):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = \
            (<ovrVector3f>self).c_data[0].Normalized()

        return to_return

    def lerp(self, ovrVector3f b, float f):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = \
            (<ovrVector3f>self).c_data[0].Lerp(b.c_data[0], f)

        return to_return

    def project_to(self, ovrVector3f b):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = \
            (<ovrVector3f>self).c_data[0].ProjectTo(
            b.c_data[0])

        return to_return

    def project_to_plane(self, ovrVector3f normal):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = \
            (<ovrVector3f>self).c_data[0].ProjectToPlane(
                normal.c_data[0])

        return to_return


cdef class ovrFovPort(object):
    cdef ovr_capi.ovrFovPort* c_data
    cdef ovr_capi.ovrFovPort  c_ovrFovPort

    def __cinit__(self,
                  float up_tan=0.0,
                  float down_tan=0.0,
                  float left_tan=0.0,
                  float right_tan=0.0):
        self.c_data = &self.c_ovrFovPort

    @property
    def up_tan(self):
        return <float>self.c_data[0].UpTan

    @up_tan.setter
    def up_tan(self, float value):
        self.c_data[0].UpTan = value

    @property
    def down_tan(self):
        return <float>self.c_data[0].DownTan

    @down_tan.setter
    def down_tan(self, float value):
        self.c_data[0].DownTan = value

    @property
    def left_tan(self):
        return <float>self.c_data[0].LeftTan

    @left_tan.setter
    def left_tan(self, float value):
        self.c_data[0].LeftTan = value

    @property
    def right_tan(self):
        return <float>self.c_data[0].RightTan

    @right_tan.setter
    def right_tan(self, float value):
        self.c_data[0].RightTan = value

    @staticmethod
    def max(ovrFovPort a, ovrFovPort b):
        cdef ovrFovPort to_return = ovrFovPort()
        (<ovrFovPort>to_return).c_data[0].UpTan = cmath.fmax(
            a.c_data[0].UpTan, b.c_data[0].UpTan)
        (<ovrFovPort>to_return).c_data[0].DownTan = cmath.fmax(
            a.c_data[0].DownTan, b.c_data[0].DownTan)
        (<ovrFovPort>to_return).c_data[0].LeftTan = cmath.fmax(
            a.c_data[0].LeftTan, b.c_data[0].LeftTan)
        (<ovrFovPort>to_return).c_data[0].RightTan = cmath.fmax(
            a.c_data[0].RightTan, b.c_data[0].RightTan)

        return to_return

    @staticmethod
    def min(ovrFovPort a, ovrFovPort b):
        cdef ovrFovPort to_return = ovrFovPort()
        (<ovrFovPort>to_return).c_data[0].UpTan = cmath.fmin(
            a.c_data[0].UpTan, b.c_data[0].UpTan)
        (<ovrFovPort>to_return).c_data[0].DownTan = cmath.fmin(
            a.c_data[0].DownTan, b.c_data[0].DownTan)
        (<ovrFovPort>to_return).c_data[0].LeftTan = cmath.fmin(
            a.c_data[0].LeftTan, b.c_data[0].LeftTan)
        (<ovrFovPort>to_return).c_data[0].RightTan = cmath.fmin(
            a.c_data[0].RightTan, b.c_data[0].RightTan)

        return to_return

cdef class ovrQuatf:
    cdef ovr_math.Quatf* c_data
    cdef ovr_math.Quatf  c_Quatf

    def __cinit__(self, *args):
        self.c_data = &self.c_Quatf

        cdef int nargin = <int>len(args)  # get number of arguments
        if nargin == 0:
            self.c_data[0] = ovr_math.Quatf.Identity()
        elif nargin == 1:
            if isinstance(ovrMatrix4f, args[0]):
                self.c_data[0] = ovr_math.Quatf(
                    (<ovrMatrix4f>args[0]).c_data[0])
            elif isinstance(ovrMatrix4f, args[0]):
                self.c_data[0] = ovr_math.Quatf(
                    (<ovrQuatf>args[0]).c_data[0])
        elif nargin == 2:
            # quaternion from axis and angle
            if isinstance(args[0], ovrVector3f) and \
                    isinstance(args[1], (int, float)):
                self.c_data[0] = ovr_math.Quatf(
                    (<ovrVector3f>args[0]).c_data[0], <float>args[1])
        elif nargin == 4:
            self.c_data[0] = ovr_math.Quatf(
                <float>args[0],
                <float>args[1],
                <float>args[2],
                <float>args[3])

    @property
    def x(self):
        return self.c_data.x

    @x.setter
    def x(self, float value):
        self.c_data.x = value

    @property
    def y(self):
        return self.c_data.y

    @y.setter
    def y(self, float value):
        self.c_data.y = value

    @property
    def z(self):
        return self.c_data.z

    @z.setter
    def z(self, float value):
        self.c_data.z = value

    @property
    def w(self):
        return self.c_data.w

    @w.setter
    def w(self, float value):
        self.c_data.w = value

    def as_tuple(self):
        return self.c_data.x, self.c_data.y, self.c_data.z, self.c_data.w

    def as_list(self):
        return [self.c_data.x, self.c_data.y, self.c_data.z, self.c_data.w]

    def __len__(self):
        return 4

    def __neg__(self):
        cdef ovrQuatf to_return = ovrQuatf(
            -self.c_data.x, -self.c_data.y, -self.c_data.z, -self.c_data.w)

        return to_return

    @staticmethod
    def identity():
        cdef ovrQuatf to_return = ovrQuatf(0.0, 0.0, 0.0, 1.0)
        return to_return

    def __eq__(self, ovrQuatf b):
        return (<ovrQuatf>self).c_data[0] == b.c_data[0]

    def __ne__(self, ovrQuatf b):
        return (<ovrQuatf>self).c_data[0] != b.c_data[0]

    def __add__(ovrQuatf a, ovrQuatf b):
        cdef ovrQuatf to_return = ovrQuatf()
        (<ovrQuatf>to_return).c_data[0] = a.c_data[0] + b.c_data[0]

        return to_return

    def __iadd__(self, ovrQuatf b):
        self.c_data.x += b.c_data.x
        self.c_data.y += b.c_data.y
        self.c_data.z += b.c_data.z
        self.c_data.w += b.c_data.w

        return self

    def __sub__(ovrQuatf a, ovrQuatf b):
        cdef ovrQuatf to_return = ovrQuatf()
        (<ovrQuatf>to_return).c_data[0] = a.c_data[0] - b.c_data[0]

        return to_return

    def __isub__(self, ovrQuatf b):
        self.c_data.x -= b.c_data.x
        self.c_data.y -= b.c_data.y
        self.c_data.z -= b.c_data.z
        self.c_data.w -= b.c_data.w

        return self

    def __mul__(ovrQuatf a, object b):
        if isinstance(b, ovrVector3f):
            return a.rotate(b)
        elif isinstance(b, ovrQuatf):
            # quaternion multiplication
            return ovrQuatf(
                a.c_data.w * b.c_data.x +
                a.c_data.x * b.c_data.w +
                a.c_data.y * b.c_data.z -
                a.c_data.z * b.c_data.y,
                a.c_data.w * b.c_data.y -
                a.c_data.x * b.c_data.z +
                a.c_data.y * b.c_data.w +
                a.c_data.z * b.c_data.x,
                a.c_data.w * b.c_data.z +
                a.c_data.x * b.c_data.y -
                a.c_data.y * b.c_data.x +
                a.c_data.z * b.c_data.w,
                a.c_data.w * b.c_data.w -
                a.c_data.x * b.c_data.x -
                a.c_data.y * b.c_data.y -
                a.c_data.z * b.c_data.z)
        elif isinstance(b, (int, float)):
            return ovrQuatf(
                a.c_data.x * <float>b,
                a.c_data.y * <float>b,
                a.c_data.z * <float>b,
                a.c_data.w * <float>b)

    def __imul__(self, object b):
        if isinstance(b, ovrQuatf):
            self.c_data.x *= b.c_data.x
            self.c_data.y *= b.c_data.y
            self.c_data.z *= b.c_data.z
            self.c_data.w *= b.c_data.w
        elif isinstance(b, (int, float)):
            self.c_data.x *= <float>b
            self.c_data.y *= <float>b
            self.c_data.z *= <float>b
            self.c_data.w *= <float>b

        return self

    def __truediv__(ovrQuatf a, float s):
        cdef ovrQuatf to_return = ovrQuatf()
        (<ovrQuatf>to_return).c_data[0] = a.c_data[0] / s

        return to_return

    def __itruediv__(self, float s):
        (<ovrQuatf>self).c_data[0] = (<ovrQuatf>self).c_data[0] / s

        return self

    def is_equal(self, ovrQuatf b, float tolerance = 0.0):
        return  self.abs(self.dot(b)) >= <float>1 - tolerance

    def abs(self, float v):
        return <float>(<ovrQuatf>self).c_data[0].Abs(v)

    def imag(self):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = \
            (<ovrQuatf>self).c_data[0].Imag()

        return to_return

    def length_sq(self):
        return <float>(<ovrQuatf>self).c_data[0].LengthSq()

    def length(self):
        return <float>(<ovrQuatf>self).c_data[0].Length()

    def distance(self, ovrQuatf q):
        return <float>(<ovrQuatf>self).c_data[0].Distance(q.c_data[0])

    def distance_sq(self, ovrQuatf q):
        return <float>(<ovrQuatf>self).c_data[0].DistanceSq(q.c_data[0])

    def dot(self, ovrQuatf q):
        return <float>(<ovrQuatf>self).c_data[0].Dot(q.c_data[0])

    def angle(self, ovrQuatf q):
        return <float>(<ovrQuatf>self).c_data[0].Angle(q.c_data[0])

    def is_normalized(self):
        return <bint>(<ovrQuatf>self).c_data[0].IsNormalized()

    def normalize(self):
        (<ovrQuatf>self).c_data[0].Normalize()

        return self

    def normalized(self):
        cdef ovrQuatf to_return = ovrQuatf()
        (<ovrQuatf>to_return).c_data[0] = \
            (<ovrQuatf>self).c_data[0].Normalized()

        return to_return

    def conj(self):
        cdef ovrQuatf to_return = ovrQuatf()
        (<ovrQuatf>to_return).c_data[0] = (<ovrQuatf>self).c_data[0].Conj()

        return to_return

    @staticmethod
    def align(ovrVector3f align_to, ovrVector3f v):
        cdef ovrQuatf to_return = ovrQuatf()
        (<ovrQuatf>to_return).c_data[0] = ovr_math.Quatf.Align(
            align_to.c_data[0], v.c_data[0])

        return to_return

    def rotate(self, ovrVector3f v):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = \
            (<ovrQuatf>self).c_data[0].Rotate(v.c_data[0])

        return to_return

    def inverse_rotate(self, ovrVector3f v):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = \
            (<ovrQuatf>self).c_data[0].InverseRotate(v.c_data[0])

        return to_return

    def inverted(self):
        cdef ovrQuatf to_return = ovrQuatf()
        (<ovrQuatf>to_return).c_data[0] = (<ovrQuatf>self).c_data[0].Inverted()

        return to_return

    def inverse(self):
        cdef ovrQuatf to_return = ovrQuatf()
        (<ovrQuatf>to_return).c_data[0] = (<ovrQuatf>self).c_data[0].Inverse()

        return to_return

    def invert(self):
        (<ovrQuatf>self).c_data[0].Inverse()

        return self

    def __invert__(self):
        return self.inverse()


cdef class ovrPosef:
    cdef ovr_math.Posef* c_data
    cdef ovr_math.Posef  c_Posef

    cdef ovrVector3f field_position
    cdef ovrQuatf field_orientation

    def __cinit__(self, *args, **kwargs):
        self.c_data = &self.c_Posef

        # create property objects
        self.field_position = ovrVector3f()
        self.field_position.c_data = &(<ovrPosef>self).c_data[0].Translation
        self.field_orientation = ovrQuatf()
        self.field_orientation.c_data = &(<ovrPosef>self).c_data[0].Rotation

        cdef int nargin = <int>len(args)  # get number of arguments
        if nargin == 0:
            pass
        elif nargin == 2:
            if isinstance(args[0], ovrVector3f) and \
                    isinstance(args[1], ovrQuatf):
                self.field_position.c_data[0] = (<ovrVector3f>args[0]).c_data[0]
                self.field_orientation.c_data[0] = (<ovrQuatf>args[1]).c_data[0]

    @property
    def rotation(self):
        return self.field_orientation

    @rotation.setter
    def rotation(self, ovrQuatf value):
        (<ovrPosef>self).c_data[0].Rotation = value.c_data[0]

    @property
    def translation(self):
        return self.field_position

    @translation.setter
    def translation(self, ovrVector3f value):
        (<ovrPosef>self).c_data[0].Translation = value.c_data[0]

    def rotate(self, ovrVector3f v):
        return self.rotation.rotate(v)

    def inverse_rotate(self, ovrVector3f v):
        return self.inverse_rotate(v)

    def translate(self, ovrVector3f v):
        return v + self.translation

    def transform(self, ovrVector3f v):
        return self.rotate(v) + self.translation

    def inverse_transform(self, ovrVector3f v):
        return self.inverse_transform(v) + self.translation

    def transform_normal(self, ovrVector3f v):
        return self.rotate(v)

    def inverse_transform_normal(self, ovrVector3f v):
        return self.inverse_rotate(v)

    def apply(self, ovrVector3f v):
        return self.transform(v)

    def __mul__(ovrPosef this, ovrPosef other):
        cdef ovrPosef to_return = ovrPosef(
            <ovrQuatf>this.rotation * <ovrQuatf>other.rotation,
            this.apply(other.rotation))

        return to_return

    def inverted(self):
        cdef ovrQuatf inv = self.rotation.inverted()
        cdef ovrPosef to_return = ovrPosef(inv, inv.rotate(-self.translation))

    def normalized(self):
        cdef ovrPosef to_return = ovrPosef(
            self.rotation.normalized(),
            self.translation)

        return to_return

    def normalize(self):
        self.rotation.normalize()


cdef class ovrMatrix4f:
    """ovrMatrix4f

    4x4 Matrix typically used for 3D transformations. By default, all matrices
    are right handed. Values are stored in row-major order. Transformations
    are applied left-to-right.

    """

    cdef ovr_math.Matrix4f* c_data
    cdef ovr_math.Matrix4f  c_Matrix4f

    def __cinit__(self, *args):
        self.c_data = &self.c_Matrix4f

        cdef int nargin = <int>len(args)
        if nargin == 0:
            self.c_data[0] = ovr_math.Matrix4f.Identity()
        elif nargin == 1:
            if isinstance(args[0], ovrQuatf):
                self.c_data[0] = ovr_math.Matrix4f(
                    <ovr_math.Quatf>(<ovrQuatf>args[0]).c_data[0])
            elif isinstance(args[0], ovrPosef):
                self.c_data[0] = ovr_math.Matrix4f(
                    <ovr_math.Posef>(<ovrPosef>args[0]).c_data[0])
            elif isinstance(args[0], ovrMatrix4f):
                self.c_data[0] = ovr_math.Matrix4f(
                    <ovr_math.Matrix4f>(<ovrMatrix4f>args[0]).c_data[0])
        elif nargin == 9:
            self.c_data[0] = ovr_math.Matrix4f(
                <float>args[0],
                <float>args[1],
                <float>args[2],
                <float>args[3],
                <float>args[4],
                <float>args[5],
                <float>args[6],
                <float>args[7],
                <float>args[8])
        elif nargin == 16:
            self.c_data[0] = ovr_math.Matrix4f(
                <float>args[0],
                <float>args[1],
                <float>args[2],
                <float>args[3],
                <float>args[4],
                <float>args[5],
                <float>args[6],
                <float>args[7],
                <float>args[8],
                <float>args[9],
                <float>args[10],
                <float>args[11],
                <float>args[12],
                <float>args[13],
                <float>args[14],
                <float>args[15])

    @property
    def M(self):
        return self.c_data.M

    @staticmethod
    def identity():
        cdef ovrMatrix4f to_return = ovrMatrix4f()
        return to_return

    def set_identity(self):
        self.c_data[0] = ovr_math.Matrix4f()

    def set_x_basis(self, ovrVector3f v):
        (<ovrMatrix4f>self).c_data[0].SetXBasis(v.c_data[0])

    def get_x_basis(self):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = \
            (<ovr_math.Matrix4f>self.c_data[0]).GetXBasis()

        return to_return

    @property
    def x_basis(self):
        return self.get_x_basis()

    @x_basis.setter
    def x_basis(self, object v):
        if isinstance(v, ovrVector3f):
            self.set_x_basis(<ovrVector3f>v)
        elif isinstance(v, (list, tuple)):
            self.set_x_basis(ovrVector3f(<float>v[0], <float>v[1], <float>v[2]))

    def set_y_basis(self, ovrVector3f v):
        (<ovrMatrix4f>self).c_data[0].SetYBasis(v.c_data[0])

    def get_y_basis(self):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = \
            (<ovr_math.Matrix4f>self.c_data[0]).GetYBasis()

        return to_return

    @property
    def y_basis(self):
        return self.get_y_basis()

    @y_basis.setter
    def y_basis(self, object v):
        if isinstance(v, ovrVector3f):
            self.set_y_basis(<ovrVector3f>v)
        elif isinstance(v, (list, tuple)):
            self.set_y_basis(ovrVector3f(<float>v[0], <float>v[1], <float>v[2]))

    def set_z_basis(self, ovrVector3f v):
        (<ovrMatrix4f>self).c_data[0].SetZBasis(v.c_data[0])

    def get_z_basis(self):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = \
            (<ovr_math.Matrix4f>self.c_data[0]).GetZBasis()

        return to_return

    @property
    def z_basis(self):
        return self.get_y_basis()

    @z_basis.setter
    def z_basis(self, object v):
        if isinstance(v, ovrVector3f):
            self.set_z_basis(<ovrVector3f>v)
        elif isinstance(v, (list, tuple)):
            self.set_z_basis(ovrVector3f(<float>v[0], <float>v[1], <float>v[2]))

    @property
    def ctypes(self):
        return (GL.GLfloat * 16)(
            self.c_data.M[0][0],
            self.c_data.M[1][0],
            self.c_data.M[2][0],
            self.c_data.M[3][0],
            self.c_data.M[0][1],
            self.c_data.M[1][1],
            self.c_data.M[2][1],
            self.c_data.M[3][1],
            self.c_data.M[0][2],
            self.c_data.M[1][2],
            self.c_data.M[2][2],
            self.c_data.M[3][2],
            self.c_data.M[0][3],
            self.c_data.M[1][3],
            self.c_data.M[2][3],
            self.c_data.M[3][3])

    def __eq__(self, ovrMatrix4f b):
        return (<ovrMatrix4f>self).c_data[0] == b.c_data[0]

    def __ne__(self, ovrMatrix4f b):
        return not (<ovrMatrix4f>self).c_data[0] == b.c_data[0]

    def __add__(ovrMatrix4f a, ovrMatrix4f b):
        cdef ovrMatrix4f to_return = ovrMatrix4f()
        (<ovrMatrix4f>to_return).c_data[0] = a.c_data[0] + b.c_data[0]

        return to_return

    def __iadd__(self, ovrMatrix4f b):
        cdef ovrMatrix4f to_return = ovrMatrix4f()
        (<ovrMatrix4f>self).c_data[0] = \
            (<ovrMatrix4f>self).c_data[0] + b.c_data[0]

        return self

    def __sub__(ovrMatrix4f a, ovrMatrix4f b):
        cdef ovrMatrix4f to_return = ovrMatrix4f()
        (<ovrMatrix4f>to_return).c_data[0] = a.c_data[0] - b.c_data[0]

        return to_return

    def __isub__(self, ovrMatrix4f b):
        cdef ovrMatrix4f to_return = ovrMatrix4f()
        (<ovrMatrix4f>self).c_data[0] = \
            (<ovrMatrix4f>self).c_data[0] - b.c_data[0]

        return self

    @staticmethod
    def multiply(ovrMatrix4f d, ovrMatrix4f a, ovrMatrix4f b):
        ovr_math.Matrix4f.Multiply(&d.c_data[0], a.c_data[0], b.c_data[0])

    def __mul__(ovrMatrix4f a, object b):
        cdef ovrMatrix4f to_return = ovrMatrix4f()
        if isinstance(b, ovrMatrix4f):
            (<ovrMatrix4f>to_return).c_data[0] = \
                (<ovrMatrix4f>a).c_data[0] * (<ovrMatrix4f>b).c_data[0]
        elif isinstance(b, (int, float)):
            (<ovrMatrix4f>to_return).c_data[0] = \
                (<ovrMatrix4f>a).c_data[0] * <float>b

        return to_return

    def __imul__(self, object b):
        if isinstance(b, ovrMatrix4f):
            (<ovrMatrix4f>self).c_data[0] = \
                (<ovrMatrix4f>self).c_data[0] * (<ovrMatrix4f>b).c_data[0]
        elif isinstance(b, (int, float)):
            (<ovrMatrix4f>self).c_data[0] = \
                (<ovrMatrix4f>self).c_data[0] * <float>b

        return self

    def __truediv__(ovrMatrix4f a, float b):
        cdef ovrMatrix4f to_return = ovrMatrix4f()
        (<ovrMatrix4f>to_return).c_data[0] = a.c_data[0] * <float>b

        return to_return

    def __itruediv__(self, float b):
        (<ovrMatrix4f>self).c_data[0] = (<ovrMatrix4f>self).c_data[0] / <float>b

        return self

    def transform(self, ovrVector3f v):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = \
            (<ovr_math.Matrix4f>self.c_data[0]).Transform(v.c_data[0])

        return to_return

    def transposed(self):
        cdef ovrMatrix4f to_return = ovrMatrix4f()
        (<ovrMatrix4f>to_return).c_data[0] = \
            (<ovr_math.Matrix4f>self.c_data[0]).Transposed()

        return to_return

    def transpose(self):
        (<ovr_math.Matrix4f>self.c_data[0]).Transpose()

        return self

    def determinant(self):
        cdef float det = (<ovrMatrix4f>self).c_data[0].Determinant()
        return det

    def adjugated(self):
        cdef ovrMatrix4f to_return = ovrMatrix4f()
        (<ovrMatrix4f>to_return).c_data[0] = \
            (<ovr_math.Matrix4f>self.c_data[0]).Adjugated()

        return to_return

    def inverted(self):
        cdef ovrMatrix4f to_return = ovrMatrix4f()
        (<ovrMatrix4f>to_return).c_data[0] = \
            (<ovr_math.Matrix4f>self.c_data[0]).Inverted()

        return to_return

    def invert(self):
        (<ovr_math.Matrix4f>self.c_data[0]).Transpose()

        return self

    def inverted_homogeneous_transform(self):
        cdef ovrMatrix4f to_return = ovrMatrix4f()
        (<ovrMatrix4f>to_return).c_data[0] = \
            (<ovr_math.Matrix4f>self.c_data[0]).InvertedHomogeneousTransform()

        return to_return

    def invert_homogeneous_transform(self):
        (<ovr_math.Matrix4f>self.c_data[0]).InvertHomogeneousTransform()

        return self

    @staticmethod
    def translation(object v):
        cdef ovrMatrix4f to_return = ovrMatrix4f()
        if isinstance(v, ovrVector3f):
            (<ovrMatrix4f>to_return).c_data[0] = \
                ovr_math.Matrix4f.Translation((<ovrVector3f>v).c_data[0])
        elif isinstance(v, (list, tuple)):
            (<ovrMatrix4f>to_return).c_data[0] = \
                ovr_math.Matrix4f.Translation(
                    ovrVector3f(<float>v[0], <float>v[1], <float>v[2]).c_data[0])

        return to_return

    def set_translation(self, ovrVector3f v):
        (<ovr_math.Matrix4f>self.c_data[0]).SetTranslation(v.c_data[0])

        return self

    def get_translation(self):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = \
            (<ovr_math.Matrix4f>self.c_data[0]).GetTranslation()

        return to_return

    @staticmethod
    def scaling(ovrVector3f v):
        cdef ovrMatrix4f to_return = ovrMatrix4f()
        (<ovrMatrix4f>to_return).c_data[0] = \
            ovr_math.Matrix4f.Scaling((<ovrVector3f>v).c_data[0])

        return to_return

    def distance(self, ovrMatrix4f m2):
        cdef float distance = \
            (<ovr_math.Matrix4f>self.c_data[0]).Distance(m2.c_data[0])

        return distance

    @staticmethod
    def rotation_x(float angle=0.0):
        cdef ovrMatrix4f to_return = ovrMatrix4f()
        (<ovrMatrix4f>to_return).c_data[0] = ovr_math.Matrix4f.RotationX(angle)

        return to_return

    @staticmethod
    def rotation_y(float angle=0.0):
        cdef ovrMatrix4f to_return = ovrMatrix4f()
        (<ovrMatrix4f>to_return).c_data[0] = ovr_math.Matrix4f.RotationY(angle)

        return to_return

    @staticmethod
    def rotation_z(float angle=0.0):
        cdef ovrMatrix4f to_return = ovrMatrix4f()
        (<ovrMatrix4f>to_return).c_data[0] = ovr_math.Matrix4f.RotationZ(angle)

        return to_return

    @staticmethod
    def look_at(ovrVector3f eye, ovrVector3f at, ovrVector3f up):
        cdef ovrMatrix4f to_return = ovrMatrix4f()
        (<ovrMatrix4f>to_return).c_data[0] = ovr_math.Matrix4f.LookAtRH(
            eye.c_data[0], at.c_data[0], up.c_data[0])

        return to_return

    @staticmethod
    def perspective(float yfov, float aspect, float znear, float zfar):
        cdef ovrMatrix4f to_return = ovrMatrix4f()
        (<ovrMatrix4f>to_return).c_data[0] = ovr_math.Matrix4f.PerspectiveRH(
            yfov, aspect, znear, zfar)

        return to_return

    @staticmethod
    def ortho_2d(float w, float h):
        cdef ovrMatrix4f to_return = ovrMatrix4f()
        (<ovrMatrix4f>to_return).c_data[0] = ovr_math.Matrix4f.Ortho2D(w, h)

        return to_return

# --------------------
# Swap Chain Functions
# --------------------
#
cdef dict texture_format_lut = {
    'R8G8B8A8_UNORM': ovr_capi.OVR_FORMAT_R8G8B8A8_UNORM,
    'R8G8B8A8_UNORM_SRGB': ovr_capi.OVR_FORMAT_R8G8B8A8_UNORM_SRGB,
    'B8G8R8A8_UNORM': ovr_capi.OVR_FORMAT_B8G8R8A8_UNORM,
    'R16G16B16A16_FLOAT': ovr_capi.OVR_FORMAT_R16G16B16A16_FLOAT,
    'R11G11B10_FLOAT': ovr_capi.OVR_FORMAT_R11G11B10_FLOAT,
    'D16_UNORM': ovr_capi.OVR_FORMAT_D16_UNORM,
    'D24_UNORM_S8_UINT': ovr_capi.OVR_FORMAT_D24_UNORM_S8_UINT,
    'D32_FLOAT': ovr_capi.OVR_FORMAT_D32_FLOAT,
    'D32_FLOAT_S8X24_UINT': ovr_capi.OVR_FORMAT_D32_FLOAT_S8X24_UINT
}

texture_formats = list(texture_format_lut.keys())

def alloc_swap_chain(ovrTextureSwapChainDesc swap_desc):
    """Allocate a new swap chain object with the specified parameters. If
    successful, an integer is returned which is used to reference the swap
    chain. You can allocate up-to 32 swap chains.

    :param width: int
    :param height: int
    :return: int

    """
    global _swap_chain_, _ptr_session_
    # get the first available swap chain, unallocated chains will test as NULL
    cdef int i, sc
    for i in range(32):
        if _swap_chain_[i] is NULL:
            sc = i
            break
    else:
        raise IndexError("Maximum number of swap chains initialized!")

    # create the swap chain
    cdef ovr_capi.ovrResult result = ovr_capi_gl.ovr_CreateTextureSwapChainGL(
        _ptr_session_, &swap_desc.c_ovrTextureSwapChainDesc, &_swap_chain_[sc])

    if debug_mode:
        check_result(result)

    # return the handle
    return sc

# Free or destroy a swap chain. The handle will be made available after this
# call.
#
def free_swap_chain(int sc):
    """Free or destroy a swap chain. The handle will be made available after
    this call.

    :param sc: int
    :return:

    """
    global _swap_chain_, _ptr_session_
    ovr_capi.ovr_DestroyTextureSwapChain(_ptr_session_, _swap_chain_[sc])
    _swap_chain_[sc] = NULL

# Get the next available texture in the specified swap chain. Use the returned
# value as a frame buffer texture.
#
def get_swap_chain_buffer(int sc):
    cdef int current_idx = 0
    cdef unsigned int tex_id = 0
    cdef ovr_capi.ovrResult result

    global _swap_chain_

    # get the current texture index within the swap chain
    result = ovr_capi.ovr_GetTextureSwapChainCurrentIndex(
        _ptr_session_, _swap_chain_[sc], &current_idx)

    if debug_mode:
        check_result(result)

    # get the next available texture ID from the swap chain
    result = ovr_capi_gl.ovr_GetTextureSwapChainBufferGL(
        _ptr_session_, _swap_chain_[sc], current_idx, &tex_id)

    if debug_mode:
        check_result(result)

    return tex_id

# -----------------
# Session Functions
# -----------------
#
cpdef bint is_oculus_service_running(int timeout_milliseconds=100):
    cdef ovr_capi_util.ovrDetectResult result = ovr_capi_util.ovr_Detect(
        timeout_milliseconds)

    return <bint>result.IsOculusServiceRunning

cpdef bint is_hmd_connected(int timeout_milliseconds=100):
    cdef ovr_capi_util.ovrDetectResult result = ovr_capi_util.ovr_Detect(
        timeout_milliseconds)

    return <bint>result.IsOculusHMDConnected

cpdef void start_session():
    """Start a new session. Control is handed over to the application from
    Oculus Home. 
    
    :return: None 
    
    """
    global _ptr_session_
    cdef ovr_capi.ovrResult result = 0
    result = ovr_capi.ovr_Initialize(NULL)
    result = ovr_capi.ovr_Create(&_ptr_session_, &_ptr_luid_)
    if ovr_errorcode.OVR_FAILURE(result):
        ovr_capi.ovr_Shutdown()

    # get HMD descriptor
    global _hmd_desc_
    _hmd_desc_ = ovr_capi.ovr_GetHmdDesc(_ptr_session_)

    # configure VR data with HMD descriptor information
    #global _eye_render_desc_, _hmd_to_eye_view_pose_
    #_eye_render_desc_[0] = ovr_capi.ovr_GetRenderDesc(
    #    _ptr_session_, ovr_capi.ovrEye_Left, _hmd_desc_.DefaultEyeFov[0])
    #_eye_render_desc_[1] = ovr_capi.ovr_GetRenderDesc(
    #    _ptr_session_, ovr_capi.ovrEye_Right, _hmd_desc_.DefaultEyeFov[1])
    #_hmd_to_eye_view_pose_[0] = _eye_render_desc_[0].HmdToEyePose
    #_hmd_to_eye_view_pose_[1] = _eye_render_desc_[1].HmdToEyePose

    # prepare the render layer
    global _eye_layer_
    _eye_layer_.Header.Type = ovr_capi.ovrLayerType_EyeFov
    _eye_layer_.Header.Flags = \
        ovr_capi.ovrLayerFlag_TextureOriginAtBottomLeft | \
        ovr_capi.ovrLayerFlag_HighQuality
    _eye_layer_.ColorTexture[0] = NULL
    _eye_layer_.ColorTexture[1] = NULL

    # setup layer FOV settings, these are computed earlier
    #_eye_layer_.Fov[0] = _eye_render_desc_[0].Fov
    #_eye_layer_.Fov[1] = _eye_render_desc_[1].Fov

cpdef void end_session():
    """End the current session. 
    
    Clean-up routines are executed that destroy all swap chains and mirror 
    texture buffers, afterwards control is returned to Oculus Home. This must be 
    called after every successful 'create_session' call.
    
    :return: None 
    
    """
    # free all swap chains
    global _ptr_session_, _swap_chain_, _mirror_texture_
    cdef int i = 0
    for i in range(32):
        if not _swap_chain_[i] is NULL:
            ovr_capi.ovr_DestroyTextureSwapChain(
                _ptr_session_, _swap_chain_[i])
            _swap_chain_[i] = NULL

    # destroy the mirror texture
    ovr_capi.ovr_DestroyMirrorTexture(_ptr_session_, _mirror_texture_)

    # destroy the current session and shutdown
    ovr_capi.ovr_Destroy(_ptr_session_)
    ovr_capi.ovr_Shutdown()

cdef class ovrHmdDesc(object):
    cdef ovr_capi.ovrHmdDesc c_ovrHmdDesc

    def __cinit__(self, *args, **kwargs):
        pass

    @property
    def type(self):
        return <int>self.c_data[0].Type

    @property
    def product_name(self):
        return self.c_ovrHmdDesc.ProductName.decode('utf-8')

    @property
    def manufacturer(self):
        return self.c_ovrHmdDesc.Manufacturer.decode('utf-8')

    @property
    def vendor_id(self):
        return <int>self.c_ovrHmdDesc.VendorId

    @property
    def product_id(self):
        return <int>self.c_ovrHmdDesc.ProductId

    @property
    def serial_number(self):
        return self.c_ovrHmdDesc.SerialNumber.decode('utf-8')

    @property
    def firmware_major(self):
        return <int>self.c_ovrHmdDesc.FirmwareMajor

    @property
    def firmware_minor(self):
        return <int>self.c_ovrHmdDesc.FirmwareMinor

    @property
    def available_hmd_caps(self):
        return <int>self.c_ovrHmdDesc.AvailableHmdCaps

    @property
    def default_hmd_caps(self):
        return <int>self.c_ovrHmdDesc.DefaultHmdCaps

    @property
    def available_tracking_caps(self):
        return <int>self.c_ovrHmdDesc.AvailableTrackingCaps

    @property
    def default_tracking_caps(self):
        return <int>self.c_ovrHmdDesc.DefaultTrackingCaps

    @property
    def default_eye_fov(self):
        cdef ovrFovPort default_fov_left = ovrFovPort()
        cdef ovrFovPort default_fov_right = ovrFovPort()

        (<ovrFovPort>default_fov_left).c_data[0] = \
            self.c_ovrHmdDesc.DefaultEyeFov[0]
        (<ovrFovPort>default_fov_right).c_data[0] = \
            self.c_ovrHmdDesc.DefaultEyeFov[1]

        return default_fov_left, default_fov_right

    @property
    def max_eye_fov(self):
        cdef ovrFovPort max_fov_left = ovrFovPort()
        cdef ovrFovPort max_fov_right = ovrFovPort()

        (<ovrFovPort>max_fov_left).c_data[0] = self.c_ovrHmdDesc.MaxEyeFov[0]
        (<ovrFovPort>max_fov_right).c_data[0] = self.c_ovrHmdDesc.MaxEyeFov[1]

        return max_fov_left, max_fov_right

    @property
    def resolution(self):
        cdef ovr_capi.ovrSizei resolution = self.c_ovrHmdDesc.Resolution

        return resolution.x, resolution.y

    @property
    def display_refresh_rate(self):
        return self.c_ovrHmdDesc.DisplayRefreshRate


cpdef ovrHmdDesc get_hmd_desc():
    """Get general information about the connected HMD. Information such as the
    serial number can identify a specific unit, etc.
    
    :return: ovrHmdDesc 
    
    """
    global _ptr_session_, _hmd_desc_
    cdef ovrHmdDesc to_return = ovrHmdDesc()
    _hmd_desc_ = ovr_capi.ovr_GetHmdDesc(_ptr_session_)
    (<ovrHmdDesc>to_return).c_ovrHmdDesc = _hmd_desc_

    return to_return

# ---------------------------------
# Rendering Configuration Functions
# ---------------------------------
#
# layer header flags
ovrLayerFlag_HighQuality = 0x01
ovrLayerFlag_TextureOriginAtBottomLeft = 0x02
ovrLayerFlag_HeadLocked = 0x04

cdef class ovrTextureSwapChain:
    """Texture swap chain.

    """
    cdef int* c_data
    cdef int  c_swap_id

    def __cinit__(self, *args, **kwargs):
        self.c_data = &self.c_swap_id

    @property
    def index(self):
        return <int>self.c_data[0]

# Texture types supported by the PC version of LibOVR
#
ovrTexture_2D = ovr_capi.ovrTexture_2D
ovrTexture_Cube = ovr_capi.ovrTexture_Cube

# Texture formats supported by OpenGL
#
OVR_FORMAT_R8G8B8A8_UNORM = ovr_capi.OVR_FORMAT_R8G8B8A8_UNORM
OVR_FORMAT_R8G8B8A8_UNORM_SRGB = ovr_capi.OVR_FORMAT_R8G8B8A8_UNORM_SRGB
OVR_FORMAT_B8G8R8A8_UNORM = ovr_capi.OVR_FORMAT_B8G8R8A8_UNORM
OVR_FORMAT_B8G8R8_UNORM = ovr_capi.OVR_FORMAT_B8G8R8_UNORM
OVR_FORMAT_R16G16B16A16_FLOAT = ovr_capi.OVR_FORMAT_R16G16B16A16_FLOAT
OVR_FORMAT_R11G11B10_FLOAT = ovr_capi.OVR_FORMAT_R11G11B10_FLOAT
OVR_FORMAT_D16_UNORM = ovr_capi.OVR_FORMAT_D16_UNORM
OVR_FORMAT_D24_UNORM_S8_UINT = ovr_capi.OVR_FORMAT_D24_UNORM_S8_UINT
OVR_FORMAT_D32_FLOAT = ovr_capi.OVR_FORMAT_D32_FLOAT
OVR_FORMAT_D32_FLOAT_S8X24_UINT = ovr_capi.OVR_FORMAT_D32_FLOAT_S8X24_UINT

cdef class ovrTextureSwapChainDesc:
    """ovrTextureSwapChainDesc

    """
    # no data pointer here
    cdef ovr_capi.ovrTextureSwapChainDesc c_ovrTextureSwapChainDesc

    def __cinit__(
            self,
            int type=ovrTexture_2D,
            int _format=OVR_FORMAT_R8G8B8A8_UNORM_SRGB,
            int width=800,
            int height=600,
            int array_size=1,
            int mip_levels=1,
            int sample_count=1,
            bint static_image=False):

        self.c_ovrTextureSwapChainDesc.Type = <ovr_capi.ovrTextureType>type
        self.c_ovrTextureSwapChainDesc.Format = <ovr_capi.ovrTextureFormat>_format
        self.c_ovrTextureSwapChainDesc.ArraySize = array_size
        self.c_ovrTextureSwapChainDesc.Width = width
        self.c_ovrTextureSwapChainDesc.Height = height
        self.c_ovrTextureSwapChainDesc.MipLevels = mip_levels
        self.c_ovrTextureSwapChainDesc.SampleCount = sample_count
        self.c_ovrTextureSwapChainDesc.StaticImage = <ovr_capi.ovrBool>static_image

        # these can't be set right now
        self.c_ovrTextureSwapChainDesc.MiscFlags = ovr_capi.ovrTextureMisc_None
        self.c_ovrTextureSwapChainDesc.BindFlags = ovr_capi.ovrTextureBind_None

    @property
    def type(self):
        return <int>self.c_ovrTextureSwapChainDesc.Type

    @type.setter
    def type(self, int value):
        self.c_ovrTextureSwapChainDesc.Type = <ovr_capi.ovrTextureType>value

    @property
    def format(self):
        return <int>self.c_ovrTextureSwapChainDesc.Format

    @format.setter
    def format(self, int value):
        self.c_ovrTextureSwapChainDesc.Format = <ovr_capi.ovrTextureFormat>value

    @property
    def array_size(self):
        return <int>self.c_ovrTextureSwapChainDesc.ArraySize

    @array_size.setter
    def array_size(self, int value):
        self.c_ovrTextureSwapChainDesc.ArraySize = value

    @property
    def width(self):
        return <int>self.c_ovrTextureSwapChainDesc.Width

    @width.setter
    def width(self, int value):
        self.c_ovrTextureSwapChainDesc.Width = value

    @property
    def height(self):
        return <int>self.c_ovrTextureSwapChainDesc.Height

    @height.setter
    def height(self, int value):
        self.c_ovrTextureSwapChainDesc.Height = value

    @property
    def mip_levels(self):
        return <int>self.c_ovrTextureSwapChainDesc.MipLevels

    @mip_levels.setter
    def mip_levels(self, int value):
        self.c_ovrTextureSwapChainDesc.MipLevels = value

    @property
    def sample_count(self):
        return <int>self.c_ovrTextureSwapChainDesc.SampleCount

    @sample_count.setter
    def sample_count(self, int value):
        self.c_ovrTextureSwapChainDesc.SampleCount = value

    @property
    def static_image(self):
        return <bint>self.c_ovrTextureSwapChainDesc.StaticImage

    @static_image.setter
    def static_image(self, bint value):
        self.c_ovrTextureSwapChainDesc.StaticImage = <ovr_capi.ovrBool>value

cpdef int create_texture_swap_chain_gl(
        ovrTextureSwapChainDesc swap_desc):
    """Allocate a new swap chain object with the specified parameters. If
    successful, an integer is returned which is used to reference the swap
    chain. You can allocate up-to 32 swap chains.
    
    The swap chain is configured by applying settings to a 
    ovrTextureSwapChainDesc object and passing it as 'swap_desc'. 

    :param swap_desc: ovrTextureSwapChainDesc
    :return: int

    """
    global _swap_chain_, _ptr_session_
    # get the first available swap chain, unallocated chains will test as NULL
    cdef int i, sc
    for i in range(32):
        if _swap_chain_[i] is NULL:
            sc = i
            break
    else:
        raise IndexError("Maximum number of swap chains initialized!")

    # create the swap chain
    cdef ovr_capi.ovrResult result = ovr_capi_gl.ovr_CreateTextureSwapChainGL(
        _ptr_session_, &swap_desc.c_ovrTextureSwapChainDesc, &_swap_chain_[sc])

    if debug_mode:
        check_result(result)

    # return the handle
    return sc

cpdef ovrTextureSwapChainDesc get_texture_swap_chain_desc(
        ovrTextureSwapChain swap_chain):
    global _swap_chain_
    cdef ovrTextureSwapChainDesc to_return = ovrTextureSwapChainDesc()

    # create the swap chain
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetTextureSwapChainDesc(
        _ptr_session_,
        _swap_chain_[swap_chain.c_data[0]],
        &to_return.c_ovrTextureSwapChainDesc)

    if debug_mode:
        check_result(result)

    return to_return

# types
ovrLayerType_EyeFov = ovr_capi.ovrLayerType_EyeFov

# layer header flags
ovrLayerFlag_HighQuality = ovr_capi.ovrLayerFlag_HighQuality
ovrLayerFlag_TextureOriginAtBottomLeft = ovr_capi.ovrLayerFlag_TextureOriginAtBottomLeft
ovrLayerFlag_HeadLocked = ovr_capi.ovrLayerFlag_HeadLocked

ovrEye_Left = ovr_capi.ovrEye_Left
ovrEye_Right = ovr_capi.ovrEye_Right
ovrEye_Count = ovr_capi.ovrEye_Count

cpdef ovrFovPort get_symmetric_fov():
    """Compute a symmetric field-of-view for the given headset.
    
    :return: tuple
    
    """
    global _hmd_desc_
    cdef ovr_capi.ovrFovPort fov_left = _hmd_desc_.DefaultEyeFov[0]
    cdef ovr_capi.ovrFovPort fov_right = _hmd_desc_.DefaultEyeFov[1]

    cdef ovr_capi.ovrFovPort fov_max
    fov_max.UpTan = <float>cmath.fmax(
        fov_left.c_data[0].UpTan, fov_right.c_data[0].UpTan)
    fov_max.DownTan = <float>cmath.fmax(
        fov_left.c_data[0].DownTan, fov_right.c_data[0].DownTan)
    fov_max.LeftTan = <float>cmath.fmax(
        fov_left.c_data[0].LeftTan, fov_right.c_data[0].LeftTan)
    fov_max.RightTan = <float>cmath.fmax(
        fov_left.c_data[0].RightTan, fov_right.c_data[0].RightTan)

    cdef float combined_htan_horz = max(fov_max.LeftTan, fov_max.RightTan)
    cdef float combined_htan_vert = max(fov_max.LeftTan, fov_max.RightTan)

    cdef ovrFovPort to_return = ovrFovPort()
    cdef ovr_capi.ovrFovPort* fov_both = (<ovrFovPort>to_return).c_data
    fov_both.LeftTan = fov_both.RightTan = combined_htan_horz
    fov_both.UpTan = fov_both.DownTan = combined_htan_vert

    return to_return

cpdef ovrSizei get_fov_texture_size(
        int eye_type,
        ovrFovPort fov,
        float texels_per_pixel=1.0):
    """Compute the recommended buffer (texture) size for a specified 
    configuration.
    
    Returns a tuple with the dimensions of the required texture (w, h). The 
    values can be used when configuring a render buffer which will ultimately
    be used to draw to the HMD buffers.
    
    :return: None 
    
    """
    cdef ovrSizei to_return = ovrSizei()
    (<ovrSizei>to_return).c_data[0] = ovr_capi.ovr_GetFovTextureSize(
        _ptr_session_,
        <ovr_capi.ovrEyeType>eye_type,
        fov.c_data[0],
        texels_per_pixel)

    return to_return

cpdef void configure_eye_render_desc(int eye_type, ovrFovPort fov):
    """Compute eye render descriptors for a given eye. 
    
    Each eye has an internal 'ovrEyeRenderDesc' structure which stores computed
    information which is not accessible directly from Python. You must call this
    function twice (for each eye) to fully configure the descriptors.

    :param eye_type: int
    :param fov: ovrFovPort
    :return: None
    
    """
    global _eye_render_desc_, _eye_layer_, _hmd_to_eye_view_pose_
    _eye_render_desc_[eye_type] = ovr_capi.ovr_GetRenderDesc(
        _ptr_session_,
        <ovr_capi.ovrEyeType>eye_type,
        fov.c_data[0])

    # set the initial eye pose
    _hmd_to_eye_view_pose_[eye_type] = _eye_render_desc_[eye_type].HmdToEyePose

    # set the render layer FOV to what is computed
    _eye_layer_.Fov[eye_type] = _eye_render_desc_[eye_type].Fov

cpdef tuple get_buffer_size(str fov_type='recommended',
                            float texel_per_pixel=1.0):
    """Compute the recommended buffer (texture) size for a specified 
    configuration.
    
    Returns a tuple with the dimensions of the required texture (w, h). The 
    values can be used when configuring a render buffer which will ultimately
    be used to draw to the HMD buffers.
    
    :return: None 
    
    """
    # get the buffer size for the specified FOV type and buffer layout
    cdef ovr_capi.ovrSizei rec_tex0_size, rec_tex1_size, buffer_size
    if fov_type == 'recommended':
        rec_tex0_size = ovr_capi.ovr_GetFovTextureSize(
            _ptr_session_,
            ovr_capi.ovrEye_Left,
            _hmd_desc_.DefaultEyeFov[0],
            texel_per_pixel)
        rec_tex1_size = ovr_capi.ovr_GetFovTextureSize(
            _ptr_session_,
            ovr_capi.ovrEye_Right,
            _hmd_desc_.DefaultEyeFov[1],
            texel_per_pixel)
    elif fov_type == 'max':
        rec_tex0_size = ovr_capi.ovr_GetFovTextureSize(
            _ptr_session_,
            ovr_capi.ovrEye_Left,
            _hmd_desc_.MaxEyeFov[0],
            texel_per_pixel)
        rec_tex1_size = ovr_capi.ovr_GetFovTextureSize(
            _ptr_session_,
            ovr_capi.ovrEye_Right,
            _hmd_desc_.MaxEyeFov[1],
            texel_per_pixel)

    buffer_size.w  = rec_tex0_size.w + rec_tex1_size.w
    buffer_size.h = max(rec_tex0_size.h, rec_tex1_size.h)

    return buffer_size.w, buffer_size.h

cpdef void set_render_viewport(str eye, int x, int y, int width, int height):
    """
    
    :param x: int
    :param y: int
    :param width: int
    :param height: int 
    :param eye: str
    :return: None
    
    """
    cdef int buffer
    if eye == 'left':
        buffer = 0
    elif eye == 'right':
        buffer = 1

    global _eye_layer_
    _eye_layer_.Viewport[buffer].Pos.x = x
    _eye_layer_.Viewport[buffer].Pos.y = y
    _eye_layer_.Viewport[buffer].Size.w = width
    _eye_layer_.Viewport[buffer].Size.h = height

cpdef void set_render_swap_chain(str eye, object swap_chain):
    """
    
    :param swap_chain: int
    :param eye: str
    :return: 
    
    """
    cdef int buffer
    if eye == 'left':
        buffer = 0
    elif eye == 'right':
        buffer = 1

    # set the swap chain textures
    global _eye_layer_

    if not swap_chain is None:
        _eye_layer_.ColorTexture[buffer] = _swap_chain_[<int>swap_chain]
    else:
        _eye_layer_.ColorTexture[buffer] = NULL


cpdef tuple get_render_viewport(str eye='left'):
    global _ptr_session_, _eye_layer_
    if eye == 'left':
        return (<int>_eye_layer_.Viewport[0].Pos.x,
                <int>_eye_layer_.Viewport[0].Pos.y,
                <int>_eye_layer_.Viewport[0].Size.w,
                <int>_eye_layer_.Viewport[0].Size.h)
    elif eye == 'right':
        return (<int>_eye_layer_.Viewport[1].Pos.x,
                <int>_eye_layer_.Viewport[1].Pos.y,
                <int>_eye_layer_.Viewport[1].Size.w,
                <int>_eye_layer_.Viewport[1].Size.h)

# ---------------------------------
# VR Tracking Classes and Functions
# ---------------------------------
#
cdef class ovrPoseStatef(object):
    """Pose state data.

    """
    cdef ovr_capi.ovrPoseStatef* c_data
    cdef ovr_capi.ovrPoseStatef  c_ovrPoseStatef

    cdef ovrPosef field_the_pose
    cdef ovrVector3f field_angular_velocity
    cdef ovrVector3f field_linear_velocity
    cdef ovrVector3f field_angular_acceleration
    cdef ovrVector3f field_linear_acceleration

    def __cinit__(self, *args, **kwargs):
        self.c_data = &self.c_ovrPoseStatef

        self.field_angular_velocity = ovrVector3f()
        self.field_linear_velocity = ovrVector3f()
        self.field_angular_acceleration = ovrVector3f()
        self.field_linear_acceleration = ovrVector3f()

    @property
    def the_pose(self):
        cdef ovrPosef to_return = ovrPosef()
        (<ovrPosef>to_return).c_data[0] = <ovr_math.Posef>self.c_data[0].ThePose

        return to_return

    @property
    def angular_velocity(self):
        self.field_angular_velocity.c_data[0] = \
            (<ovr_math.Vector3f>self.c_data[0].AngularVelocity)

        return self.field_angular_velocity

    @property
    def linear_velocity(self):
        self.field_linear_velocity.c_data[0] = \
            (<ovr_math.Vector3f>self.c_data[0].LinearVelocity)

        return self.field_linear_velocity

    @property
    def angular_acceleration(self):
        self.field_angular_acceleration.c_data[0] = \
            (<ovr_math.Vector3f>self.c_data[0].AngularAcceleration)

        return self.field_angular_acceleration

    @property
    def linear_acceleration(self):
        self.field_linear_acceleration.c_data[0] = \
            (<ovr_math.Vector3f>self.c_data[0].LinearAcceleration)

        return self.field_linear_acceleration

    @property
    def time_in_seconds(self):
        return <double>self.c_data[0].TimeInSeconds


cdef class TrackingStateData(object):
    """Structure which stores tracking state information. All attributes are
    read-only, returning a copy of the data in the accessed field.

    """
    cdef ovr_capi.ovrTrackingState* c_data
    cdef ovr_capi.ovrTrackingState  c_ovrTrackingState

    def __cinit__(self, *args, **kwargs):
        self.c_data = &self.c_ovrTrackingState

    @property
    def head_pose(self):
        cdef ovrPoseStatef to_return = ovrPoseStatef()
        (<ovrPoseStatef>to_return).c_data[0] = self.c_data[0].HeadPose

        return to_return

    @property
    def status_flags(self):
        return <unsigned int>self.c_data[0].StatusFlags

    @property
    def is_head_position_tracked(self):
        cdef unsigned int status_flags = self.c_data[0].StatusFlags
        cdef unsigned int status_bits = ovr_capi.ovrStatus_PositionTracked

        return <bint>(status_flags & status_bits) == status_bits

    @property
    def is_head_orientation_tracked(self):
        cdef unsigned int status_flags = self.c_data[0].StatusFlags
        cdef unsigned int status_bits = ovr_capi.ovrStatus_OrientationTracked

        return <bint>(status_flags & status_bits) == status_bits

    @property
    def is_head_tracked(self):
        cdef unsigned int status_flags = self.c_data[0].StatusFlags
        cdef unsigned int status_bits = ovr_capi.ovrStatus_OrientationTracked
        status_bits |= ovr_capi.ovrStatus_PositionTracked

        return <bint>(status_flags & status_bits) == status_bits

    @property
    def hand_poses(self):
        cdef ovrPoseStatef left_hand_pose = ovrPoseStatef()
        (<ovrPoseStatef>left_hand_pose).c_data[0] = self.c_data[0].HandPoses[0]

        cdef ovrPoseStatef right_hand_pose = ovrPoseStatef()
        (<ovrPoseStatef>right_hand_pose).c_data[0] = self.c_data[0].HandPoses[1]

        return left_hand_pose, right_hand_pose

    @property
    def hand_status_flags(self):
        return <unsigned int>self.c_data[0].HandStatusFlags[0], \
               <unsigned int>self.c_data[0].HandStatusFlags[1]

cpdef TrackingStateData get_tracking_state(
        double abs_time,
        bint latency_marker=True):

    cdef ovr_capi.ovrBool use_marker = \
        ovr_capi.ovrTrue if latency_marker else ovr_capi.ovrFalse

    cdef ovr_capi.ovrTrackingState ts = ovr_capi.ovr_GetTrackingState(
        _ptr_session_, abs_time, use_marker)

    cdef TrackingStateData to_return = TrackingStateData()
    (<TrackingStateData>to_return).c_data[0] = ts

    return to_return

cpdef void set_tracking_origin_type(str origin='floor'):
    """Set the tracking origin type. Can either be 'floor' or 'eye'.
    
    :param origin: str
    :return: 
    
    """
    global _ptr_session_
    cdef ovr_capi.ovrResult result
    if origin == 'floor':
        result = ovr_capi.ovr_SetTrackingOriginType(
            _ptr_session_, ovr_capi.ovrTrackingOrigin_FloorLevel)
    elif origin == 'eye':
        result = ovr_capi.ovr_SetTrackingOriginType(
            _ptr_session_, ovr_capi.ovrTrackingOrigin_EyeLevel)

    if debug_mode:
        check_result(result)

cpdef str get_tracking_origin_type():
    """Get the current tracking origin type.
    
    :return: str
    
    """
    global _ptr_session_
    cdef ovr_capi.ovrTrackingOrigin origin = ovr_capi.ovr_GetTrackingOriginType(
        _ptr_session_)

    if origin == ovr_capi.ovrTrackingOrigin_FloorLevel:
        return 'floor'
    elif origin == ovr_capi.ovrTrackingOrigin_EyeLevel:
        return 'eye'

cpdef void recenter_tracking_origin():
    """Recenter the tracking origin.
    
    :return: None
    
    """
    global _ptr_session_
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_RecenterTrackingOrigin(
        _ptr_session_)

    if debug_mode:
        check_result(result)

cpdef void specify_tracking_origin(ovrPosef origin_pose):
    """Specify a custom tracking origin.
    
    :param origin_pose: ovrVector3f
    :return: 
    """
    global _ptr_session_
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_SpecifyTrackingOrigin(
        _ptr_session_, <ovr_capi.ovrPosef>origin_pose.c_data[0])

    if debug_mode:
        check_result(result)

cpdef tuple calc_eye_poses(TrackingStateData tracking_state):
    """Calculate eye poses from tracking state data.
    
    Poses are stored internally for conversion to transformation matrices by 
    calling 'get_eye_view_matrix'. Should be called at least once per frame 
    after 'wait_to_begin_frame' but before 'begin_frame' to minimize 
    motion-to-photon latency.
    
    :param tracking_state: TrackingStateData
    :return: 
    
    """
    global _hmd_to_eye_view_pose_

    ovr_capi_util.ovr_CalcEyePoses2(
        tracking_state.c_data[0].HeadPose.ThePose,
        _hmd_to_eye_view_pose_,
        _eye_layer_.RenderPose)

    cdef ovrPosef eye_pose0 = ovrPosef()
    cdef ovrPosef eye_pose1 = ovrPosef()
    (<ovrPosef>eye_pose0).c_data[0] = <ovr_math.Posef>_eye_layer_.RenderPose[0]
    (<ovrPosef>eye_pose1).c_data[0] = <ovr_math.Posef>_eye_layer_.RenderPose[1]

    return eye_pose0, eye_pose1

cpdef ovrMatrix4f get_eye_view_matrix(ovrPosef eye_pose):
    """Get the view matrix from the last calculated head pose. This should be
    called once per frame if real-time head tracking is desired.
    
    :param eye: str
    :return: 
    
    """
    cdef ovrVector3f pos = ovrVector3f()
    cdef ovrMatrix4f rot = ovrMatrix4f()
    pos.c_data[0] = <ovr_math.Vector3f>eye_pose.c_data.Translation
    rot.c_data[0] = ovr_math.Matrix4f(<ovr_math.Quatf>eye_pose.c_data.Rotation)

    cdef ovrVector3f final_up = \
        (<ovrVector3f>rot).transform(ovrVector3f(0, 1, 0))
    cdef ovrVector3f final_forward = \
        (<ovrVector3f>rot).transform(ovrVector3f(0, 0, -1))
    cdef ovrMatrix4f view = \
        ovrMatrix4f.look_at(pos, pos + final_forward, final_up)

    return view

cpdef ovrMatrix4f get_eye_projection_matrix(
        str eye='left',
        float near_clip=0.2,
        float far_clip=1000.0):
    """Get the projection matrix for a specified eye. These do not need to be
    computed more than once per session unless the render layer descriptors are 
    updated, or the clipping planes have been changed.
    
    :param eye: str 
    :param near_clip: float
    :param far_clip: float
    :return: 
    
    """
    cdef int buffer = 0 if eye == 'left' else 1
    global _eye_layer_

    cdef ovrMatrix4f proj = ovrMatrix4f()
    (<ovrMatrix4f>proj).c_data[0] = \
        <ovr_math.Matrix4f>ovr_capi_util.ovrMatrix4f_Projection(
            _eye_layer_.Fov[buffer],
            near_clip,
            far_clip,
            ovr_capi_util.ovrProjection_ClipRangeOpenGL)

    return proj

# -------------------------
# Frame Rendering Functions
# -------------------------
#
cpdef double get_display_time(unsigned int frame_index=0, bint predicted=True):
    cdef double t_secs
    if predicted:
        t_secs = ovr_capi.ovr_GetPredictedDisplayTime(
            _ptr_session_, frame_index)
    else:
        t_secs = ovr_capi.ovr_GetTimeInSeconds()

    return t_secs

cpdef int wait_to_begin_frame(unsigned int frame_index):
    cdef ovr_capi.ovrResult result = 0
    result = ovr_capi.ovr_WaitToBeginFrame(_ptr_session_, frame_index)

    return <int>result

cpdef int begin_frame(unsigned int frame_index):
    result = ovr_capi.ovr_BeginFrame(_ptr_session_, frame_index)

    return <int>result

cpdef void commit_swap_chain(int sc):
    global _ptr_session_, _swap_chain_
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_CommitTextureSwapChain(
        _ptr_session_,
        _swap_chain_[sc])

    if debug_mode:
        check_result(result)

cpdef void end_frame(unsigned int frame_index=0):
    global _eye_layer_
    cdef ovr_capi.ovrLayerHeader* layers = &_eye_layer_.Header
    result = ovr_capi.ovr_EndFrame(
        _ptr_session_,
        frame_index,
        NULL,
        &layers,
        <unsigned int>1)

    if debug_mode:
        check_result(result)

    global _frame_index_
    _frame_index_ += 1

# ------------------------
# Mirror Texture Functions
# ------------------------
#
cpdef void setup_mirror_texture(int width=800, int height=600):
    """Create a mirror texture buffer.
    
    :param width: int 
    :param height: int 
    :return: None
    
    """
    cdef ovr_capi.ovrMirrorTextureDesc mirror_desc
    mirror_desc.Format = ovr_capi.OVR_FORMAT_R8G8B8A8_UNORM_SRGB
    mirror_desc.Width = width
    mirror_desc.Height = height
    mirror_desc.MiscFlags = ovr_capi.ovrTextureMisc_None
    mirror_desc.MirrorOptions = ovr_capi.ovrMirrorOption_PostDistortion

    global _mirror_texture_
    cdef ovr_capi.ovrResult result = ovr_capi_gl.ovr_CreateMirrorTextureGL(
        _ptr_session_, &mirror_desc, &_mirror_texture_)

    if debug_mode:
        check_result(result)

cpdef unsigned int get_mirror_texture():
    """Get the mirror texture handle.
    
    :return: 
    """
    cdef unsigned int out_tex_id
    cdef ovr_capi.ovrResult result = \
        ovr_capi_gl.ovr_GetMirrorTextureBufferGL(
            _ptr_session_,
            _mirror_texture_,
            &out_tex_id)

    return <unsigned int>out_tex_id

# ------------------------
# Session Status Functions
# ------------------------
#
cpdef void update_session_status():
    """Update session status information. Must be called at least once every 
    render cycle.
    
    :return: None 
    
    """
    global _ptr_session_, _session_status_
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetSessionStatus(
        _ptr_session_, &_session_status_)

cpdef bint is_visible():
    global _session_status_
    return (<ovr_capi.ovrSessionStatus>_session_status_).IsVisible

cpdef bint hmd_present():
    global _session_status_
    return (<ovr_capi.ovrSessionStatus>_session_status_).HmdPresent

cpdef bint display_lost():
    global _session_status_
    return (<ovr_capi.ovrSessionStatus>_session_status_).DisplayLost

cpdef bint should_quit():
    global _session_status_
    return (<ovr_capi.ovrSessionStatus>_session_status_).ShouldQuit

cpdef bint should_recenter():
    global _session_status_
    return (<ovr_capi.ovrSessionStatus>_session_status_).ShouldRecenter

cpdef bint has_input_focus():
    global _session_status_
    return (<ovr_capi.ovrSessionStatus>_session_status_).HasInputFocus

cpdef bint overlay_present():
    global _session_status_
    return (<ovr_capi.ovrSessionStatus>_session_status_).OverlayPresent

cpdef bint depth_requested():
    global _session_status_
    return (<ovr_capi.ovrSessionStatus>_session_status_).DepthRequested

# -------------------------
# HID Classes and Functions
# -------------------------
#
cdef class ovrInputState(object):
    """Class storing the state of an input device. Fields can only be updated
    by calling 'get_input_state()'.

    """
    cdef ovr_capi.ovrInputState* c_data
    cdef ovr_capi.ovrInputState c_ovrInputState

    def __cinit__(self, *args, **kwargs):
        self.c_data = &self.c_ovrInputState

    @property
    def time_in_seconds(self):
        return <double>self.c_data.TimeInSeconds

    @property
    def buttons(self):
        return self.c_data[0].Buttons

    @property
    def touches(self):
        return self.c_data[0].Touches

    @property
    def index_trigger(self):
        cdef float index_trigger_left = self.c_data[0].IndexTrigger[0]
        cdef float index_trigger_right = self.c_data[0].IndexTrigger[1]

        return index_trigger_left, index_trigger_right

    @property
    def hand_trigger(self):
        cdef float hand_trigger_left = self.c_data[0].HandTrigger[0]
        cdef float hand_trigger_right = self.c_data[0].HandTrigger[1]

        return hand_trigger_left, hand_trigger_right

    @property
    def thumbstick(self):
        cdef float thumbstick_x0 = self.c_data[0].Thumbstick[0].x
        cdef float thumbstick_y0 = self.c_data[0].Thumbstick[0].y
        cdef float thumbstick_x1 = self.c_data[0].Thumbstick[1].x
        cdef float thumbstick_y1 = self.c_data[0].Thumbstick[1].y

        return (thumbstick_x0, thumbstick_y0), (thumbstick_x1, thumbstick_y1)

    @property
    def controller_type(self):
        cdef int ctrl_type = <int>self.c_data[0].ControllerType
        if ctrl_type == ovr_capi.ovrControllerType_XBox:
            return 'xbox'
        elif ctrl_type == ovr_capi.ovrControllerType_Remote:
            return 'remote'
        elif ctrl_type == ovr_capi.ovrControllerType_Touch:
            return 'touch'
        elif ctrl_type == ovr_capi.ovrControllerType_LTouch:
            return 'ltouch'
        elif ctrl_type == ovr_capi.ovrControllerType_RTouch:
            return 'rtouch'
        else:
            return None

    @property
    def index_trigger_no_deadzone(self):
        cdef float index_trigger_left = self.c_data[0].IndexTriggerNoDeadzone[0]
        cdef float index_trigger_right = self.c_data[0].IndexTriggerNoDeadzone[1]

        return index_trigger_left, index_trigger_right

    @property
    def hand_trigger_no_deadzone(self):
        cdef float hand_trigger_left = self.c_data[0].HandTriggerNoDeadzone[0]
        cdef float hand_trigger_right = self.c_data[0].HandTriggerNoDeadzone[1]

        return hand_trigger_left, hand_trigger_right

    @property
    def thumbstick_no_deadzone(self):
        cdef float thumbstick_x0 = self.c_data[0].ThumbstickNoDeadzone[0].x
        cdef float thumbstick_y0 = self.c_data[0].ThumbstickNoDeadzone[0].y
        cdef float thumbstick_x1 = self.c_data[0].ThumbstickNoDeadzone[1].x
        cdef float thumbstick_y1 = self.c_data[0].ThumbstickNoDeadzone[1].y

        return (thumbstick_x0, thumbstick_y0), (thumbstick_x1, thumbstick_y1)

    @property
    def index_trigger_raw(self):
        cdef float index_trigger_left = self.c_data[0].IndexTriggerRaw[0]
        cdef float index_trigger_right = self.c_data[0].IndexTriggerRaw[1]

        return index_trigger_left, index_trigger_right

    @property
    def hand_trigger_raw(self):
        cdef float hand_trigger_left = self.c_data[0].HandTriggerRaw[0]
        cdef float hand_trigger_right = self.c_data[0].HandTriggerRaw[1]

        return hand_trigger_left, hand_trigger_right

    @property
    def thumbstick_no_raw(self):
        cdef float thumbstick_x0 = self.c_data[0].ThumbstickRaw[0].x
        cdef float thumbstick_y0 = self.c_data[0].ThumbstickRaw[0].y
        cdef float thumbstick_x1 = self.c_data[0].ThumbstickRaw[1].x
        cdef float thumbstick_y1 = self.c_data[0].ThumbstickRaw[1].y

        return (thumbstick_x0, thumbstick_y0), (thumbstick_x1, thumbstick_y1)

cpdef object get_input_state(str controller, object state_out=None):
    """Get a controller state as an object. If a 'InputStateData' object is
    passed to 'state_out', that object will be updated.
    
    :param controller: str
    :param state_out: InputStateData or None
    :return: InputStateData or None
    
    """
    cdef ovr_capi.ovrControllerType ctrl_type
    if controller == 'xbox':
        ctrl_type = ovr_capi.ovrControllerType_XBox
    elif controller == 'remote':
        ctrl_type = ovr_capi.ovrControllerType_Remote
    elif controller == 'touch':
        ctrl_type = ovr_capi.ovrControllerType_Touch
    elif controller == 'left_touch':
        ctrl_type = ovr_capi.ovrControllerType_LTouch
    elif controller == 'right_touch':
        ctrl_type = ovr_capi.ovrControllerType_RTouch

    # create a controller state object and set its data
    global _ptr_session_
    cdef ovr_capi.ovrInputState* ptr_state
    cdef ovrInputState to_return = ovrInputState()

    if state_out is None:
        ptr_state = &(<ovrInputState>to_return).c_ovrInputState
    else:
        ptr_state = &(<ovrInputState>state_out).c_ovrInputState

    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetInputState(
        _ptr_session_,
        ctrl_type,
        ptr_state)

    if state_out is None:
        return None

    return to_return

cpdef double poll_controller(str controller):
    """Poll and update specified controller's state data. The time delta in 
    seconds between the current and previous controller state is returned.
    
    :param controller: str or None
    :return: double
    
    """
    global _ptr_session_, _ctrl_states_, _ctrl_states_prev_
    cdef ovr_capi.ovrInputState* ptr_ctrl = NULL
    cdef ovr_capi.ovrInputState* ptr_ctrl_prev = NULL

    cdef ovr_capi.ovrControllerType ctrl_type
    if controller == 'xbox':
        ctrl_type = ovr_capi.ovrControllerType_XBox
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.xbox]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.xbox]
    elif controller == 'remote':
        ctrl_type = ovr_capi.ovrControllerType_Remote
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.remote]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.remote]
    elif controller == 'touch':
        ctrl_type = ovr_capi.ovrControllerType_Touch
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.touch]
    elif controller == 'left_touch':
        ctrl_type = ovr_capi.ovrControllerType_LTouch
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.left_touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.left_touch]
    elif controller == 'right_touch':
        ctrl_type = ovr_capi.ovrControllerType_RTouch
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.right_touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.right_touch]

    # copy the previous control state
    ptr_ctrl_prev[0] = ptr_ctrl[0]

    # update the current controller state
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetInputState(
        _ptr_session_,
        ctrl_type,
        ptr_ctrl)

    if debug_mode:
        check_result(result)

    # return the time delta between the last time the controller was polled
    return ptr_ctrl[0].TimeInSeconds - ptr_ctrl_prev[0].TimeInSeconds

cpdef double get_controller_abs_time(str controller):
    """Get the absolute time the state of the specified controller was last 
    updated.
    
    :param controller: str or None
    :return: float
    
    """
    # get pointer to control state
    global _ctrl_states_
    cdef ovr_capi.ovrInputState* ptr_ctrl_state = NULL
    if controller == 'xbox':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.xbox]
    elif controller == 'remote':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.remote]
    elif controller == 'touch':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.touch]
    elif controller == 'left_touch':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.left_touch]
    elif controller == 'right_touch':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.right_touch]

    return ptr_ctrl_state[0].TimeInSeconds

cpdef tuple get_index_trigger_values(str controller, bint dead_zone=False):
    """Get index trigger values for a specified controller.
    
    :param controller: str
    :param deadzone: boolean
    :return: tuple
    
    """
    # get pointer to control state
    global _ctrl_states_
    cdef ovr_capi.ovrInputState* ptr_ctrl_state = NULL
    if controller == 'xbox':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.xbox]
    elif controller == 'remote':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.remote]
    elif controller == 'touch':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.touch]
    elif controller == 'left_touch':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.left_touch]
    elif controller == 'right_touch':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.right_touch]

    cdef float index_trigger_left = 0.0
    cdef float index_trigger_right = 0.0

    # get the value with or without the deadzone
    if not dead_zone:
        index_trigger_left = ptr_ctrl_state[0].IndexTriggerNoDeadzone[0]
        index_trigger_right = ptr_ctrl_state[0].IndexTriggerNoDeadzone[1]
    else:
        index_trigger_left = ptr_ctrl_state[0].IndexTrigger[0]
        index_trigger_right = ptr_ctrl_state[0].IndexTrigger[1]

    return index_trigger_left, index_trigger_right

cpdef tuple get_hand_trigger_values(str controller, bint dead_zone=False):
    """Get hand trigger values for a specified controller.
    
    :param controller: str
    :param deadzone: boolean
    :return: tuple
    
    """
    # get pointer to control state
    global _ctrl_states_
    cdef ovr_capi.ovrInputState* ptr_ctrl_state = NULL
    if controller == 'xbox':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.xbox]
    elif controller == 'remote':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.remote]
    elif controller == 'touch':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.touch]
    elif controller == 'left_touch':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.left_touch]
    elif controller == 'right_touch':
        ptr_ctrl_state = &_ctrl_states_[<int>LibOVRControllers.right_touch]

    cdef float hand_trigger_left = 0.0
    cdef float hand_trigger_right = 0.0

    # get the value with or without the deadzone
    if not dead_zone:
        hand_trigger_left = ptr_ctrl_state[0].HandTriggerNoDeadzone[0]
        hand_trigger_right = ptr_ctrl_state[0].HandTriggerNoDeadzone[1]
    else:
        hand_trigger_left = ptr_ctrl_state[0].HandTrigger[0]
        hand_trigger_right = ptr_ctrl_state[0].HandTrigger[1]

    return hand_trigger_left, hand_trigger_right

cdef float clip_input_range(float val):
    """Constrain an analog input device's range between -1.0 and 1.0. This is 
    only accessible from module functions.
    
    :param val: float
    :return: float
    
    """
    if val > 1.0:
        val = 1.0
    elif val < 1.0:
        val = 1.0

    return val

cpdef tuple get_thumbstick_values(str controller, bint dead_zone=False):
    """Get thumbstick values for a specified controller.
    
    :param controller: 
    :param dead_zone: 
    :return: tuple
    
    """
    # get pointer to control state
    global _ctrl_states_
    cdef ovr_capi.ovrInputState* ptr_ctrl = NULL
    cdef ovr_capi.ovrInputState* ptr_ctrl_prev = NULL
    if controller == 'xbox':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.xbox]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.xbox]
    elif controller == 'remote':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.remote]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.remote]
    elif controller == 'touch':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.touch]
    elif controller == 'left_touch':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.left_touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.left_touch]
    elif controller == 'right_touch':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.right_touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.right_touch]

    cdef float thumbstick0_x = 0.0
    cdef float thumbstick0_y = 0.0
    cdef float thumbstick1_x = 0.0
    cdef float thumbstick1_y = 0.0

    # get the value with or without the deadzone
    if not dead_zone:
        thumbstick0_x = ptr_ctrl[0].Thumbstick[0].x
        thumbstick0_y = ptr_ctrl[0].Thumbstick[0].y
        thumbstick1_x = ptr_ctrl[0].Thumbstick[1].x
        thumbstick1_y = ptr_ctrl[0].Thumbstick[1].y
    else:
        thumbstick0_x = ptr_ctrl[0].ThumbstickNoDeadzone[0].x
        thumbstick0_y = ptr_ctrl[0].ThumbstickNoDeadzone[0].y
        thumbstick1_x = ptr_ctrl[0].ThumbstickNoDeadzone[1].x
        thumbstick1_y = ptr_ctrl[0].ThumbstickNoDeadzone[1].y

    # clip range
    thumbstick0_x = clip_input_range(thumbstick0_x)
    thumbstick0_y = clip_input_range(thumbstick0_y)
    thumbstick1_x = clip_input_range(thumbstick1_x)
    thumbstick1_y = clip_input_range(thumbstick1_y)

    return (thumbstick0_x, thumbstick0_y), (thumbstick1_x, thumbstick1_y)

cpdef bint get_buttons(str controller, object button_names, str trigger='continuous'):
    """Get the state of a specified button for a given controller. 
    
    Buttons to test are specified using their string names. Argument
    'button_names' accepts a single string or a list. If a list is specified,
    the returned value will reflect whether all buttons were triggered at the
    time the controller was polled last. 
    
    An optional trigger mode may be specified which defines the button's
    activation criteria. Be default, trigger='continuous' which will return the
    immediate state of the button is used. Using 'rising' will return True once 
    when the button is first pressed, whereas 'falling' will return True once 
    the button is released.
    
    :param controller: str
    :param touch_names: str, tuple or list
    :param trigger: str
    :return: boolean
    
    """
    # get pointer to control state
    global _ctrl_states_
    cdef ovr_capi.ovrInputState* ptr_ctrl = NULL
    cdef ovr_capi.ovrInputState* ptr_ctrl_prev = NULL
    if controller == 'xbox':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.xbox]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.xbox]
    elif controller == 'remote':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.remote]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.remote]
    elif controller == 'touch':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.touch]
    elif controller == 'left_touch':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.left_touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.left_touch]
    elif controller == 'right_touch':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.right_touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.right_touch]

    cdef unsigned int button_bits = 0x00000000
    cdef int i, N
    if isinstance(button_names, str):  # don't loop if a string is specified
        button_bits |= ctrl_button_lut[button_names]
    elif isinstance(button_names, (tuple, list)):
        # loop over all names and combine them
        N = <int>len(button_names)
        for i in range(N):
            button_bits |= ctrl_button_lut[button_names[i]]

    # test if the button was pressed
    cdef bint pressed
    if trigger == 'continuous':
        pressed = (ptr_ctrl.Buttons & button_bits) == button_bits
    elif trigger == 'rising' or trigger == 'pressed':
        # rising edge, will trigger once when pressed
        pressed = (ptr_ctrl.Buttons & button_bits) == button_bits and \
            (ptr_ctrl_prev.Buttons & button_bits) != button_bits
    elif trigger == 'falling' or trigger == 'released':
        # falling edge, will trigger once when released
        pressed = (ptr_ctrl.Buttons & button_bits) != button_bits and \
            (ptr_ctrl_prev.Buttons & button_bits) == button_bits
    else:
        raise ValueError("Invalid trigger mode specified.")

    return pressed

cpdef bint get_touches(str controller, object touch_names, str trigger='continuous'):
    """Get touches for a specified device.
    
    Touches reveal information about the user's hand pose, for instance, whether 
    a pointing or pinching gesture is being made. Oculus Touch controllers are
    required for this functionality.

    Touch points to test are specified using their string names. Argument
    'touch_names' accepts a single string or a list. If a list is specified,
    the returned value will reflect whether all touches were triggered at the
    time the controller was polled last. 
    
    :param controller: str
    :param touch_names: str, tuple or list
    :param trigger: str
    :return: boolean
    
    """
    # get pointer to control state
    global _ctrl_states_
    cdef ovr_capi.ovrInputState* ptr_ctrl = NULL
    cdef ovr_capi.ovrInputState* ptr_ctrl_prev = NULL
    if controller == 'xbox':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.xbox]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.xbox]
    elif controller == 'remote':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.remote]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.remote]
    elif controller == 'touch':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.touch]
    elif controller == 'left_touch':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.left_touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.left_touch]
    elif controller == 'right_touch':
        ptr_ctrl = &_ctrl_states_[<int>LibOVRControllers.right_touch]
        ptr_ctrl_prev = &_ctrl_states_prev_[<int>LibOVRControllers.right_touch]

    cdef unsigned int touch_bits = 0x00000000
    cdef int i, N
    if isinstance(touch_names, str):  # don't loop if a string is specified
        touch_bits |= ctrl_button_lut[touch_names]
    elif isinstance(touch_names, (tuple, list)):
        # loop over all names and combine them
        N = <int>len(touch_names)
        for i in range(N):
            touch_bits |= ctrl_button_lut[touch_names[i]]

    # test if the button was pressed
    cdef bint touched
    if trigger == 'continuous':
        touched = (ptr_ctrl.Buttons & touch_bits) == touch_bits
    elif trigger == 'rising' or trigger == 'pressed':
        # rising edge, will trigger once when pressed
        touched = (ptr_ctrl.Buttons & touch_bits) == touch_bits and \
            (ptr_ctrl_prev.Buttons & touch_bits) != touch_bits
    elif trigger == 'falling' or trigger == 'released':
        # falling edge, will trigger once when released
        touched = (ptr_ctrl.Buttons & touch_bits) != touch_bits and \
            (ptr_ctrl_prev.Buttons & touch_bits) == touch_bits
    else:
        raise ValueError("Invalid trigger mode specified.")

    return touched

# List of controller names that are available to the user. These are handled by
# the SDK, additional joysticks, keyboards and mice must be accessed by some
# other method.
#
controller_names = ['xbox', 'remote', 'touch', 'left_touch', 'right_touch']

cpdef list get_connected_controller_types():
    """Get a list of currently connected controllers. You can check if a
    controller is attached by testing for its membership in the list using its
    name.
    
    :return: list  
    
    """
    cdef unsigned int result = ovr_capi.ovr_GetConnectedControllerTypes(
        _ptr_session_)

    cdef list ctrl_types = list()
    if (result & ovr_capi.ovrControllerType_XBox) == \
            ovr_capi.ovrControllerType_XBox:
        ctrl_types.append('xbox')
    elif (result & ovr_capi.ovrControllerType_Remote) == \
            ovr_capi.ovrControllerType_Remote:
        ctrl_types.append('remote')
    elif (result & ovr_capi.ovrControllerType_Touch) == \
            ovr_capi.ovrControllerType_Touch:
        ctrl_types.append('touch')
    elif (result & ovr_capi.ovrControllerType_LTouch) == \
            ovr_capi.ovrControllerType_LTouch:
        ctrl_types.append('left_touch')
    elif (result & ovr_capi.ovrControllerType_RTouch) == \
            ovr_capi.ovrControllerType_RTouch:
        ctrl_types.append('right_touch')

    return ctrl_types

# -------------------------------
# Performance/Profiling Functions
# -------------------------------
#
cdef class ovrPerfStatsPerCompositorFrame(object):
    cdef ovr_capi.ovrPerfStatsPerCompositorFrame* c_data
    cdef ovr_capi.ovrPerfStatsPerCompositorFrame  c_ovrPerfStatsPerCompositorFrame

    def __cinit__(self, *args, **kwargs):
        self.c_data = &self.c_ovrPerfStatsPerCompositorFrame

    @property
    def hmd_vsync_index(self):
        return self.c_data[0].HmdVsyncIndex

    @property
    def app_frame_index(self):
        return self.c_data[0].AppFrameIndex

    @property
    def app_dropped_frame_count(self):
        return self.c_data[0].AppDroppedFrameCount

    @property
    def app_queue_ahead_time(self):
        return self.c_data[0].AppQueueAheadTime

    @property
    def app_cpu_elapsed_time(self):
        return self.c_data[0].AppCpuElapsedTime

    @property
    def app_gpu_elapsed_time(self):
        return self.c_data[0].AppGpuElapsedTime

    @property
    def compositor_frame_index(self):
        return self.c_data[0].CompositorFrameIndex

    @property
    def compositor_latency(self):
        return self.c_data[0].CompositorLatency

    @property
    def compositor_cpu_elapsed_time(self):
        return self.c_data[0].CompositorCpuElapsedTime

    @property
    def compositor_gpu_elapsed_time(self):
        return self.c_data[0].CompositorGpuElapsedTime

    @property
    def compositor_cpu_start_to_gpu_end_elapsed_time(self):
        return self.c_data[0].CompositorCpuStartToGpuEndElapsedTime

    @property
    def compositor_gpu_end_to_vsync_elapsed_time(self):
        return self.c_data[0].CompositorGpuEndToVsyncElapsedTime


cdef class ovrPerfStats(object):
    cdef ovr_capi.ovrPerfStats* c_data
    cdef ovr_capi.ovrPerfStats  c_ovrPerfStats
    cdef list perf_stats

    def __cinit__(self, *args, **kwargs):
        self.c_data = &self.c_ovrPerfStats

        # initialize performance stats list
        self.perf_stats = list()
        cdef int i, N
        N = <int>ovr_capi.ovrMaxProvidedFrameStats
        for i in range(N):
            self.perf_stats.append(ovrPerfStatsPerCompositorFrame())
            (<ovrPerfStatsPerCompositorFrame>self.perf_stats[i]).c_data[0] = \
                self.c_data[0].FrameStats[i]

    @property
    def frame_stats_count(self):
        return self.c_data[0].FrameStatsCount

    @property
    def any_frame_stats_dropped(self):
        return <bint>self.c_data[0].AnyFrameStatsDropped

    @property
    def frame_stats(self):
        cdef int i, N
        N = self.c_data[0].FrameStatsCount
        for i in range(N):
            (<ovrPerfStatsPerCompositorFrame>self.perf_stats[i]).c_data[0] = \
                self.c_data[0].FrameStats[i]

        return self.perf_stats

    @property
    def adaptive_gpu_performance_scale(self):
        return <bint>self.c_data[0].AdaptiveGpuPerformanceScale

    @property
    def asw_is_available(self):
        return <bint>self.c_data[0].AswIsAvailable


cpdef ovrPerfStats get_frame_stats():
    """Get most recent performance stats, returns an object with fields
    corresponding to various performance stats reported by the SDK.
    
    :return: dict 
    
    """
    global _ptr_session_

    cdef ovrPerfStats to_return = ovrPerfStats()
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetPerfStats(
        _ptr_session_,
        &(<ovrPerfStats>to_return).c_data[0])

    if debug_mode:
        check_result(result)

    return to_return

cpdef void reset_frame_stats():
    """Flushes backlog of frame stats.
    
    :return: None 
    
    """
    global _ptr_session_
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_ResetPerfStats(
        _ptr_session_)

    if debug_mode:
        check_result(result)

# List of available performance HUD modes.
#
available_hud_modes = [
    'Off',
    'PerfSummary',
    'LatencyTiming',
    'AppRenderTiming',
    'CompRenderTiming',
    'AswStats',
    'VersionInfo']

cpdef void perf_hud_mode(str mode='Off'):
    """Display a performance HUD with a specified mode.
    
    :param mode: str 
    :return: None
    
    """
    global _ptr_session_
    cdef int perf_hud_mode = 0

    if mode == 'Off':
        perf_hud_mode = <int>ovr_capi.ovrPerfHud_Off
    elif mode == 'PerfSummary':
        perf_hud_mode = <int>ovr_capi.ovrPerfHud_PerfSummary
    elif mode == 'LatencyTiming':
        perf_hud_mode = <int>ovr_capi.ovrPerfHud_LatencyTiming
    elif mode == 'AppRenderTiming':
        perf_hud_mode = <int>ovr_capi.ovrPerfHud_AppRenderTiming
    elif mode == 'CompRenderTiming':
        perf_hud_mode = <int>ovr_capi.ovrPerfHud_CompRenderTiming
    elif mode == 'AswStats':
        perf_hud_mode = <int>ovr_capi.ovrPerfHud_AswStats
    elif mode == 'VersionInfo':
        perf_hud_mode = <int>ovr_capi.ovrPerfHud_VersionInfo

    cdef ovr_capi.ovrBool ret = ovr_capi.ovr_SetInt(
        _ptr_session_, b"PerfHudMode", perf_hud_mode)

# -----------------------------
# Boundary and Safety Functions
# -----------------------------
#
cdef ovr_capi.ovrBoundaryLookAndFeel _boundary_style_

cpdef void set_boundry_color(float r, float g, float b):
    global _ptr_session_, _boundary_style_

    cdef ovr_capi.ovrColorf color
    color.r = r
    color.g = g
    color.b = b

    _boundary_style_.Color = color

    cdef ovr_capi.ovrResult result = ovr_capi.ovr_SetBoundaryLookAndFeel(
        _ptr_session_,
        &_boundary_style_)

    if debug_mode:
        check_result(result)

cpdef void reset_boundry_color():
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_ResetBoundaryLookAndFeel(
        _ptr_session_)

    if debug_mode:
        check_result(result)

cpdef bint is_boundry_visible():
    cdef ovr_capi.ovrBool is_visible
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetBoundaryVisible(
        _ptr_session_, &is_visible)

    if debug_mode:
        check_result(result)

    return <bint>is_visible

cpdef void show_boundry(bint show=True):
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_RequestBoundaryVisible(
        _ptr_session_, <ovr_capi.ovrBool>show)

    if debug_mode:
        check_result(result)

# -----------------------
# Miscellaneous Functions
# -----------------------
#
cpdef float get_player_height():
    global _ptr_session_
    cdef float to_return  = ovr_capi.ovr_GetFloat(
        _ptr_session_,
        b"PlayerHeight",
        <float>1.778)

    return to_return

cpdef float get_eye_height():
    global _ptr_session_
    cdef float to_return  = ovr_capi.ovr_GetFloat(
        _ptr_session_,
        b"EyeHeight",
        <float>1.675)

    return to_return

cpdef tuple get_neck_eye_distance():
    global _ptr_session_
    cdef float vals[2]

    cdef unsigned int ret  = ovr_capi.ovr_GetFloatArray(
        _ptr_session_,
        b"NeckEyeDistance",
        vals,
        <unsigned int>2)

    return <float>vals[0], <float>vals[1]

cpdef tuple get_eye_to_nose_distance():
    global _ptr_session_
    cdef float vals[2]

    cdef unsigned int ret  = ovr_capi.ovr_GetFloatArray(
        _ptr_session_,
        b"EyeToNoseDist",
        vals,
        <unsigned int>2)

    return <float>vals[0], <float>vals[1]