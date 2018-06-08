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
cdef ovr_capi.ovrLayerEyeFov _eye_layer_

# Arrays to store device poses.
#
cdef ovr_capi.ovrTrackedDeviceType[9] _device_types_
cdef ovr_capi.ovrPoseStatef[9] _device_poses_

# Session status
#
cdef ovr_capi.ovrSessionStatus _session_status_

# Input states for supported HIDs
#
cdef ovr_capi.ovrInputState _ctrl_state_[2]

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


def alloc_swap_chain(int width, int height):
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

    # configure the swap chain
    cdef ovr_capi.ovrTextureSwapChainDesc config
    config.Type = ovr_capi.ovrTexture_2D
    config.Format = ovr_capi.OVR_FORMAT_R8G8B8A8_UNORM_SRGB
    config.Width = width
    config.Height = height
    config.StaticImage = ovr_capi.ovrFalse
    config.ArraySize = config.MipLevels = config.SampleCount = 1
    config.MiscFlags = ovr_capi.ovrTextureMisc_None
    config.BindFlags = ovr_capi.ovrTextureBind_None

    # create the swap chain
    cdef ovr_capi.ovrResult result = ovr_capi_gl.ovr_CreateTextureSwapChainGL(
        _ptr_session_, &config, &_swap_chain_[sc])

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
    global _eye_render_desc_, _hmd_to_eye_view_pose_
    _eye_render_desc_[0] = ovr_capi.ovr_GetRenderDesc(
        _ptr_session_, ovr_capi.ovrEye_Left, _hmd_desc_.DefaultEyeFov[0])
    _eye_render_desc_[1] = ovr_capi.ovr_GetRenderDesc(
        _ptr_session_, ovr_capi.ovrEye_Right, _hmd_desc_.DefaultEyeFov[1])
    _hmd_to_eye_view_pose_[0] = _eye_render_desc_[0].HmdToEyePose
    _hmd_to_eye_view_pose_[1] = _eye_render_desc_[1].HmdToEyePose

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

cpdef dict get_hmd_info():
    """Get general information about the connected HMD. Information such as the
    serial number can identify a specific unit, etc.
    
    :return: dict 
    
    """
    global _hmd_desc_
    # return general HMD information from descriptor
    cdef dict hmd_info = dict()
    hmd_info["ProductName"] = _hmd_desc_.ProductName.decode('utf-8')
    hmd_info["Manufacturer"] = _hmd_desc_.Manufacturer.decode('utf-8')
    hmd_info["VendorId"] = <int>_hmd_desc_.VendorId
    hmd_info["ProductId"] = <int>_hmd_desc_.ProductId
    hmd_info["SerialNumber"] = _hmd_desc_.SerialNumber.decode('utf-8')
    hmd_info["FirmwareMajor"] = <int>_hmd_desc_.FirmwareMajor
    hmd_info["FirmwareMinor"] = <int>_hmd_desc_.FirmwareMinor
    hmd_info["DisplayRefreshRate"] = <float>_hmd_desc_.DisplayRefreshRate

    return hmd_info

# ---------------------------------
# Rendering Configuration Functions
# ---------------------------------
#
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

cpdef void setup_render_layer(int width, int height, int sc):
    global _ptr_session_, _eye_layer_
    # configure render layer
    _eye_layer_.Header.Type = ovr_capi.ovrLayerType_EyeFov
    _eye_layer_.Header.Flags = ovr_capi.ovrLayerFlag_TextureOriginAtBottomLeft

    # setup layer FOV settings, these are computed earlier
    _eye_layer_.Fov[0] = _eye_render_desc_[0].Fov
    _eye_layer_.Fov[1] = _eye_render_desc_[1].Fov

    # set the viewport
    _eye_layer_.Viewport[0].Pos.x = 0
    _eye_layer_.Viewport[0].Pos.y = 0
    _eye_layer_.Viewport[0].Size.w = width / 2
    _eye_layer_.Viewport[0].Size.h = height
    _eye_layer_.Viewport[1].Pos.x = width / 2
    _eye_layer_.Viewport[1].Pos.y = 0
    _eye_layer_.Viewport[1].Size.w = width / 2
    _eye_layer_.Viewport[1].Size.h = height

    # set the swap chain textures
    _eye_layer_.ColorTexture[0] = _swap_chain_[sc]
    _eye_layer_.ColorTexture[1] = NULL  # NULL for now

cpdef tuple get_render_layer_viewport(str eye='left'):
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

# ------------
# VR Functions
# ------------
#
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
    
    :param origin: ovrVector3f
    :return: 
    """
    global _ptr_session_
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_SpecifyTrackingOrigin(
        _ptr_session_, <ovr_capi.ovrPosef>origin_pose.c_data[0])

    if debug_mode:
        check_result(result)


cpdef void calc_eye_poses(double abs_time, bint time_stamp=True):
    """Calculate eye poses. 
    
    Poses are stored internally for conversion to transformation matrices by 
    calling 'get_eye_view_matrix'. Should be called at least once per frame 
    after 'wait_to_begin_frame' but before 'begin_frame' to minimize 
    motion-to-photon latency.
    
    :param abs_time: 
    :param time_stamp: 
    :return: 
    """
    cdef ovr_capi.ovrBool use_marker = \
        ovr_capi.ovrTrue if time_stamp else ovr_capi.ovrFalse

    cpdef ovr_capi.ovrTrackingState hmd_state = ovr_capi.ovr_GetTrackingState(
        _ptr_session_, abs_time, use_marker)

    ovr_capi_util.ovr_CalcEyePoses2(
        hmd_state.HeadPose.ThePose,
        _hmd_to_eye_view_pose_,
        _eye_layer_.RenderPose)

cpdef ovrMatrix4f get_eye_view_matrix(str eye='left'):
    """Get the view matrix from the last calculated head pose. This should be
    called once per frame if real-time head tracking is desired.
    
    :param eye: str
    :return: 
    
    """
    cdef int buffer = 0 if eye == 'left' else 1
    global _eye_layer_

    cdef ovrVector3f pos = ovrVector3f()
    pos.c_data[0] = \
        <ovr_math.Vector3f>_eye_layer_.RenderPose[buffer].Position
    cdef ovrMatrix4f rot = ovrMatrix4f()
    rot.c_data[0] = \
        ovr_math.Matrix4f(
            <ovr_math.Quatf>_eye_layer_.RenderPose[buffer].Orientation)

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

cpdef int wait_to_begin_frame(unsigned int frame_index=0):
    cdef ovr_capi.ovrResult result = 0
    result = ovr_capi.ovr_WaitToBeginFrame(_ptr_session_, frame_index)

    return <int>result

cpdef int begin_frame(unsigned int frame_index=0):
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
cdef class InputStateData(object):
    cdef ovr_capi.ovrInputState* c_data
    cdef ovr_capi.ovrInputState  c_ovrInputState

    def __cinit_(self, *args, **kwargs):
        self.c_data = &self.c_ovrInputState

    @property
    def time_in_seconds(self):
        return <double>(<ovr_capi.ovrInputState>self).c_data[0].TimeInSeconds

    @property
    def buttons(self):
        return <unsigned int>(<ovr_capi.ovrInputState>self).c_data[0].Buttons

    @property
    def touches(self):
        return <unsigned int>(<ovr_capi.ovrInputState>self).c_data[0].Touches

    @property
    def index_trigger(self):
        return (<float>(<ovr_capi.ovrInputState>self).c_data[0].IndexTrigger[0],
                <float>(<ovr_capi.ovrInputState>self).c_data[0].IndexTrigger[1])

    @property
    def hand_trigger(self):
        return (<float>(<ovr_capi.ovrInputState>self).c_data[0].HandTrigger[0],
                <float>(<ovr_capi.ovrInputState>self).c_data[0].HandTrigger[1])

    @property
    def thumbstick(self):
        return (
            (<float>(<ovr_capi.ovrInputState>self).c_data[0].Thumbstick[0].x,
             <float>(<ovr_capi.ovrInputState>self).c_data[0].Thumbstick[0].y),
            (<float>(<ovr_capi.ovrInputState>self).c_data[0].Thumbstick[1].x,
             <float>(<ovr_capi.ovrInputState>self).c_data[0].Thumbstick[1].y))

cpdef InputStateData get_input_state(str controller='xbox'):
    global _ptr_session_, _ctrl_state_

    cdef ovr_capi.ovrControllerType ctrl_type
    if controller == 'xbox':
        ctrl_type = ovr_capi.ovrControllerType_XBox
    elif controller == 'remote':
        ctrl_type = ovr_capi.ovrControllerType_Remote

    cdef InputStateData to_return = InputStateData()
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetInputState(
        _ptr_session_,
        ctrl_type,
        (<InputStateData>to_return).c_data)

    if debug_mode:
        check_result(result)

    return to_return

cpdef double poll_remote_controller_state():
    global _ptr_session_, _ctrl_state_
    cdef ovr_capi.ovrResult result = ovr_capi.ovr_GetInputState(
        _ptr_session_, ovr_capi.ovrControllerType_Remote, &_ctrl_state_[1])

    if debug_mode:
        check_result(result)

    return (<ovr_capi.ovrInputState>_ctrl_state_[1]).TimeInSeconds
