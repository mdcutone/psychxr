cimport ovr_capi, ovr_capi_gl, ovr_errorcode, ovr_capi_util
cimport ovr_math
from libc.stdint cimport uintptr_t, uint32_t, int32_t
from libcpp cimport nullptr
from libc.stdlib cimport malloc, free
cimport libc.math as cmath

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

# texture swap chain
#
cdef ovr_capi.ovrTextureSwapChain _swap_chain_ = NULL
cdef ovr_capi.ovrMirrorTexture _mirror_texture_ = NULL

# VR related structures to store head pose and other data used across frames.
#
cdef ovr_capi.ovrEyeRenderDesc[2] _eye_render_desc_
cdef ovr_capi.ovrPosef[2] _hmd_to_eye_view_pose_
cdef ovr_capi.ovrLayerEyeFov _eye_layer_

# Arrays to store device poses.
#
cdef ovr_capi.ovrTrackedDeviceType[9] _device_types_
cdef ovr_capi.ovrPoseStatef[9] _device_poses_

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

# PsychXR related functions and types
#
cdef double MATH_DOUBLE_PI = 3.14159265358979323846
cdef float MATH_FLOAT_PI = <float>MATH_DOUBLE_PI
cdef double MATH_DOUBLE_PIOVER2 = <double>0.5 * MATH_DOUBLE_PI
cdef float MATH_FLOAT_PIOVER2 = <float>MATH_DOUBLE_PIOVER2
cdef float MATH_FLOAT_SMALLESTNONDENORMAL = 1.1754943508222875e-38
cdef float MATH_FLOAT_HUGENUMBER = 1.8446742974197924e19

cdef float acos(float x):
    cdef float to_return = 0.0
    if x > 1.0:
        to_return = 0.0
    else:
        if x < -1.0:
            to_return = MATH_FLOAT_PI
        else:
            to_return = cmath.acos(x)

    return to_return

cdef float asin(float x):
    cdef float to_return = 0.0
    if x > 1.0:
        to_return = 0.0
    else:
        if x < -1.0:
            to_return = MATH_FLOAT_PI
        else:
            to_return = cmath.asin(x)

    return to_return


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

        cdef ovrVector3f axis, unit_axis
        cdef float angle, sin_half_angle = 0.0
        cdef int nargin = <int>len(args)  # get number of arguments
        if nargin == 0:
            self.c_data.x = 0.0
            self.c_data.y = 0.0
            self.c_data.z = 0.0
            self.c_data.w = 0.0
        elif nargin == 1:
            if isinstance(ovrMatrix4f, args[0]):
                self.init_from_rotation_matrix(args[0])

        elif nargin == 2:
            # quaternion from axis and angle
            if isinstance(args[0], ovrVector3f) and \
                    isinstance(args[1], (int, float)):
                axis = args[0]
                angle = <float>args[1]  # make sure this is a float
                if axis.length_sq() == <float>0:
                    assert angle == <float>0
                    self.c_data.x = self.c_data.y = self.c_data.z = <float>0
                    self.c_data.w = <float>1
                else:
                    unit_axis = axis.normalized()
                    half_angle = cmath.sin(angle * <float>0.5)
                    self.c_data.w = cmath.cos(angle * <float>0.5)
                    self.c_data.x = unit_axis.c_data.x * sin_half_angle
                    self.c_data.y = unit_axis.c_data.y * sin_half_angle
                    self.c_data.z = unit_axis.c_data.z * sin_half_angle
            elif isinstance(args[0], ovrVector3f) and \
                    isinstance(args[1], ovrVector3f):
                self._init_from_segment(args[0], args[1])
        elif nargin == 4:
            self.c_data.x = <float>args[0]
            self.c_data.y = <float>args[1]
            self.c_data.z = <float>args[2]
            self.c_data.w = <float>args[3]

    cdef init_from_rotation_matrix(self, ovrMatrix4f m):
        # port of Oculus SDK C++ routine found in OVR_Math.h, starting at line
        # 1570
        cdef float trace = 0.0
        cdef float s = 0.0
        if trace > <float>0.0:
            s = cmath.sqrt(trace + <float>1) * <float>2
            self.c_data.w = <float>0.25 * s
            self.c_data.x = (m.M[2][1] - m.M[1][2]) / s
            self.c_data.y = (m.M[0][2] - m.M[2][0]) / s
            self.c_data.z = (m.M[1][0] - m.M[0][1]) / s
        elif m.M[0][0] > m.M[1][1] and m.M[0][0] > m.M[2][2]:
            s = cmath.sqrt(
                <float>1 + m.M[0][0] - m.M[1][1] - m.M[2][2]) * <float>2
            self.c_data.w = (m.M[2][1] - m.M[1][2]) / s
            self.c_data.x = <float>0.25 * s
            self.c_data.y = (m.M[0][1] + m.M[1][0]) / s
            self.c_data.z = (m.M[2][0] + m.M[0][2]) / s
        elif m.M[1][1] > m.M[2][2]:
            s = cmath.sqrt(
                <float>1 + m.M[1][1] - m.M[0][0] - m.M[2][2]) * <float>2
            self.c_data.w = (m.M[0][2] - m.M[2][0]) / s
            self.c_data.x = (m.M[0][1] + m.M[1][0]) / s
            self.c_data.y = <float>0.25 * s
            self.c_data.z = (m.M[1][2] + m.M[2][1]) / s
        else:
            s = cmath.sqrt(
                <float>1 + m.M[2][2] - m.M[0][0] - m.M[1][1]) * <float>2
            self.c_data.w = (m.M[1][0] - m.M[0][1]) / s
            self.c_data.x = (m.M[0][2] + m.M[2][0]) / s
            self.c_data.y = (m.M[1][2] + m.M[2][1]) / s
            self.c_data.z = <float>0.25 * s

        assert self.is_normalized()

    cdef _init_from_segment(self, ovrVector3f from_vec, ovrVector3f to_vec):
        # Port of Oculus SDK C++ routine found in OVR_Math.h, starting at line
        # 1614
        cdef float cx = from_vec.c_data.y * to_vec.c_data.z - \
                        from_vec.c_data.z * to_vec.c_data.y
        cdef float cy = from_vec.c_data.z * to_vec.c_data.x - \
                        from_vec.c_data.x * to_vec.c_data.z
        cdef float cz = from_vec.c_data.x * to_vec.c_data.y - \
                        from_vec.c_data.y * to_vec.c_data.x
        cdef float dot = from_vec.c_data.x * to_vec.c_data.x + \
                         from_vec.c_data.y * to_vec.c_data.y + \
                         from_vec.c_data.z * to_vec.c_data.z
        cdef float cross_len_sq = cx * cx + cy * cy + cz * cz
        cdef float magnitude = cmath.sqrt(dot * dot + cross_len_sq)
        cdef float cw = dot + magnitude
        cdef float sx = 0.0
        cdef float sz = 0.0
        cdef float rcp_len = 0.0

        if cw < MATH_FLOAT_SMALLESTNONDENORMAL:
            sx = to_vec.c_data.y * to_vec.c_data.y + \
                 to_vec.c_data.z * to_vec.c_data.z
            sz = to_vec.c_data.x * to_vec.c_data.x + \
                 to_vec.c_data.y * to_vec.c_data.y
            if sx > sz:
                if sx >= MATH_FLOAT_SMALLESTNONDENORMAL:
                    rcp_len = <float>1.0 / cmath.sqrt(sx)
                else:
                    rcp_len = MATH_FLOAT_HUGENUMBER

                self.c_data.x = <float>0
                self.c_data.y = to_vec.c_data.z * rcp_len
                self.c_data.z = -to_vec.c_data.y * rcp_len
                self.c_data.w = <float>0
            else:
                if sz >= MATH_FLOAT_SMALLESTNONDENORMAL:
                    rcp_len = <float>1.0 / cmath.sqrt(sz)
                else:
                    rcp_len = MATH_FLOAT_HUGENUMBER

                self.c_data.x = to_vec.c_data.y * rcp_len
                self.c_data.y = -to_vec.c_data.x * rcp_len
                self.c_data.z = <float>0
                self.c_data.w = <float>0
        else:
            if sz >= MATH_FLOAT_SMALLESTNONDENORMAL:
                rcp_len = <float>1.0 / cmath.sqrt(sz)
            else:
                rcp_len = MATH_FLOAT_HUGENUMBER

            self.c_data.x = cx * rcp_len
            self.c_data.y = cy * rcp_len
            self.c_data.z = cz * rcp_len
            self.c_data.w = cw * rcp_len

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
        return self.c_data.x == b.c_data.x and \
               self.c_data.y == b.c_data.y and \
               self.c_data.z == b.c_data.z and \
               self.c_data.w == b.c_data.w

    def __ne__(self, ovrQuatf b):
        return self.c_data.x != b.c_data.x or \
               self.c_data.y != b.c_data.y or \
               self.c_data.z != b.c_data.z or \
               self.c_data.w != b.c_data.w


    def __add__(ovrQuatf a, ovrQuatf b):
        cdef ovrQuatf to_return = ovrQuatf(
            a.c_data.x + b.c_data.x,
            a.c_data.y + b.c_data.y,
            a.c_data.z + b.c_data.z,
            a.c_data.w + b.c_data.w)

        return to_return

    def __iadd__(self, ovrQuatf b):
        self.c_data.x += b.c_data.x
        self.c_data.y += b.c_data.y
        self.c_data.z += b.c_data.z
        self.c_data.w += b.c_data.w

        return self

    def __sub__(ovrQuatf a, ovrQuatf b):
        cdef ovrQuatf to_return = ovrQuatf(
            a.c_data.x - b.c_data.x,
            a.c_data.y - b.c_data.y,
            a.c_data.z - b.c_data.z,
            a.c_data.w - b.c_data.w)

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

    def __truediv__(ovrQuatf a, object s):
        cdef float rcp = <float>1 / <float>s
        cdef ovrQuatf to_return = ovrQuatf(
            a.c_data.x * rcp,
            a.c_data.y * rcp,
            a.c_data.z * rcp,
            a.c_data.w * rcp)

        return to_return

    def __itruediv__(self, object s):
        cdef float rcp = <float>1 / <float>s
        self.c_data.x *= rcp
        self.c_data.y *= rcp
        self.c_data.z *= rcp
        self.c_data.w *= rcp

        return self

    def is_equal(self, ovrQuatf b, float tolerance = 0.0):
        return  self.abs(self.dot(b)) >= <float>1 - tolerance

    def abs(self, object v):
        cdef float to_return = 0.0
        if <float>v >= 0:
            to_return = v
        else:
            to_return = -v

        return to_return

    def imag(self):
        cdef ovrVector3f to_return = ovrVector3f(
            self.c_data.x,
            self.c_data.y,
            self.c_data.z)

        return to_return

    def length_sq(self):
        return self.c_data.x * self.c_data.x + \
               self.c_data.y * self.c_data.y + \
               self.c_data.z * self.c_data.z + \
               self.c_data.z * self.c_data.z

    def length(self):
        return cmath.sqrt(self.length_sq())

    def distance(self, ovrQuatf q):
        # Port of Oculus SDK C++ routine found in OVR_Math.h, starting at line
        # 1764
        cdef float d1 = (self - q).length()
        cdef float d2 = (self + q).length()

        return d1 if d1 < d2 else d2

    def distance_sq(self, ovrQuatf q):
        # Port of Oculus SDK C++ routine found in OVR_Math.h, starting at line
        # 1770
        cdef float d1 = (self - q).length_sq()
        cdef float d2 = (self + q).length_sq()

        return d1 if d1 < d2 else d2

    def dot(self, ovrQuatf q):
        return self.c_data.x * q.c_data.x + \
               self.c_data.y * q.c_data.y + \
               self.c_data.z * q.c_data.z + \
               self.c_data.w * q.c_data.w

    def angle(self, ovrQuatf q):
        return <float>2 * acos(self.abs(self.dot(q)))

    def is_normalized(self):
        return cmath.fabs(self.length_sq() - <float>1) < 0

    def normalize(self):
        cdef float s = self.length()
        if s != <float>0:
            s = <float>1 / s

        self *= s

    def normalized(self):
        cdef float s = self.length()
        if s != <float>0:
            s = <float>1 / s

        return self * s

    def conj(self):
        cdef ovrQuatf to_return = ovrQuatf(
            -self.c_data.x, -self.c_data.y, -self.c_data.z, self.c_data.w)
        return to_return

    @staticmethod
    def align(ovrVector3f align_to, ovrVector3f v):
        # Port of Oculus SDK C++ routine found in OVR_Math.h, starting at line
        # 1848
        assert align_to.is_normalized() and v.is_normalized()

        cdef ovrVector3f bisector = align_to + v
        bisector.normalize()

        cdef float cos_half_angle = v.dot(bisector)
        cdef ovrVector3f imag
        cdef float inv_length = 0.0

        if cos_half_angle > <float>0:
            imag = v.cross(bisector)
            return ovrQuatf(imag.x, imag.y, imag.z, cos_half_angle)
        else:
            if cmath.fabs(v.c_data.x) > cmath.fabs(v.c_data.y):
                inv_length = cmath.sqrt(v.c_data.x * v.c_data.x +
                                        v.c_data.z * v.c_data.z)
                if inv_length > <float>0:
                    inv_length = <float>1 / inv_length

                return ovrQuatf(-v.c_data.z * inv_length,
                                0,
                                v.c_data.x * inv_length,
                                0)
            else:
                inv_length = cmath.sqrt(
                    v.c_data.y * v.c_data.y + v.c_data.z * v.c_data.z)
                if inv_length > <float>0:
                    inv_length = <float>1 / inv_length
                return ovrQuatf(0,
                                v.c_data.z * inv_length,
                                -v.c_data.y * inv_length,
                                0)

    def rotate(self, ovrVector3f v):
        # Port of Oculus SDK C++ routine found in OVR_Math.h, starting at line
        # 1929
        assert v.is_normalized()
        cdef float uvx = <float>2 * (self.c_data.y * v.c_data.z -
                                     self.c_data.z * v.c_data.y)
        cdef float uvy = <float>2 * (self.c_data.z * v.c_data.x -
                                     self.c_data.x * v.c_data.z)
        cdef float uvz = <float>2 * (self.c_data.x * v.c_data.y -
                                     self.c_data.y * v.c_data.x)

        cdef ovrVector3f to_return = ovrVector3f(
            v.c_data.x + self.c_data.w *
            uvx + self.c_data.y *
            uvz - self.c_data.z * uvy,
            v.c_data.y + self.c_data.w *
            uvy + self.c_data.z *
            uvx - self.c_data.x * uvz,
            v.c_data.z + self.c_data.w *
            uvz + self.c_data.x *
            uvy - self.c_data.y * uvx)

        return to_return

    def inverse_rotate(self, ovrVector3f v):
        # Port of Oculus SDK C++ routine found in OVR_Math.h, starting at line
        # 1948
        assert v.is_normalized()
        cdef float uvx = <float>2 * (self.c_data.y * v.c_data.z -
                                     self.c_data.z * v.c_data.y)
        cdef float uvy = <float>2 * (self.c_data.z * v.c_data.x -
                                     self.c_data.x * v.c_data.z)
        cdef float uvz = <float>2 * (self.c_data.x * v.c_data.y -
                                     self.c_data.y * v.c_data.x)

        cdef ovrVector3f to_return = ovrVector3f(
            v.c_data.x - self.c_data.w *
            uvx + self.c_data.y *
            uvz - self.c_data.z * uvy,
            v.c_data.y - self.c_data.w *
            uvy + self.c_data.z *
            uvx - self.c_data.x * uvz,
            v.c_data.z - self.c_data.w *
            uvz + self.c_data.x *
            uvy - self.c_data.y * uvx)

    def inverted(self):
        cdef ovrQuatf to_return = ovrQuatf(-self.c_data.x,
                                           -self.c_data.y,
                                           -self.c_data.z,
                                            self.c_data.w)
        return to_return

    def inverse(self):
        cdef ovrQuatf to_return = ovrQuatf(-self.c_data.x,
                                           -self.c_data.y,
                                           -self.c_data.z,
                                            self.c_data.w)
        return to_return

    def invert(self):
        self.c_data.x = -self.c_data.x
        self.c_data.y = -self.c_data.y
        self.c_data.z = -self.c_data.z

        return self

    def __invert__(self):
        return self.invert()


cdef class ovrPosef:
    cdef ovr_capi.ovrPosef* c_data
    cdef ovr_capi.ovrPosef  c_ovrPosef

    cdef ovrVector3f field_position
    cdef ovrQuatf field_orientation

    def __cinit__(self, *args, **kwargs):
        self.c_data = &self.c_ovrPosef

        # create property objects
        self.field_position.c_data = &self.c_data.pos
        self.field_orientation.c_data = &self.c_data.orientation

        cdef int nargin = <int>len(args)  # get number of arguments
        if nargin == 0:
            pass
        elif nargin == 2:
            if isinstance(args[0], ovrVector3f) and isinstance(args[1], ovrQuatf):
                self.field_position.c_data[0] = args[0].c_data[0]
                self.field_orientation.c_data[0] = args[1].c_data[0]

    @property
    def rotation(self):
        return self.field_orientation

    @rotation.setter
    def rotation(self, ovrQuatf value):
        self.field_orientation.c_data[0] = value.c_data[0]

    @property
    def translation(self):
        return self.field_position

    @translation.setter
    def translation(self, ovrVector3f value):
        self.field_position.c_data[0] = value.c_data[0]

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



cdef class TextureSwapChain(object):
    cdef ovr_capi.ovrTextureSwapChain texture_swap_chain

    def __init__(self, object size, **kwargs):
        pass

    def __cinit__(self, object size, **kwargs):

        # get hmd descriptor
        global _hmd_desc_, _ptr_session_

        # initialize swap chain texture
        cdef ovr_capi.ovrTextureSwapChainDesc swap_desc
        swap_desc.Type = ovr_capi.ovrTexture_2D
        swap_desc.Format = ovr_capi.OVR_FORMAT_R8G8B8A8_UNORM_SRGB
        swap_desc.Width = <int>size[0]
        swap_desc.Height = <int>size[1]
        swap_desc.StaticImage = ovr_capi.ovrFalse
        swap_desc.ArraySize = swap_desc.MipLevels = swap_desc.SampleCount = 1
        swap_desc.MiscFlags = ovr_capi.ovrTextureMisc_None
        swap_desc.BindFlags = ovr_capi.ovrTextureBind_None

        # create the texture swap chain
        cdef ovr_capi.ovrResult result = 0
        result = ovr_capi_gl.ovr_CreateTextureSwapChainGL(
            _ptr_session_, &swap_desc, &self.texture_swap_chain)

        if debug_mode:
            check_result(result)

    def get_size(self):
        global _ptr_session_

        cdef ovr_capi.ovrTextureSwapChainDesc swap_desc
        ovr_capi.ovr_GetTextureSwapChainDesc(
            _ptr_session_,
            self.texture_swap_chain,
            &swap_desc)

        return swap_desc.Width, swap_desc.Height

    def get_buffer(self):
        cdef int current_idx = 0
        cdef unsigned int tex_id = 0

        global _ptr_session_
        ovr_capi.ovr_GetTextureSwapChainCurrentIndex(
            _ptr_session_,
            self.texture_swap_chain,
            &current_idx)

        ovr_capi_gl.ovr_GetTextureSwapChainBufferGL(
            _ptr_session_,
            self.texture_swap_chain,
            current_idx,
            &tex_id)

        return tex_id

    def commit(self):
        ovr_capi.ovr_CommitTextureSwapChain(
            _ptr_session_,
            self.texture_swap_chain)


cdef class RenderLayer(object):
    cdef ovr_capi.ovrLayerEyeFov _eye_layer
    cdef TextureSwapChain swap_chain

    def __init__(self, *args, **kwargs):
        pass

    def __cinit__(
            self,
            TextureSwapChain swap_chain,
            bint high_quality = True,
            bint head_locked = False,
            float texels_per_pixel = 1.0,
            **kwargs):

        # get hmd descriptor
        global _hmd_desc_, _ptr_session_

        # configure render layer
        # self._eye_layer.Header.Type = ovr_capi.ovrLayerType_EyeFov
        # self._eye_layer.Header.Flags = ovr_capi.ovrLayerFlag_TextureOriginAtBottomLeft
        # self._eye_layer.Fov[0] = _eye_render_desc_[0].Fov
        # self._eye_layer.Fov[1] = _eye_render_desc_[1].Fov
        # self._eye_layer.Viewport[0].Pos.x = 0
        # self._eye_layer.Viewport[0].Pos.y = 0
        # self._eye_layer.Viewport[0].Size.w = buffer_size[0] / 2
        # self._eye_layer.Viewport[0].Size.h = buffer_size[1]
        # self._eye_layer.Viewport[1].Pos.x = buffer_size[0] / 2
        # self._eye_layer.Viewport[1].Pos.y = 0
        # self._eye_layer.Viewport[1].Size.w = buffer_size[0] / 2
        # self._eye_layer.Viewport[1].Size.h = buffer_size[1]
        #
        # self._eye_layer.ColorTexture[0] = self.texture_swap_chain
        # self._eye_layer.ColorTexture[1] = NULL

# ----------------
# Module Functions
# ----------------
#
cpdef dict start_session():
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

    # return HMD information from descriptor
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

cpdef void end_session():
    ovr_capi.ovr_DestroyTextureSwapChain(_ptr_session_, _swap_chain_)
    ovr_capi.ovr_Destroy(_ptr_session_)
    ovr_capi.ovr_Shutdown()

cpdef tuple get_buffer_size(float texels_per_pixel=1.0):
    cdef ovr_capi.ovrSizei rec_tex0_size, rec_tex1_size, buffer_size

    rec_tex0_size = ovr_capi.ovr_GetFovTextureSize(
        _ptr_session_,
        ovr_capi.ovrEye_Left,
        _hmd_desc_.DefaultEyeFov[0],
        texels_per_pixel)
    rec_tex1_size = ovr_capi.ovr_GetFovTextureSize(
        _ptr_session_,
        ovr_capi.ovrEye_Right,
        _hmd_desc_.DefaultEyeFov[1],
        texels_per_pixel)

    buffer_size.w  = rec_tex0_size.w + rec_tex1_size.w
    buffer_size.h = max(rec_tex0_size.h, rec_tex1_size.h)

    return buffer_size.w, buffer_size.h

cpdef void setup_render_layer(TextureSwapChain swap_chain):
    buffer_size = swap_chain.get_size()

    # setup the render layer
    _eye_layer_.Header.Type = ovr_capi.ovrLayerType_EyeFov
    _eye_layer_.Header.Flags = ovr_capi.ovrLayerFlag_TextureOriginAtBottomLeft
    _eye_layer_.ColorTexture[0] = swap_chain.texture_swap_chain
    _eye_layer_.ColorTexture[1] = NULL
    _eye_layer_.Fov[0] = _eye_render_desc_[0].Fov
    _eye_layer_.Fov[1] = _eye_render_desc_[1].Fov
    _eye_layer_.Viewport[0].Pos.x = 0
    _eye_layer_.Viewport[0].Pos.y = 0
    _eye_layer_.Viewport[0].Size.w = buffer_size[0] / 2
    _eye_layer_.Viewport[0].Size.h = buffer_size[1]
    _eye_layer_.Viewport[1].Pos.x = buffer_size[0] / 2
    _eye_layer_.Viewport[1].Pos.y = 0
    _eye_layer_.Viewport[1].Size.w = buffer_size[0] / 2
    _eye_layer_.Viewport[1].Size.h = buffer_size[1]

cpdef void setup_mirror_texture(int width=800, int height=600):
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

cpdef void calc_eye_poses(double abs_time, bint time_stamp=True):

    cdef ovr_capi.ovrBool use_marker = 0
    if time_stamp:
        use_marker = ovr_capi.ovrTrue
    else:
        use_marker = ovr_capi.ovrFalse

    cpdef ovr_capi.ovrTrackingState hmd_state = ovr_capi.ovr_GetTrackingState(
        _ptr_session_, abs_time, use_marker)

    ovr_capi_util.ovr_CalcEyePoses2(
        hmd_state.HeadPose.ThePose,
        _hmd_to_eye_view_pose_,
        _eye_layer_.RenderPose)

cpdef int begin_frame(unsigned int frame_index=0):
    result = ovr_capi.ovr_BeginFrame(_ptr_session_, frame_index)

    return <int>result

cpdef unsigned int get_texture_swap_buffer():
    cdef int current_idx = 0
    cdef unsigned int tex_id = 0
    ovr_capi.ovr_GetTextureSwapChainCurrentIndex(
        _ptr_session_, _swap_chain_, &current_idx)
    ovr_capi_gl.ovr_GetTextureSwapChainBufferGL(
        _ptr_session_, _swap_chain_, current_idx, &tex_id)

    return tex_id

cpdef unsigned int get_mirror_texture():
    cdef unsigned int out_tex_id
    cdef ovr_capi.ovrResult result = \
        ovr_capi_gl.ovr_GetMirrorTextureBufferGL(
            _ptr_session_,
            _mirror_texture_,
            &out_tex_id)

    return <unsigned int>out_tex_id

cpdef void end_frame(unsigned int frame_index=0):
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