cimport ovr_capi, ovr_capi_gl, ovr_errorcode, ovr_capi_util
from libc.stdint cimport uintptr_t, uint32_t, int32_t
from libcpp cimport nullptr
from libc.stdlib cimport malloc, free
cimport libc.math as cmath
cimport rift

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

cdef float acos(float x):
    cdef float to_return = 0.0
    if x > 1.0:
        to_return = 0.0
    else:
        if x < -1.0:
            to_return = MATH_FLOAT_PI
        else:
            to_return = rift.acosf(x)

    return to_return

cdef float asin(float x):
    cdef float to_return = 0.0
    if x > 1.0:
        to_return = 0.0
    else:
        if x < -1.0:
            to_return = MATH_FLOAT_PI
        else:
            to_return = rift.asinf(x)

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

        cdef int nargin = len(args)  # get number of arguments
        if nargin == 0:
            self.c_data.Pos.x = self.c_data.Pos.y = 0
            self.c_data.Size.w = self.c_data.Size.h = 0
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


cdef class ovrVector2f:
    cdef ovr_capi.ovrVector2f* c_data
    cdef ovr_capi.ovrVector2f  c_ovrVector2f

    def __cinit__(self, float x=0.0, float y=0.0):
        self.c_data = &self.c_ovrVector2f

        self.c_data.x = x
        self.c_data.y = y

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

    def as_tuple(self):
        return self.c_data.x, self.c_data.y

    def as_list(self):
        return list(self.c_data.x, self.c_data.y)

    def __len__(self):
        return 2

    def __eq__(self, ovrVector2f b):
        return self.c_data.x == b.c_data.x and self.c_data.y == b.c_data.y

    def __ne__(self, ovrVector2f b):
        return self.c_data.x != b.c_data.x or self.c_data.y != b.c_data.y

    def __add__(ovrVector2f a, ovrVector2f b):
        cdef ovrVector2f to_return = ovrVector2f(
            a.c_data.x + b.c_data.x,
            a.c_data.y + b.c_data.y)

        return to_return

    def __iadd__(self, ovrVector2f b):
        self.c_data.x += b.c_data.x
        self.c_data.y += b.c_data.y

    def __sub__(ovrVector2f a, ovrVector2f b):
        cdef ovrVector2f to_return = ovrVector2f(
            a.c_data.x - b.c_data.x,
            a.c_data.y - b.c_data.y)

        return to_return

    def __isub__(self, ovrVector2i b):
        self.c_data.x -= b.c_data.x
        self.c_data.y -= b.c_data.y

    def __neg__(self):
        cdef ovrVector2f to_return = ovrVector2f(-self.c_data.x, -self.c_data.y)

        return to_return

    def __mul__(ovrVector2f a, object b):
        cdef ovrVector2f to_return
        if isinstance(b, ovrVector2f):
            to_return = ovrVector2f(a.c_data.x * b.c_data.x,
                                    a.c_data.y * b.c_data.y)
        elif isinstance(b, (int, float)):
            to_return = ovrVector2f(a.c_data.x * <float>b,
                                    a.c_data.y * <float>b)

        return to_return

    def __imul__(self, object b):
        cdef ovrVector2f to_return
        if isinstance(b, ovrVector2f):
            self.c_data.x *= b.c_data.x
            self.c_data.y *= b.c_data.y
        elif isinstance(b, (int, float)):
            self.c_data.x *= <float>b
            self.c_data.y *= <float>b

    def __truediv__(ovrVector2f a, object b):
        cdef float rcp = <float>1 / <float>b
        cdef ovrVector2f to_return = ovrVector2f(
            a.c_data.x * rcp,
            a.c_data.y * rcp)

        return to_return

    def __itruediv__(self, object b):
        cdef float rcp = <float>1 / <float>b
        self.c_data.x *= rcp
        self.c_data.y *= rcp

    @staticmethod
    def min(ovrVector2f a, ovrVector2f b):
        cdef ovrVector2f to_return = ovrVector2f(
            a.c_data.x if a.c_data.x < b.c_data.x else b.c_data.x,
            a.c_data.y if a.c_data.y < b.c_data.y else b.c_data.y)

        return to_return

    @staticmethod
    def max(ovrVector2f a, ovrVector2f b):
        cdef ovrVector2f to_return = ovrVector2f(
            a.c_data.x if a.c_data.x > b.c_data.x else b.c_data.x,
            a.c_data.y if a.c_data.y > b.c_data.y else b.c_data.y)

        return to_return

    def clamped(self, float max_mag):
        cdef float mag_squared = self.length_sq()
        if mag_squared > max_mag * max_mag:
            return self * (max_mag / cmath.sqrt(mag_squared))

        return self

    def is_equal(self, ovrVector2f b, float tolerance = 0.0):
        return cmath.fabs(b.c_data.x - self.c_data.x) <= tolerance and \
            cmath.fabs(b.c_data.y - self.c_data.y) <= tolerance

    def compare(self, ovrVector2f b, float tolerance = 0):
        return self.is_equal(b, tolerance)

    def __getitem__(self, int idx):
        assert 0 <= idx < 2
        cdef float* ptr_val = &self.c_data.x + idx

        return <float>ptr_val[0]

    def __setitem__(self, int idx, float val):
        assert 0 <= idx < 2
        cdef float* ptr_val = &self.c_data.x + idx
        ptr_val[0] = val

    def entrywise_multiply(self, ovrVector2f b):
        cdef ovrVector2f to_return = ovrVector2f(
            self.c_data.x * b.c_data.x,
            self.c_data.y * b.c_data.y)

        return to_return

    def dot(self, ovrVector2f b):
        cdef float dot_prod = \
            self.c_data.x * b.c_data.x + self.c_data.y * b.c_data.y

        return <float>dot_prod

    def angle(self, ovrVector2f b):
        cdef float div = self.length_sq() * b.length_sq()
        assert div != <float>0
        cdef float to_return = self.dot(b) / cmath.sqrt(div)

        return to_return

    def length_sq(self):
        return \
            <float>(self.c_data.x * self.c_data.x + self.c_data.y * self.c_data.y)

    def length(self):
        return <float>cmath.sqrt(self.length_sq())

    def distance_sq(self, ovrVector2f b):
        return (self - b).length_sq()

    def distance(self, ovrVector2f b):
        return (self - b).length()

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

    def lerp(self, ovrVector2f b, float f):
        return self * (<float>1 - f) + b * f

    def project_to(self, ovrVector2f b):
        cdef float l2 = self.length_sq()
        assert l2 != <float>0

        return b * (self.dot(b) / l2)

    def is_clockwise(self, ovrVector2f b):
        return (self.c_data.x * b.c_data.y - self.c_data.y * b.c_data.x) < 0


cdef class ovrVector3f(object):
    cdef ovr_capi.ovrVector3f* c_data
    cdef ovr_capi.ovrVector3f  c_ovrVector3f

    def __init__(self, *args, **kwargs):
        pass

    def __cinit__(self, *args, **kwargs):
        self.c_data = &self.c_ovrVector3f

        cdef int nargin = len(args)  # get number of arguments
        if nargin == 0:
            self.c_data.x = 0.0
            self.c_data.y = 0.0
            self.c_data.z = 0.0
        elif nargin == 3:
            self.c_data.x = <float>args[0]
            self.c_data.y = <float>args[1]
            self.c_data.z = <float>args[2]

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

    def __eq__(self, ovrVector3f b):
        return self.c_data.x == b.c_data.x and \
               self.c_data.y == b.c_data.y and \
               self.c_data.z == b.c_data.z

    def __ne__(self, ovrVector3f b):
        return self.c_data.x != b.c_data.x or \
               self.c_data.y != b.c_data.y or \
               self.c_data.z != b.c_data.z


    def __add__(ovrVector3f a, ovrVector3f b):
        cdef ovrVector3f to_return = ovrVector3f(
            a.c_data.x + b.c_data.x,
            a.c_data.y + b.c_data.y,
            a.c_data.y + b.c_data.z)

        return to_return

    def __iadd__(self, ovrVector3f b):
        self.c_data.x += b.c_data.x
        self.c_data.y += b.c_data.y
        self.c_data.z += b.c_data.z

        return self

    def __sub__(ovrVector3f a, ovrVector3f b):
        cdef ovrVector3f to_return = ovrVector3f(
            a.c_data.x - b.c_data.x,
            a.c_data.y - b.c_data.y,
            a.c_data.y - b.c_data.z)

        return to_return

    def __isub__(self, ovrVector3f b):
        self.c_data.x -= b.c_data.x
        self.c_data.y -= b.c_data.y
        self.c_data.z -= b.c_data.z

        return self

    def __neg__(self):
        cdef ovrVector3f to_return = ovrVector3f(
            -self.c_data.x, -self.c_data.y, -self.c_data.z)

        return to_return

    def __mul__(ovrVector3f a, object b):
        cdef ovrVector3f to_return
        if isinstance(b, ovrVector3f):
            to_return = ovrVector3f(a.c_data.x * b.c_data.x,
                                    a.c_data.y * b.c_data.y,
                                    a.c_data.z * b.c_data.z)
        elif isinstance(b, (int, float)):
            to_return = ovrVector3f(a.c_data.x * <float>b,
                                    a.c_data.y * <float>b,
                                    a.c_data.z * <float>b)

        return to_return

    def __imul__(self, object b):
        cdef ovrVector3f to_return
        if isinstance(b, ovrVector3f):
            self.c_data.x *= b.c_data.x
            self.c_data.y *= b.c_data.y
            self.c_data.z *= b.c_data.z
        elif isinstance(b, (int, float)):
            self.c_data.x *= <float>b
            self.c_data.y *= <float>b
            self.c_data.z *= <float>b

        return self

    def __truediv__(ovrVector3f a, object b):
        cdef float rcp = <float>1 / <float>b
        cdef ovrVector3f to_return = ovrVector3f(
            a.c_data.x * rcp,
            a.c_data.y * rcp,
            a.c_data.z * rcp)

        return to_return

    def __itruediv__(self, object b):
        cdef float rcp = <float>1 / <float>b
        self.c_data.x *= rcp
        self.c_data.y *= rcp
        self.c_data.z *= rcp

        return self

    @staticmethod
    def min(ovrVector3f a, ovrVector3f b):
        cdef ovrVector3f to_return = ovrVector3f(
            a.c_data.x if a.c_data.x < b.c_data.x else b.c_data.x,
            a.c_data.y if a.c_data.y < b.c_data.y else b.c_data.y,
            a.c_data.z if a.c_data.z < b.c_data.z else b.c_data.z)

        return to_return

    @staticmethod
    def max(ovrVector3f a, ovrVector3f b):
        cdef ovrVector3f to_return = ovrVector3f(
            a.c_data.x if a.c_data.x > b.c_data.x else b.c_data.x,
            a.c_data.y if a.c_data.y > b.c_data.y else b.c_data.y,
            a.c_data.z if a.c_data.z > b.c_data.z else b.c_data.z)

        return to_return

    @staticmethod
    def max(ovrVector3f a, ovrVector3f b):
        cdef ovrVector3f to_return = ovrVector3f(
            a.c_data.x if a.c_data.x > b.c_data.x else b.c_data.x,
            a.c_data.y if a.c_data.y > b.c_data.y else b.c_data.y)

        return to_return

    def clamped(self, float max_mag):
        cdef float mag_squared = self.length_sq()
        if mag_squared > max_mag * max_mag:
            return self * (max_mag / cmath.sqrt(mag_squared))

        return self

    def is_equal(self, ovrVector3f b, float tolerance = 0.0):
        return cmath.fabs(b.c_data.x - self.c_data.x) <= tolerance and \
            cmath.fabs(b.c_data.y - self.c_data.y) <= tolerance

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
        cdef ovrVector3f to_return = ovrVector3f(
            self.c_data.x * b.c_data.x,
            self.c_data.y * b.c_data.y,
            self.c_data.z * b.c_data.z)

        return to_return

    def dot(self, ovrVector3f b):
        cdef float dot_prod = \
            self.c_data.x * b.c_data.x + \
            self.c_data.y * b.c_data.y + \
            self.c_data.z * b.c_data.z

        return <float>dot_prod

    def cross(self, ovrVector3f b):
        cdef ovrVector3f to_return = ovrVector3f(
            self.c_data.y * b.c_data.z - self.c_data.z * b.c_data.y,
            self.c_data.z * b.c_data.x - self.c_data.x * b.c_data.z,
            self.c_data.x * b.c_data.y - self.c_data.y * b.c_data.x)

        return to_return

    def angle(self, ovrVector2f b):
        cdef float div = self.length_sq() * b.length_sq()
        assert div != <float>0
        cdef float to_return = acos(self.dot(b) / cmath.sqrt(div))

        return to_return

    def length_sq(self):
        return self.c_data.x * self.c_data.x + \
               self.c_data.y * self.c_data.y + \
               self.c_data.z * self.c_data.z

    def length(self):
        return <float>cmath.sqrt(self.length_sq())

    def distance_sq(self, ovrVector3f b):
        return (self - b).length_sq()

    def distance(self, ovrVector3f b):
        return (self - b).length()

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

    def lerp(self, ovrVector2f b, float f):
        return self * (<float>1 - f) + b * f

    def project_to(self, ovrVector2f b):
        cdef float l2 = self.length_sq()
        assert l2 != <float>0

        return b * (self.dot(b) / l2)

    def is_clockwise(self, ovrVector2f b):
        return (self.c_data.x * b.c_data.y - self.c_data.y * b.c_data.x) < 0


cdef class ovrPoint3f(ovrVector3f):
    def __cinit__(self, *args):
        super(ovrVector3f, self).__init__(self)


cdef class ovrVector4f:
    cdef ovr_capi.ovrVector4f* c_data
    cdef ovr_capi.ovrVector4f  c_ovrVector4f

    def __init__(self, *args, **kwargs):
        pass

    def __cinit__(self, *args, **kwargs):
        self.c_data = &self.c_ovrVector4f

        cdef int nargin = len(args)  # get number of arguments
        if nargin == 0:
            self.c_data.x = 0.0
            self.c_data.y = 0.0
            self.c_data.z = 0.0
            self.c_data.w = 0.0
        elif nargin == 4:
            self.c_data.x = <float>args[0]
            self.c_data.y = <float>args[1]
            self.c_data.z = <float>args[2]
            self.c_data.w = <float>args[3]

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

    def __len__(self):
        return 4


cdef class ovrQuatf:
    cdef ovr_capi.ovrQuatf* c_data
    cdef ovr_capi.ovrQuatf  c_ovrQuatf

    def __cinit__(self, float x=0.0, float y=0.0, float z=0.0, float w=0.0):
        self.c_data = &self.c_ovrQuatf

        self.c_data.x = x
        self.c_data.y = y
        self.c_data.z = z
        self.c_data.w = w

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


cdef class ovrMatrix4f:
    cdef ovr_capi.ovrMatrix4f* c_data
    cdef ovr_capi.ovrMatrix4f  c_ovrMatrix4f

    def __cinit__(self):
        self.c_data = &self.c_ovrMatrix4f

    @property
    def M(self):
        return self.c_data.M

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