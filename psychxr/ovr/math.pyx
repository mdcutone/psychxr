from math cimport *
cimport libc.math as cmath
import OpenGL.GL as GL

cdef class ovrVector2i:
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

    def asTuple(self):
        return self.c_data.x, self.c_data.y

    def asList(self):
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

    def isEqual(self, ovrVector2i b, int tolerance = 0):
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

    def entrywiseMultiply(self, ovrVector2i b):
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

    def lengthSq(self):
        return \
            <int>(self.c_data.x * self.c_data.x + self.c_data.y * self.c_data.y)

    def length(self):
        return <int>cmath.sqrt(self.length_sq())

    def distanceSq(self, ovrVector2i b):
        return (self - b).length_sq()

    def distance(self, ovrVector2i b):
        return (self - b).length()

    def isNormalized(self):
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

    def projectTo(self, ovrVector2i b):
        cdef int l2 = self.length_sq()
        assert l2 != <int>0

        return b * (self.dot(b) / l2)

    def isClockwise(self, ovrVector2i b):
        return (self.c_data.x * b.c_data.y - self.c_data.y * b.c_data.x) < 0


cdef class ovrSizei:
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

    def asTuple(self):
        return self.c_data.w, self.c_data.h

    def asList(self):
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

    def toVector(self):
        cdef ovrVector2i to_return = ovrVector2i(self.c_data.w, self.c_data.h)

        return to_return

    def asVector(self):
        return self.to_vector()


cdef class ovrRecti:

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

    def asTuple(self):
        return self.c_data.Pos.x, self.c_data.Pos.y, \
               self.c_data.Size.w, self.c_data.Size.h

    def asList(self):
        return [self.c_data.Pos.x, self.c_data.Pos.y,
                self.c_data.Size.w, self.c_data.Size.h]

    def __len__(self):
        return 4

    def getPos(self):
        cdef ovrVector2i to_return = ovrVector2i(self.c_data.Pos.x,
                                                 self.c_data.Pos.y)

        return to_return

    def getSize(self):
        cdef ovrSizei to_return = ovrSizei(self.c_data.Size.w,
                                           self.c_data.Size.h)

        return to_return

    def setPos(self, ovrVector2i pos):
        self.c_data.Pos.x = pos.c_data.x
        self.c_data.Pos.y = pos.c_data.y

    def setSize(self, ovrSizei size):
        self.c_data.Size.w = size.c_data.w
        self.c_data.Size.h = size.c_data.h

    def __eq__(self, ovrRecti b):
        return self.c_data.Pos.x == b.c_data.Pos.x and \
               self.c_data.Pos.y == b.c_data.Pos.y and \
               self.c_data.Size.w == b.c_data.Size.w and \
               self.c_data.Size.h == b.c_data.Size.h

    def __ne__(self, ovrRecti b):
        return not self.__eq__(b)


cdef class ovrVector3f:
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

    def isEqual(self, ovrVector3f b, float tolerance = 0.0):
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

    def entrywiseMultiply(self, ovrVector3f b):
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

    def lengthSq(self):
        return <float>(<ovrVector3f>self).c_data[0].LengthSq()

    def length(self):
        return <float>(<ovrVector3f>self).c_data[0].Length()

    def distanceSq(self, ovrVector3f b):
        return <float>(<ovrVector3f>self).c_data[0].DistanceSq(b.c_data[0])

    def distance(self, ovrVector3f b):
        return <float>(<ovrVector3f>self).c_data[0].Distance(b.c_data[0])

    def isNormalized(self):
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

    def projectTo(self, ovrVector3f b):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = \
            (<ovrVector3f>self).c_data[0].ProjectTo(
            b.c_data[0])

        return to_return

    def projectToPlane(self, ovrVector3f normal):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = \
            (<ovrVector3f>self).c_data[0].ProjectToPlane(
                normal.c_data[0])

        return to_return


cdef class ovrFovPort:
    def __cinit__(self,
                  float up_tan=0.0,
                  float down_tan=0.0,
                  float left_tan=0.0,
                  float right_tan=0.0):
        self.c_data = &self.c_ovrFovPort

    @property
    def UpTan(self):
        return <float>self.c_data[0].UpTan

    @UpTan.setter
    def UpTan(self, float value):
        self.c_data[0].UpTan = value

    @property
    def DownTan(self):
        return <float>self.c_data[0].DownTan

    @DownTan.setter
    def DownTan(self, float value):
        self.c_data[0].DownTan = value

    @property
    def LeftTan(self):
        return <float>self.c_data[0].LeftTan

    @LeftTan.setter
    def LeftTan(self, float value):
        self.c_data[0].LeftTan = value

    @property
    def RightTan(self):
        return <float>self.c_data[0].RightTan

    @RightTan.setter
    def RightTan(self, float value):
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

    def asTuple(self):
        return self.c_data.x, self.c_data.y, self.c_data.z, self.c_data.w

    def asList(self):
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
            return a.rotate(<ovrVector3f>b)
        elif isinstance(b, ovrQuatf):
            # quaternion multiplication
            return ovrQuatf(
                a.c_data.w * (<ovrQuatf>b).c_data.x +
                a.c_data.x * (<ovrQuatf>b).c_data.w +
                a.c_data.y * (<ovrQuatf>b).c_data.z -
                a.c_data.z * (<ovrQuatf>b).c_data.y,
                a.c_data.w * (<ovrQuatf>b).c_data.y -
                a.c_data.x * (<ovrQuatf>b).c_data.z +
                a.c_data.y * (<ovrQuatf>b).c_data.w +
                a.c_data.z * (<ovrQuatf>b).c_data.x,
                a.c_data.w * (<ovrQuatf>b).c_data.z +
                a.c_data.x * (<ovrQuatf>b).c_data.y -
                a.c_data.y * (<ovrQuatf>b).c_data.x +
                a.c_data.z * (<ovrQuatf>b).c_data.w,
                a.c_data.w * (<ovrQuatf>b).c_data.w -
                a.c_data.x * (<ovrQuatf>b).c_data.x -
                a.c_data.y * (<ovrQuatf>b).c_data.y -
                a.c_data.z * (<ovrQuatf>b).c_data.z)
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

    def isEqual(self, ovrQuatf b, float tolerance = 0.0):
        return  self.abs(self.dot(b)) >= <float>1 - tolerance

    def abs(self, float v):
        return <float>(<ovrQuatf>self).c_data[0].Abs(v)

    def imag(self):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = \
            (<ovrQuatf>self).c_data[0].Imag()

        return to_return

    def lengthSq(self):
        return <float>(<ovrQuatf>self).c_data[0].LengthSq()

    def length(self):
        return <float>(<ovrQuatf>self).c_data[0].Length()

    def distance(self, ovrQuatf q):
        return <float>(<ovrQuatf>self).c_data[0].Distance(q.c_data[0])

    def distanceSq(self, ovrQuatf q):
        return <float>(<ovrQuatf>self).c_data[0].DistanceSq(q.c_data[0])

    def dot(self, ovrQuatf q):
        return <float>(<ovrQuatf>self).c_data[0].Dot(q.c_data[0])

    def angle(self, ovrQuatf q):
        return <float>(<ovrQuatf>self).c_data[0].Angle(q.c_data[0])

    def isNormalized(self):
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

    def inverseRotate(self, ovrVector3f v):
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
    """ovrPosef

    Class to represent a pose using a vector and quaternion.

    """
    def __cinit__(self, *args, **kwargs):
        self.c_data = &self.c_Posef

        cdef int nargin = <int>len(args)  # get number of arguments
        if nargin == 0:
            self.c_data[0].Translation = ovr_math.Vector3f(0.0, 0.0, 0.0)
            self.c_data[0].Rotation = ovr_math.Quatf.Identity()
        elif nargin == 1:
            if isinstance(args[0], ovrVector3f):
                self.c_data[0].Translation = (<ovrVector3f>args[0]).c_data[0]
            elif isinstance(args[0], ovrQuatf):
                self.c_data[0].Rotation = (<ovrQuatf>args[0]).c_data[0]
        elif nargin == 2:
            if isinstance(args[0], ovrVector3f) and \
                    isinstance(args[1], ovrQuatf):
                self.c_data[0].Translation = (<ovrVector3f>args[0]).c_data[0]
                self.c_data[0].Rotation = (<ovrQuatf>args[1]).c_data[0]

    @property
    def rotation(self):
        cdef ovrQuatf to_return = ovrQuatf()
        (<ovrQuatf>to_return).c_data[0] = self.c_data[0].Rotation

        return to_return

    @rotation.setter
    def rotation(self, ovrQuatf value):
        (<ovrPosef>self).c_data[0].Rotation = value.c_data[0]

    @property
    def translation(self):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = self.c_data[0].Translation

        return to_return

    @translation.setter
    def translation(self, ovrVector3f value):
        (<ovrPosef>self).c_data[0].Translation = value.c_data[0]

    def rotate(self, ovrVector3f v):
        return self.rotation.rotate(v)

    def inverseRotate(self, ovrVector3f v):
        return self.rotation.inverseRotate(v)

    def translate(self, ovrVector3f v):
        return v + self.translation

    def transform(self, ovrVector3f v):
        return self.rotate(v) + self.translation

    def inverseTransform(self, ovrVector3f v):
        return self.inverseRotate(v - self.translation)

    def transformNormal(self, ovrVector3f v):
        return self.rotate(v)

    def inverseTransformNormal(self, ovrVector3f v):
        return self.inverseRotate(v)

    def apply(self, ovrVector3f v):
        return self.transform(v)

    def __mul__(ovrPosef a, ovrPosef b):
        cdef ovrPosef to_return = ovrPosef(
            a.apply(b.translation),
            <ovrQuatf>a.rotation * <ovrQuatf>b.rotation)

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

    def setIdentity(self):
        self.c_data[0] = ovr_math.Matrix4f()

    def setXBasis(self, ovrVector3f v):
        (<ovrMatrix4f>self).c_data[0].SetXBasis(v.c_data[0])

    def getXBasis(self):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = \
            (<ovr_math.Matrix4f>self.c_data[0]).GetXBasis()

        return to_return

    @property
    def xBasis(self):
        return self.getXBasis()

    @xBasis.setter
    def xBasis(self, object v):
        if isinstance(v, ovrVector3f):
            self.setXBasis(<ovrVector3f>v)
        elif isinstance(v, (list, tuple)):
            self.setXBasis(ovrVector3f(<float>v[0], <float>v[1], <float>v[2]))

    def setYBasis(self, ovrVector3f v):
        (<ovrMatrix4f>self).c_data[0].SetYBasis(v.c_data[0])

    def getYBasis(self):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = \
            (<ovr_math.Matrix4f>self.c_data[0]).GetYBasis()

        return to_return

    @property
    def yBasis(self):
        return self.getYBasis()

    @yBasis.setter
    def yBasis(self, object v):
        if isinstance(v, ovrVector3f):
            self.setYBasis(<ovrVector3f>v)
        elif isinstance(v, (list, tuple)):
            self.setYBasis(ovrVector3f(<float>v[0], <float>v[1], <float>v[2]))

    def setZBasis(self, ovrVector3f v):
        (<ovrMatrix4f>self).c_data[0].SetZBasis(v.c_data[0])

    def getZBasis(self):
        cdef ovrVector3f to_return = ovrVector3f()
        (<ovrVector3f>to_return).c_data[0] = \
            (<ovr_math.Matrix4f>self.c_data[0]).GetZBasis()

        return to_return

    @property
    def zBasis(self):
        return self.getZBasis()

    @zBasis.setter
    def zBasis(self, object v):
        if isinstance(v, ovrVector3f):
            self.setZBasis(<ovrVector3f>v)
        elif isinstance(v, (list, tuple)):
            self.setZBasis(ovrVector3f(<float>v[0], <float>v[1], <float>v[2]))

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

    def invertedHomogeneousTransform(self):
        cdef ovrMatrix4f to_return = ovrMatrix4f()
        (<ovrMatrix4f>to_return).c_data[0] = \
            (<ovr_math.Matrix4f>self.c_data[0]).InvertedHomogeneousTransform()

        return to_return

    def invertHomogeneousTransform(self):
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

    def setTranslation(self, ovrVector3f v):
        (<ovr_math.Matrix4f>self.c_data[0]).SetTranslation(v.c_data[0])

        return self

    def getTranslation(self):
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
    def rotationX(float angle=0.0):
        cdef ovrMatrix4f to_return = ovrMatrix4f()
        (<ovrMatrix4f>to_return).c_data[0] = ovr_math.Matrix4f.RotationX(angle)

        return to_return

    @staticmethod
    def rotationY(float angle=0.0):
        cdef ovrMatrix4f to_return = ovrMatrix4f()
        (<ovrMatrix4f>to_return).c_data[0] = ovr_math.Matrix4f.RotationY(angle)

        return to_return

    @staticmethod
    def rotationZ(float angle=0.0):
        cdef ovrMatrix4f to_return = ovrMatrix4f()
        (<ovrMatrix4f>to_return).c_data[0] = ovr_math.Matrix4f.RotationZ(angle)

        return to_return

    @staticmethod
    def lookAt(ovrVector3f eye, ovrVector3f at, ovrVector3f up):
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
    def ortho2d(float w, float h):
        cdef ovrMatrix4f to_return = ovrMatrix4f()
        (<ovrMatrix4f>to_return).c_data[0] = ovr_math.Matrix4f.Ortho2D(w, h)

        return to_return