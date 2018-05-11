cimport ovr_capi


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


cdef class ovrRecti:
    cdef ovr_capi.ovrRecti* c_data
    cdef ovr_capi.ovrRecti  c_ovrRecti

    # nested field objects
    cdef ovrVector2i obj_pos
    cdef ovrSizei obj_size

    def __cinit__(self, int x=0, int y=0, int w=0, int h=0):
        self.c_data = &self.c_ovrRecti
        self.obj_pos = ovrVector2i()
        self.obj_size = ovrSizei()

        (<ovrVector2i>self.obj_pos).c_data = &self.c_data.Pos
        (<ovrSizei>self.obj_size).c_data = &self.c_data.Size

    @property
    def Pos(self):
        return <ovrVector2i>self.obj_pos

    @property
    def Size(self):
        return <ovrSizei>self.obj_size


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


cdef class ovrVector3f:
    cdef ovr_capi.ovrVector3f* c_data
    cdef ovr_capi.ovrVector3f  c_ovrVector3f

    def __cinit__(self, float x=0.0, float y=0.0, float z=0.0):
        self.c_data = &self.c_ovrVector3f

        self.c_data.x = x
        self.c_data.y = y
        self.c_data.z = z

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