cimport _vrlinalg

cdef class vec2i:
    cdef _vrlinalg.vec2i* c_data
    cdef _vrlinalg.vec2i  c_vec2i

    def __cinit__(self, int x=0, int y=0):
        self.c_data = &self.c_vec2i

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
        return [self.c_data.x, self.c_data.y]


cpdef void vec2i_add(vec2i r, vec2i a, vec2i b):
    _vrlinalg.vec2i_add(r.c_data, a.c_data[0], b.c_data[0])

cpdef void vec2i_sub(vec2i r, vec2i a, vec2i b):
    _vrlinalg.vec2i_sub(r.c_data, a.c_data[0], b.c_data[0])

cpdef void vec2i_scale(vec2i r, vec2i v, int s):
    _vrlinalg.vec2i_scale(r.c_data, v.c_data[0], s)


cdef class vec2f:
    cdef _vrlinalg.vec2f* c_data
    cdef _vrlinalg.vec2f  c_vec2f

    def __cinit__(self, float x=0, float y=0):
        self.c_data = &self.c_vec2f

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
        return [self.c_data.x, self.c_data.y]


cpdef void vec2f_add(vec2f r, vec2f a, vec2f b):
    _vrlinalg.vec2f_add(r.c_data, a.c_data[0], b.c_data[0])

cpdef void vec2f_sub(vec2f r, vec2f a, vec2f b):
    _vrlinalg.vec2f_sub(r.c_data, a.c_data[0], b.c_data[0])

cpdef void vec2f_scale(vec2f r, vec2f v, float s):
    _vrlinalg.vec2f_scale(r.c_data, v.c_data[0], s)

cpdef float vec2f_dot(vec2f a, vec2f b):
    return _vrlinalg.vec2f_dot(a.c_data[0], b.c_data[0])