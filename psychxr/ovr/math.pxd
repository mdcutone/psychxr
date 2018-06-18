cimport ovr_math
cimport ovr_capi

cdef class ovrVector2i:
    cdef ovr_capi.ovrVector2i* c_data
    cdef ovr_capi.ovrVector2i  c_ovrVector2i

cdef class ovrSizei:
    cdef ovr_capi.ovrSizei* c_data
    cdef ovr_capi.ovrSizei  c_ovrSizei

cdef class ovrRecti:
    cdef ovr_capi.ovrRecti* c_data
    cdef ovr_capi.ovrRecti  c_ovrRecti

cdef class ovrVector3f:
    cdef ovr_math.Vector3f* c_data
    cdef ovr_math.Vector3f  c_Vector3f

cdef class ovrFovPort:
    cdef ovr_capi.ovrFovPort* c_data
    cdef ovr_capi.ovrFovPort  c_ovrFovPort

cdef class ovrQuatf:
    cdef ovr_math.Quatf* c_data
    cdef ovr_math.Quatf  c_Quatf

cdef class ovrPosef:
    cdef ovr_math.Posef* c_data
    cdef ovr_math.Posef  c_Posef

cdef class ovrMatrix4f:
    cdef ovr_math.Matrix4f* c_data
    cdef ovr_math.Matrix4f  c_Matrix4f