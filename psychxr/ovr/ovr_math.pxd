from libcpp cimport bool

cdef extern from "OVR_Math.h" namespace "OVR":

    ctypedef enum Axis:
        Axis_X = 0,
        Axis_Y = 1,
        Axis_Z = 2

    ctypedef enum RotateDirection:
        Rotate_CCW = 1,
        Rotate_CW = -1

    ctypedef enum HandedSystem:
        Handed_R = 1,
        Handed_L = -1

    ctypedef enum AxisDirection:
        Axis_Up = 2,
        Axis_Down = -2,
        Axis_Right = 1,
        Axis_Left = -1,
        Axis_In = 3,
        Axis_Out = -3

    cdef cppclass Mathf "OVR::Mathf":
        ctypedef float OtherFloatType

    # this is unused for now
    cdef cppclass Vector2i "OVR::Vector2i":
        CppVector2i() except +
        CppVector2i(int, int) except +
        int x, y
        @staticmethod
        Vector2i Zero() except +
        bool operator==(Vector2i& b)
        bool operator!=(Vector2i& b)
        Vector2i operator+(Vector2i& b)
        Vector2i operator-(Vector2i& b)
        Vector2i operator-()
        Vector2i operator*(int s)
        Vector2i operator/(int s)
        @staticmethod
        Vector2i Min(const Vector2i& a, const Vector2i& b) except +
        @staticmethod
        Vector2i Max(const Vector2i& a, const Vector2i& b) except +
        Vector2i Clamped(int maxMag) except +
        bool IsEqual(const Vector2i& b, int tolerance)
        bool Compare(const Vector2i& b, int tolerance)
        int& operator[](int idx)
        Vector2i EntrywiseMultiply(const Vector2i& b)
        Vector2i operator*(const Vector2i& b)
        Vector2i operator/(const Vector2i& b)
        int Dot(const Vector2i& b)
        int Angle(const Vector2i& b)
        int LengthSq()
        int Length()
        int DistanceSq(const Vector2i b)
        int Distance(const Vector2i b)
        bool IsNormalized()
        void Normalize()
        Vector2i Normalized()
        Vector2i Lerp(const Vector2i& b, int f)
        Vector2i ProjectTo(const Vector2i& b)
        bool IsClockwise(const Vector2i& b)

    cdef cppclass Vector2f "OVR::Vector2f":
        Vector2f() except +
        Vector2f(float, float) except +
        float x, y
        @staticmethod
        Vector2f Zero() except +
        bool operator==(Vector2f& b)
        bool operator!=(Vector2f& b)
        Vector2f operator+(Vector2f& b)
        Vector2f operator-(Vector2f& b)
        Vector2f operator-()
        Vector2f operator*(float s)
        Vector2f operator/(float s)
        @staticmethod
        Vector2f Min(const Vector2f& a, const Vector2f& b) except +
        @staticmethod
        Vector2f Max(const Vector2f& a, const Vector2f& b) except +
        Vector2f Clamped(float maxMag) except +
        bool IsEqual(const Vector2f& b, float tolerance)
        bool Compare(const Vector2f& b, float tolerance)
        int& operator[](int idx)
        Vector2f EntrywiseMultiply(const Vector2f& b)
        Vector2f operator*(const Vector2f& b)
        Vector2f operator/(const Vector2f& b)
        float Dot(const Vector2f& b)
        float Angle(const Vector2f& b)
        float LengthSq()
        float Length()
        float DistanceSq(const Vector2f b)
        float Distance(const Vector2f b)
        bool IsNormalized()
        void Normalize()
        Vector2f Normalized()
        Vector2f Lerp(const Vector2f& b, float f)
        Vector2f ProjectTo(const Vector2f& b)
        bool IsClockwise(const Vector2f& b)

    cdef cppclass Vector3f "OVR::Vector3f":
        Vector3f() except +
        Vector3f(float, float, float) except +
        float x, y, z
        @staticmethod
        Vector3f Zero() except +
        bool operator==(Vector3f& b)
        bool operator!=(Vector3f& b)
        Vector3f operator+(Vector3f& b)
        Vector3f operator-(Vector3f& b)
        Vector3f operator-()
        Vector3f operator*(float s)
        #Vector3f operator*=(float s)
        Vector3f operator/(float s)
        #Vector3f operator/=(float s)
        @staticmethod
        Vector3f Min(const Vector3f& a, const Vector3f& b) except +
        @staticmethod
        Vector3f Max(const Vector3f& a, const Vector3f& b) except +
        Vector3f Clamped(float maxMag) except +
        bool IsEqual(const Vector3f& b, float tolerance)
        bool Compare(const Vector3f& b, float tolerance)
        int& operator[](int idx)
        Vector3f EntrywiseMultiply(const Vector3f& b)
        Vector3f operator*(const Vector3f& b)
        Vector3f operator/(const Vector3f& b)
        float Dot(const Vector3f& b)
        Vector3f Cross(const Vector3f& b)
        float Angle(const Vector3f& b)
        float LengthSq()
        float Length()
        float DistanceSq(const Vector3f b)
        float Distance(const Vector3f b)
        bool IsNormalized()
        void Normalize()
        Vector3f Normalized()
        Vector3f Lerp(const Vector3f& b, float f)
        Vector3f ProjectTo(const Vector3f& b)
        Vector3f ProjectToPlane(const Vector3f& normal)
        bool IsNan()
        bool IsFinite()

    cdef cppclass Vector4f "OVR::Vector4f":
        Vector4f() except +
        CppVector4(Vector3f& v, float w_) except +
        Vector4f(float x_, float y_, float z_, float w_) except +
        float x, y, z, w
        @staticmethod
        Vector4f Zero() except +
        Vector4f operator=(Vector3f& b)
        bool operator==(Vector4f& b)
        bool operator!=(Vector4f& b)
        Vector4f operator+(Vector4f& b)
        Vector4f operator-(Vector4f& b)
        Vector4f operator-()
        Vector4f operator*(float s)
        Vector4f operator/(float s)
        @staticmethod
        Vector4f Min(const Vector4f& a, const Vector4f& b) except +
        @staticmethod
        Vector4f Max(const Vector4f& a, const Vector4f& b) except +
        Vector4f Clamped(float maxMag) except +
        bool IsEqual(const Vector4f& b, float tolerance)
        bool Compare(const Vector4f& b, float tolerance)
        int& operator[](int idx)
        Vector4f EntrywiseMultiply(const Vector4f& b)
        Vector4f operator*(const Vector4f& b)
        Vector4f operator/(const Vector4f& b)
        float Dot(const Vector4f& b)
        float LengthSq()
        float Length()
        bool IsNormalized()
        void Normalize()
        Vector4f Normalized()
        Vector4f Lerp(const Vector4f& b, float f)

    cdef cppclass Quatf "OVR::Quatf":
        Quatf() except +
        Quatf(Vector3f& axis, float angle) except +
        Quatf(float, float, float, float) except +
        Quatf(Matrix4f& m)
        Quatf(Quatf& q)
        float x, y, z, w
        Quatf operator-()
        @staticmethod
        Quatf Identity()
        void GetAxisAngle(Vector3f* axis, float* angle)
        Vector3f ToRotationVector()
        Vector3f FastToRotationVector()
        @staticmethod
        Quatf FromRotationVector(Vector3f v)
        @staticmethod
        Quatf FastFromRotationVector(Vector3f v, bool normalize)
        bool operator==(Quatf& b)
        bool operator!=(Quatf& b)
        Quatf operator+(Quatf& b)
        Quatf operator-(Quatf& b)
        Quatf operator*(float s)
        Quatf operator/(float s)
        bool IsEqual(const Quatf& b, float tolerance)
        bool IsEqualMatchHemisphere(Quatf b, float tolerance)
        @staticmethod
        float Abs(float v)
        Vector3f Imag()
        float Length()
        float LengthSq()
        float Distance(Quatf& q)
        float DistanceSq(Quatf& q)
        float Dot(Quatf& q)
        float Angle(Quatf& q)
        float Angle()
        bool IsNormalized()
        void Normalize()
        Quatf Normalized()
        void EnsureSameHemisphere(Quatf& o)
        Quatf Conj()
        Quatf operator*(Quatf& b)
        Quatf PowNormalized(float p)
        @staticmethod
        Quatf Align(Vector3f& alignTo, Vector3f& v)
        Quatf GetSwingTwist(Vector3f& axis, Quatf* twist)
        Quatf Lerp(Quatf& b, float s)
        Quatf Slerp(const Quatf b, float s)
        Quatf FastSlerp(Quatf& b, float s)
        Vector3f Rotate(Vector3f& v)
        Vector3f InverseRotate(Vector3f& v)
        Quatf Inverted()
        Quatf Inverse()
        void Invert()
        #Quatf TimeIntegrate(Vector3f& angularVelocity, float dt)
        #Quatf TimeIntegrate(
        #        Vector3f& angularVelocity,
        #        Vector3f& angularAcceleration,
        #        float dt)
        void GetYawPitchRoll(float* yaw, float* pitch, float* roll)
        void GetEulerAngles(float* a, float* b, float* c)
        void GetEulerAnglesABA(float* a, float* b, float* c)
        bool IsNan()
        bool IsFinite()

    cdef cppclass Matrix4f "OVR::Matrix4f":
        Matrix4f() except +
        Matrix4f(float, float, float, float, float, float, float, float,
                 float, float, float, float, float, float, float, float
                 ) except +
        Matrix4f(float, float, float, float, float, float, float, float,
                 float) except +
        Matrix4f(Quatf& q)
        Matrix4f(Posef& p)
        Matrix4f(Matrix4f& m)
        float M[4][4]
        @staticmethod
        Matrix4f Identity()
        void SetIdentity()
        void SetXBasis(Vector3f& v)
        Vector3f GetXBasis()
        void SetYBasis(Vector3f& v)
        Vector3f GetYBasis()
        void SetZBasis(Vector3f& v)
        Vector3f GetZBasis()
        bool operator==(Matrix4f& b)
        Matrix4f operator+(Matrix4f& b)
        Matrix4f operator-(Matrix4f& b)
        @staticmethod
        Matrix4f& Multiply(Matrix4f* d, Matrix4f& a, Matrix4f& b)
        Matrix4f operator*(Matrix4f& b)
        Matrix4f operator*(float s)
        Matrix4f operator/(float s)
        Vector3f Transform(Vector3f& v)
        Vector4f Transform(Vector4f& v)
        Matrix4f Transposed()
        void Transpose()
        float SubDet(const size_t* rows, const size_t* cols)
        float Cofactor(size_t I, size_t J)
        float Determinant()
        Matrix4f Adjugated()
        Matrix4f Inverted()
        void Invert()
        Matrix4f InvertedHomogeneousTransform()
        void InvertHomogeneousTransform()
        void ToEulerAngles()
        void ToEulerAnglesABA()
        #@staticmethod
        #Matrix4f AxisConversion(const WorldAxes& to, const WorldAxes& from)
        #@staticmethod
        #Matrix4f Translation(float x, float y, float z)
        @staticmethod
        Matrix4f Translation(Vector3f& v)
        void SetTranslation(Vector3f& v)
        Vector3f GetTranslation()
        @staticmethod
        Matrix4f Scaling(Vector3f& v)
        #@staticmethod
        #Matrix4f Scaling(float x, float y, float z)
        #@staticmethod
        #Matrix4f Scaling(float s)
        float Distance(Matrix4f& m2)
        @staticmethod
        Matrix4f RotationAxis(Axis A, float angle, RotateDirection d, HandedSystem s)
        @staticmethod
        Matrix4f RotationX(float angle)
        @staticmethod
        Matrix4f RotationY(float angle)
        @staticmethod
        Matrix4f RotationZ(float angle)
        @staticmethod
        Matrix4f LookAtRH(Vector3f& eye, Vector3f& at, Vector3f& up)
        @staticmethod
        Matrix4f LookAtLH(Vector3f& eye, Vector3f& at, Vector3f& up)
        @staticmethod
        Matrix4f PerspectiveRH(
                float yfov, float aspect, float znear, float zfar)
        @staticmethod
        Matrix4f PerspectiveLH(
                float yfov, float aspect, float znear, float zfar)
        @staticmethod
        Matrix4f Ortho2D(float w, float h)

    cdef cppclass Posef "OVR::Posef":
        Posef() except +
        Posef(Quatf orientation, Vector3f pos) except +
        @staticmethod
        Posef Identity()
        void SetIdentity()
        void SetInvalid()
        bool IsEqual(Posef b, float tolerance)
        bool IsEqualMatchHemisphere(Posef b, float tolerance)
        Quatf Rotation
        Vector3f Translation
        Vector3f Rotate(Vector3f v)
        Vector3f InverseRotate(Vector3f v)
        Vector3f Translate(Vector3f v)
        Vector3f Transform(Vector3f v)
        Vector3f InverseTransform(Vector3f v)
        Vector3f TransformNormal(Vector3f v)
        Vector3f InverseTransformNormal(Vector3f v)
        Vector3f Apply(Vector3f v)
        Posef operator*(Posef& b)
        Posef Inverted()
        Posef Lerp(Posef b, float s)
        Posef FastLerp(Posef b, float s)
        Posef TimeIntegrate(Vector3f linearVelocity, Vector3f angularVelocity, float dt)
        Posef TimeIntegrate(Vector3f linearVelocity, Vector3f linearAcceleration, Vector3f angularVelocity, Vector3f angularAcceleration, float dt)
        Posef Normalized()
        void Normalize()
        bool IsNan()
        bool IsFinite()

    cdef cppclass Sizei "OVR::Sizei":
        Sizei() except +
        Sizei(int s) except +
        Sizei(int w_, int h_) except +
        int w, h
        bool operator==(Sizei& b)
        bool operator!=(Sizei& b)
        Sizei operator+(Sizei& b)
        Sizei operator-(Sizei& b)
        Sizei operator-()
        Sizei operator*(Sizei& b)
        Sizei operator*(int b)
        Sizei operator/(Sizei& b)
        Sizei operator/(int b)
        bool operator==(Sizei& b)
        bool operator!=(Sizei& b)
        int Area()
        Vector2i ToVector()

    cdef cppclass Recti "OVR::Recti":
        Recti() except +
        Recti(Vector2i& pos, Sizei& sz) except +
        Recti(int, int, int, int) except +
        int x, y
        int w, h
        Vector2i GetPos()
        Sizei GetSize()
        void SetPos(const Vector2i &pos)
        void SetSize(const Sizei& sz)