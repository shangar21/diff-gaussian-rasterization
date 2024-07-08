#ifndef TYPES_H
#define TYPES_H

#include <cuda_fp16.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <ATen/ATen.h>

// Define float3 and float4 for __half type
struct half3 {
    __half x, y, z;

    __host__ __device__
    half3() : x(__float2half(0.0f)), y(__float2half(0.0f)), z(__float2half(0.0f)) {}

    __host__ __device__
    half3(__half x, __half y, __half z) : x(x), y(y), z(z) {}
};

struct half4 {
    __half x, y, z, w;

    __host__ __device__
    half4() : x(__float2half(0.0f)), y(__float2half(0.0f)), z(__float2half(0.0f)), w(__float2half(0.0f)) {}

    __host__ __device__
    half4(__half x, __half y, __half z, __half w) : x(x), y(y), z(z), w(w) {}
};

// Define glm::vec3 and glm::vec4 for __half type
namespace glm {
    template<>
    struct vec<3, __half, defaultp> {
        __half x, y, z;

        vec() : x(__float2half(0.0f)), y(__float2half(0.0f)), z(__float2half(0.0f)) {}

        vec(__half x, __half y, __half z) : x(x), y(y), z(z) {}

        // Conversion constructor from glm::vec3<float>
        vec(const vec<3, float, defaultp>& v) : x(__float2half(v.x)), y(__float2half(v.y)), z(__float2half(v.z)) {}

        operator vec<3, float, defaultp>() const {
            return vec<3, float, defaultp>(__half2float(x), __half2float(y), __half2float(z));
        }
    };

    template<>
    struct vec<4, __half, defaultp> {
        __half x, y, z, w;

        vec() : x(__float2half(0.0f)), y(__float2half(0.0f)), z(__float2half(0.0f)), w(__float2half(0.0f)) {}

        vec(__half x, __half y, __half z, __half w) : x(x), y(y), z(z), w(w) {}

        // Conversion constructor from glm::vec4<float>
        vec(const vec<4, float, defaultp>& v) : x(__float2half(v.x)), y(__float2half(v.y)), z(__float2half(v.z)), w(__float2half(v.w)) {}

        operator vec<4, float, defaultp>() const {
            return vec<4, float, defaultp>(__half2float(x), __half2float(y), __half2float(z), __half2float(w));
        }
    };
}

typedef glm::vec<3, __half, glm::defaultp> half3_vec;
typedef glm::vec<4, __half, glm::defaultp> half4_vec;
//typedef float floatp;
//typedef float2 floatp2;
//typedef float3 floatp3;
//typedef float4 floatp4;
//typedef glm::vec3 vec3p;
//typedef glm::vec4 vec4p;
//typedef at::Half floatp;
//typedef glm::tvec2<at::Half> floatp2;
//typedef glm::tvec3<at::Half> floatp3;
//typedef glm::tvec4<at::Half> floatp4;
//typedef glm::tvec3<at::Half> vec3p;
//typedef glm::tvec4<at::Half> vec4p;
typedef __half floatp;
typedef __half2 floatp2;
typedef half3_vec floatp3;
typedef half4_vec floatp4;
typedef half3_vec vec3p;
typedef half4_vec vec4p;

#endif // TYPES_H

