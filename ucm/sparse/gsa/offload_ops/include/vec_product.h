#ifndef VEC_PRODUCT_H
#define VEC_PRODUCT_H

#include <cstdint>
#include <string>

// SIMD头文件
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    #define ARCH_X86 1
    #include <immintrin.h>
    #ifdef _WIN32
        #include <malloc.h>
        #include <intrin.h>
    #else
        #include <cpuid.h>
    #endif
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
    #define ARCH_ARM 1
    #include <arm_neon.h>
    #ifdef _WIN32
        #include <malloc.h>
    #endif
#else
    #define ARCH_GENERIC 1
    #ifdef _WIN32
        #include <malloc.h>
    #endif
#endif

namespace VecProduct {

/**
* 常量定义
*/

namespace SimdLevel {
    constexpr int32_t NONE   = 0;
    constexpr int32_t BASIC  = 1;
    constexpr int32_t AVX2   = 2;
    constexpr int32_t AVX512 = 3;
}

namespace SimdElements {
    constexpr uint32_t VEC4  = 4;
    constexpr uint32_t VEC8  = 8;
    constexpr uint32_t VEC16 = 16;
}

constexpr uint32_t SCALAR_UNROLL_FACTOR = 4;

/**
* 编译时SIMD选择
*/

namespace CompileTimeSimd {
#ifdef ARCH_X86
    #ifdef __AVX512F__
        constexpr int32_t SELECTED_LEVEL = SimdLevel::AVX512;
        constexpr const char* SIMD_NAME = "x86_64 AVX512F";
    #elif defined(__AVX2__)
        constexpr int32_t SELECTED_LEVEL = SimdLevel::AVX2;
        constexpr const char* SIMD_NAME = "x86_64 AVX2";
    #elif defined(__SSE2__)
        constexpr int32_t SELECTED_LEVEL = SimdLevel::BASIC;
        constexpr const char* SIMD_NAME = "x86_64 SSE2";
    #else
        constexpr int32_t SELECTED_LEVEL = SimdLevel::NONE;
        constexpr const char* SIMD_NAME = "x86_64 Scalar";
    #endif
#elif defined(ARCH_ARM)
    constexpr int32_t SELECTED_LEVEL = SimdLevel::BASIC;
    constexpr const char* SIMD_NAME = "ARM NEON";
#else
    constexpr int32_t SELECTED_LEVEL = SimdLevel::NONE;
    constexpr const char* SIMD_NAME = "Generic Scalar";
#endif
}

/**
* @brief VecProduct 主类
*
*/
class VecProduct {
public:
    VecProduct() = default;
    
    /**
    * @brief 禁用拷贝构造和赋值
    */
    VecProduct(const VecProduct&) = delete;
    VecProduct& operator=(const VecProduct&) = delete;

    /**
    * @brief 计算双向量点积
    */
    inline float VectorDot(const float* q, const float* k, uint32_t dim) const
    {
        return VectorDotImpl(q, k, dim);
    }
    
    /**
    * @brief 计算向量组均值
    */
    inline void VectorMean(const float* vectors, float* mean, uint32_t dim, uint32_t numK) const
    {
        VectorMeanImpl(vectors, mean, dim, numK);
    }

private:

   static float VectorDotImpl(const float* __restrict q, const float* __restrict k, uint32_t dim);

   static void VectorMeanImpl(const float* __restrict vectors, float* __restrict mean, uint32_t dim, uint32_t numK);
};

}

#endif
