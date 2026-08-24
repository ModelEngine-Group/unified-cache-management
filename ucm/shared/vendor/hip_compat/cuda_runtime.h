/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Author: Jeff Daily <jeff.daily@amd.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */

/*
 * ROCm/HIP compatibility shim. On a ROCm build the per-backend CMake puts this
 * directory ahead of the toolchain includes, so every existing
 * `#include <cuda_runtime.h>` resolves here instead of the (absent) NVIDIA
 * header. We pull in the HIP runtime and alias the small set of cuda* runtime
 * symbols the KV-transfer backend uses to their hip* equivalents, so the
 * device-backend sources compile unchanged. The NVIDIA path never sees this
 * file (its include dir points at the real CUDA toolkit).
 */
#ifndef UNIFIEDCACHE_HIP_COMPAT_CUDA_RUNTIME_H
#define UNIFIEDCACHE_HIP_COMPAT_CUDA_RUNTIME_H

#include <cstdlib>
#include <cstring>

#include <hip/hip_runtime.h>

using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
using cudaEvent_t = hipEvent_t;

static constexpr hipError_t cudaSuccess = hipSuccess;
static constexpr hipMemcpyKind cudaMemcpyHostToDevice = hipMemcpyHostToDevice;
static constexpr hipMemcpyKind cudaMemcpyDeviceToHost = hipMemcpyDeviceToHost;
static constexpr unsigned int cudaStreamNonBlocking = hipStreamNonBlocking;
static constexpr unsigned int cudaHostRegisterDefault = hipHostRegisterDefault;

#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocHost hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaSetDevice hipSetDevice
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamAddCallback hipStreamAddCallback
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaHostRegister hipHostRegister
#define cudaHostUnregister hipHostUnregister
#define cudaHostGetDevicePointer hipHostGetDevicePointer

#endif
