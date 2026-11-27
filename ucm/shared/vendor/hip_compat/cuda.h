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
 * ROCm/HIP compatibility shim for the CUDA driver-API header. The sparse
 * Hamming-distance extension includes <cuda.h> only to pull in the runtime
 * declarations it shares with <cuda_runtime.h>; it uses no driver-API entry
 * points. On a ROCm build we map it onto the runtime shim so the include
 * resolves without the (absent) NVIDIA driver header.
 */
#ifndef UNIFIEDCACHE_HIP_COMPAT_CUDA_H
#define UNIFIEDCACHE_HIP_COMPAT_CUDA_H

#include "cuda_runtime.h"

#endif
