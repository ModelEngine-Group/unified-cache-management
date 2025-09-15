/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#include "device.h"
#include "aclrt_device.h"
#include "cuda_device.h"
#include "logger/logger.h"
#include "simu_device.h"

namespace UC {

std::unique_ptr<IDevice> Device::Make(const int32_t deviceId, const size_t bufferSize, const size_t bufferNumber)
{
    try {
#ifdef ASCEND_AVAILABLE
        return std::make_unique<AclrtDevice>(deviceId, bufferSize, bufferNumber);
#endif
#ifdef CUDA_AVAILABLE
        return std::make_unique<CudaDevice>(deviceId, bufferSize, bufferNumber);
#endif
        return std::make_unique<SimuDevice>(deviceId, bufferSize, bufferNumber);
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to instantinate the device({},{},{}) object.", e.what(), deviceId, bufferSize,
                 bufferNumber);
        return nullptr;
    }
}

} // namespace UC
