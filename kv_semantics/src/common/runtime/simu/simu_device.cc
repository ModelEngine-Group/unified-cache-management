/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
#include <fmt/format.h>
#include "simu_buffer.h"
#include "simu_trans.h"
#include "types.h"

namespace UC::Trans {

UC::ASU::Status Device::Init() { return UC::ASU::Status::OK(); }

UC::ASU::Status Device::Setup(int32_t deviceId)
{
    if (deviceId < 0) {
        return UC::ASU::Status::Error(UC::ASU::StatusCode::INVALID_ARGUMENT,
                                      fmt::format("invalid device id({})", deviceId));
    }
    return UC::ASU::Status::OK();
}

UC::ASU::Status Device::Reset(int32_t deviceId)
{
    if (deviceId < 0) {
        return UC::ASU::Status::Error(UC::ASU::StatusCode::INVALID_ARGUMENT,
                                      fmt::format("invalid device id({})", deviceId));
    }
    return UC::ASU::Status::OK();
}

UC::ASU::Status Device::Finalize() { return UC::ASU::Status::OK(); }

std::unique_ptr<Buffer> Device::MakeBuffer()
{
    try {
        return std::make_unique<SimuBuffer>();
    } catch (...) {
        return nullptr;
    }
}

std::unique_ptr<Trans> Device::MakeTrans()
{
    try {
        return std::make_unique<SimuTrans>();
    } catch (...) {
        return nullptr;
    }
}

}  // namespace UC::Trans
