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
#include "simu_device.h"
#include "logger/logger.h"

namespace UC {

Status SimuDevice::Setup()
{
    auto status = IBufferedDevice::Setup();
    if (status.Failure()) { return status; }
    if (!this->_backend.Setup([](auto& task) { task(); })) { return Status::Error(); }
    return Status::OK();
}

Status SimuDevice::H2DAsync(std::byte* dst, const std::byte* src, const size_t count)
{
    if (dst == nullptr || src == nullptr || count == 0) {
        UC_ERROR("Invalid params: count={}.", count);
        return Status::InvalidParam();
    }
    this->_backend.Push([=] { std::copy(src, src + count, dst); });
    return Status::OK();
}

Status SimuDevice::D2HAsync(std::byte* dst, const std::byte* src, const size_t count)
{
    if (dst == nullptr || src == nullptr || count == 0) {
        UC_ERROR("Invalid params: count={}.", count);
        return Status::InvalidParam();
    }
    this->_backend.Push([=] { std::copy(src, src + count, dst); });
    return Status::OK();
}

Status SimuDevice::AppendCallback(std::function<void(bool)> cb)
{
    this->_backend.Push([=] { cb(true); });
    return Status::OK();
}

std::shared_ptr<std::byte> SimuDevice::MakeBuffer(const size_t size)
{
    return std::shared_ptr<std::byte>((std::byte*)malloc(size), free);
}

} // namespace UC
