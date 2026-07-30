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
#include <acl/acl.h>
#include "ascend_buffer.h"
#include "ascend_stream.h"
#include "trans/device.h"

namespace UC::Trans {

Status Device::Init()
{
    const auto ret = aclInit(nullptr);
    if (ret == ACL_SUCCESS) {
        deviceRuntimeOwned_ = true;
        return Status::OK();
    }
    if (ret == ACL_ERROR_REPEAT_INITIALIZE) { return Status::OK(); }
    return Status{ret, std::to_string(ret)};
}

Status Device::Setup(int32_t deviceId)
{
    if (deviceId < 0) { return Status::Error(fmt::format("invalid device id({})", deviceId)); }
    auto ret = aclrtSetDevice(deviceId);
    if (ret == ACL_SUCCESS) {
        deviceId_ = deviceId;
        return Status::OK();
    }
    return Status{ret, std::to_string(ret)};
}

Status Device::Reset()
{
    if (!deviceRuntimeOwned_) { return Status::OK(); }
    if (deviceId_ < 0) { return Status::OK(); }
    const auto ret = aclrtResetDevice(deviceId_);
    if (ret != ACL_SUCCESS) { return Status{ret, std::to_string(ret)}; }
    deviceId_ = -1;
    return Status::OK();
}

Status Device::Finalize()
{
    if (!deviceRuntimeOwned_) { return Status::OK(); }
    const auto ret = aclFinalize();
    if (ret != ACL_SUCCESS) { return Status{ret, std::to_string(ret)}; }
    deviceRuntimeOwned_ = false;
    return Status::OK();
}

std::unique_ptr<Stream> Device::MakeStream()
{
    std::unique_ptr<Stream> stream = nullptr;
    try {
        stream = std::make_unique<AscendStream>();
    } catch (...) {
        return nullptr;
    }
    if (stream->Setup().Success()) { return stream; }
    return nullptr;
}

std::shared_ptr<Stream> Device::MakeSharedStream()
{
    std::shared_ptr<Stream> stream = nullptr;
    try {
        stream = std::make_shared<AscendStream>();
    } catch (...) {
        return nullptr;
    }
    if (stream->Setup().Success()) { return stream; }
    return nullptr;
}

std::unique_ptr<Stream> Device::MakeSMStream() { return nullptr; }

std::unique_ptr<Buffer> Device::MakeBuffer()
{
    try {
        return std::make_unique<AscendBuffer>();
    } catch (...) {
        return nullptr;
    }
}

} // namespace UC::Trans
