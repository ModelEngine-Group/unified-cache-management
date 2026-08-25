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
#if UCM_RUNTIME_ASCEND_IO_AGGREGATION
#include "io_aggregation/ascend_io_aggregation_stream.h"
#endif
#if UCM_RUNTIME_ASCEND_SDMA_DIRECT
#include "sdma_direct/ascend_sdma_direct_stream.h"
#endif
#include "ascend_stream.h"
#include "trans/device.h"

namespace UC::Trans {

Status Device::Init()
{
    const auto ret = aclInit(nullptr);
    if (ret == ACL_SUCCESS) { return Status::OK(); }
    if (ret == ACL_ERROR_REPEAT_INITIALIZE) { return Status::DuplicateKey(); }
    return Status{ret, std::to_string(ret)};
}

Status Device::Setup(int32_t deviceId)
{
    if (deviceId < 0) { return Status::Error(fmt::format("invalid device id({})", deviceId)); }
    auto ret = aclrtSetDevice(deviceId);
    if (ret == ACL_SUCCESS) { return Status::OK(); }
    return Status{ret, std::to_string(ret)};
}

Status Device::Reset(int32_t deviceId)
{
    if (deviceId < 0) { return Status::Error(fmt::format("invalid device id({})", deviceId)); }
    const auto ret = aclrtResetDevice(deviceId);
    return ret == ACL_SUCCESS ? Status::OK() : Status{ret, std::to_string(ret)};
}

Status Device::Finalize()
{
    const auto ret = aclFinalize();
    return ret == ACL_SUCCESS ? Status::OK() : Status{ret, std::to_string(ret)};
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

std::shared_ptr<Stream> Device::MakeIoAggregationStream()
{
#if UCM_RUNTIME_ASCEND_IO_AGGREGATION
    std::shared_ptr<AscendIoAggregationStream> stream = nullptr;
    try {
        stream = std::make_shared<AscendIoAggregationStream>();
    } catch (...) {
        return nullptr;
    }
    if (stream->Setup().Success()) { return stream; }
#else
#endif
    return nullptr;
}

std::shared_ptr<Stream> Device::MakeSdmaDirectStream()
{
#if UCM_RUNTIME_ASCEND_SDMA_DIRECT
    std::shared_ptr<AscendSdmaDirectStream> stream = nullptr;
    try {
        stream = std::make_shared<AscendSdmaDirectStream>();
    } catch (...) {
        return nullptr;
    }
    if (stream->Setup().Success()) { return stream; }
    return nullptr;
#else
    return nullptr;
#endif
}

std::unique_ptr<Stream> Device::MakeGdrStream() { return nullptr; }

std::unique_ptr<Stream> Device::MakeSMStream() { return nullptr; }

std::unique_ptr<Buffer> Device::MakeBuffer()
{
    try {
        return std::make_unique<AscendBuffer>();
    } catch (...) {
        return nullptr;
    }
}

}  // namespace UC::Trans
