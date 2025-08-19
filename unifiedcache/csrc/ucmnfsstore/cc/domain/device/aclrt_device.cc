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
#include "aclrt_device.h"
#include <acl/acl.h>
#include "logger/logger.h"

namespace UC {

template <typename Api, typename... Args>
Status AclrtApi(const char* caller, const char* file, const size_t line, const char* name, Api&& api, Args&&..args)
{
    auto ret = api(args...);
    if (ret != ACL_SUCCESS) {
        UC_ERROR("ACL ERROR: api={}, code={}, caller={},{}:{}.", name, ret, caller, basename(file), line);
        return Status::OsApiError();
    }
    return Status::OK();
}
#define ACLRT_API(api, ...) AclrtApi(__FUNCTION__, __FILE__, __LINE__, #api, api, __VA_ARGS__)

AclrtDevice::~AclrtDevice()
{
    if (this->_cbThread.joinable()) {
        auto tid = this->_cbThread.native_handle();
        (void)ACLRT_API(aclrtUnSubscribeReport, tid, this->_stream);
        this->_stop = true;
        this->_cbThread.join();
    }
    if (this->_stream) {
        (void)ACLRT_API(aclrtDestroyStream, this->_stream);
        this->_stream = nullptr;
    }
    (void)ACLRT_API(aclrtResetDevice, this->_deviceId);
}

Status AclrtDevice::Setup()
{
    auto status = Status::OK();
    if ((status = ACLRT_API(aclrtSetDevice, this->_deviceId)).Failure()) { return status; }
    if ((status = IBufferedDevice::Setup()).Failure()) { return status; }
    if ((status = ACLRT_API(aclrtCreateStream, &this->_stream)).Failure()) { return status; }
    this->_cbThread = std::thread([this] {
        while (!this->_stop) { (void)aclrtProcessReport(10); }
    });
    auto tid = this->_cbThread.native_handle();
    if ((status = ACLRT_API(aclrtSubscribeReport, tid, this->_stream)).Failure()) { return status; }
    return Status::OK();
}

Status AclrtDevice::H2DAsync(std::byte* dst, const std::byte* src, const size_t count)
{
    return ACLRT_API(aclrtMemcpyAsync, dst, count, src, count, ACL_MEMCPY_HOST_TO_DEVICE, this->_stream);
}

Status AclrtDevice::D2HAsync(std::byte* dst, const std::byte* src, const size_t count)
{
    return ACLRT_API(aclrtMemcpyAsync, dst, count, src, count, ACL_MEMCPY_DEVICE_TO_HOST, this->_stream);
}

struct Closure {
    std::function<void(bool)> cb;
    explicit Closure(std::function<void(bool)> cb) : cb{cb} {}
};

static void Trampoline(void* data)
{
    auto c = (Closure*)data;
    c->cb(true);
    delete c;
}

Status AclrtDevice::AppendCallback(std::function<void(bool)> cb)
{
    auto* c = new (std::nothrow) Closure(cb);
    if (!c) {
        UC_ERROR("Failed to make closure for append cb.");
        return Status::OutOfMemory();
    }
    return ACLRT_API(aclrtLaunchCallback, Trampoline, (void*)c, ACL_CALLBACK_NO_BLOCK, this->_stream);
}

std::shared_ptr<std::byte> AclrtDevice::MakeBuffer(const size_t size)
{
    std::byte* host = nullptr;
    auto status = ACLRT_API(aclrtMallocHost, (void**)&host);
    if (status.Success()) { return std::shared_ptr<std::byte>(host, aclrtFreeHost); }
    return nullptr;
}

} // namespace UC
