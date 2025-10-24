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
#include <array>
#include <atomic>
#include <thread>
#include "ibuffered_device.h"
#include "logger/logger.h"
#include "thread/latch.h"

namespace UC {

template <typename Api, typename... Args>
Status AscendApi(const char* caller, const char* file, const size_t line, const char* name,
                 Api&& api, Args&&... args)
{
    auto ret = api(args...);
    if (ret != ACL_SUCCESS) {
        UC_ERROR("ACL ERROR: api={}, code={}, caller={},{}:{}.", name, ret, caller, basename(file),
                 line);
        return Status::OsApiError();
    }
    return Status::OK();
}
#define ASCEND_API(api, ...) AscendApi(__FUNCTION__, __FILE__, __LINE__, #api, api, __VA_ARGS__)

class AscendDevice : public IBufferedDevice {
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

public:
    AscendDevice(const int32_t deviceId, const size_t bufferSize, const size_t bufferNumber)
        : IBufferedDevice{deviceId, bufferSize, bufferNumber}, stop_{false}, stream_{nullptr}
    {
    }
    ~AscendDevice() override
    {
        if (this->cbThread_.joinable()) {
            auto tid = this->cbThread_.native_handle();
            (void)aclrtUnSubscribeReport(tid, this->stream_);
            this->stop_ = true;
            this->cbThread_.join();
        }
        if (this->stream_) {
            (void)aclrtDestroyStream(this->stream_);
            this->stream_ = nullptr;
        }
        (void)aclrtResetDevice(this->deviceId);
    }
    Status Setup() override
    {
        auto status = Status::OK();
        if ((status = ASCEND_API(aclrtSetDevice, this->deviceId)).Failure()) { return status; }
        if ((status = IBufferedDevice::Setup()).Failure()) { return status; }
        if ((status = ASCEND_API(aclrtCreateStream, &this->stream_)).Failure()) { return status; }
        this->cbThread_ = std::thread([this] {
            while (!this->stop_) { (void)aclrtProcessReport(10); }
        });
        auto tid = this->cbThread_.native_handle();
        if ((status = ASCEND_API(aclrtSubscribeReport, tid, this->stream_)).Failure()) {
            return status;
        }
        return Status::OK();
    }
    Status H2DSync(std::byte* dst, const std::byte* src, const size_t count) override
    {
        return ASCEND_API(aclrtMemcpy, dst, count, src, count, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    Status D2HSync(std::byte* dst, const std::byte* src, const size_t count) override
    {
        return ASCEND_API(aclrtMemcpy, dst, count, src, count, ACL_MEMCPY_DEVICE_TO_HOST);
    }
    Status H2DAsync(std::byte* dst, const std::byte* src, const size_t count) override
    {
        return ASCEND_API(aclrtMemcpyAsync, dst, count, src, count, ACL_MEMCPY_HOST_TO_DEVICE,
                          this->stream_);
    }
    Status D2HAsync(std::byte* dst, const std::byte* src, const size_t count) override
    {
        return ASCEND_API(aclrtMemcpyAsync, dst, count, src, count, ACL_MEMCPY_DEVICE_TO_HOST,
                          this->stream_);
    }
    Status AppendCallback(std::function<void(bool)> cb) override
    {
        auto* c = new (std::nothrow) Closure(cb);
        if (!c) {
            UC_ERROR("Failed to make closure for append cb.");
            return Status::OutOfMemory();
        }
        return ASCEND_API(aclrtLaunchCallback, Trampoline, (void*)c, ACL_CALLBACK_NO_BLOCK,
                          this->stream_);
    }
    Status Synchronized() override { return ASCEND_API(aclrtSynchronizeStream, this->stream_); }

protected:
    std::shared_ptr<std::byte> MakeBuffer(const size_t size) override
    {
        std::byte* host = nullptr;
        auto status = ASCEND_API(aclrtMallocHost, (void**)&host, size);
        if (status.Success()) { return std::shared_ptr<std::byte>(host, aclrtFreeHost); }
        return nullptr;
    }

private:
    std::atomic_bool stop_;
    void* stream_;
    std::thread cbThread_;
};

std::unique_ptr<IDevice> DeviceFactory::Make(const int32_t deviceId, const size_t bufferSize,
                                             const size_t bufferNumber)
{
    try {
        return std::make_unique<AscendDevice>(deviceId, bufferSize, bufferNumber);
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to make ascend device({},{},{}).", e.what(), deviceId, bufferSize,
                 bufferNumber);
        return nullptr;
    }
}

} // namespace UC
