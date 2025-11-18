/**
 * MIT License
 *
 * Copyright (c) 2025 MThreads Technologies Co., Ltd. All rights reserved.
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
#include <musa_runtime.h>
#include "ibuffered_device.h"
#include "logger/logger.h"

template <>
struct fmt::formatter<musaError_t> : formatter<int32_t> {
    auto format(musaError_t err, format_context& ctx) const -> format_context::iterator
    {
        return formatter<int32_t>::format(err, ctx);
    }
};

namespace UC {

template <typename Api, typename... Args>
Status MusaApi(const char* caller, const char* file, const size_t line, const char* name, Api&& api,
               Args&&... args)
{
    auto ret = api(args...);
    if (ret != musaSuccess) {
        UC_ERROR("MUSA ERROR: api={}, code={}, err={}, caller={},{}:{}.", name, ret,
                 musaGetErrorString(ret), caller, basename(file), line);
        return Status::OsApiError();
    }
    return Status::OK();
}
#define MUSA_API(api, ...) MusaApi(__FUNCTION__, __FILE__, __LINE__, #api, api, __VA_ARGS__)

class MusaDevice : public IBufferedDevice {
    struct Closure {
        std::function<void(bool)> cb;
        explicit Closure(std::function<void(bool)> cb) : cb{cb} {}
    };

    static void Trampoline(musaStream_t stream, musaError_t ret, void* data)
    {
        (void)stream;
        auto c = (Closure*)data;
        c->cb(ret == musaSuccess);
        delete c;
    }

public:
    MusaDevice(const int32_t deviceId, const size_t bufferSize, const size_t bufferNumber)
        : IBufferedDevice{deviceId, bufferSize, bufferNumber}, stream_{nullptr}
    {
    }
    Status Setup() override
    {
        auto status = Status::OK();
        if ((status = MUSA_API(musaSetDevice, this->deviceId)).Failure()) { return status; }
        if ((status = IBufferedDevice::Setup()).Failure()) { return status; }
        if ((status = MUSA_API(musaStreamCreate, (musaStream_t*)&this->stream_)).Failure()) {
            return status;
        }
        return status;
    }
    Status H2DSync(std::byte* dst, const std::byte* src, const size_t count) override
    {
        return MUSA_API(musaMemcpy, dst, src, count, musaMemcpyHostToDevice);
    }
    Status D2HSync(std::byte* dst, const std::byte* src, const size_t count) override
    {
        return MUSA_API(musaMemcpy, dst, src, count, musaMemcpyDeviceToHost);
    }

    Status H2DAsync(std::byte* dst, const std::byte* src, const size_t count) override
    {
        return MUSA_API(musaMemcpyAsync, dst, src, count, musaMemcpyHostToDevice,
                        (musaStream_t)this->stream_);
    }
    Status D2HAsync(std::byte* dst, const std::byte* src, const size_t count) override
    {
        return MUSA_API(musaMemcpyAsync, dst, src, count, musaMemcpyDeviceToHost,
                        (musaStream_t)this->stream_);
    }
    Status AppendCallback(std::function<void(bool)> cb) override
    {
        auto* c = new (std::nothrow) Closure(cb);
        if (!c) {
            UC_ERROR("Failed to make closure for append cb.");
            return Status::OutOfMemory();
        }
        auto status =
            MUSA_API(musaStreamAddCallback, (musaStream_t)this->stream_, Trampoline, (void*)c, 0);
        if (status.Failure()) { delete c; }
        return status;
    }

    Status Synchronized() override { return MUSA_API(musaStreamSynchronize, this->stream_); }

    Status H2DBatchSync(std::byte* dArr[], const std::byte* hArr[], const size_t number,
                        const size_t count) override
    {
        for (size_t i = 0; i < number; i++) {
            auto status = this->H2DSync(dArr[i], hArr[i], count);
            if (status.Failure()) { return status; }
        }
        return Status::OK();
    }
    Status D2HBatchSync(std::byte* hArr[], const std::byte* dArr[], const size_t number,
                        const size_t count) override
    {
        for (size_t i = 0; i < number; i++) {
            auto status = this->D2HSync(hArr[i], dArr[i], count);
            if (status.Failure()) { return status; }
        }
        return Status::OK();
    }

protected:
    std::shared_ptr<std::byte> MakeBuffer(const size_t size) override
    {
        std::byte* host = nullptr;
        auto ret = musaMallocHost((void**)&host, size);
        if (ret != musaSuccess) {
            UC_ERROR("MUSA ERROR: api=musaMallocHost, code={}.", ret);
            return nullptr;
        }
        return std::shared_ptr<std::byte>(host, musaFreeHost);
    }

private:
    musaStream_t stream_;

};

std::unique_ptr<IDevice> DeviceFactory::Make(const int32_t deviceId, const size_t bufferSize,
                                             const size_t bufferNumber)
{
    try {
        return std::make_unique<MusaDevice>(deviceId, bufferSize, bufferNumber);
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to make musa device({},{},{}).", e.what(), deviceId, bufferSize,
                 bufferNumber);
        return nullptr;
    }
}

} // namespace UC
