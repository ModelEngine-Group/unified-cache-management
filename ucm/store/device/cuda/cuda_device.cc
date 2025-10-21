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
#include <cuda_runtime.h>
#include "ibuffered_device.h"
#include "logger/logger.h"
#include <cufile.h>
#include <mutex>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include "sharded_handle_recorder.h"

template <>
struct fmt::formatter<cudaError_t> : formatter<int32_t> {
    auto format(cudaError_t err, format_context& ctx) const -> format_context::iterator
    {
        return formatter<int32_t>::format(err, ctx);
    }
};

namespace UC {

static Status CreateCuFileHandle(const std::string& path, int flags, CUfileHandle_t& cuFileHandle, int& fd)
{
    fd = open(path.c_str(), flags, 0644);
    if (fd < 0) {
        UC_ERROR("Failed to open file {}: {}", path, strerror(errno));
        return Status::Error();
    }

    CUfileDescr_t cfDescr{};
    cfDescr.handle.fd = fd;
    cfDescr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUfileError_t err = cuFileHandleRegister(&cuFileHandle, &cfDescr);
    if (err.err != CU_FILE_SUCCESS) {
        UC_ERROR("Failed to register cuFile handle for {}: error {}",
                 path, static_cast<int>(err.err));
        close(fd);
        fd = -1;
        return Status::Error();
    }

    return Status::OK();
}

template <typename Api, typename... Args>
Status CudaApi(const char* caller, const char* file, const size_t line, const char* name, Api&& api,
               Args&&... args)
{
    auto ret = api(args...);
    if (ret != cudaSuccess) {
        UC_ERROR("CUDA ERROR: api={}, code={}, err={}, caller={},{}:{}.", name, ret,
                 cudaGetErrorString(ret), caller, basename(file), line);
        return Status::OsApiError();
    }
    return Status::OK();
}
#define CUDA_API(api, ...) CudaApi(__FUNCTION__, __FILE__, __LINE__, #api, api, __VA_ARGS__)

class CudaDevice : public IBufferedDevice {
    struct Closure {
        std::function<void(bool)> cb;
        explicit Closure(std::function<void(bool)> cb) : cb{cb} {}
    };

    static void Trampoline(cudaStream_t stream, cudaError_t ret, void* data)
    {
        (void)stream;
        auto c = (Closure*)data;
        c->cb(ret == cudaSuccess);
        delete c;
    }

    static std::once_flag gdsOnce_;
    static void InitGdsOnce();

public:
    CudaDevice(const int32_t deviceId, const size_t bufferSize, const size_t bufferNumber)
        : IBufferedDevice{deviceId, bufferSize, bufferNumber}, stream_{nullptr}
    {
    }

    ~CudaDevice() {
        CuFileHandleRecorder::Instance().ClearAll([](CUfileHandle_t h, int fd) {
            cuFileHandleDeregister(h);
            if (fd >= 0) {
                close(fd);
            }
        });

        if (stream_ != nullptr) {
            cudaStreamDestroy((cudaStream_t)stream_);
        }
    }

    Status Setup(bool transferUseDirect) override
    {
        if(transferUseDirect) {InitGdsOnce();}
        auto status = Status::OK();
        if ((status = CUDA_API(cudaSetDevice, this->deviceId)).Failure()) { return status; }
        if ((status = IBufferedDevice::Setup(transferUseDirect)).Failure()) { return status; }
        if ((status = CUDA_API(cudaStreamCreate, (cudaStream_t*)&this->stream_)).Failure()) {
            return status;
        }
        return status;
    }
    Status H2DAsync(std::byte* dst, const std::byte* src, const size_t count) override
    {
        return CUDA_API(cudaMemcpyAsync, dst, src, count, cudaMemcpyHostToDevice,
                        (cudaStream_t)this->stream_);
    }
    Status D2HAsync(std::byte* dst, const std::byte* src, const size_t count) override
    {
        return CUDA_API(cudaMemcpyAsync, dst, src, count, cudaMemcpyDeviceToHost,
                        (cudaStream_t)this->stream_);
    }
    Status AppendCallback(std::function<void(bool)> cb) override
    {
        auto* c = new (std::nothrow) Closure(cb);
        if (!c) {
            UC_ERROR("Failed to make closure for append cb.");
            return Status::OutOfMemory();
        }
        auto status =
            CUDA_API(cudaStreamAddCallback, (cudaStream_t)this->stream_, Trampoline, (void*)c, 0);
        if (status.Failure()) { delete c; }
        return status;
    }
    Status S2D(const std::string& path, void* address, const size_t length, const size_t file_offset, const size_t dev_offset) override
    {
        CUfileHandle_t cuFileHandle = nullptr;
        auto status = CuFileHandleRecorder::Instance().Get(path, cuFileHandle,
            [&path](CUfileHandle_t& handle, int& fd) -> Status {
                return CreateCuFileHandle(path, O_RDONLY | O_DIRECT, handle, fd);
            });
        if (status.Failure()) {
            return status;
        }
        ssize_t bytesRead = cuFileRead(cuFileHandle, address, length, file_offset, dev_offset);
        if (bytesRead < 0 || (size_t)bytesRead != length) {
            UC_ERROR("cuFileRead failed for {}: expected {}, got {}", path, length, bytesRead);
            return Status::Error();
        }
        return Status::OK();
    }
     Status D2S(const std::string& path, void* address, const size_t length, const size_t file_offset, const size_t dev_offset) override
    {
        CUfileHandle_t cuFileHandle = nullptr;
        auto status = CuFileHandleRecorder::Instance().Get(path, cuFileHandle,
            [&path](CUfileHandle_t& handle, int& fd) -> Status {
                return CreateCuFileHandle(path, O_WRONLY | O_CREAT | O_DIRECT, handle, fd);
            });
        if (status.Failure()) {
            return status;
        }
        ssize_t bytesWrite = cuFileWrite(cuFileHandle, address, length, file_offset, dev_offset);
        if (bytesWrite < 0 || (size_t)bytesWrite != length) {
            UC_ERROR("cuFileWrite failed for {}: expected {}, got {}", path, length, bytesWrite);
            return Status::Error();
        }
        return Status::OK();
    }

protected:
    std::shared_ptr<std::byte> MakeBuffer(const size_t size) override
    {
        std::byte* host = nullptr;
        auto ret = cudaMallocHost((void**)&host, size);
        if (ret != cudaSuccess) {
            UC_ERROR("CUDA ERROR: api=cudaMallocHost, code={}.", ret);
            return nullptr;
        }
        return std::shared_ptr<std::byte>(host, cudaFreeHost);
    }

private:
    void* stream_;
};

std::unique_ptr<IDevice> DeviceFactory::Make(const int32_t deviceId, const size_t bufferSize,
                                             const size_t bufferNumber)
{
    try {
        return std::make_unique<CudaDevice>(deviceId, bufferSize, bufferNumber);
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to make cuda device({},{},{}).", e.what(), deviceId, bufferSize,
                 bufferNumber);
        return nullptr;
    }
}

std::once_flag CudaDevice::gdsOnce_{};
void CudaDevice::InitGdsOnce()
{
    std::call_once(gdsOnce_, [] (){
        CUfileError_t ret = cuFileDriverOpen();
        if (ret.err == CU_FILE_SUCCESS) {
            UC_INFO("GDS driver initialized successfully");
        } else {
            UC_ERROR("GDS driver initialized unsuccessfully");
        }
    });
}
} // namespace UC
