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
#include "idevice.h"
#include "logger/logger.h"
#include "thread/latch.h"

namespace UC {

#define ASCEND_STREAM_NUMBER (8)
#define ASCEND_REPORT_PROCESS_TIMEOUT_MS (10)

class AscendDevice : public IDevice {
    void* _addr;
    std::atomic_bool _stop;
    std::thread _cbThread;
    std::array<aclrtStream, ASCEND_STREAM_NUMBER> _streams;

    void* AllocHost(const size_t size)
    {
        void* ptr = nullptr;
        auto ret = aclrtMallocHost(&ptr, size);
        if (ret != ACL_SUCCESS) {
            UC_ERROR("ACL ERROR: api=aclrtMallocHost, code={}.", ret);
            return nullptr;
        }
        return ptr;
    }
    void FreeHost(void* ptr)
    {
        auto ret = aclrtFreeHost(ptr);
        if (ret != ACL_SUCCESS) { UC_WARN("ACL ERROR: api=aclrtFreeHost, code={}.", ret); }
    }
    Status Synchornize()
    {
        auto status = Status::OK();
        Latch waiter{ASCEND_STREAM_NUMBER};
        for (auto& s : this->_streams) {
            if (status.Failure()) {
                waiter.Done();
                continue;
            }
            auto ret = aclrtLaunchCallback([](void* ud) { ((Latch*)ud)->Done(); }, &waiter, ACL_CALLBACK_NO_BLOCK, s);
            if (ret != ACL_SUCCESS) {
                status = Status::Error();
                UC_ERROR("ACL ERROR: api=aclrtLaunchCallback, code={}.", ret);
            }
        }
        waiter.Wait();
        return status;
    }

public:
    AscendDevice(const int32_t deviceId, const size_t bufferSize, const size_t bufferNumber)
        : IDevice{deviceId, bufferSize, bufferNumber}, _addr{nullptr}, _stop{false}
    {
        this->_cbThread = std::thread([this] {
            while (!this->_stop) { (void)aclrtProcessReport(ASCEND_REPORT_PROCESS_TIMEOUT_MS); }
        });
    }
    ~AscendDevice() override
    {
        auto tid = this->_cbThread.native_handle();
        for (auto& s : this->_streams) {
            if (!s) { continue; }
            auto ret = aclrtUnSubscribeReport(tid, s);
            if (ret != ACL_SUCCESS) { UC_WARN("ACL ERROR: api=aclrtUnSubscribeReport, code={}.", ret); }
            ret = aclrtDestroyStream(s);
            if (ret != ACL_SUCCESS) { UC_WARN("ACL ERROR: api=aclrtDestroyStream, code={}.", ret); }
        }
        this->_stop = true;
        this->_cbThread.join();
        if (this->_addr) {
            this->FreeHost(this->_addr);
            this->_addr = nullptr;
        }
        auto ret = aclrtResetDevice(this->deviceId);
        if (ret != ACL_SUCCESS) { UC_WARN("ACL ERROR: api=aclrtResetDevice, code={}.", ret); }
    }
    Status Setup() override
    {
        if (this->deviceId < 0) {
            UC_ERROR("Invalid xpu id({}).", this->deviceId);
            return Status::InvalidParam();
        }
        auto ret = aclrtSetDevice(this->deviceId);
        if (ret != ACL_SUCCESS) {
            UC_ERROR("ACL ERROR: api=aclrtSetDevice, code={}.", ret);
            return Status::Error();
        }
        auto tid = this->_cbThread.native_handle();
        for (auto& s : this->_streams) {
            ret = aclrtCreateStream(&s);
            if (ret != ACL_SUCCESS) {
                UC_ERROR("ACL ERROR: api=aclrtCreateStream, code={}.", ret);
                return Status::Error();
            }
            ret = aclrtSubscribeReport(tid, s);
            if (ret != ACL_SUCCESS) {
                UC_ERROR("ACL ERROR: api=aclrtSubscribeReport, code={}.", ret);
                return Status::Error();
            }
        }
        auto reservedMemSize = this->bufferNumber * this->bufferSize;
        if (reservedMemSize != 0) {
            this->_addr = this->AllocHost(reservedMemSize);
            if (!this->_addr) { return Status::OutOfMemory(); }
        }
        return Status::OK();
    }
    void* GetBuffer(const size_t idx) override
    {
        if (idx < this->bufferNumber) { return ((uint8_t*)this->_addr) + this->bufferSize * idx; }
        return this->AllocHost(this->bufferSize);
    }
    void PutBuffer(const size_t idx, void* ptr) override
    {
        if (idx < this->bufferNumber) { return; }
        this->FreeHost(ptr);
    }
    Status H2DBatch(const uintptr_t* from, uintptr_t* to, const size_t number, const size_t size) override
    {
        for (size_t i = 0; i < number; i++) {
            auto ret = aclrtMemcpyAsync((void*)to[i], size, (void*)from[i], size, ACL_MEMCPY_HOST_TO_DEVICE,
                                        this->_streams[i % ASCEND_STREAM_NUMBER]);
            if (ret != ACL_SUCCESS) {
                UC_ERROR("ACL ERROR: api=aclrtMemcpyAsync, code={}.", ret);
                (void)this->Synchornize();
                return Status::Error();
            }
        }
        return this->Synchornize();
    }
    Status D2HBatch(const uintptr_t* from, uintptr_t* to, const size_t number, const size_t size) override
    {
        for (size_t i = 0; i < number; i++) {
            auto ret = aclrtMemcpyAsync((void*)to[i], size, (void*)from[i], size, ACL_MEMCPY_DEVICE_TO_HOST,
                                        this->_streams[i % ASCEND_STREAM_NUMBER]);
            if (ret != ACL_SUCCESS) {
                UC_ERROR("ACL ERROR: api=aclrtMemcpyAsync, code={}.", ret);
                (void)this->Synchornize();
                return Status::Error();
            }
        }
        return this->Synchornize();
    }
};

std::unique_ptr<IDevice> DeviceFactory::Make(const int32_t deviceId, const size_t bufferSize, const size_t bufferNumber)
{
    try {
        return std::make_unique<AscendDevice>(deviceId, bufferSize, bufferNumber);
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to make ascend device({},{},{}).", e.what(), deviceId, bufferSize, bufferNumber);
        return nullptr;
    }
}

} // namespace UC
