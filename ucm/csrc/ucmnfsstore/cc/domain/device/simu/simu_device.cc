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
#include <thread>
#include <vector>
#include "idevice.h"
#include "logger/logger.h"

namespace UC {

class SimuDevice : public IDevice {
    void* _addr;
    void* Alloc(const size_t size) { return malloc(size); }
    void Free(void* ptr) { free(ptr); }

public:
    SimuDevice(const int32_t deviceId, const size_t bufferSize, const size_t bufferNumber)
        : IDevice{deviceId, bufferSize, bufferNumber}, _addr{nullptr}
    {
    }
    ~SimuDevice() override
    {
        if (this->_addr) {
            this->Free(this->_addr);
            this->_addr = nullptr;
        }
    }
    Status Setup() override
    {
        auto reservedMemSize = this->bufferNumber * this->bufferSize;
        if (reservedMemSize != 0) {
            this->_addr = this->Alloc(reservedMemSize);
            if (!this->_addr) {
                UC_ERROR("Out of memory({}B).", reservedMemSize);
                return Status::OutOfMemory();
            }
        }
        return Status::OK();
    }
    void* GetBuffer(const size_t idx) override
    {
        if (idx < this->bufferNumber) { return ((uint8_t*)this->_addr) + this->bufferSize * idx; }
        auto ptr = this->Alloc(this->bufferSize);
        if (!ptr) { UC_ERROR("Out of memory({}B).", this->bufferSize); }
        return ptr;
    }
    void PutBuffer(const size_t idx, void* ptr) override
    {
        if (idx < this->bufferNumber) { return; }
        this->Free(ptr);
    }
    Status H2DBatch(const uintptr_t* from, uintptr_t* to, const size_t number, const size_t size) override
    {
        return D2HBatch(from, to, number, size);
    }
    Status D2HBatch(const uintptr_t* from, uintptr_t* to, const size_t number, const size_t size) override
    {
        constexpr size_t nPerThread = 1024;
        std::vector<std::thread> workers;
        for (size_t start = 0; start < number; start += nPerThread) {
            auto end = std::min(start + nPerThread, number);
            workers.emplace_back([=] {
                for (auto i = start; i < end; i++) {
                    auto src = (const std::byte*)from[i];
                    auto dst = (std::byte*)to[i];
                    std::copy(src, src + size, dst);
                }
            });
        }
        for (auto& worker : workers) { worker.join(); }
        return Status::OK();
    }
};

std::unique_ptr<IDevice> DeviceFactory::Make(const int32_t deviceId, const size_t bufferSize, const size_t bufferNumber)
{
    try {
        return std::make_unique<SimuDevice>(deviceId, bufferSize, bufferNumber);
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to make simu device({},{},{}).", e.what(), deviceId, bufferSize, bufferNumber);
        return nullptr;
    }
}

} // namespace UC
