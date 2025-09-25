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
#include "dramstore.h"
#include "logger/logger.h"
#include "status/status.h"

namespace UC {

class DramStoreImpl : public DramStore {
public:
    int32_t Setup(const size_t ioSize, const size_t capacity, const int32_t deviceId) { return -1; }
    int32_t Alloc(const std::string& block) override { return -1; }
    bool Lookup(const std::string& block) override { return false; }
    void Commit(const std::string& block, const bool success) override {}
    std::list<int32_t> Alloc(const std::list<std::string>& blocks) override
    {
        return std::list<int32_t>();
    }
    std::list<bool> Lookup(const std::list<std::string>& blocks) override
    {
        return std::list<bool>();
    }
    void Commit(const std::list<std::string>& blocks, const bool success) override {}
    size_t Submit(Task&& task) override { return 0; }
    int32_t Wait(const size_t task) override { return -1; }
    int32_t Check(const size_t task, bool& finish) override { return -1; }
};

int32_t DramStore::Setup(const Config& config)
{
    auto impl = new (std::nothrow) DramStoreImpl();
    if (!impl) {
        UC_ERROR("Out of memory.");
        return Status::OutOfMemory().Underlying();
    }
    this->impl_ = impl;
    return impl->Setup(config.ioSize, config.capacity, config.deviceId);
}

} // namespace UC
