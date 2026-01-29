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
#include <atomic>
#include "ucmstore_v1.h"

namespace UC::EmptyStore {

class EmptyStore : public StoreV1 {
public:
    Status Setup(const Detail::Dictionary& config) { return Status::OK(); }
    std::string Readme() const { return "EmptyStore"; }
    Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId* blocks, size_t num)
    {
        return std::vector<uint8_t>(num, false);
    }
    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, size_t num) { return -1; }
    void Prefetch(const Detail::BlockId* blocks, size_t num) {}
    Expected<Detail::TaskHandle> Load(Detail::TaskDesc task) { return NextId(); }
    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task) { return NextId(); }
    Expected<bool> Check(Detail::TaskHandle taskId) { return true; }
    Status Wait(Detail::TaskHandle taskId) { return Status::OK(); }

private:
    static Detail::TaskHandle NextId() noexcept
    {
        static std::atomic<Detail::TaskHandle> id{1};
        return id.fetch_add(1, std::memory_order_relaxed);
    };
};

}  // namespace UC::EmptyStore

extern "C" UC::StoreV1* MakeEmptyStore() { return new UC::EmptyStore::EmptyStore(); }
