/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>
#include "store.h"

namespace UC::Store {

namespace {

/* Behavior-configurable store plugin for assembler tests, driven entirely
 * through the public Store interface and loaded by path. Setup keys:
 *  - "piped_lower_store" (Store*, injected by the assembler): when set,
 *    Lookup delegates to the wired lower tier, making the pipe wiring
 *    observable end to end;
 *  - "present" (int64): the leading block count Lookup reports as hits,
 *    all blocks when absent;
 *  - "fail_setup" (int64): non-zero makes Setup fail, for error paths. */
class StubStore : public Store {
public:
    Status Setup(const Dictionary& config) override
    {
        (void)config.Get<Store*>("piped_lower_store", lower_);
        std::int64_t present = 0;
        if (config.GetInt64("present", present)) { present_ = static_cast<std::size_t>(present); }
        std::int64_t failSetup = 0;
        if (config.GetInt64("fail_setup", failSetup) && failSetup != 0) {
            return Status::General("stub setup failure");
        }
        return Status::Ok();
    }

    Expected<std::vector<std::uint8_t>> Lookup(const std::vector<BlockId>& blockIds) override
    {
        if (lower_ != nullptr) { return lower_->Lookup(blockIds); }
        std::vector<std::uint8_t> hits(blockIds.size(), 0);
        for (std::size_t i = 0; i < hits.size() && i < present_; i++) { hits[i] = 1; }
        return hits;
    }

    void Touch(const std::vector<BlockId>& /*blockIds*/, const RequestContext& /*context*/) override
    {
    }

    Status Load(std::size_t /*jobId*/, JobDescriptor /*job*/, AddressedToken /*token*/) override
    {
        return Status::Ok();
    }

    Status Dump(std::size_t /*jobId*/, JobDescriptor /*job*/, AddressedToken /*token*/) override
    {
        return Status::Ok();
    }

    std::vector<std::pair<std::size_t, bool>> GetFinished() override { return {}; }

    std::vector<std::pair<std::size_t, bool>> Wait(
        const std::vector<std::size_t>& /*jobIds*/) override
    {
        return {};
    }

private:
    Store* lower_ = nullptr;
    std::size_t present_ = static_cast<std::size_t>(-1);
};

}  // namespace

extern "C" UC_EXPORT ::UC::Store::Store* UCCreateStore() { return new (std::nothrow) StubStore(); }

extern "C" UC_EXPORT void UCDestroyStore(::UC::Store::Store* store) { delete store; }

}  // namespace UC::Store
