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
#pragma once

#include "store.h"

namespace UC::Store {

class PosixStore : public Store {
public:
    Status Setup(const Dictionary& /*config*/) override
    {
        return Status::Unsupported("PosixStore::Setup is not implemented");
    }

    Expected<std::vector<std::uint8_t>> Lookup(const std::vector<BlockId>& /*blockIds*/) override
    {
        return Status::Unsupported("PosixStore::Lookup is not implemented");
    }

    void Touch(const std::vector<BlockId>& /*blockIds*/, const RequestContext& /*context*/) override
    {
    }

    Status Load(std::size_t /*jobId*/, JobDescriptor /*job*/, AddressedToken /*token*/) override
    {
        return Status::Unsupported("PosixStore::Load is not implemented");
    }

    Status Dump(std::size_t /*jobId*/, JobDescriptor /*job*/, AddressedToken /*token*/) override
    {
        return Status::Unsupported("PosixStore::Dump is not implemented");
    }

    std::vector<std::pair<std::size_t, bool>> GetFinished() override { return {}; }

    std::vector<std::pair<std::size_t, bool>> Wait(
        const std::vector<std::size_t>& /*jobIds*/) override
    {
        return {};
    }
};

}  // namespace UC::Store
