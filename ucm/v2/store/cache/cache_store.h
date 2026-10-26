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

#include "prefetchable.h"
#include "store.h"
#include "token_addressed.h"

namespace UC::Store {

class CacheStore : public Store, public TokenAddressed, public Prefetchable {
public:
    Status Setup(const Dictionary& /*config*/) override
    {
        return Status::Unsupported("CacheStore::Setup is not implemented");
    }

    Expected<std::vector<std::uint8_t>> Lookup(const std::vector<BlockId>& /*blockIds*/) override
    {
        return Status::Unsupported("CacheStore::Lookup is not implemented");
    }

    using Store::LookupOnPrefix;

    using Store::LookupOnReverse;

    void Touch(const std::vector<BlockId>& /*blockIds*/, const RequestContext& /*context*/) override
    {
    }

    Status Load(std::size_t /*jobId*/, JobDescriptor /*job*/) override
    {
        return Status::Unsupported("CacheStore::Load is token-bearing only (TokenAddressed::Load)");
    }

    Status Dump(std::size_t /*jobId*/, JobDescriptor /*job*/) override
    {
        return Status::Unsupported("CacheStore::Dump is token-bearing only (TokenAddressed::Dump)");
    }

    std::vector<std::pair<std::size_t, bool>> GetFinished() override { return {}; }

    std::vector<std::pair<std::size_t, bool>> Wait(
        const std::vector<std::size_t>& /*jobIds*/) override
    {
        return {};
    }

    Prefetchable* AsPrefetchable() override { return this; }

    TokenAddressed* AsTokenAddressed() override { return this; }

    /*== TokenAddressed extension ==*/

    Expected<AddressedToken> Pin(const std::vector<BlockId>& /*blockIds*/) override
    {
        return Status::Unsupported("CacheStore::Pin is not implemented");
    }

    void Unpin(const std::vector<BlockId>& /*blockIds*/) override {}

    Expected<AddressedToken> Alloc(const std::vector<BlockId>& /*blockIds*/,
                                   const RequestContext& /*context*/) override
    {
        return Status::Unsupported("CacheStore::Alloc is not implemented");
    }

    void Commit(const std::vector<BlockId>& /*blockIds*/, bool /*success*/) override {}

    Status Load(std::size_t /*jobId*/, JobDescriptor /*job*/, AddressedToken /*token*/) override
    {
        return Status::Unsupported("CacheStore::Load is not implemented");
    }

    Status Dump(std::size_t /*jobId*/, JobDescriptor /*job*/, AddressedToken /*token*/) override
    {
        return Status::Unsupported("CacheStore::Dump is not implemented");
    }

    /*== Prefetchable extension ==*/

    Expected<std::optional<std::vector<std::uint8_t>>> Lookup(
        const RequestId& /*requestId*/, const std::vector<BlockId>& /*blockIds*/) override
    {
        return Status::Unsupported("CacheStore::Lookup is not implemented");
    }

    using Prefetchable::LookupOnPrefix;

    using Prefetchable::LookupOnReverse;

    Expected<AddressedToken> PreparePrefetch(const RequestId& /*requestId*/,
                                             const RequestContext& /*context*/) override
    {
        return Status::Unsupported("CacheStore::PreparePrefetch is not implemented");
    }

    Status CompletePrefetch(const AddressedToken& /*plan*/, bool /*success*/) override
    {
        return Status::Unsupported("CacheStore::CompletePrefetch is not implemented");
    }

    Status Prefetch(std::size_t /*jobId*/, AddressedToken /*plan*/) override
    {
        return Status::Unsupported("CacheStore::Prefetch is not implemented");
    }
};

}  // namespace UC::Store
