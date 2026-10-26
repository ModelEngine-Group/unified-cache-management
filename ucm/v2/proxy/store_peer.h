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

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>
#include "routing/strategy.h"
#include "store_entry.h"
#include "store_scheduler.h"
#include "store_worker.h"
#include "type/request.h"

namespace UC::Proxy {

class StorePeer : public StoreScheduler, public StoreWorker {
public:
    StorePeer(std::vector<std::unique_ptr<StoreEntry>> branches,
              std::unique_ptr<RoutingStrategy> routing);

    /*== Scheduler ==*/
    Status OnNewRequest(const RequestContext& request) override;
    void OnRequestFinished(const RequestId& requestId) override;
    Expected<std::optional<std::vector<std::uint8_t>>> Lookup(
        const RequestId& requestId, const std::vector<BlockId>& keys) override;
    Expected<std::optional<std::int64_t>> LookupOnPrefix(const RequestId& requestId,
                                                         const std::vector<BlockId>& keys) override;
    Expected<std::optional<std::int64_t>> LookupOnReverse(
        const RequestId& requestId, const std::vector<BlockId>& keys) override;
    void Touch(const RequestId& requestId, const std::vector<BlockId>& keys) override;
    Expected<AddressedToken> PreparePrefetch(const RequestId& requestId) override;
    void CompletePrefetch(const RequestId& requestId, bool success) override;
    Expected<AddressedToken> PrepareGet(const RequestId& requestId,
                                        const std::vector<BlockId>& keys) override;
    void CompleteGet(const RequestId& requestId) override;
    Expected<AddressedToken> PreparePut(const RequestId& requestId,
                                        const std::vector<BlockId>& keys) override;
    void CompletePut(const RequestId& requestId, bool success) override;

    /*== Worker ==*/
    Status SubmitPrefetch(std::size_t jobId, AddressedToken token) override;
    Status SubmitGet(std::size_t jobId, std::vector<BlockPayload> payloads, AddressedToken token,
                     const IoContext& ctx) override;
    Status SubmitPut(std::size_t jobId, std::vector<BlockPayload> payloads, AddressedToken token,
                     const IoContext& ctx) override;
    std::vector<IoResult> GetFinished() override;
    std::vector<IoResult> Wait(const std::vector<std::size_t>& jobIds) override;

private:
    struct RequestState {
        std::size_t branch = 0;
    };

    Expected<RequestState*> RequireRequest(const RequestId& requestId);

    std::vector<std::unique_ptr<StoreEntry>> branches_;
    std::unique_ptr<RoutingStrategy> routing_;
    std::unordered_map<RequestId, RequestState> requests_;
    std::unordered_map<std::size_t, std::size_t> inFlightJobs_;
    /* In-flight job count per branch; drives GetFinished's branch skipping. */
    std::vector<std::size_t> inFlightPerBranch_;
};

}  // namespace UC::Proxy
