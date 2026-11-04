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
#include <cstdint>
#include <vector>
#include "strategy.h"
#include "strategy_registry.h"

namespace UC::Proxy {
namespace {

class TtlRoutingStrategy : public RoutingStrategy {
public:
    Status Setup(const Dictionary& config, std::size_t branchCount) override
    {
        std::vector<std::int64_t> thresholds;
        if (!config.GetNumbers<std::int64_t>("thresholds", thresholds) || thresholds.empty()) {
            return Status::InvalidParam("ttl: 'thresholds' must be a non-empty int array");
        }
        for (std::size_t i = 1; i < thresholds.size(); i++) {
            if (thresholds[i - 1] <= thresholds[i]) {
                return Status::InvalidParam("ttl: 'thresholds' must be strictly descending");
            }
        }
        if (branchCount != thresholds.size() + 1) {
            return Status::InvalidParam(
                "ttl: {} thresholds carve {} segments; the peer has {} branches", thresholds.size(),
                thresholds.size() + 1, branchCount);
        }
        thresholds_ = std::move(thresholds);
        return Status::Ok();
    }

    std::size_t Select(const RequestContext& request) override
    {
        if (!request.ttl) { return thresholds_.size(); }
        for (std::size_t i = 0; i < thresholds_.size(); i++) {
            if (request.ttl->count() > thresholds_[i]) { return i; }
        }
        return thresholds_.size();
    }

private:
    std::vector<std::int64_t> thresholds_;
};

UC_PROXY_REGISTER_ROUTING_STRATEGY(TtlRoutingStrategy, "ttl");

}  // namespace
}  // namespace UC::Proxy
