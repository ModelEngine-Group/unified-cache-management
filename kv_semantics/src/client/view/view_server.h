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

#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>
#include "kv_transport.h"

namespace kv {

struct AsuClientConfig;

// GlobalView carries the routing membership and view metadata.
struct GlobalView {
    std::uint64_t viewEpoch{0};
    std::uint64_t viewId{0};
    std::unordered_map<AsuId, AsuInfo> asuMap;
    std::uint64_t createTimeMs{0};
    std::uint64_t expireTimeMs{0};
};

// ViewServer owns global view fetching and refresh decisions.
class ViewServer {
public:
    // Destroys the view server interface.
    virtual ~ViewServer() = default;
    // Fetches the current global view.
    virtual Status GetGlobalView(GlobalView& view) = 0;
    // Returns whether a fetched view should replace the published view.
    virtual bool ShouldPublishView(const GlobalView& publishedView,
                                   const GlobalView& fetchedView) const;
    // Returns whether an operation status should schedule view refresh.
    virtual bool ShouldRefreshView(const Status& status) const;
    // Returns whether any task status should schedule view refresh.
    virtual bool ShouldRefreshView(const TaskResult& result) const;
};

using ViewServerFactory = std::function<std::shared_ptr<ViewServer>(const AsuClientConfig&)>;

std::shared_ptr<ViewServer> CreateDefaultViewServer(const AsuClientConfig& config);
GlobalView BuildConfigGlobalView(const AsuClientConfig& config);

}  // namespace kv
