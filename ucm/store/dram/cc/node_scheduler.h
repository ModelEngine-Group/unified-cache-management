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
#ifndef UNIFIEDCACHE_DRAM_STORE_CC_NODE_SCHEDULER_H
#define UNIFIEDCACHE_DRAM_STORE_CC_NODE_SCHEDULER_H

#include <atomic>
#include <memory>
#include <unordered_map>
#include <vector>
#include "messages.h"
#include "status/status.h"

namespace UC::Dram {

class NodeScheduler final {
public:
    NodeScheduler(NodeSchedulerConfig config, NodeDependencies dependencies);
    ~NodeScheduler();

    NodeScheduler(const NodeScheduler&) = delete;
    NodeScheduler& operator=(const NodeScheduler&) = delete;

    Status Start();
    void Shutdown();

    // Consumes on success during runtime.
    Status Post(Request& request);
    // Reliable during runtime. Events concurrent with terminal Shutdown may be discarded.
    void Publish(NodeId nodeId, NodeEvent event);

private:
    struct Runner;

    Runner& GetRunner(NodeId nodeId) const noexcept;
    void RunActors(Runner& runner) noexcept;
    void JoinAll();

    NodeSchedulerConfig config_;
    NodeDependencies dependencies_;
    std::unordered_map<NodeId, Runner*> nodes_;
    std::vector<std::unique_ptr<Runner>> runners_;
    std::atomic<bool> acceptingMessages_{false};
};

}  // namespace UC::Dram

#endif  // UNIFIEDCACHE_DRAM_STORE_CC_NODE_SCHEDULER_H
