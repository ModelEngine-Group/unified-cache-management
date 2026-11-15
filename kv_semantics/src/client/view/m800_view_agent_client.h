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
#include <memory>
#include <string>
#include <vector>
#include "view/m800_view_common.h"

namespace kv {

// C++ representation of kv_client_agent_node_view after ownership has been
// moved out of the C boundary.
struct M800AgentNodeView {
    std::int32_t osdId{-1};
    bool up{false};
    bool in{false};
    std::vector<std::string> addrs;
};

// Process-local boundary between M800ViewMasterServer and libkv_agent.so.
// The default implementation wraps dlopen/dlsym and the C ABI in Shard 3;
// tests and other masters can inject a fake implementation.
class M800ViewAgentClient {
public:
    virtual ~M800ViewAgentClient() = default;

    // Calls kv_client_agent_init with the config captured by the factory.
    // This establishes the Ceph connection and authenticates the client.
    virtual Status Initialize() = 0;
    // Calls kv_client_agent_start and starts the agent-internal refresher.
    virtual Status Start() = 0;
    // Calls kv_client_agent_stop. Destroy and dlclose remain RAII-owned by
    // the implementation and must execute after Stop.
    virtual void Stop() = 0;

    virtual Status GetEpoch(std::uint64_t& epoch) const = 0;
    virtual Status GetAllNsNames(std::vector<std::string>& nsNames) const = 0;
    virtual Status GetViewByNs(const std::string& nsName, bool up, bool in,
                               std::vector<M800AgentNodeView>& views) const = 0;
};

// Creates a configured agent client. The default factory dlopens
// agentLibPath with RTLD_NOW | RTLD_LOCAL, resolves all required C symbols,
// checks KV_CLIENT_AGENT_ABI_VERSION, and creates the opaque handle. It does
// not initialize or start the agent.
class M800ViewAgentClientFactory {
public:
    virtual ~M800ViewAgentClientFactory() = default;

    virtual Status Create(const M800ViewServerConfig& config,
                          std::unique_ptr<M800ViewAgentClient>& client) const = 0;
};

}  // namespace kv
