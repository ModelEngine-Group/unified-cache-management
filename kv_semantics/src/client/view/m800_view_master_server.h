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

#include "view/m800_view_agent_client.h"
#include "view/view_http_server.h"
#include "view/view_server.h"
#include "view/view_universe.h"

namespace kv {

// Master-side service boundary. It loads the Ceph agent, converts Ceph views
// to ViewUniverse, and serves the read-only HTTP contract.
class M800ViewMasterServer : public ViewServer {
public:
    // Creates and initializes the agent client, starts the agent, fetches the
    // initial universe, then starts the read-only HTTP service. HTTP must not
    // accept Worker requests before the initial universe is available.
    virtual Status Start() = 0;
    // Stops the HTTP service first, then stops the agent client. Stop is
    // idempotent and safe to call when Start failed.
    virtual void Stop() = 0;

    // M800 masters must return a real multi-namespace universe.
    virtual Status GetAllViews(ViewUniverse& universe) = 0;
};

}  // namespace kv
