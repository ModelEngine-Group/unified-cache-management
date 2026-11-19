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

#include <memory>
#include <unordered_map>
#include "view/m800_view_common.h"
#include "view/view_server.h"

namespace kv {

// Slave-side service boundary. It fetches the text protocol from the master
// and exposes the decoded universe. Epoch acceptance is owned exclusively by
// ViewUniverse::PublishAll; the slave keeps no independent epoch cache.
class M800ViewSlaveServer : public ViewServer {
public:
    // Initializes the master HTTP client and performs the initial universe
    // fetch. Start must fail if the initial fetch cannot complete.
    virtual Status Start() = 0;

    // M800 slaves must decode and return the complete multi-namespace view
    // map. The output is left unchanged when fetching fails.
    virtual Status GetAllViews(std::unordered_map<KvNsId, GlobalView>& views) override = 0;
    // Fetches one namespace view; this is the single-namespace query entry
    // point for a multi-namespace service.
    virtual Status GetGlobalViewByKvnsId(KvNsId nsId, GlobalView& view) override = 0;
    // M800 serves multiple namespaces; the legacy single-view entry point is
    // intentionally unsupported. Callers must use GetAllViews.
    virtual Status GetGlobalView(GlobalView& view) override = 0;

    virtual void Stop() = 0;
};

std::shared_ptr<M800ViewSlaveServer> CreateM800ViewSlaveServer(const M800ViewServerConfig& config);

}  // namespace kv
