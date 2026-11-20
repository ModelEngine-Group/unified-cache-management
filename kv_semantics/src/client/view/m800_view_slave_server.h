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

#include "view/view_server.h"
#include "view/view_universe.h"

namespace kv {

// Slave-side service boundary. It fetches the text protocol from the master
// and exposes the decoded universe.
class M800ViewSlaveServer : public ViewServer {
public:
    // Initializes the master HTTP client and performs the initial universe
    // fetch. Start must fail if the initial fetch cannot complete; later
    // refreshes are controlled by RefreshScheduler.
    virtual Status Start() = 0;

    // Stops fetch activity. Stop is idempotent and safe to call when Start
    // failed.
    virtual void Stop() = 0;

    // M800 slaves must decode the master's full universe.
    virtual Status GetAllViews(ViewUniverse& universe) = 0;

    // M800 serves multiple namespaces, so the legacy single-namespace entry
    // point is intentionally unsupported. Use GetAllViews instead; this method
    // must not modify view.
    virtual Status GetGlobalView(GlobalView& view) override
    {
        (void)view;
        return Status::Error(StatusCode::UNSUPPORTED,
                             "M800 view slave serves multiple namespaces; use GetAllViews");
    }
};

}  // namespace kv
