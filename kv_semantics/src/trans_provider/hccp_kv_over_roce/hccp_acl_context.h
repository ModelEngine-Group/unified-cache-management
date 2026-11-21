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
#include <string>
#include "kv_types.h"

#ifdef UCM_ASU_ENABLE_HCCP_KV_OVER_ROCE_PROVIDER

#include <acl/acl.h>

namespace kv {

// ACL device/context the provider binds around every operation. The device id
// comes from TransportConfig::deviceId, which the client keeps consistent
// between a transport and its provider.
struct HccpAclBinding {
    std::int32_t deviceId{-1};
    // Written once by ResolveHccpAclBinding, before the provider is shared with
    // any other thread, and only read afterwards.
    aclrtContext context{nullptr};
    std::string source;
};

// Resolves the provider device and captures its ACL context. The calling
// thread's previous context is restored before returning. Fails when no context
// can be captured, so a resolved binding always has a non-null context.
Status ResolveHccpAclBinding(const TransportConfig& config, HccpAclBinding& binding);

// Binds an already resolved provider context for one provider entry point and
// restores the caller's context on exit. A null binding context is rejected;
// only Capture() populates it.
class ScopedHccpAclContext final {
public:
    ScopedHccpAclContext(const char* stage, HccpAclBinding& binding);
    ~ScopedHccpAclContext();

    // One-time initial bind: switches to binding.deviceId and captures the
    // resulting context into the binding, failing when none can be captured.
    // Must run before the binding is shared with other threads.
    static Status Capture(const char* stage, HccpAclBinding& binding);

    ScopedHccpAclContext(const ScopedHccpAclContext&) = delete;
    ScopedHccpAclContext& operator=(const ScopedHccpAclContext&) = delete;
    ScopedHccpAclContext(ScopedHccpAclContext&&) = delete;
    ScopedHccpAclContext& operator=(ScopedHccpAclContext&&) = delete;

    const Status& status() const { return status_; }

private:
    struct InitialBind {};
    ScopedHccpAclContext(const char* stage, HccpAclBinding& binding, InitialBind);
    Status Bind();
    Status BindInitial();

    const char* stage_{nullptr};
    HccpAclBinding& binding_;
    aclrtContext callerContext_{nullptr};
    std::int32_t callerDevice_{-1};
    bool contextChanged_{false};
    Status status_;
};

}  // namespace kv

#endif  // UCM_ASU_ENABLE_HCCP_KV_OVER_ROCE_PROVIDER
