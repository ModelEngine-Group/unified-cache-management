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
    aclrtContext context{nullptr};
    std::string source;
};

// Resolves the provider device and captures its ACL context. The calling
// thread's previous device/context is restored before returning.
Status ResolveHccpAclBinding(const TransportConfig& config, HccpAclBinding& binding);

// Binds the provider context for one provider entry point and restores the
// caller's context on exit. A null binding context falls back to
// aclrtSetDevice and captures the resulting context for later calls.
class ScopedHccpAclContext final {
public:
    ScopedHccpAclContext(const char* stage, HccpAclBinding& binding);
    ~ScopedHccpAclContext();

    ScopedHccpAclContext(const ScopedHccpAclContext&) = delete;
    ScopedHccpAclContext& operator=(const ScopedHccpAclContext&) = delete;
    ScopedHccpAclContext(ScopedHccpAclContext&&) = delete;
    ScopedHccpAclContext& operator=(ScopedHccpAclContext&&) = delete;

    const Status& status() const { return status_; }

private:
    Status Bind();

    const char* stage_{nullptr};
    HccpAclBinding& binding_;
    aclrtContext callerContext_{nullptr};
    std::int32_t callerDevice_{-1};
    bool contextChanged_{false};
    Status status_;
};

}  // namespace kv

#endif  // UCM_ASU_ENABLE_HCCP_KV_OVER_ROCE_PROVIDER
