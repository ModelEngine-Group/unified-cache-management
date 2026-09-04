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
#include <map>
#include <memory>
#include <string>
#include "acl/acl.h"
#include "core/transport.h"

namespace hixl {
class Hixl;
}

namespace transport {

// Owns the resources of one HIXL engine. Calls into the engine are made by HixlTransport.
class HixlInstance final {
public:
    HixlInstance(Endpoint local_endpoint, int32_t device_id);
    ~HixlInstance();

    HixlInstance(const HixlInstance&) = delete;
    HixlInstance& operator=(const HixlInstance&) = delete;

    Status Initialize(const std::map<std::string, std::string>& options);
    void Finalize();

    hixl::Hixl& Engine();
    aclrtContext Context() const;
    const Endpoint& LocalEndpoint() const;
    int32_t LogicalDeviceId() const;
    int32_t PhysicalDeviceId() const;

private:
    Endpoint local_endpoint_;
    int32_t device_id_ = -1;
    int32_t physical_device_id_ = -1;
    aclrtContext context_ = nullptr;
    std::unique_ptr<hixl::Hixl> engine_;
};

}  // namespace transport
