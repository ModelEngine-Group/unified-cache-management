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
#include "protocols/hixl/hixl_instance.h"
#include <utility>
#include "common/acl_runtime_context.h"
#include "hixl/hixl.h"
#include "logger/logger.h"

namespace transport {

HixlInstance::HixlInstance(Endpoint local_endpoint, int32_t device_id)
    : local_endpoint_(std::move(local_endpoint)), device_id_(device_id)
{
}

HixlInstance::~HixlInstance() { Finalize(); }

Status HixlInstance::Initialize(const std::map<std::string, std::string>& options)
{
    if (engine_) { return Status::OK(); }
    const auto set_device_status = aclrtSetDevice(device_id_);
    if (set_device_status != ACL_ERROR_NONE) {
        return Status::Error(fmt::format("aclrtSetDevice({}) returned {}", device_id_,
                                         static_cast<int>(set_device_status)));
    }
    const auto context_status = aclrtGetCurrentContext(&context_);
    if (context_status != ACL_ERROR_NONE || context_ == nullptr) {
        return Status::Error(fmt::format("aclrtGetCurrentContext(device={}) returned {}",
                                         device_id_, static_cast<int>(context_status)));
    }
    const auto physical_status = aclrtGetPhyDevIdByLogicDevId(device_id_, &physical_device_id_);
    if (physical_status != ACL_ERROR_NONE) {
        return Status::Error(fmt::format("aclrtGetPhyDevIdByLogicDevId({}) returned {}", device_id_,
                                         static_cast<int>(physical_status)));
    }

    std::map<hixl::AscendString, hixl::AscendString> hixl_options;
    for (const auto& item : options) {
        hixl_options.emplace(item.first.c_str(), item.second.c_str());
    }
    auto engine = std::make_unique<hixl::Hixl>();
    const auto local_engine = local_endpoint_.ToString();
    const auto status = engine->Initialize(local_engine.c_str(), hixl_options);
    if (status != hixl::SUCCESS) {
        return Status::Error(
            fmt::format("Initialize(\"{}\") returned {}", local_engine, static_cast<int>(status)));
    }
    engine_ = std::move(engine);
    UC_DEBUG(
        "[Transport][HIXL] instance initialized: engine={} logical_device={} physical_device={}",
        local_engine, device_id_, physical_device_id_);
    return Status::OK();
}

void HixlInstance::Finalize()
{
    if (!engine_) { return; }
    WithAclRuntimeContext context(context_);
    (void)engine_->Finalize();
    engine_.reset();
    context_ = nullptr;
}

hixl::Hixl& HixlInstance::Engine() { return *engine_; }

aclrtContext HixlInstance::Context() const { return context_; }

const Endpoint& HixlInstance::LocalEndpoint() const { return local_endpoint_; }

int32_t HixlInstance::LogicalDeviceId() const { return device_id_; }

int32_t HixlInstance::PhysicalDeviceId() const { return physical_device_id_; }

}  // namespace transport
