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
#include "hccp_acl_context.h"

#ifdef UCM_ASU_ENABLE_HCCP_KV_OVER_ROCE_PROVIDER

#include <limits>
#include <utility>
#include "logger.h"
#include "parser_common.h"

namespace kv {
namespace {

Status AclError(const char* stage, const std::string& operation, aclError ret)
{
    std::string message = std::string("HccpKvOverRoceTransProvider: ") + stage + " " + operation +
                          " failed ret=" + std::to_string(static_cast<int>(ret));
    if (const char* recent = aclGetRecentErrMsg(); recent != nullptr && recent[0] != '\0') {
        message += " msg=";
        message += recent;
    }
    KV_ERROR("{}", message);
    return Status::Error(StatusCode::INTERNAL_ERROR, std::move(message));
}

std::int32_t ResolveConfiguredDevice(const TransportConfig& config)
{
    if (config.deviceId >= 0) { return config.deviceId; }

    const auto configured = GetConfigAttr(config, {"device_id", "deviceId", "logical_device_id"});
    if (configured.empty()) { return -1; }

    const auto parsed = ParseConfigUint32(configured, 0);
    if (parsed > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        return -1;
    }
    return static_cast<std::int32_t>(parsed);
}

}  // namespace

Status ResolveHccpAclBinding(const TransportConfig& config, HccpAclBinding& binding)
{
    binding = HccpAclBinding{};
    binding.deviceId = ResolveConfiguredDevice(config);
    binding.source = config.deviceId >= 0 ? "config.deviceId" : "config.attr";
    if (binding.deviceId < 0) {
        std::int32_t currentDevice = -1;
        if (aclrtGetDevice(&currentDevice) == ACL_SUCCESS && currentDevice >= 0) {
            binding.deviceId = currentDevice;
            binding.source = "current_acl_device";
        } else {
            binding.deviceId = 0;
            binding.source = "default_device_0";
            KV_WARN(
                "HccpKvOverRoceTransProvider: no device id configured and no current ACL device; "
                "falling back to device 0");
        }
    }

    ScopedHccpAclContext scope("ResolveDevice", binding);
    if (!scope.status().ok()) { return scope.status(); }

    KV_INFO("HccpKvOverRoceTransProvider: ACL binding device_id={} context={} source={}",
            binding.deviceId, static_cast<const void*>(binding.context), binding.source);
    return Status::OK();
}

ScopedHccpAclContext::ScopedHccpAclContext(const char* stage, HccpAclBinding& binding)
    : stage_(stage), binding_(binding)
{
    status_ = Bind();
}

ScopedHccpAclContext::~ScopedHccpAclContext()
{
    if (!contextChanged_ || callerContext_ == nullptr) { return; }

    const auto ret = aclrtSetCurrentContext(callerContext_);
    if (ret != ACL_SUCCESS) {
        KV_WARN(
            "HccpKvOverRoceTransProvider: failed to restore caller ACL context stage={} "
            "context={} device_id={} ret={}",
            stage_, static_cast<const void*>(callerContext_), callerDevice_, static_cast<int>(ret));
    }
}

Status ScopedHccpAclContext::Bind()
{
    if (binding_.deviceId < 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "HccpKvOverRoceTransProvider: no ACL device selected");
    }

    const auto getContextRet = aclrtGetCurrentContext(&callerContext_);
    if (getContextRet != ACL_SUCCESS || callerContext_ == nullptr) { callerContext_ = nullptr; }

    if (binding_.context != nullptr) {
        if (callerContext_ == binding_.context) { return Status::OK(); }

        const auto ret = aclrtSetCurrentContext(binding_.context);
        if (ret != ACL_SUCCESS) {
            return AclError(stage_, "aclrtSetCurrentContext", ret);
        }
        contextChanged_ = true;
        return Status::OK();
    }

    // No context captured yet. Bind the device once and remember the resulting
    // context so later calls only need aclrtSetCurrentContext.
    if (aclrtGetDevice(&callerDevice_) != ACL_SUCCESS) { callerDevice_ = -1; }
    if (callerDevice_ != binding_.deviceId || callerContext_ == nullptr) {
        const auto initRet = aclInit(nullptr);
        if (initRet != ACL_SUCCESS && initRet != ACL_ERROR_REPEAT_INITIALIZE) {
            return AclError(stage_, "aclInit", initRet);
        }

        const auto setRet = aclrtSetDevice(binding_.deviceId);
        if (setRet != ACL_SUCCESS) {
            return AclError(stage_, "aclrtSetDevice device_id=" + std::to_string(binding_.deviceId),
                            setRet);
        }
        contextChanged_ = true;
    }

    aclrtContext context = nullptr;
    if (aclrtGetCurrentContext(&context) == ACL_SUCCESS && context != nullptr) {
        binding_.context = context;
    }
    return Status::OK();
}

}  // namespace kv

#endif  // UCM_ASU_ENABLE_HCCP_KV_OVER_ROCE_PROVIDER
