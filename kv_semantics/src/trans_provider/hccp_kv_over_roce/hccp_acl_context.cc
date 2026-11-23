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

// ParseConfigUint64 falls back on syntax errors and out-of-range values. This
// sentinel is far outside every range the provider accepts, so a fallback
// result unambiguously means "the configured value is not a valid number".
constexpr std::uint64_t kInvalidConfigValue = std::numeric_limits<std::uint64_t>::max();

std::int32_t ResolveConfiguredDevice(const TransportConfig& config)
{
    if (config.deviceId >= 0) { return config.deviceId; }

    const auto configured = GetConfigAttr(config, {"device_id", "deviceId", "logical_device_id"});
    if (configured.empty()) { return -1; }

    const auto parsed = ParseConfigUint64(configured, kInvalidConfigValue);
    if (parsed == kInvalidConfigValue ||
        parsed > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        KV_WARN(
            "HccpKvOverRoceTransProvider: ignoring invalid device_id attribute value=\"{}\" "
            "(expected an integer in [0, {}]); treating it as unconfigured",
            configured, static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()));
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

    const auto captureStatus = ScopedHccpAclContext::Capture("ResolveDevice", binding);
    if (!captureStatus.ok()) { return captureStatus; }

    KV_INFO("HccpKvOverRoceTransProvider: ACL binding device_id={} context={} source={}",
            binding.deviceId, static_cast<const void*>(binding.context), binding.source);
    return Status::OK();
}

ScopedHccpAclContext::ScopedHccpAclContext(const char* stage, HccpAclBinding& binding)
    : stage_(stage), binding_(binding)
{ status_ = Bind(); }

ScopedHccpAclContext::ScopedHccpAclContext(const char* stage, HccpAclBinding& binding, InitialBind)
    : stage_(stage), binding_(binding)
{ status_ = BindInitial(); }

Status ScopedHccpAclContext::Capture(const char* stage, HccpAclBinding& binding)
{
    ScopedHccpAclContext scope(stage, binding, InitialBind{});
    return scope.status();
}

ScopedHccpAclContext::~ScopedHccpAclContext()
{
    // BindInitial may have switched the thread's device. Restore it first so the
    // caller's device/context pair is coherent again; aclrtSetCurrentContext can
    // then restore a custom caller context. A thread that had no ACL device when
    // the binding was captured cannot be unbound, because ACL has no such API:
    // it stays on the provider device.
    if (deviceChanged_ && callerDevice_ >= 0 && callerDevice_ != binding_.deviceId) {
        const auto ret = aclrtSetDevice(callerDevice_);
        if (ret != ACL_SUCCESS) {
            KV_WARN(
                "HccpKvOverRoceTransProvider: failed to restore caller ACL device stage={} "
                "device_id={} ret={}",
                stage_, callerDevice_, static_cast<int>(ret));
        }
    }

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
    if (binding_.context == nullptr) {
        return Status::Error(StatusCode::NOT_INITIALIZED,
                             "HccpKvOverRoceTransProvider: ACL context is not resolved "
                             "device_id=" +
                                 std::to_string(binding_.deviceId));
    }

    if (aclrtGetCurrentContext(&callerContext_) != ACL_SUCCESS) { callerContext_ = nullptr; }
    if (aclrtGetDevice(&callerDevice_) != ACL_SUCCESS) { callerDevice_ = -1; }

    if (callerContext_ == binding_.context) { return Status::OK(); }

    const auto ret = aclrtSetCurrentContext(binding_.context);
    if (ret != ACL_SUCCESS) { return AclError(stage_, "aclrtSetCurrentContext", ret); }
    contextChanged_ = true;
    return Status::OK();
}

Status ScopedHccpAclContext::BindInitial()
{
    if (binding_.deviceId < 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "HccpKvOverRoceTransProvider: no ACL device selected");
    }
    if (binding_.context != nullptr) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "HccpKvOverRoceTransProvider: ACL context is already resolved");
    }

    // No context captured yet. Bind the device once and capture the resulting
    // context; the binding is immutable after this point.
    if (aclrtGetCurrentContext(&callerContext_) != ACL_SUCCESS) { callerContext_ = nullptr; }
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
        // Remember what the capture changed so the destructor can restore it.
        deviceChanged_ = callerDevice_ != binding_.deviceId;
        contextChanged_ = true;
    }

    aclrtContext context = nullptr;
    const auto getRet = aclrtGetCurrentContext(&context);
    if (getRet != ACL_SUCCESS) { return AclError(stage_, "aclrtGetCurrentContext", getRet); }
    if (context == nullptr) {
        return Status::Error(StatusCode::INTERNAL_ERROR,
                             "HccpKvOverRoceTransProvider: no ACL context after binding "
                             "device_id=" +
                                 std::to_string(binding_.deviceId));
    }
    binding_.context = context;
    return Status::OK();
}

}  // namespace kv

#endif  // UCM_ASU_ENABLE_HCCP_KV_OVER_ROCE_PROVIDER
