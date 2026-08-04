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

#include "src/ub/raii.h"
#include <cstdint>
#include <sys/types.h>
#include "src/runtime/hccp_v2_loader.h"
#include "src/ub/log.h"

namespace umc::comm {

namespace {

using HandleDisposer = UbErrorCode (*)(HandleKind, void*, const HandleAssoc&);

UbErrorCode DefaultHccpV2Disposer(HandleKind kind, void* handle, const HandleAssoc& assoc)
{
    using V2 = ::umc::comm::v2::DlHccpV2Api;
    if (handle == nullptr) return UbErrorCode::Ok;

    int rc = 0;
    switch (kind) {
        case HandleKind::kCtx: rc = V2::RaCtxDeinit(handle); break;
        case HandleKind::kQp: rc = V2::RaCtxQpDestroy(handle); break;
        case HandleKind::kCq: rc = V2::RaCtxCqDestroy(assoc.ctxHandle, handle); break;
        case HandleKind::kChan: rc = V2::RaCtxChanDestroy(assoc.ctxHandle, handle); break;
        case HandleKind::kLmem: rc = V2::RaCtxLmemUnregister(assoc.ctxHandle, handle); break;
        case HandleKind::kRmem: rc = V2::RaCtxRmemUnimport(assoc.ctxHandle, handle); break;
        case HandleKind::kRemQp: rc = V2::RaCtxQpUnimport(assoc.ctxHandle, handle); break;
        case HandleKind::kTokenId: rc = V2::RaCtxTokenIdFree(assoc.ctxHandle, handle); break;
        case HandleKind::kTsdProc: {
            const auto subPid = static_cast<pid_t>(reinterpret_cast<uintptr_t>(handle));
            if (subPid <= 0) {
                UB_LOG_ERROR("HCCP V2 dispose FAILED: invalid TSD subPid={} device={}",
                             static_cast<int>(subPid), assoc.logicDeviceId);
                return UbErrorCode::HccpV2HandleInvalid;
            }
            uint32_t r = V2::TsdProcessClose(static_cast<uint32_t>(assoc.logicDeviceId), subPid);
            rc = static_cast<int>(r);
            break;
        }
        case HandleKind::kInvalid:
        default:
            UB_LOG_WARN("HccpV2Handle::Reset called with invalid kind={}", static_cast<int>(kind));
            return UbErrorCode::HccpV2HandleInvalid;
    }

    if (rc != 0) {
        UB_LOG_ERROR(
            "HCCP V2 dispose FAILED (resource may leak on device): "
            "kind={} ctx={} handle={} rc={}",
            static_cast<int>(kind), assoc.ctxHandle, handle, rc);
        return UbErrorCode::HccpV2HandleInvalid;
    }
    return UbErrorCode::Ok;
}

HandleDisposer& DisposerRef()
{
    static HandleDisposer d = &DefaultHccpV2Disposer;
    return d;
}

}  // namespace

UbErrorCode HccpV2Handle::Reset()
{
    if (handle_ == nullptr) return UbErrorCode::Ok;
    UbErrorCode r = DisposerRef()(kind_, handle_, assoc_);
    if (r == UbErrorCode::Ok) {
        handle_ = nullptr;
        kind_ = HandleKind::kInvalid;
    }
    return r;
}

}  // namespace umc::comm
