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
#include <utility>
#include "src/protocol/ub_error_code.h"

namespace umc::comm {

enum class HandleKind : int32_t {
    kInvalid = 0,
    kCtx = 1,      // RaCtxInit ↔ RaCtxDeinit
    kQp = 2,       // RaCtxQpCreate ↔ RaCtxQpDestroy
    kCq = 3,       // RaCtxCqCreate ↔ RaCtxCqDestroy
    kChan = 4,     // RaCtxChanCreate ↔ RaCtxChanDestroy
    kLmem = 5,     // RaCtxLmemRegister ↔ RaCtxLmemUnregister
    kRmem = 6,     // RaCtxRmemImport ↔ RaCtxRmemUnimport
    kRemQp = 7,    // RaCtxQpImport ↔ RaCtxQpUnimport
    kTokenId = 8,  // RaCtxTokenIdAlloc ↔ RaCtxTokenIdFree
    kTsdProc = 9,  // TsdProcessOpen ↔ TsdProcessClose
};

struct HandleAssoc {
    void* ctxHandle{nullptr};
    int32_t logicDeviceId{-1};
};

class HccpV2Handle {
public:
    HccpV2Handle() = default;
    HccpV2Handle(HandleKind kind, void* h, HandleAssoc assoc)
        : kind_(kind), handle_(h), assoc_(assoc)
    {
    }

    HccpV2Handle(const HccpV2Handle&) = delete;
    HccpV2Handle& operator=(const HccpV2Handle&) = delete;

    HccpV2Handle(HccpV2Handle&& o) noexcept { Steal(std::move(o)); }
    HccpV2Handle& operator=(HccpV2Handle&& o) noexcept = delete;

    ~HccpV2Handle() { (void)Dispose(); }

    void* Raw() const { return handle_; }
    HandleKind Kind() const { return kind_; }
    explicit operator bool() const { return handle_ != nullptr; }

    UbErrorCode Reset();

    UbErrorCode Adopt(HccpV2Handle&& other) noexcept
    {
        if (handle_ != nullptr) { return UbErrorCode::HccpV2HandleInvalid; }
        Steal(std::move(other));
        return UbErrorCode::Ok;
    }

    void* Release()
    {
        void* h = handle_;
        handle_ = nullptr;
        kind_ = HandleKind::kInvalid;
        return h;
    }

private:
    UbErrorCode Dispose();
    void Steal(HccpV2Handle&& o)
    {
        kind_ = o.kind_;
        handle_ = o.handle_;
        assoc_ = o.assoc_;
        o.kind_ = HandleKind::kInvalid;
        o.handle_ = nullptr;
    }

    HandleKind kind_{HandleKind::kInvalid};
    void* handle_{nullptr};
    HandleAssoc assoc_{};
};

using CtxHandleRAII = HccpV2Handle;
using QpHandleRAII = HccpV2Handle;
using CqHandleRAII = HccpV2Handle;
using ChanHandleRAII = HccpV2Handle;
using LmemHandleRAII = HccpV2Handle;
using RmemHandleRAII = HccpV2Handle;
using RemQpHandleRAII = HccpV2Handle;
using TokenIdHandleRAII = HccpV2Handle;
using TsdProcRAII = HccpV2Handle;

}  // namespace umc::comm
