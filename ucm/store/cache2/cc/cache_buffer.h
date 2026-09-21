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

#include <atomic>
#include <cstddef>
#include <utility>
#include "ctrl_layout.h"
#include "ctrl_strategy.h"
#include "data_strategy.h"
#include "global_config.h"
#include "status/status.h"
#include "type/types.h"

namespace UC::Cache2 {

class Buffer {
    CtrlStrategy ctrl_;
    DataStrategy data_;

public:
    class Handle {
        friend class Buffer;
        Buffer* buf_{nullptr};
        size_t slotIdx_{kInvalid};
        bool owner_{false};

        Handle(Buffer* buf, size_t slotIdx, bool owner)
            : buf_(buf), slotIdx_(slotIdx), owner_(owner)
        {
        }

    public:
        Handle(const Handle&) = delete;
        Handle& operator=(const Handle&) = delete;
        Handle(Handle&& o) noexcept : buf_(o.buf_), slotIdx_(o.slotIdx_), owner_(o.owner_)
        {
            o.buf_ = nullptr;
            o.slotIdx_ = kInvalid;
            o.owner_ = false;
        }
        Handle& operator=(Handle&& o) noexcept
        {
            Handle tmp(std::move(o));
            Swap(tmp);
            return *this;
        }
        ~Handle()
        {
            if (buf_ != nullptr && slotIdx_ != kInvalid) {
                buf_->ctrl_.Layout().SlotMetaArr()[slotIdx_].reference.fetch_sub(
                    1, std::memory_order_release);
            }
        }
        bool Owner() const { return owner_; }
        bool HostAccessible() const { return buf_->data_.HostAccessibleOf(slotIdx_); }
        void* Data() { return buf_->data_.DataAt(slotIdx_); }
        void* DeviceData() { return buf_->data_.DeviceDataAt(slotIdx_); }
        CtrlLayout::SlotMeta::State GetState() const
        {
            return buf_->ctrl_.Layout().SlotMetaArr()[slotIdx_].state.load(
                std::memory_order_acquire);
        }
        void MarkReady()
        {
            if (Owner()) {
                buf_->ctrl_.Layout().SlotMetaArr()[slotIdx_].state.store(
                    CtrlLayout::SlotMeta::State::Ready, std::memory_order_release);
            }
        }
        void MarkFailed()
        {
            if (Owner()) {
                buf_->ctrl_.Layout().SlotMetaArr()[slotIdx_].state.store(
                    CtrlLayout::SlotMeta::State::Failed, std::memory_order_release);
            }
        }

    private:
        void Swap(Handle& o) noexcept
        {
            std::swap(buf_, o.buf_);
            std::swap(slotIdx_, o.slotIdx_);
            std::swap(owner_, o.owner_);
        }
    };
    Status Setup(const Config& cfg) { return Status::Unsupported(); }
    Handle Get(const Detail::BlockId& blockId, size_t offset, bool allowReserved = false)
    {
        return Handle{this, 0, true};
    }
    void Prealloc(const Detail::BlockId& blockId, size_t offset, bool allowReserved = false) {}
    bool Exist(const Detail::BlockId& blockId, size_t offset) { return false; }
    void Touch(const Detail::BlockId* blocks, size_t num) {}
};

}  // namespace UC::Cache2
