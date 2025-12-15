/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#ifndef UNIFIEDCACHE_CACHE_STORE_CC_TRANS_BUFFER_H
#define UNIFIEDCACHE_CACHE_STORE_CC_TRANS_BUFFER_H

#include <limits>
#include <memory>
#include "global_config.h"
#include "status/status.h"
#include "type/types.h"

namespace UC::CacheStore {

class BufferStrategy;

class TransBuffer {
    using Index = std::size_t;
    static constexpr Index npos = std::numeric_limits<Index>::max();
    std::shared_ptr<BufferStrategy> strategy_{nullptr};

public:
    class Handle {
    public:
        Handle() = default;
        Handle(const Handle& other) : buf_{other.buf_}, pos_{other.pos_}, owner_{other.owner_}
        {
            if (Valid()) { buf_->Acquire(pos_); }
        }
        Handle& operator=(const Handle& other)
        {
            if (this != &other) {
                Handle tmp(other);
                Swap(tmp);
            }
            return *this;
        }
        Handle(Handle&& other) noexcept : buf_{other.buf_}, pos_{other.pos_}, owner_{other.owner_}
        {
            other.buf_ = nullptr;
            other.pos_ = npos;
            other.owner_ = false;
        }
        Handle& operator=(Handle&& other) noexcept
        {
            Handle tmp(std::move(other));
            Swap(tmp);
            return *this;
        }
        ~Handle()
        {
            if (Valid()) { buf_->Release(pos_); }
        }
        explicit operator bool() const { return Valid(); }
        bool Owner() const { return owner_; }
        void* Data()
        {
            if (Valid()) { return buf_->DataAt(pos_); }
            return nullptr;
        }
        bool Ready() const { return buf_->Ready(pos_); };
        void MarkReady() { buf_->MarkReady(pos_); };

    private:
        friend class TransBuffer;
        Handle(TransBuffer* buf, Index pos, bool owner) : buf_{buf}, pos_{pos}, owner_{owner} {};
        bool Valid() const { return buf_ && pos_ != npos; }
        void Swap(Handle& other) noexcept
        {
            std::swap(buf_, other.buf_);
            std::swap(pos_, other.pos_);
            std::swap(owner_, other.owner_);
        }
        TransBuffer* buf_{nullptr};
        Index pos_{npos};
        bool owner_{false};
    };

public:
    Status Setup(const Config& config);
    Handle Get(const Detail::BlockId& blockId, size_t shardIdx);

private:
    size_t FindAt(size_t iBucket, const Detail::BlockId& blockId, size_t shardIdx, bool& owner);
    size_t Alloc(const Detail::BlockId& blockId, size_t shardIdx, size_t iBucket);
    void MoveTo(size_t iBucket, size_t iNode);
    void Remove(size_t iBucket, size_t iNode);
    void* DataAt(Index pos);
    void Acquire(Index pos);
    void Release(Index pos);
    bool Ready(Index pos);
    void MarkReady(Index pos);
};

}  // namespace UC::CacheStore

#endif
