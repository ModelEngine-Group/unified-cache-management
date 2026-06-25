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
#ifndef UNIFIEDCACHE_MOONCAKE_STORE_CC_SHARE_BUFFER_H
#define UNIFIEDCACHE_MOONCAKE_STORE_CC_SHARE_BUFFER_H

#include <cstddef>
#include <memory>
#include <string>
#include "status/status.h"
#include "trans/buffer.h"

namespace UC::MooncakeStore {

class ShareBuffer {
public:
    class Reader {
        std::string key_;
        size_t length_;
        bool shared_;
        void* addr_;

    public:
        bool Shared() const noexcept { return shared_; }
        // Returns OK if data is ready, Retry if another process is loading,
        // DuplicateKey if this process should load data into GetData().
        Status Ready4Read();
        void MarkLoaded();
        void MarkFailed();
        void* GetData();

    private:
        Reader(const std::string& key, const size_t length, const bool shared, void* addr)
            : key_{key}, length_{length}, shared_{shared}, addr_{addr}
        {
        }
        friend class ShareBuffer;
        Status Ready4ReadOnSharedBuffer();
    };

public:
    Status Setup(size_t blockSize, size_t blockNumber, const std::string& uniqueId);
    ~ShareBuffer();
    std::shared_ptr<Reader> MakeReader(const std::string& key);

private:
    size_t DataOffset() const;
    size_t ShmSize() const;
    Status InitShmBuffer(int fd);
    Status LoadShmBuffer(int fd);
    size_t AcquireBlock(const std::string& key);
    void ReleaseBlock(size_t index);
    void* BlockAt(size_t index);
    std::shared_ptr<Reader> MakeSharedReader(const std::string& key, size_t position);
    std::shared_ptr<Reader> MakeLocalReader(const std::string& key);

private:
    size_t blockSize_{0};
    size_t blockNumber_{0};
    std::string shmName_;
    void* addr_{nullptr};
    std::unique_ptr<Trans::Buffer> tmpBufMaker_{nullptr};
};

}  // namespace UC::MooncakeStore

#endif
