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
#ifndef UNIFIEDCACHE_SHARE_BUFFER_H
#define UNIFIEDCACHE_SHARE_BUFFER_H

#include <cstddef>
#include <memory>
#include <string>
#include "file/ifile.h"
#include "status/status.h"

namespace UC {

class ShareBuffer {
public:
    class Reader {
        std::string block_;
        std::string path_;
        size_t length_;
        bool ioDirect_;
        size_t nSharer_;
        void* addr_;

    public:
        Status Ready4Read();
        uintptr_t GetData();

    private:
        Reader(const std::string& block, const std::string& path, const size_t length,
               const bool ioDirect, const size_t nSharer, void* addr)
            : block_{block}, path_{path}, length_{length}, ioDirect_{ioDirect}, nSharer_{nSharer},
              addr_{addr}
        {
        }
        friend class ShareBuffer;
    };

public:
    Status Setup(const size_t blockSize, const size_t blockNumber, const bool ioDirect,
                 const size_t nSharer);
    ~ShareBuffer();
    std::shared_ptr<Reader> MakeReader(const std::string& block, const std::string& path);

private:
    size_t DataOffset() const;
    size_t ShmSize() const;
    Status InitShmBuffer(IFile* file);
    Status LoadShmBuffer(IFile* file);
    size_t AcquireBlock(const std::string& block);
    void ReleaseBlock(const size_t index);
    void* BlockAt(const size_t index);

private:
    size_t blockSize_;
    size_t blockNumber_;
    bool ioDirect_;
    size_t nSharer_;
    std::string shmName_;
    void* addr_;
};

} // namespace UC

#endif
