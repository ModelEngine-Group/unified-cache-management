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
#ifndef UNIFIEDCACHE_IFILE_H
#define UNIFIEDCACHE_IFILE_H

#include <fcntl.h>
#include <string>
#include "status/status.h"

namespace UC {

class IFile {
public:
    class AccessMode {
    public:
        static constexpr int32_t READ = R_OK;
        static constexpr int32_t WRITE = W_OK;
        static constexpr int32_t EXIST = F_OK;
        static constexpr int32_t EXECUTE = X_OK;
    };
    class OpenFlag {
    public:
        static constexpr uint32_t READ_ONLY = O_RDONLY;
        static constexpr uint32_t WRITE_ONLY = O_WRONLY;
        static constexpr uint32_t READ_WRITE = O_RDWR;
        static constexpr uint32_t CREATE = O_CREAT;
        static constexpr uint32_t DIRECT = O_DIRECT;
        static constexpr uint32_t APPEND = O_APPEND;
    };

public:
    IFile(const std::string& path) : _path{path} {}
    virtual ~IFile() = default;
    const std::string& Path() const { return this->_path; }
    virtual Status MkDir() = 0;
    virtual void RmDir() = 0;
    virtual Status Rename(const std::string& newName) = 0;
    virtual Status Access(const int32_t mode) = 0;
    virtual Status Open(const uint32_t flags) = 0;
    virtual void Close() = 0;
    virtual void Remove() = 0;
    virtual Status Seek2End() = 0;
    virtual Status Read(void* buffer, size_t size, off64_t offset = -1) = 0;
    virtual Status Write(const void* buffer, size_t size, off64_t offset = -1) = 0;
    virtual Status Lock() = 0;
    virtual Status Lock(uint32_t retryCnt, uint32_t intervalUs) = 0;
    virtual void Unlock() = 0;
    virtual Status MMap(off64_t offset, size_t length, void*& addr, bool wr) = 0;
    virtual Status Stat(struct stat* buffer) = 0;
    virtual Status Truncate(size_t length) = 0;

private:
    std::string _path;
};

} // namespace UC

#endif
