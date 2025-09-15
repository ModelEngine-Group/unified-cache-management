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
#ifndef UNIFIEDCACHE_POSIX_FILE_H
#define UNIFIEDCACHE_POSIX_FILE_H

#include "ifile.h"

namespace UC {

class PosixFile : public IFile {
public:
    PosixFile(const std::string& path) : IFile{path}, _handle{-1} {}
    ~PosixFile() override;
    Status MkDir() override;
    Status RmDir() override;
    Status Rename(const std::string& newName) override;
    Status Access(const int32_t mode) override;
    Status Open(const uint32_t flags) override;
    void Close() override;
    void Remove() override;
    Status Read(void* buffer, size_t size, off64_t offset = -1) override;
    Status Write(const void* buffer, size_t size, off64_t offset = -1) override;
    Status Truncate(size_t length) override;

private:
    int32_t _handle;
};

} // namespace UC

#endif // UNIFIEDCACHE_POSIX_FILE_H