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
#ifndef UNIFIEDCACHE_POSIX_STORE_CC_POSIX_FILE_H
#define UNIFIEDCACHE_POSIX_STORE_CC_POSIX_FILE_H

#include <fcntl.h>
#include "status/status.h"

namespace UC::PosixStore {

class PosixFile {
public:
    struct AccessMode {
        static constexpr int32_t READ = R_OK;
        static constexpr int32_t WRITE = W_OK;
        static constexpr int32_t EXIST = F_OK;
        static constexpr int32_t EXECUTE = X_OK;
    };
    struct OpenFlag {
        static constexpr uint32_t READ_ONLY = O_RDONLY;
        static constexpr uint32_t WRITE_ONLY = O_WRONLY;
        static constexpr uint32_t READ_WRITE = O_RDWR;
        static constexpr uint32_t CREATE = O_CREAT;
        static constexpr uint32_t DIRECT = O_DIRECT;
        static constexpr uint32_t APPEND = O_APPEND;
        static constexpr uint32_t EXCL = O_EXCL;
    };

private:
    std::string path_{};
    int32_t handle_{-1};

public:
    explicit PosixFile(std::string path) : path_{std::move(path)} {}
    ~PosixFile();
    const std::string& Path() const { return path_; }
    Status MkDir();
    Status RmDir();
    Status Rename(const std::string& newName);
    Status Access(const int32_t mode);
    Status Open(const uint32_t flags);
    void Close();
    void Remove();
    Status Read(void* buffer, size_t size, off64_t offset);
    Status Write(const void* buffer, size_t size, off64_t offset);
};

}  // namespace UC::PosixStore

#endif
