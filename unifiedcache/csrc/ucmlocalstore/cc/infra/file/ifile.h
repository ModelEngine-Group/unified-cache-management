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
#ifndef UCM_LOCAL_STORE_IFILE_H
#define UCM_LOCAL_STORE_IFILE_H

#include <string>
#include <string_view>
#include <sys/fcntl.h>
#include <sys/stat.h>
#include "status/status.h"

namespace UCM {

enum class OpenFlag : int32_t {
    RDWR = O_RDWR,
    CREAT = O_CREAT,
    EXCL = O_EXCL,
};

inline OpenFlag operator|(OpenFlag l, OpenFlag r) {
    return static_cast<OpenFlag>(static_cast<int32_t>(l) | static_cast<int32_t>(r));
}

using FileStat = struct stat64;

inline int32_t Unwrap(OpenFlag of) { return static_cast<int32_t>(of); }

class IFile {
public:
    IFile(std::string_view filepath) : _filepath{filepath} {}
    virtual ~IFile() {}

    std::string_view Filepath() const { return this->_filepath; }

    virtual Status ShmOpen(OpenFlag of) = 0;
    virtual Status Truncate(uint64_t size) = 0;
    virtual Status MMap(void** addr, uint64_t len, int32_t prot, int32_t flags, uint64_t offset = 0) = 0;
    virtual Status Stat(FileStat& stat) const = 0;

private:
    std::string _filepath;
};

} // namespace UCM

#endif // UCM_LOCAL_STORE_IFILE_H
