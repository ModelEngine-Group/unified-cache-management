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
#include "posix_file.h"
#include <unistd.h>
#include <sys/mman.h>
#include "logger/logger.h"

namespace UCM {

PosixFile::~PosixFile()
{
    if (this->_fd != -1) {
        close(this->_fd);
    }
}

Status PosixFile::ShmOpen(OpenFlag of)
{
    auto flag = static_cast<int>(Unwrap(of));
    this->_fd = shm_open(this->Filepath().data(), flag, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
    auto eno = errno;
    if (this->_fd == -1) {
        if (eno == EEXIST) {
            return Status::Exist();
        }
        UCM_ERROR("Failed to open shared memory file, filepath={}, errno={}", this->Filepath().data(), eno);
        return Status::OsApiError();
    }
    return Status::OK();
}

Status PosixFile::Truncate(uint64_t size)
{
    auto ret = ftruncate(this->_fd, static_cast<off_t>(size));
    auto eno = errno;
    if (ret != 0) {
        UCM_ERROR("Failed to truncate file, filepath={}, errno={}", this->Filepath().data(), eno);
        return Status::OsApiError();
    }
    return Status::OK();
}

Status PosixFile::MMap(void** addr, uint64_t len, int32_t prot, int32_t flags, uint64_t offset)
{
    *addr = mmap(NULL, static_cast<size_t>(len), static_cast<int>(prot),
                 static_cast<int>(flags), this->_fd, static_cast<off_t>(offset));
    auto eno = errno;
    if (*addr == MAP_FAILED) {
        UCM_ERROR("Failed to mmap file, filepath={}, len={}, prot={}, flags={}, offset={}, errno={}",
                  this->Filepath().data(), len, prot, flags, offset, eno);
        *addr = nullptr;
        return Status::OsApiError();
    }
    return Status::OK();
}

Status PosixFile::Stat(FileStat& stat) const
{
    auto ret = fstat64(this->_fd, &stat);
    auto eno = errno;
    if (ret == -1) {
        UCM_ERROR("Failed to stat file, filepath={}, errno={}", this->Filepath().data(), eno);
        return Status::OsApiError();
    }
    return Status::OK();
}

} // namespace UCM