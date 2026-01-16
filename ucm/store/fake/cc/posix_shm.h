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
#ifndef UNIFIEDCACHE_FAKE_STORE_CC_POSIX_SHM_H
#define UNIFIEDCACHE_FAKE_STORE_CC_POSIX_SHM_H

#include <cstdint>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include "status/status.h"

namespace UC::FakeStore {

class PosixShm {
public:
    struct OpenFlag {
        static constexpr uint32_t READ_WRITE = O_RDWR;
        static constexpr uint32_t CREATE = O_CREAT;
        static constexpr uint32_t EXCL = O_EXCL;
    };

private:
    std::string name_{};
    int32_t handle_{-1};

public:
    PosixShm(std::string name) : name_{std::move(name)} {}
    ~PosixShm()
    {
        if (handle_ != -1) { close(handle_); }
    }
    Status ShmOpen(const uint32_t flags)
    {
        static constexpr auto NewFilePerm = (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
        handle_ = shm_open(name_.c_str(), flags, NewFilePerm);
        auto eno = errno;
        if (handle_ >= 0) { return Status::OK(); }
        if (eno == EEXIST) { return Status::DuplicateKey(); }
        return Status{eno, std::to_string(eno)};
    }
    Status Truncate(size_t length)
    {
        auto ret = ftruncate64(handle_, length);
        auto eno = errno;
        if (ret == 0) { return Status::OK(); }
        return Status{eno, std::to_string(eno)};
    }
    Status MMap(void*& addr, size_t size, bool write, bool read, bool shared)
    {
        auto prot = PROT_NONE;
        if (write) { prot |= PROT_WRITE; }
        if (read) { prot |= PROT_READ; }
        auto flags = 0;
        if (shared) { flags |= MAP_SHARED; }
        addr = mmap(nullptr, size, prot, flags, handle_, 0);
        auto eno = errno;
        if (addr != MAP_FAILED) { return Status::OK(); }
        return Status{eno, std::to_string(eno)};
    }
    static void MUnmap(void* addr, size_t size) { munmap(addr, size); }
    void ShmUnlink() { shm_unlink(name_.c_str()); }
};

}  // namespace UC::FakeStore

#endif
