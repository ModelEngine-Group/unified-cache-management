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
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/xattr.h>
#include <unistd.h>
#include "logger/logger.h"

namespace UC {

PosixFile::~PosixFile()
{
    if (this->_handle != -1) {
        this->Close();
    }
}

Status PosixFile::MkDir()
{
    constexpr auto permission = (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    auto ret = mkdir(this->Path().c_str(), permission);
    auto eno = errno;
    if (ret != 0) {
        if (eno == EEXIST) {
            return Status::DuplicateKey();
        } else {
            UC_ERROR("Failed to create directory, path: {}, errcode: {}, errno: {}.", this->Path(), ret, eno);
            return Status::OsApiError();
        }
    }
    return Status::OK();
}

void PosixFile::RmDir()
{
    auto ret = rmdir(this->Path().c_str());
    auto eno = errno;
    if (ret != 0) {
        UC_ERROR("Failed to remove directory, path: {}, errcode: {}, errno: {}.", this->Path(), ret, eno);
    }
}

Status PosixFile::Rename(const std::string& newName)
{
    auto ret = rename(this->Path().c_str(), newName.c_str());
    auto eno = errno;
    if (ret != 0) {
        if (eno == ENOENT) {
            return Status::NotFound();
        } else {
            UC_ERROR("Failed to rename file, old path: {}, new path: {}, errno: {}.",
                     this->Path(), newName, eno);
            return Status::OsApiError();
        }
    }
    return Status::OK();
}

Status PosixFile::Access(const int32_t mode)
{
    auto ret = access(this->Path().c_str(), mode);
    auto eno = errno;
    if (ret != 0) {
        if (eno == ENOENT) {
            return Status::NotFound();
        } else {
            UC_ERROR("Failed to access file, path: {}, mode: {}, errcode: {}, errno: {}.",
                     this->Path(), mode, ret, eno);
            return Status::OsApiError();
        }
    }
    return Status::OK();
}

Status PosixFile::Open(const uint32_t flags)
{
    auto eno = 0;
    this->_openMode = flags;
    constexpr auto permission = (S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
    this->_handle = open(this->Path().c_str(), this->_openMode, permission);
    eno = errno;
    if (!this->_handle) {
        UC_ERROR("Failed to open file, path: {}, flags: {}, errno: {}.",
                 this->Path(), this->_openMode, eno);
        return Status::OsApiError();
    }
    return Status::OK();
}

void PosixFile::Close()
{
    close(this->_handle);
    this->_handle = -1;
}

void PosixFile::Remove()
{
    auto ret = remove(this->Path().c_str());
    auto eno = errno;
    if (ret != 0) {
        if (eno == ENOENT) {
            UC_WARN("Failed to remove file, path: {}, file not found.", this->Path());
        }
    }
}

Status PosixFile::Seek2End()
{
    auto ret = lseek64(this->_handle, 0, SEEK_END);
    auto eno = errno;
    if (ret < 0) {
        UC_ERROR("Failed to seek to end of file, path: {}, errno: {}.", this->Path(), ret, eno);
        return Status::OsApiError();
    }
    return Status::OK();
}

Status PosixFile::Read(void* buffer, size_t size, off64_t offset)
{
    ssize_t nBytes = -1;
    if (offset != -1) {
        nBytes = pread(this->_handle, buffer, size, offset);
    } else {
        nBytes = read(this->_handle, buffer, size);
    }
    auto eno = errno;
    if (nBytes != static_cast<ssize_t>(size)) {
        UC_ERROR("Failed to read file, path: {}, size: {}, offset: {}, errno: {}.",
                 this->Path(), size, offset, eno);
        return Status::OsApiError();
    }
    return Status::OK();
}

Status PosixFile::Write(const void* buffer, size_t size, off64_t offset)
{
    ssize_t nBytes = -1;
    if (offset != -1) {
        nBytes = pwrite(this->_handle, buffer, size, offset);
    } else {
        nBytes = write(this->_handle, buffer, size);
    }
    auto eno = errno;
    if (nBytes != static_cast<ssize_t>(size)) {
        UC_ERROR("Failed to write file, path: {}, size: {}, offset: {}, errno: {}.",
                 this->Path(), size, offset, eno);
        return Status::OsApiError();
    }
    return Status::OK();
}

Status PosixFile::Lock()
{
    struct flock lock = {0};
    lock.l_type = F_WRLCK;
    lock.l_whence = SEEK_SET;
    lock.l_start = 0;
    lock.l_len = 0; // Lock the whole file
    auto ret = fcntl(this->_handle, F_SETLK, &lock);
    auto eno = errno;
    if (ret != 0) {
        if (eno == EACCES || eno == EAGAIN) {
            return Status::Retry();
        } else {
            UC_ERROR("Failed to lock file, path: {}, errno: {}.", this->Path(), eno);
            return Status::OsApiError();
        }
    }
    return Status::OK();
}

Status PosixFile::Lock(uint32_t retryCnt, uint32_t interval)
{
    uint32_t retry = 0;
    auto ret = this->Lock();
    while (ret == Status::Retry() && retry < retryCnt) {
        usleep(interval);
        ret = this->Lock();
        ++retry;
    }
    if (ret.Failure()) {
        UC_ERROR("Failed to lock file after {} retries, path: {}, status: {}.", retry, this->Path(), ret);
    }
    return ret;
}

void PosixFile::Unlock()
{
    struct flock lock = {0};
    lock.l_type = F_UNLCK;
    lock.l_whence = SEEK_SET;
    lock.l_start = 0;
    lock.l_len = 0; // Unlock the whole file
    auto ret = fcntl(this->_handle, F_SETLK, &lock);
    auto eno = errno;
    if (ret != 0) {
        UC_ERROR("Failed to unlock file, path: {}, errno: {}.", this->Path(), eno);
    }
}

Status PosixFile::MMap(off64_t offset, size_t length, void*& addr, bool wr)
{
    auto prot = PROT_READ;
    if (wr) {
        prot |= PROT_WRITE;
    }
    auto flags = MAP_SHARED | MAP_POPULATE;
    auto ptr = mmap(nullptr, length, prot, flags, this->_handle, offset);
    auto eno = errno;
    if (ptr == MAP_FAILED || ptr == nullptr) {
        UC_ERROR("Failed to mmap file, path: {}, offset: {}, length: {}, errno: {}.",
                 this->Path(), offset, length, eno);
        return Status::OsApiError();
    }
    addr = ptr;
    return Status::OK();
}

Status PosixFile::Stat(struct stat* buffer)
{
    auto ret = stat(this->Path().c_str(), buffer);
    auto eno = errno;
    if (ret != 0) {
        UC_ERROR("Failed to stat file, path: {}, errno: {}.", this->Path(), eno);
        return Status::OsApiError();
    }
    return Status::OK();
}

Status PosixFile::Truncate(size_t length)
{
    auto ret = ftruncate(this->_handle, length);
    auto eno = errno;
    if (ret != 0) {
        UC_ERROR("Failed to truncate file, path: {}, length: {}, errno: {}.", this->Path(), length, eno);
        return Status::OsApiError();
    }
    return Status::OK();
}

} // namespace UC