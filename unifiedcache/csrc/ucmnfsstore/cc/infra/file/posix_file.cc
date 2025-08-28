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

PosixFile::~PosixFile() { this->Close(); }

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

Status PosixFile::RmDir()
{
    auto ret = rmdir(this->Path().c_str());
    auto eno = errno;
    if (ret != 0) {
        if (eno != ENOTEMPTY) { UC_WARN("Failed to remove directory, path: {}.", this->Path()); }
        return Status::OsApiError();
    }
    return Status::OK();
}

Status PosixFile::Rename(const std::string& newName)
{
    auto ret = rename(this->Path().c_str(), newName.c_str());
    auto eno = errno;
    if (ret != 0) {
        if (eno == ENOENT) {
            return Status::NotFound();
        } else {
            UC_ERROR("Failed to rename file, old path: {}, new path: {}, errno: {}.", this->Path(), newName, eno);
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
            UC_ERROR("Failed to access file, path: {}, mode: {}, errcode: {}, errno: {}.", this->Path(), mode, ret,
                     eno);
            return Status::OsApiError();
        }
    }
    return Status::OK();
}

Status PosixFile::Open(const uint32_t flags)
{
    constexpr auto permission = (S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
    this->_handle = open(this->Path().c_str(), flags, permission);
    auto eno = errno;
    auto status = this->_handle >= 0 ? Status::OK() : Status::OsApiError();
    if (status.Failure()) {
        UC_ERROR("Failed({},{}) to open file({}) with flags({}).", eno, status, this->Path(), flags);
    }
    return status;
}

void PosixFile::Close()
{
    if (this->_handle != -1) { close(this->_handle); }
    this->_handle = -1;
}

void PosixFile::Remove()
{
    auto ret = remove(this->Path().c_str());
    auto eno = errno;
    if (ret != 0) {
        if (eno == ENOENT) { UC_WARN("Failed to remove file, path: {}, file not found.", this->Path()); }
    }
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
        UC_ERROR("Failed to read file, path: {}, size: {}, offset: {}, errno: {}.", this->Path(), size, offset, eno);
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
        UC_ERROR("Failed to write file, path: {}, size: {}, offset: {}, errno: {}.", this->Path(), size, offset, eno);
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
