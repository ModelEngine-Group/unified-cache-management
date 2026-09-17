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
#include <cstring>
#include <sys/stat.h>
#include <unistd.h>

namespace UC::PosixStore {

static constexpr auto NewFilePerm = (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
static constexpr auto NewDirPerm = (S_IRWXU | S_IRWXG | S_IROTH);

static Status FileError(const char* operation, const std::string& path, int error,
                        const Status& status = Status::OsApiError())
{
    return {
        status.Underlying(),
        fmt::format("{}('{}') failed: errno={} ({})", operation, path, error, std::strerror(error)),
        error};
}

PosixFile::~PosixFile()
{
    if (handle_ != -1) { Close(); }
}

Status PosixFile::MkDir()
{
    const auto dir = path_.c_str();
    auto ret = mkdir(dir, NewDirPerm);
    auto eno = errno;
    if (ret != 0) [[unlikely]] {
        if (eno == EEXIST) { return Status::DuplicateKey(); }
        return FileError("mkdir", path_, eno);
    }
    chmod(dir, NewDirPerm);
    return Status::OK();
}

Status PosixFile::RmDir()
{
    auto ret = rmdir(path_.c_str());
    auto eno = errno;
    if (ret != 0) [[unlikely]] { return FileError("rmdir", path_, eno); }
    return Status::OK();
}

Status PosixFile::Rename(const std::string& newName)
{
    auto ret = rename(path_.c_str(), newName.c_str());
    auto eno = errno;
    if (ret != 0) [[unlikely]] {
        if (eno == ENOENT) { return Status::NotFound(); }
        return FileError("rename", path_, eno);
    }
    return Status::OK();
}

Status PosixFile::Access(const int32_t mode)
{
    auto ret = access(path_.c_str(), mode);
    auto eno = errno;
    if (ret != 0) [[unlikely]] {
        if (eno == ENOENT) { return FileError("access", path_, eno, Status::NotFound()); }
        return FileError("access", path_, eno);
    }
    return Status::OK();
}

Status PosixFile::Open(const uint32_t flags)
{
#ifdef UCM_ENABLE_TEST_HOOKS
    auto hook = TestHooks::GetOpenHook();
    handle_ = hook ? hook(path_, flags, NewFilePerm) : open(path_.c_str(), flags, NewFilePerm);
#else
    handle_ = open(path_.c_str(), flags, NewFilePerm);
#endif
    auto eno = errno;
    if (handle_ < 0) [[unlikely]] {
        if (eno == EEXIST) { return FileError("open", path_, eno, Status::DuplicateKey()); }
        if (eno == ENOENT) { return FileError("open", path_, eno, Status::NotFound()); }
        return FileError("open", path_, eno);
    }
    return Status::OK();
}

Status PosixFile::Close()
{
    if (handle_ < 0) { return Status::OK(); }
    const auto ret = close(handle_);
    const auto eno = errno;
    handle_ = -1;
    if (ret != 0) { return FileError("close", path_, eno); }
    return Status::OK();
}

Status PosixFile::Remove()
{
    auto ret = remove(path_.c_str());
    auto eno = errno;
    if (ret == 0 || eno == ENOENT) { return Status::OK(); }
    return FileError("remove", path_, eno);
}

Status PosixFile::Read(void* buffer, size_t size, off64_t offset)
{
    ssize_t nBytes = -1;
    if (offset != -1) {
        nBytes = pread(handle_, buffer, size, offset);
    } else {
        nBytes = read(handle_, buffer, size);
    }
    auto eno = errno;
    const auto operation = offset == -1 ? "read" : "pread";
    if (nBytes < 0) [[unlikely]] { return FileError(operation, path_, eno); }
    if (nBytes != static_cast<ssize_t>(size)) [[unlikely]] {
        return {Status::NotFound().Underlying(),
                fmt::format("{}('{}') short read: expected {} bytes, got {}", operation, path_,
                            size, nBytes)};
    }
    return Status::OK();
}

Status PosixFile::Write(const void* buffer, size_t size, off64_t offset)
{
    ssize_t nBytes = -1;
    if (offset != -1) {
        nBytes = pwrite(handle_, buffer, size, offset);
    } else {
        nBytes = write(handle_, buffer, size);
    }
    auto eno = errno;
    const auto operation = offset == -1 ? "write" : "pwrite";
    if (nBytes < 0) [[unlikely]] { return FileError(operation, path_, eno); }
    if (nBytes != static_cast<ssize_t>(size)) [[unlikely]] {
        return Status::OsApiError(fmt::format("{}('{}') short write: expected {} bytes, got {}",
                                              operation, path_, size, nBytes));
    }
    return Status::OK();
}

Status PosixFile::Sync()
{
    auto ret = fsync(handle_);
    auto eno = errno;
    if (ret != 0) [[unlikely]] { return FileError("fsync", path_, eno); }
    return Status::OK();
}

}  // namespace UC::PosixStore
