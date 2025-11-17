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
#include "file.h"
#include "logger/logger.h"
#include "posix_file.h"

namespace UC {

using FileImpl = PosixFile;

std::unique_ptr<IFile> File::Make(const std::string& path)
{
    try {
        return std::make_unique<FileImpl>(path);
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to make file({}) pointer.", e.what(), path);
        return nullptr;
    }
}

Status File::MkDir(const std::string& path) { return FileImpl{path}.MkDir(); }

Status File::RmDir(const std::string& path) { return FileImpl{path}.RmDir(); }

Status File::Rename(const std::string& path, const std::string& newName)
{
    return FileImpl{path}.Rename(newName);
}

Status File::Access(const std::string& path, const int32_t mode)
{
    return FileImpl{path}.Access(mode);
}

Status File::Stat(const std::string& path, IFile::FileStat& st)
{
    FileImpl file{path};
    auto status = file.Open(IFile::OpenFlag::READ_ONLY);
    if (status.Failure()) { return status; }
    status = file.Stat(st);
    file.Close();
    return status;
}

Status File::Read(const std::string& path, const size_t offset, const size_t length,
                  uintptr_t address, const bool directIo)
{
    FileImpl file{path};
    Status status = Status::OK();
    auto flags = directIo ? IFile::OpenFlag::READ_ONLY | IFile::OpenFlag::DIRECT
                          : IFile::OpenFlag::READ_ONLY;
    if ((status = file.Open(flags)).Failure()) { return status; }
    if ((status = file.Read((void*)address, length, offset)).Failure()) { return status; }
    return status;
}

Status File::Write(const std::string& path, const size_t offset, const size_t length,
                   const uintptr_t address, const bool directIo, const bool create = false)
{
    FileImpl file{path};
    Status status = Status::OK();
    auto flags = IFile::OpenFlag::WRITE_ONLY;
    if (directIo) { flags |= IFile::OpenFlag::DIRECT; }
    if (create) { flags |= IFile::OpenFlag::CREATE; }
    if ((status = file.Open(flags)).Failure()) { return status; }
    if ((status = file.Write((const void*)address, length, offset)).Failure()) { return status; }
    return status;
}

void File::MUnmap(void* addr, size_t size) { FileImpl{{}}.MUnmap(addr, size); }

void File::ShmUnlink(const std::string& path) { FileImpl{path}.ShmUnlink(); }

void File::Remove(const std::string& path) { FileImpl{path}.Remove(); }

} // namespace UC
