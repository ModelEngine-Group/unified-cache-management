/**
/* MIT License
/*
/* Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
/*
/* Permission is hereby granted, free of charge, to any person obtaining a copy
/* of this software and associated documentation files (the "Software"), to deal
/* in the Software without restriction, including without limitation the rights
/* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/* copies of the Software, and to permit persons to whom the Software is
/* furnished to do so, subject to the following conditions:
/*
/* The above copyright notice and this permission notice shall be included in all
/* copies or substantial portions of the Software.
/*
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
/* SOFTWARE.
 * */
#ifndef UNIFIEDCACHE_POSIX_FILE_H
#define UNIFIEDCACHE_POSIX_FILE_H

#include "ifile.h"

namespace UC {
class PosixFile : public IFile {
public:
    PosixFile(const std::string& path) : IFile(path), _handle(-1) {}
    virtual ~PosixFile() override {
        if (this->_handle != -1) {
            this->Close();
        }
        this->Close(); 
    }
    // todo 
    virtual Status MkDir() override;
    virtual Status RmDir() override;
    virtual Status Rename(const std::string& newName) override;
    virtual Status Access(const int32_t type) override;
    virtual Status Open(const uint32_t mode) override;
    virtual void Close() override;
    virtual void Remove() override;
    virtual Status Seek2End() override;
    virtual Status Read(void* bufferPtrBase, size_t size, off64_t fileOffset = -1) override;
    virtual Status Write(const void* bufferPtrBase, size_t size, off64_t fileOffset = -1) override;
    virtual Status Lock() override;
    virtual Status Lock(uint32_t retryCnt, uint32_t intervalUs) override;
    virtual Status Unlock() override;
    virtual Status MMap(off64_t offset, size_t length, void*& addr, bool enableWr) override;
    virtual Status Stat(struct stat* buf) override;
    virtual Status Truncate(size_t length) override;

private:
    int32_t _handle;
    uint32_t _openMode;
};

} // namespace UC

#endif // UNIFIEDCACHE_POSIX_FILE_H