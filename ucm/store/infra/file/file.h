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
#ifndef UNIFIEDCACHE_FILE_H
#define UNIFIEDCACHE_FILE_H

#include <memory>
#include "ifile.h"

namespace UC {

class File {
public:
    static std::unique_ptr<IFile> Make(const std::string& path);
    static Status MkDir(const std::string& path);
    static Status RmDir(const std::string& path);
    static Status Rename(const std::string& path, const std::string& newName);
    static Status Access(const std::string& path, const int32_t mode);
    static Status Read(const std::string& path, const size_t offset, const size_t length,
                       uintptr_t address);
    static Status Write(const std::string& path, const size_t offset, const size_t length,
                        const uintptr_t address);
    static void MUnmap(void* addr, size_t size);
    static void ShmUnlink(const std::string& path);
};

} // namespace UC

#endif
