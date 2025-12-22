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
#ifndef UNIFIEDCACHE_SPACE_LAYOUT_H
#define UNIFIEDCACHE_SPACE_LAYOUT_H

#include <string>
#include <vector>
#include "status/status.h"

namespace UC {

class SpaceLayout {
public:
    Status Setup(const std::vector<std::string>& storageBackends, bool shardDataDir);
    std::string DataFilePath(const std::string& blockId, bool activated) const;
    Status Commit(const std::string& blockId, bool success) const;

private:
    std::vector<std::string> RelativeRoots() const;
    Status AddStorageBackend(const std::string& path);
    Status AddFirstStorageBackend(const std::string& path);
    Status AddSecondaryStorageBackend(const std::string& path);
    std::string StorageBackend(const std::string& blockId) const;
    std::string DataParentName(const std::string& blockFile, bool activated) const;
    std::string DataFileRoot() const;
    std::string TempFileRoot() const;
    std::string DataFileName(const std::string& blockId) const;

private:
    std::vector<std::string> storageBackends_;
    bool shardDataDir_;
};

}  // namespace UC

#endif
