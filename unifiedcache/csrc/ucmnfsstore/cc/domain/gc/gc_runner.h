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
#ifndef UNIFIEDCACHE_GC_RUNNER_H
#define UNIFIEDCACHE_GC_RUNNER_H

#include <cstdint>
#include <queue>
#include <filesystem>

namespace UC {

struct FileInfo {
    std::filesystem::path path;
    std::filesystem::file_time_type lastModified;
    uint64_t fileSize;

    FileInfo(const std::filesystem::path& path,
             const std::filesystem::file_time_type& time,
             uint64_t size)
        : path(path), lastModified(time), fileSize(size) {}

    bool operator>(const FileInfo& other) const {
        return lastModified > other.lastModified;
    }
};

class GCRunner {

public:
    GCRunner(const uint64_t threshold, const float percent) : _threshold(threshold), _percent(percent) {}
    void operator()();

private:
    std::priority_queue<FileInfo, std::vector<FileInfo>, std::greater<>> _minHeap;
    uint64_t _threshold;
    float _percent;
};

} // namespace UC

#endif
