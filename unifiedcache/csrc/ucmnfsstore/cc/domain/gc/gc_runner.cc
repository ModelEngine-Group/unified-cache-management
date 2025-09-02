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
#include "gc_runner.h"
#include "space/space_manager.h"
#include "template/singleton.h"
#include "logger/logger.h"

namespace UC {

void GCRunner::operator()()
{
    auto spmg = Singleton<SpaceManager>::Instance();
    uint64_t usedSpace = spmg->GetUsedSpace();
    if (usedSpace < this->_threshold) {
        return;
    }

    UC_INFO("Gc start, use: {} bytes, exceed: {} bytes", usedSpace, usedSpace - this->_threshold);
    auto layout = spmg->GetSpaceLayout();
    auto backends = layout->GetStorageBackends();
    size_t removeCount = static_cast<size_t>(this->_minHeap.size() * this->_percent);
    for (auto& backend : backends) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(backend)) {
            if (entry.is_regular_file() && entry.path().extension() == ".dat") {
                auto ftime = std::filesystem::last_write_time(entry.path());
                uint64_t fsize = static_cast<uint64_t>(std::filesystem::file_size(entry.path()));
                if (this->_minHeap.size() < removeCount) {
                    this->_minHeap.emplace(entry.path(), ftime, fsize);
                } else if (ftime < this->_minHeap.top().lastModified) {
                    this->_minHeap.pop();
                    this->_minHeap.emplace(entry.path(), ftime, fsize);
                }
            }
        }
    }

    for (size_t i = 0; i < this->_minHeap.size(); ++i) {
        FileInfo info = this->_minHeap.top();
        this->_minHeap.pop();
        std::error_code err;
        if (!std::filesystem::remove(info.path, err)) {
            UC_ERROR("Failed({}) to gc file: {}, errno: {}", info.path.string(), err.value());
            continue;
        }
        usedSpace -= info.fileSize;
    }

    spmg->SetUsedSpace(usedSpace);
    std::priority_queue<FileInfo, std::vector<FileInfo>, std::greater<>> empty;
    this->_minHeap.swap(empty);
}

} // namespace UC
