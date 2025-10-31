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

#include "space_recycle.h"
#include "logger/logger.h"
#include "file/file.h"
#include "template/topn_heap.h"

namespace UC {

constexpr float recyclePercent = 0.1f; /* recycle 10% of the capacity */
constexpr uint32_t maxRecycleNum = 10240; /* max recycle num */

struct BlockInfo {
    std::string path;
    size_t timestamp;
};

struct CmpTimestamp {
    bool operator()(const BlockInfo& lhs, const BlockInfo& rhs) const {
        return lhs.timestamp > rhs.timestamp;
    }
};

size_t GetFileTimestamp(const std::string& path) {
    IFile::FileStat st;
    if (File::Stat(path, st).Failure()) { return 0; }
    return st.st_mtim.tv_sec;
}

void RemoveBlockFile(const std::string& path) {
    File::Remove(path);
    auto pos = path.rfind('/');
    if (pos == std::string::npos) { return; }
    auto parent = path.substr(0, pos);
    File::RmDir(parent);
}

void DoRecycle(const SpaceLayout* layout, const uint32_t recycleNum,
               SpaceRecycle::RecycleOneBlockDone done) {
    auto earliestHeap = std::make_unique<TopNHeap<BlockInfo, CmpTimestamp>>(recycleNum);
    auto it = layout->CreateFilePathIterator();
    while (it) {
        auto filePath = layout->NextDataFilePath(it);
        if (filePath.empty()) { break; }
        auto timestamp = GetFileTimestamp(filePath);
        if (timestamp == 0) { continue; }
        earliestHeap->Push({filePath, timestamp});
    }
    while (!earliestHeap->Empty()) {
        RemoveBlockFile(earliestHeap->Top().path);
        if (done) { done(); }
        earliestHeap->Pop();
    }
}
SpaceRecycle::~SpaceRecycle() {
    {
        std::lock_guard<std::mutex> lock(this->mtx_);
        this->stop_ = true;
        this->cv_.notify_all();
    }
    if (this->worker_.joinable()) {
        this->worker_.join();
    }
}
Status SpaceRecycle::Setup(const SpaceLayout* layout, const size_t totalNumber,
                           RecycleOneBlockDone done) {
    this->layout_ = layout;
    this->recycleNum_ = totalNumber * recyclePercent;
    if (this->recycleNum_ == 0) {
        this->recycleNum_ = 1;
    }
    this->recycleOneBlockDone_ = done;
    if (this->recycleNum_ > maxRecycleNum) {
        this->recycleNum_ = maxRecycleNum;
    }
    return Status::OK();
}

void SpaceRecycle::Trigger()
{
    if (!this->serviceRunning_) {
        this->worker_ = std::thread(&SpaceRecycle::Recycler, this);
    }
    std::lock_guard<std::mutex> lock(this->mtx_);
    if (!this->recycling_) {
        this->recycling_ = true;
        this->cv_.notify_all();
    }
}

void SpaceRecycle::Recycler()
{
    this->serviceRunning_ = true;
    UC_INFO("Space Recycle service start successfully.");
    while (true) {
        {
            std::unique_lock<std::mutex> lock(this->mtx_);
            this->cv_.wait(lock, [this] { return this->stop_ || this->recycling_; });
            if (this->stop_) { break; }
        }
        DoRecycle(this->layout_, this->recycleNum_, this->recycleOneBlockDone_);
        {
            std::lock_guard<std::mutex> lock(this->mtx_);
            this->recycling_ = false;
        }
    }
}
} // namespace UC
