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
#ifndef UNIFIEDCACHE_POSIX_STORE_CC_SPACE_LAYOUT_H
#define UNIFIEDCACHE_POSIX_STORE_CC_SPACE_LAYOUT_H

#include <condition_variable>
#include <ctime>
#include <functional>
#include <mutex>
#include <thread>
#include "common/health_window.h"
#include "global_config.h"
#include "status/status.h"
#include "type/types.h"

namespace UC::PosixStore {

struct FileInfo {
    Detail::BlockId blockId;
    time_t mtime;
};

class SpaceLayout {
private:
    std::vector<std::string> storageBackends_;
    mutable std::vector<Common::HealthWindow> backendHealth_;
    mutable std::vector<size_t> availableBackends_;
    mutable std::mutex backendMutex_;
    std::mutex stopMutex_;
    std::condition_variable stopCv_;
    bool stop_{false};
    std::thread probeThread_;
    size_t ioTimeoutMs_{0};
    std::vector<std::string> shards_;
    bool dataDirShard_;
    size_t dataDirShardBytes_;
    bool ioDirect_{true};

public:
    ~SpaceLayout();
    Status Setup(const Config& config);
    Expected<std::string> StorageBackend(const Detail::BlockId& blockId,
                                         const std::vector<std::string>& excluded = {}) const;
    size_t BackendCount() const { return storageBackends_.size(); }
    const std::vector<std::string>& Backends() const { return storageBackends_; }
    size_t IoTimeoutMs() const { return ioTimeoutMs_; }
    void RecordIoResult(const std::string& backend, const Status& status) const;
    Status RunOnAvailableBackend(const Detail::BlockId& blockId,
                                 const std::function<Status(const std::string&)>& operation) const;
    Expected<std::string> DataFilePath(const Detail::BlockId& blockId, bool activated) const;
    std::string DataFilePath(const std::string& backend, const Detail::BlockId& blockId,
                             bool activated) const;
    Status CheckHealth() const;
    Status CommitFile(const Detail::BlockId& blockId, bool success) const;
    Status CommitFile(const std::string& backend, const Detail::BlockId& blockId,
                      bool success) const;
    Status RemoveFile(const Detail::BlockId& blockId) const;
    std::vector<std::string> SampleShards(double sampleRatio) const;
    size_t CountFilesInShard(const std::string& shard) const;
    std::vector<Detail::BlockId> GetOldestFiles(const std::string& shard, double recyclePercent,
                                                size_t maxRecycleCount) const;
    std::vector<FileInfo> GetColdestCandidates(const std::string& shard, double candidatePercent,
                                               size_t maxCandidateCount) const;
    std::string ShardOf(const Detail::BlockId& blockId) const;

private:
    std::vector<std::string> RelativeRoots() const;
    void ProbeBackends(const Common::StoreHealthConfig& config);
    void RecordHealth(size_t index, const Status& status, bool recovery) const;
    std::string FileShardName(const std::string& fileName) const
    {
        return fileName.substr(0, dataDirShardBytes_);
    }
};

}  // namespace UC::PosixStore

#endif
