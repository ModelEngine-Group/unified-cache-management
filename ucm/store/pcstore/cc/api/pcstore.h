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
#ifndef UNIFIEDCACHE_PCSTORE_H
#define UNIFIEDCACHE_PCSTORE_H

#include "trans/trans_task.h"
#include "ucmstore.h"

namespace UC {

class PcStore : CCStore<TransTask> {
public:
    struct Config {
        std::vector<std::string> storageBackends;
        size_t kvcacheBlockSize;
        bool transferEnable;
        std::string uniqueId{};
        size_t transferIoSize{262144};
        bool transferIoDirect{false};
        size_t transferLocalRankSize{1};
        int32_t transferDeviceId{-1};
        size_t transferStreamNumber{8};
        size_t transferBufferNumber{4096};
        size_t transferTimeoutMs{30000};
        bool transferScatterGatherEnable{false};

        Config(const std::vector<std::string>& storageBackends, const size_t kvcacheBlockSize,
               const bool transferEnable)
            : storageBackends{storageBackends},
              kvcacheBlockSize{kvcacheBlockSize},
              transferEnable{transferEnable}
        {
        }
    };

public:
    ~PcStore() override
    {
        if (this->impl_) { delete this->impl_; }
    }
    int32_t Setup(const Config& config);
    int32_t Alloc(const std::string& block) override { return this->impl_->Alloc(block); }
    bool Lookup(const std::string& block) override { return this->impl_->Lookup(block); }
    void Commit(const std::string& block, const bool success) override
    {
        this->impl_->Commit(block, success);
    }
    std::list<int32_t> Alloc(const std::list<std::string>& blocks) override
    {
        return this->impl_->Alloc(blocks);
    }
    std::list<bool> Lookup(const std::list<std::string>& blocks) override
    {
        return this->impl_->Lookup(blocks);
    }
    void Commit(const std::list<std::string>& blocks, const bool success) override
    {
        this->impl_->Commit(blocks, success);
    }
    size_t Submit(TransTask&& task) override { return this->impl_->Submit(std::move(task)); }
    int32_t Wait(const size_t task) override { return this->impl_->Wait(task); }
    int32_t Check(const size_t task, bool& finish) override
    {
        return this->impl_->Check(task, finish);
    }

private:
    PcStore* impl_{nullptr};
};

}  // namespace UC

#endif
