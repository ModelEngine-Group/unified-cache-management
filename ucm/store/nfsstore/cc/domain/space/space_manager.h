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
#ifndef UNIFIEDCACHE_SPACE_MANAGER_H
#define UNIFIEDCACHE_SPACE_MANAGER_H

#include <memory>
#include "space_layout.h"
#include "space_property.h"
#include "status/status.h"
#include "space_recycle.h"

namespace UC {

class SpaceManager {
public:
    Status Setup(const std::vector<std::string>& storageBackends, const size_t blockSize,
                 const bool tempDumpDirEnable, const size_t storageCapacity = 0,
                 const bool recycleEnable = false, const float recycleThresholdRatio = 0.7f);
    Status NewBlock(const std::string& blockId);
    Status CommitBlock(const std::string& blockId, bool success = true);
    bool LookupBlock(const std::string& blockId) const;
    const SpaceLayout* GetSpaceLayout() const;

private:
    Status CapacityCheck();
private:
    std::unique_ptr<SpaceLayout> layout_;
    SpaceProperty property_;
    SpaceRecycle recycle_;
    size_t blockSize_;
    size_t capacity_;
    bool recycleEnable_;
    size_t capacityRecycleThreshold_;
};

} // namespace UC

#endif
