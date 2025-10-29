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

#include "hotness_set.h"
#include "logger/logger.h"
#include "file/file.h"
#include "template/singleton.h"
namespace UC {

void HotnessSet::Insert(const std::string& blockId)
{
    std::lock_guard<std::mutex> lg(this->mutex_);
    this->pendingBlocks_.insert(blockId);
}

void HotnessSet::UpdateHotness(const SpaceLayout* spaceLayout)
{
    std::unordered_set<std::string> blocksToUpdate;
    {
        std::lock_guard<std::mutex> lg(this->mutex_);
        if (this->pendingBlocks_.empty()) {
            return;
        }
        blocksToUpdate.swap(this->pendingBlocks_);
    }

    size_t number = 0;
    for (const std::string& blockId : blocksToUpdate) {
        auto blockPath = spaceLayout->DataFilePath(blockId, false);
        auto file = File::Make(blockPath);
        if (!file) {
            UC_WARN("Failed to make file({}), blockId({}).", blockPath, blockId);
            continue;
        }
        auto status = file->UpdateTime();
        if (status.Failure()) {
            UC_WARN("Failed({}) to update time({}), blockId({}).", status, blockPath, blockId);
            continue;
        }
        number++;
    }
    if (blocksToUpdate.size() == number) {
        UC_INFO("All blocks are hotness.");
    } else {
        UC_WARN("{} of {} blocks are hotness.", blocksToUpdate.size() - number, blocksToUpdate.size());
    }
}

} // namespace UC