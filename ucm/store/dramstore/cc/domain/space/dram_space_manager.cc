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
#include "dram_space_manager.h"
#include "file/file.h"
#include "logger/logger.h"

namespace UC {

Status DramSpaceManager::Setup(uint32_t maxSize, uint32_t blockSize, uint32_t interval)
{
    if (blockSize == 0) {
        UC_ERROR("Invalid block size({}).", blockSize);
        return Status::InvalidParam();
    }
    auto status = this->layout_.Setup(maxSize, blockSize, interval);
    if (status.Failure()) { return status; }
    this->_blockSize = blockSize;
    return Status::OK();
}

Status DramSpaceManager::NewBlock(const std::string& blockId) const
{
    return Status::OK();
}

Status DramSpaceManager::CommitBlock(const std::string& blockId, bool success) const
{
    if (success) {
        this->_layout->StoredBlocksAppend(blockId);
    } 
    else {
        this->_layout->StoredBlocksErase(blockId);
    }
    return Status::OK();
}

bool DramSpaceManager::LookupBlock(const std::string& blockId) const
{
    return this->_layout->StoredBlocksExist(blockId);
}

const SpaceLayout* DramSpaceManager::GetSpaceLayout() const { return &this->layout_; }

} // namespace UC
