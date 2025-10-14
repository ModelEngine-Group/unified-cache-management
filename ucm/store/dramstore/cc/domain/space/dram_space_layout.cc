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
#include "dram_space_layout.h"
#include <algorithm>
#include <array>
#include "file/file.h"
#include "logger/logger.h"

namespace UC {

constexpr size_t blockIdSize = 16;
constexpr size_t nU64PerBlock = blockIdSize / sizeof(uint64_t);
using BlockId = std::array<uint64_t, nU64PerBlock>;
static_assert(sizeof(BlockId) == blockIdSize);

Status DramSpaceLayout::Setup(uint32_t maxSize, uint32_t blockSize, uint32_t interval)
{
    if (maxSize <= 0) {
        UC_ERROR("Invalid maxSize value.");
        return Status::InvalidParam();
    }
    _dataStorePool = nullptr;
    _dataStorePool = new char[maxSize]; // 这里内存分配的逻辑与方法是否正确？也要确认
    if (!_dataStorePool) {
        UC_ERROR("Allocate DRAM storage space failed");
        return Status::OutOfMemory();
    }
    _dataStoreMap = {};
    _storedBlocks = {};
    _blockSize = blockSize;
    _interval = interval;
    _capacity = maxSize;
    return Status::OK();
}

char* DramSpaceLayout::AllocateDataAddr(std::string blockId, std::string offset) {
    auto iter = _dataStoreMap.find(blockId + offset);
    if (iter != _dataStoreMap.end()) {
        // 已经存在，不需要重分配
        char* addr = _dataStoreMap[blockId + offset];
        _dataStoreMap.erase(blockId + offset)
        return addr;
    }
    _dataStoreMap[blockId + offset] = _dataStorePool + _curOffset;
    _curOffset = (_curOffset + _interval) % _capacity; // 这个interval的逻辑是否正确，还要再确认
    return _dataStoreMap[blockId + offset];
}

char* DramSpaceLayout::GetDataAddr(std::string blockId, std::string offset) {
    auto iter = _dataStoreMap.find(blockId + offset);
    if (iter == _dataStoreMap.end()) {
        return nullptr;
    }
    return _dataStoreMap[blockId];
}

void DramSpaceLayout::DataStoreMapAppend(std::string key, char* address) {
    _dataStoreMap[key] = address;
}

void DramSpaceLayout::StoredBlocksAppend(std::string blockId) {
    _storedBlocks.insert(blockId);
}

void DramSpaceLayout::StoredBlocksErase(std::string blockId) {
    _storedBlocks.erase(blockId);
}

bool DramSpaceLayout::StoredBlocksExist(std::string blockId) {
    return _storedBlocks.find(blockId) != _storedBlocks.end();
}

} // namespace UC
