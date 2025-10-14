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
#include "logger/logger.h"

namespace UC {

Status DramSpaceLayout::Setup(uint32_t maxSize,
                              uint32_t blockSize,
                              uint32_t interval) {
    if (maxSize == 0 || interval == 0 || blockSize == 0 ||
        blockSize % interval != 0) {
        UC_ERROR("Setup: invalid param");
        return Status::InvalidParam();
    }

    _capacity    = maxSize;
    _interval    = interval;
    _blockSize   = blockSize;
    _slotsPerBlock = blockSize / interval;
    _totalSlots  = maxSize / interval;

    try {
        _dataStorePool = new char[_capacity];
        _fifoKey.resize(_totalSlots);
    } catch (...) {
        UC_ERROR("Allocate DRAM failed");
        return Status::OutOfMemory();
    }

    _head = 0;
    _dataStoreMap.clear();
    _storedBlocks.clear();
    return Status::OK();
}

char* DramSpaceLayout::AllocateDataAddr(const std::string& blockId,
                                        const std::string& offset) {
    std::string key = MakeKey(blockId, offset);

    // 1. 曾经写过，直接返回（好像不应该出现）
    auto it = _dataStoreMap.find(key);
    if (it != _dataStoreMap.end()) return it->second;

    // 2. 没写过，需要挑一个物理槽位
    size_t slot = _head;
    char*  addr = _dataStorePool + slot * _interval;

    // 2.1 如果该槽位旧数据有效，先把它从映射里删掉
    if (!_fifoKey[slot].empty()) {
        _dataStoreMap.erase(_fifoKey[slot]);
    }

    // 2.2 占用槽位，记录映射
    _fifoKey[slot] = key;
    _dataStoreMap[key] = addr;

    // 2.3 推进 FIFO 头
    _head = (_head + 1) % _totalSlots;

    return addr;
}

char* DramSpaceLayout::GetDataAddr(const std::string& blockId,
                                   const std::string& offset) {
    std::string key = MakeKey(blockId, offset);
    auto it = _dataStoreMap.find(key);
    return (it == _dataStoreMap.end()) ? nullptr : it->second;
}

void DramSpaceLayout::DataStoreMapAppend(const std::string& key,
                                         char* address) {
    _dataStoreMap[key] = address;
}

void DramSpaceLayout::StoredBlocksAppend(const std::string& blockId) {
    _storedBlocks.insert(blockId);
}
void DramSpaceLayout::StoredBlocksErase(const std::string& blockId) {
    _storedBlocks.erase(blockId);
}
bool DramSpaceLayout::StoredBlocksExist(const std::string& blockId) const {
    return _storedBlocks.count(blockId) != 0;
}

} // namespace UC