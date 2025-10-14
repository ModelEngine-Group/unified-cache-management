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
#pragma once
#ifndef UNIFIEDCACHE_DRAM_SPACE_LAYOUT_H
#define UNIFIEDCACHE_DRAM_SPACE_LAYOUT_H

#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include "status/status.h"

namespace UC {

class DramSpaceLayout {
public:
    // maxSize   : 池字节数
    // blockSize : 一个 block 的字节数（仅用于计算 slotsPerBlock）
    // interval  : 一次存/取的数据量，也是槽位对齐大小
    Status Setup(uint32_t maxSize, uint32_t blockSize, uint32_t interval);

    // 返回可用于写入 _interval 字节的地址
    // 如果曾经写过同样 (blockId,offset) 直接返回原地址；
    // 否则按 FIFO 复用或开辟新槽。
    char* AllocateDataAddr(const std::string& blockId,
                           const std::string& offset);

    // 纯查询，不会分配
    char* GetDataAddr(const std::string& blockId,
                      const std::string& offset);

    // 下面三个函数仅用于外部把 meta 信息同步进来
    void DataStoreMapAppend(const std::string& key, char* address);
    void StoredBlocksAppend(const std::string& blockId);
    void StoredBlocksErase (const std::string& blockId);
    bool StoredBlocksExist (const std::string& blockId) const;

private:
    // 生成 _dataStoreMap 的 key
    static std::string MakeKey(const std::string& blockId,
                               const std::string& offset) {
        return blockId + "_" + offset;
    }

    char* _dataStorePool = nullptr;          // 裸池
    size_t _capacity = 0;                    // 池字节数
    size_t _interval = 0;                    // 槽字节数
    size_t _blockSize = 0;                   // block 字节数
    size_t _slotsPerBlock = 0;               // blockSize / interval
    size_t _totalSlots = 0;                  // 池槽位数

    // FIFO 循环队列
    size_t _head = 0;                        // 下一个要被复用的槽号
    std::vector<std::string> _fifoKey;       // 槽号 -> 当前占用它的 key

    // 两个索引
    std::unordered_map<std::string, char*> _dataStoreMap;
    std::set<std::string> _storedBlocks;

    // 每个 block 已经写入了多少个 offset
    std::unordered_map<std::string, size_t> _blockWritten;
};

} // namespace UC
#endif