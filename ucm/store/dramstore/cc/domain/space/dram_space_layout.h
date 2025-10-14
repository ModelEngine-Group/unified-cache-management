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
#ifndef UNIFIEDCACHE_DRAM_SPACE_LAYOUT_H
#define UNIFIEDCACHE_DRAM_SPACE_LAYOUT_H

#include <string>
#include <vector>
#include "status/status.h"
#include <map>
#include <set>

namespace UC {

class DramSpaceLayout {
public:
    Status Setup(uint32_t maxSize, uint32_t blockSize, uint32_t minLength); // TODO：这里面要先进行内存空间的初始化，通过调malloc这类函数
    char* AllocateDataAddr(std::string blockId, std::string offset);
    char* GetDataAddr(std::string blockId, std::string offset);
    void DataStoreMapAppend(std::string key, char* address);
    void StoredBlocksAppend(std::string blockId);
    void StoredBlocksErase(std::string blockId);
    bool StoredBlocksExist(std::string blockId);

private:

private:
    char* _dataStorePool; // KVCache存放的内存空间，在Setup函数中进行初始化
    std::map<std::string, char*> _dataStoreMap; // 键是 block_id+offset 的拼接，值是对应的KVCache的存放起始位置，初始化为空
    std::set<std::string> _storedBlocks; // 被存了的 blocks 的所有 blockId，初始化为空
    size_t _curOffset{0}; // 当前的内存池中下一个可用的Offset（相较于内存池中初始地址的偏移）
    size_t blockSize_;
    size_t capacity_;
    size_t minLength_;
    // 目前为了简单，如果全写满了的话，就从头再来吧，把最头上(即_dataStorePool的起始位置)的数据替换掉
};

} // namespace UC

#endif