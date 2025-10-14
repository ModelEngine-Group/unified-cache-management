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
#ifndef UNIFIEDCACHE_DRAM_SPACE_MANAGER_H
#define UNIFIEDCACHE_DRAM_SPACE_MANAGER_H

#include "dram_space_layout.h"
#include "status/status.h"

namespace UC {

class DramSpaceManager {
public:
    Status Setup(uint32_t maxSize, uint32_t blockSize, uint32_t interval);
    Status NewBlock(const std::string& blockId) const; // 也许不需要实现它。无论如何先放这里
    Status CommitBlock(const std::string& blockId, bool success = true) const; // 等一个block完全存完或者被完全删除后，调用这个方法，并更新layout_中的_storedBlocks集合
    bool LookupBlock(const std::string& blockId) const;
    const DramSpaceLayout* GetSpaceLayout() const;

private:
    DramSpaceLayout layout_;
    size_t _blockSize;
};

} // namespace UC

#endif