/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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

#include "request/request.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <ctime>
#include <string>
#include <vector>

std::vector<std::uint8_t> BuildValueBlock(const RequestOrc &entry, int testmode,
                                          std::time_t seedTime)
{
    // 只生成写入缓冲的前缀，不补齐：前缀之后的缓冲内容保持启动 memset 的 'A' 底色或
    // 上次写入的残留（缓冲复用），IO 长度由 entry.length 决定与此处无关
    if (testmode == 2) {
        // mode2：行尾 value 字段作前缀（超 length 截断；为空时返回空，不写任何字节）
        const std::size_t prefix =
            std::min(entry.value.size(), static_cast<std::size_t>(entry.length));
        return std::vector<std::uint8_t>(entry.value.begin(), entry.value.begin() + prefix);
    }

    // mode1：系统生成种子串（key128 低 64 位、高 16 位、offset、length、时间戳，空格分隔），
    // 只写一次不重复填充（超 length 截断）；用户写的 value 不生效
    std::array<char, 32> timeBuf{};
    std::tm tmNow{};
    // localtime 返回内部静态缓冲指针（标准不保证线程安全），本函数经 buildTaskRequest
    // 被多 worker 并发调用；localtime_r 写入调用方栈上缓冲（POSIX，与线程命名/绑核
    // 同一 Linux 直调先例），并发语义确定化
    if (localtime_r(&seedTime, &tmNow) != nullptr) {
        std::strftime(timeBuf.data(), timeBuf.size(), "%Y-%m-%d %H:%M:%S", &tmNow);
    }
    const std::string seed = std::to_string(entry.keyLow64) + " " +
                             std::to_string(entry.keyHigh16) + " " + std::to_string(entry.offset) +
                             " " + std::to_string(entry.length) + " " + timeBuf.data();
    const std::size_t prefix = std::min(seed.size(), static_cast<std::size_t>(entry.length));
    return std::vector<std::uint8_t>(seed.begin(), seed.begin() + prefix);
}
