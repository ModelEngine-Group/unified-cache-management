/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "request/request.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <ctime>
#include <string>
#include <vector>

std::vector<std::uint8_t> BuildValueBlock(const RequestOrc &entry, int testmode)
{
    // 块整体以 'A' 初始化（testmode=2 的填充底色；mode1 下会被生成的种子串整体覆盖）
    std::vector<std::uint8_t> block(entry.length, 0x41);

    if (testmode == 2) {
        // mode2：行尾 value 字段作前缀，其余保持 'A'
        const std::size_t prefix = std::min(entry.value.size(), static_cast<std::size_t>(entry.length));
        std::copy_n(entry.value.begin(), prefix, block.begin());
        return block;
    }

    // mode1：系统生成种子串（key128 低 64 位、高 16 位、offset、length、当前时间，空格分隔），
    // 用户写的 value 不生效；种子串重复填充至 length
    std::array<char, 32> timeBuf{};
    const std::time_t now = std::time(nullptr);
    const std::tm *tmNow = std::localtime(&now);
    if (tmNow != nullptr) {
        std::strftime(timeBuf.data(), timeBuf.size(), "%Y-%m-%d %H:%M:%S", tmNow);
    }
    const std::string seed = std::to_string(entry.keyLow64) + " " + std::to_string(entry.keyHigh16) + " " +
                             std::to_string(entry.offset) + " " + std::to_string(entry.length) + " " +
                             timeBuf.data();
    for (std::size_t pos = 0; pos < block.size(); pos += seed.size()) {
        const std::size_t chunk = std::min(seed.size(), block.size() - pos);
        std::copy_n(seed.begin(), chunk, block.begin() + pos);
    }
    return block;
}
