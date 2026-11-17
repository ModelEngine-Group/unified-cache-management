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
    const std::tm *tmNow = std::localtime(&seedTime);
    if (tmNow != nullptr) {
        std::strftime(timeBuf.data(), timeBuf.size(), "%Y-%m-%d %H:%M:%S", tmNow);
    }
    const std::string seed = std::to_string(entry.keyLow64) + " " +
                             std::to_string(entry.keyHigh16) + " " + std::to_string(entry.offset) +
                             " " + std::to_string(entry.length) + " " + timeBuf.data();
    const std::size_t prefix = std::min(seed.size(), static_cast<std::size_t>(entry.length));
    return std::vector<std::uint8_t>(seed.begin(), seed.begin() + prefix);
}
