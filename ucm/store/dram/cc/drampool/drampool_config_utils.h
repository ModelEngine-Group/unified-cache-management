/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace UC::DramPool::detail {

inline std::string Trim(const std::string& value)
{
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) { return ""; }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

inline std::uint64_t ParseUint64(const std::string& value)
{
    if (value.empty() || value.front() == '-') {
        throw std::invalid_argument("expected an unsigned integer");
    }
    std::size_t parsed = 0;
    const auto number = std::stoull(value, &parsed, 0);
    if (parsed != value.size()) { throw std::invalid_argument("trailing characters"); }
    return number;
}

inline std::uint32_t ParseUint32(const std::string& value)
{
    const auto number = ParseUint64(value);
    if (number > std::numeric_limits<std::uint32_t>::max()) {
        throw std::out_of_range("uint32 overflow");
    }
    return static_cast<std::uint32_t>(number);
}

}  // namespace UC::DramPool::detail
