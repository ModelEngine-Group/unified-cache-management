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
 * DEALINGS IN THE SOFTWARE.
 * */
#pragma once

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include "status/status.h"

namespace UC::Dram {

inline std::string Trim(const std::string& value)
{
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) { return ""; }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

inline std::string ToLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

inline Status ParseUint64(const std::string& value, std::uint64_t& output)
{
    if (value.empty() || value.front() == '-') {
        return Status::InvalidParam("expected an unsigned integer");
    }

    char* end = nullptr;
    errno = 0;
    const auto number = std::strtoull(value.c_str(), &end, 0);
    if (errno == ERANGE) { return Status::InvalidParam("uint64 overflow"); }
    if (end == value.c_str() || *end != '\0') {
        return Status::InvalidParam("expected an unsigned integer");
    }

    output = static_cast<std::uint64_t>(number);
    return Status::OK();
}

inline Status ParseUint32(const std::string& value, std::uint32_t& output)
{
    std::uint64_t number = 0;
    auto status = ParseUint64(value, number);
    if (status.Failure()) { return status; }
    if (number > std::numeric_limits<std::uint32_t>::max()) {
        return Status::InvalidParam("uint32 overflow");
    }
    output = static_cast<std::uint32_t>(number);
    return Status::OK();
}

inline Status ParseUint16(const std::string& value, std::uint16_t& output)
{
    std::uint64_t number = 0;
    auto status = ParseUint64(value, number);
    if (status.Failure()) { return status; }
    if (number > std::numeric_limits<std::uint16_t>::max()) {
        return Status::InvalidParam("uint16 overflow");
    }
    output = static_cast<std::uint16_t>(number);
    return Status::OK();
}

inline Status ParseInt64(const std::string& value, std::int64_t& output)
{
    if (value.empty()) { return Status::InvalidParam("expected an integer"); }

    char* end = nullptr;
    errno = 0;
    const auto number = std::strtoll(value.c_str(), &end, 0);
    if (errno == ERANGE) { return Status::InvalidParam("int64 overflow"); }
    if (end == value.c_str() || *end != '\0') {
        return Status::InvalidParam("expected an integer");
    }

    output = static_cast<std::int64_t>(number);
    return Status::OK();
}

inline Status ParseInt32(const std::string& value, std::int32_t& output)
{
    std::int64_t number = 0;
    auto status = ParseInt64(value, number);
    if (status.Failure()) { return status; }
    if (number < std::numeric_limits<std::int32_t>::min() ||
        number > std::numeric_limits<std::int32_t>::max()) {
        return Status::InvalidParam("int32 overflow");
    }
    output = static_cast<std::int32_t>(number);
    return Status::OK();
}

inline Status ParseInt16(const std::string& value, std::int16_t& output)
{
    std::int64_t number = 0;
    auto status = ParseInt64(value, number);
    if (status.Failure()) { return status; }
    if (number < std::numeric_limits<std::int16_t>::min() ||
        number > std::numeric_limits<std::int16_t>::max()) {
        return Status::InvalidParam("int16 overflow");
    }
    output = static_cast<std::int16_t>(number);
    return Status::OK();
}

inline Status ParseDouble(const std::string& value, double& output)
{
    if (value.empty()) { return Status::InvalidParam("expected a finite number"); }

    char* end = nullptr;
    errno = 0;
    const auto number = std::strtod(value.c_str(), &end);
    if (errno == ERANGE || end == value.c_str() || *end != '\0' || !std::isfinite(number)) {
        return Status::InvalidParam("expected a finite number");
    }

    output = number;
    return Status::OK();
}

inline Status ParseBool(const std::string& value, bool& output)
{
    const auto normalized = ToLower(value);
    if (normalized == "true" || normalized == "yes" || normalized == "on" || normalized == "1") {
        output = true;
        return Status::OK();
    }
    if (normalized == "false" || normalized == "no" || normalized == "off" || normalized == "0") {
        output = false;
        return Status::OK();
    }
    return Status::InvalidParam("expected a boolean");
}

}  // namespace UC::Dram
