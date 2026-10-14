/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once

#include <string_view>

namespace UC::Cache2 {

enum class MemfdPageMode {
    Base,
    HugeTlb2M,
    Invalid,
};

inline constexpr const char* MemfdPageModeName(MemfdPageMode mode)
{
    switch (mode) {
        case MemfdPageMode::Base: return "base";
        case MemfdPageMode::HugeTlb2M: return "hugetlb-2m";
        case MemfdPageMode::Invalid: return "invalid";
    }
    return "invalid";
}

inline constexpr MemfdPageMode ParseMemfdPageMode(std::string_view mode)
{
    if (mode == "base") { return MemfdPageMode::Base; }
    if (mode == "hugetlb-2m") { return MemfdPageMode::HugeTlb2M; }
    return MemfdPageMode::Invalid;
}

}  // namespace UC::Cache2
