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
#ifndef UNIFIEDCACHE_TEST_DETAIL_TYPES_HELPER_H
#define UNIFIEDCACHE_TEST_DETAIL_TYPES_HELPER_H

#include <random>
#include "type/types.h"

namespace UC::Test::Detail {

class TypesHelper {
public:
    static UC::Detail::BlockId MakeBlockId(const char* hex)
    {
        UC::Detail::BlockId id{};
        for (size_t i = 0; i < id.size() && hex[i]; ++i) { id[i] = static_cast<std::byte>(hex[i]); }
        return id;
    }
    static UC::Detail::BlockId MakeBlockIdRandomly()
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<std::uint8_t> dist(0, 255);
        UC::Detail::BlockId id;
        for (std::size_t i = 0; i < id.size(); ++i) { id[i] = static_cast<std::byte>(dist(gen)); }
        return id;
    }
    template <typename T, std::size_t N, typename... Args>
    static auto MakeArray(Args&&... args)
    {
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return std::array<T, N>{((void)I, T{std::forward<Args>(args)...})...};
        }(std::make_index_sequence<N>{});
    }
};

}  // namespace UC::Test::Detail

#endif
