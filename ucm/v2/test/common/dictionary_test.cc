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
#include "type/dictionary.h"
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace UC {
namespace {

TEST(DictionaryTest, SetAndGetRoundTrip)
{
    Dictionary dictionary;
    dictionary.Set("name", std::string("value"));
    dictionary.SetInt64("count", 7);
    dictionary.SetDouble("ratio", 0.5);

    std::string name;
    EXPECT_TRUE(dictionary.Get("name", name));
    EXPECT_EQ(name, "value");
    std::int64_t count = 0;
    EXPECT_TRUE(dictionary.GetInt64("count", count));
    EXPECT_EQ(count, 7);
    double ratio = 0.0;
    EXPECT_TRUE(dictionary.GetDouble("ratio", ratio));
    EXPECT_DOUBLE_EQ(ratio, 0.5);

    EXPECT_TRUE(dictionary.Contains("name"));
    EXPECT_FALSE(dictionary.Contains("missing"));
    std::string missing;
    EXPECT_FALSE(dictionary.Get("missing", missing));
}

TEST(DictionaryTest, GetFailsOnAbsentKeyOrTypeMismatch)
{
    Dictionary dictionary;
    dictionary.Set("name", std::string("value"));
    std::int64_t number = 0;
    EXPECT_FALSE(dictionary.GetInt64("name", number));
    EXPECT_EQ(number, 0);
    std::string text;
    EXPECT_FALSE(dictionary.Get("missing", text));
}

TEST(DictionaryTest, SetStoresStringLiteralsAsString)
{
    Dictionary dictionary;
    dictionary.Set("name", "literal");
    std::string value;
    EXPECT_TRUE(dictionary.Get("name", value));
    EXPECT_EQ(value, "literal");
    std::int64_t number = 0;
    EXPECT_FALSE(dictionary.GetInt64("name", number));
}

TEST(DictionaryTest, GetNumbersReplacesTargetContents)
{
    Dictionary dictionary;
    dictionary.SetNumbers("ids", std::vector<std::int64_t>{1, 2, 3});
    std::vector<std::int64_t> target{9, 9};
    ASSERT_TRUE(dictionary.GetNumbers("ids", target));
    EXPECT_EQ(target, (std::vector<std::int64_t>{1, 2, 3}));

    std::vector<std::int64_t> missing{9};
    EXPECT_FALSE(dictionary.GetNumbers("missing", missing));
    EXPECT_EQ(missing, (std::vector<std::int64_t>{9}));
}

TEST(DictionaryTest, GetNumbersCastsElementWise)
{
    Dictionary dictionary;
    dictionary.SetNumbers("ids", std::vector<std::int32_t>{1, 2, 3});
    std::vector<std::uint32_t> target{0};
    ASSERT_TRUE(dictionary.GetNumbers("ids", target));
    EXPECT_EQ(target, (std::vector<std::uint32_t>{1, 2, 3}));

    dictionary.Set("ids", std::string("not-a-number-list"));
    EXPECT_FALSE(dictionary.GetNumbers("ids", target));
    EXPECT_EQ(target, (std::vector<std::uint32_t>{1, 2, 3}));
}

}  // namespace
}  // namespace UC
