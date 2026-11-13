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
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include "status/status.h"
#include "type/dictionary.h"

namespace UC {
namespace {

TEST(DictionaryReaderTest, ReadsAllFieldKinds)
{
    Dictionary dict;
    dict.Set("name", std::string("alpha"));
    dict.SetInt64("count", 7);
    dict.SetInt64("device", -2);
    dict.Set("flag", true);
    dict.Set("role", std::string("both"));

    std::string name;
    std::size_t count = 0;
    std::int32_t device = 0;
    bool flag = false;
    std::string role;
    DictionaryReader reader{dict, "ctx"};
    reader.RequiredNonEmpty("name", name);
    reader.RequiredPositive("count", count);
    reader.RequiredInt32("device", device);
    reader.OptionalBool("flag", flag);
    reader.RequiredEnum("role", role, {"scheduler", "worker", "both"});
    EXPECT_TRUE(reader.Result().Success());
    EXPECT_EQ(name, "alpha");
    EXPECT_EQ(count, std::size_t{7});
    EXPECT_EQ(device, -2);
    EXPECT_TRUE(flag);
    EXPECT_EQ(role, "both");
}

TEST(DictionaryReaderTest, OptionalReadsKeepTargetsWhenKeysAbsent)
{
    Dictionary dict;
    std::string name{"default"};
    std::int64_t value = 42;
    bool flag = true;
    DictionaryReader reader{dict};
    reader.OptionalNonEmpty("name", name);
    reader.OptionalPositive("value", value);
    reader.OptionalBool("flag", flag);
    EXPECT_TRUE(reader.Result().Success());
    EXPECT_EQ(name, "default");
    EXPECT_EQ(value, 42);
    EXPECT_TRUE(flag);
}

TEST(DictionaryReaderTest, RequiredReadsRejectMissingKeys)
{
    Dictionary dict;
    std::string name;
    std::size_t count = 0;
    std::int32_t device = 0;
    std::string role;
    DictionaryReader reader{dict};
    reader.RequiredNonEmpty("name", name);
    reader.RequiredPositive("count", count);
    reader.RequiredInt32("device", device);
    reader.RequiredEnum("role", role, {"scheduler", "worker", "both"});
    EXPECT_EQ(reader.Result(), Status::InvalidParam());
}

TEST(DictionaryReaderTest, RequiredNonEmptyRejectsEmptyAndWrongType)
{
    Dictionary dict;
    dict.Set("name", std::string(""));
    std::string target;
    DictionaryReader empty{dict};
    empty.RequiredNonEmpty("name", target);
    EXPECT_EQ(empty.Result(), Status::InvalidParam());

    dict.SetInt64("number", 7);
    DictionaryReader wrong{dict};
    std::string number;
    wrong.RequiredNonEmpty("number", number);
    EXPECT_EQ(wrong.Result(), Status::InvalidParam());
}

TEST(DictionaryReaderTest, EnumRejectsValuesOutsideTheList)
{
    Dictionary dict;
    dict.Set("role", std::string("root"));
    std::string role;
    DictionaryReader reader{dict};
    reader.RequiredEnum("role", role, {"scheduler", "worker", "both"});
    EXPECT_EQ(reader.Result(), Status::InvalidParam());
    EXPECT_NE(reader.Result().ToString().find("scheduler/worker/both"), std::string::npos);
}

TEST(DictionaryReaderTest, PositiveReadsRejectZeroNegativeAndWrongType)
{
    Dictionary dict;
    dict.SetInt64("value", 0);
    std::size_t zero = 0;
    DictionaryReader zeroReader{dict};
    zeroReader.RequiredPositive("value", zero);
    EXPECT_EQ(zeroReader.Result(), Status::InvalidParam());

    dict.SetInt64("value", -1);
    std::size_t negative = 0;
    DictionaryReader negativeReader{dict};
    negativeReader.RequiredPositive("value", negative);
    EXPECT_EQ(negativeReader.Result(), Status::InvalidParam());

    dict.Set("value", std::string("7"));
    std::size_t wrongType = 0;
    DictionaryReader wrongReader{dict};
    wrongReader.RequiredPositive("value", wrongType);
    EXPECT_EQ(wrongReader.Result(), Status::InvalidParam());

    dict.SetInt64("value", 3);
    std::size_t optional = 9;
    DictionaryReader optionalReader{dict};
    optionalReader.OptionalPositive("value", optional);
    EXPECT_TRUE(optionalReader.Result().Success());
    EXPECT_EQ(optional, std::size_t{3});
}

TEST(DictionaryReaderTest, RequiredPositiveRejectsValuesBeyondTargetRange)
{
    Dictionary dict;
    dict.SetInt64("value", 5000000000);
    std::int32_t value = 0;
    DictionaryReader reader{dict};
    reader.RequiredPositive("value", value);
    EXPECT_EQ(reader.Result(), Status::InvalidParam());
}

TEST(DictionaryReaderTest, RequiredInt32AcceptsBoundariesAndRejectsOutOfRange)
{
    Dictionary dict;
    dict.SetInt64("value", std::numeric_limits<std::int32_t>::max());
    std::int32_t maxValue = 0;
    DictionaryReader maxReader{dict};
    maxReader.RequiredInt32("value", maxValue);
    EXPECT_TRUE(maxReader.Result().Success());
    EXPECT_EQ(maxValue, std::numeric_limits<std::int32_t>::max());

    dict.SetInt64("value", std::numeric_limits<std::int32_t>::min());
    std::int32_t minValue = 0;
    DictionaryReader minReader{dict};
    minReader.RequiredInt32("value", minValue);
    EXPECT_TRUE(minReader.Result().Success());
    EXPECT_EQ(minValue, std::numeric_limits<std::int32_t>::min());

    dict.SetInt64("value", std::int64_t{std::numeric_limits<std::int32_t>::max()} + 1);
    std::int32_t beyond = 0;
    DictionaryReader beyondReader{dict};
    beyondReader.RequiredInt32("value", beyond);
    EXPECT_EQ(beyondReader.Result(), Status::InvalidParam());
}

TEST(DictionaryReaderTest, OptionalBoolRejectsNonBoolWhenSet)
{
    Dictionary dict;
    dict.SetInt64("flag", 1);
    bool flag = false;
    DictionaryReader reader{dict};
    reader.OptionalBool("flag", flag);
    EXPECT_EQ(reader.Result(), Status::InvalidParam());
}

TEST(DictionaryReaderTest, FirstFailureWins)
{
    Dictionary dict;
    std::string first;
    std::string second;
    DictionaryReader reader{dict, "ctx"};
    reader.RequiredNonEmpty("first", first);
    reader.RequiredNonEmpty("second", second);
    ASSERT_EQ(reader.Result(), Status::InvalidParam());
    EXPECT_NE(reader.Result().ToString().find("'first'"), std::string::npos);
    EXPECT_EQ(reader.Result().ToString().find("'second'"), std::string::npos);
}

TEST(DictionaryReaderTest, ContextPrefixesErrorMessages)
{
    Dictionary dict;
    std::string name;
    DictionaryReader reader{dict, "cache_config"};
    reader.RequiredNonEmpty("unique_id", name);
    EXPECT_NE(reader.Result().ToString().find("cache_config: 'unique_id'"), std::string::npos);
}

TEST(DictionaryReaderTest, EmptyContextLeavesMessageBare)
{
    Dictionary dict;
    std::string name;
    DictionaryReader reader{dict};
    reader.RequiredNonEmpty("unique_id", name);
    EXPECT_EQ(reader.Result().ToString().find("cache_config"), std::string::npos);
    EXPECT_NE(reader.Result().ToString().find("'unique_id'"), std::string::npos);
}

}  // namespace
}  // namespace UC
