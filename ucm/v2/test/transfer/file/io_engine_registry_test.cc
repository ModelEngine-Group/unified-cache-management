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
#include "file/io/io_engine_registry.h"
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include "file/io/io_engine.h"
#include "status/status.h"

namespace UC::Transfer {
namespace {

constexpr std::size_t kIoDepth = 4;

TEST(IoEngineRegistryTest, CreateByNameInstantiatesTheNamedKind)
{
    auto created = IoEngineRegistry::Instance().Create("psync");
    ASSERT_TRUE(created.HasValue()) << created.Error().ToString();
    std::unique_ptr<IoEngine> engine(*created);
    EXPECT_TRUE(engine->Setup(IoEngine::InitParam{kIoDepth}).Success());
}

TEST(IoEngineRegistryTest, CreateByEmptyNameTakesTheBuildDefault)
{
    /* The file transfer build registers both psync and aio; aio ranks
     * higher, so it is the default, and the no-argument call is the
     * empty name. */
    auto noArgument = IoEngineRegistry::Instance().Create();
    ASSERT_TRUE(noArgument.HasValue()) << noArgument.Error().ToString();
    std::unique_ptr<IoEngine> fromDefault(*noArgument);

    auto emptyName = IoEngineRegistry::Instance().Create("");
    ASSERT_TRUE(emptyName.HasValue()) << emptyName.Error().ToString();
    std::unique_ptr<IoEngine> fromEmptyName(*emptyName);

    EXPECT_TRUE(fromDefault->Setup(IoEngine::InitParam{kIoDepth}).Success());
    EXPECT_TRUE(fromEmptyName->Setup(IoEngine::InitParam{kIoDepth}).Success());
}

TEST(IoEngineRegistryTest, CreateByUnknownNameAnswersNotFound)
{
    auto created = IoEngineRegistry::Instance().Create("nope");
    ASSERT_FALSE(created.HasValue());
    EXPECT_EQ(created.Error(), Status::NotFound());
}

}  // namespace
}  // namespace UC::Transfer
