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
#include "memory/dma/dma_engine_registry.h"
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include "memory/dma/dma_engine.h"
#include "status/status.h"

namespace UC::Transfer {
namespace {

constexpr std::size_t kMaxLength = 64 * 1024;
constexpr int32_t kDeviceId = 0;

TEST(DmaEngineRegistryTest, CreateByNameInstantiatesTheNamedKind)
{
    auto created = DmaEngineRegistry::Instance().Create("naive");
    ASSERT_TRUE(created.HasValue()) << created.Error().ToString();
    std::unique_ptr<DmaEngine> engine(*created);
    EXPECT_TRUE(engine->Setup(DmaEngine::InitParam{kDeviceId, kMaxLength}).Success());
}

TEST(DmaEngineRegistryTest, CreateByEmptyNameTakesTheBuildDefault)
{
    /* This test binary registers the simu engine under naive and the
     * fault fixture under gdr, which ranks below it: the default is the
     * naive engine, and the no-argument call is the empty name. */
    auto noArgument = DmaEngineRegistry::Instance().Create();
    ASSERT_TRUE(noArgument.HasValue()) << noArgument.Error().ToString();
    std::unique_ptr<DmaEngine> fromDefault(*noArgument);

    auto emptyName = DmaEngineRegistry::Instance().Create("");
    ASSERT_TRUE(emptyName.HasValue()) << emptyName.Error().ToString();
    std::unique_ptr<DmaEngine> fromEmptyName(*emptyName);

    EXPECT_TRUE(fromDefault->Setup(DmaEngine::InitParam{kDeviceId, kMaxLength}).Success());
    EXPECT_TRUE(fromEmptyName->Setup(DmaEngine::InitParam{kDeviceId, kMaxLength}).Success());
}

TEST(DmaEngineRegistryTest, CreateByUnknownNameAnswersNotFound)
{
    auto created = DmaEngineRegistry::Instance().Create("nope");
    ASSERT_FALSE(created.HasValue());
    EXPECT_EQ(created.Error(), Status::NotFound());
}

TEST(DmaEngineRegistryTest, CreateByUnregisteredKindNameAnswersNotFound)
{
    /* 'ffts' is a valid kind name, but this build links no factory under
     * it: only naive (simu) and gdr (the fault fixture) register. */
    auto created = DmaEngineRegistry::Instance().Create("ffts");
    ASSERT_FALSE(created.HasValue());
    EXPECT_EQ(created.Error(), Status::NotFound());
}

}  // namespace
}  // namespace UC::Transfer
