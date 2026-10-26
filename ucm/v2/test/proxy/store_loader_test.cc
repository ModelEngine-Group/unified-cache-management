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
#include "store_loader.h"
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "status/status.h"
#include "store.h"
#include "type/block.h"
#include "type/dictionary.h"

namespace UC::Proxy {
namespace {

using UC::Store::JobDescriptor;

class FakeStore : public Store::Store {
public:
    explicit FakeStore(bool* destroyed) : destroyed_(destroyed) {}

    ~FakeStore() override
    {
        if (destroyed_ != nullptr) { *destroyed_ = true; }
    }

    Status Setup(const Dictionary& /*config*/) override { return Status::Ok(); }

    Expected<std::vector<std::uint8_t>> Lookup(const std::vector<BlockId>& /*blockIds*/) override
    {
        return std::vector<std::uint8_t>{};
    }

    Status Load(std::size_t /*jobId*/, JobDescriptor /*job*/) override { return Status::Ok(); }

    Status Dump(std::size_t /*jobId*/, JobDescriptor /*job*/) override { return Status::Ok(); }

    std::vector<std::pair<std::size_t, bool>> GetFinished() override { return {}; }

    std::vector<std::pair<std::size_t, bool>> Wait(
        const std::vector<std::size_t>& /*jobIds*/) override
    {
        return {};
    }

private:
    bool* destroyed_;
};

TEST(StoreLoaderTest, LoadBuiltinNameCreatesInstance)
{
    auto loaded = StoreLoader::Load("posix");
    ASSERT_TRUE(loaded);
    EXPECT_NE(loaded.Value()->Store(), nullptr);
}

TEST(StoreLoaderTest, LoadSamePluginTwiceYieldsDistinctInstances)
{
    auto first = StoreLoader::Load("posix");
    auto second = StoreLoader::Load("posix");
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    EXPECT_NE(first.Value()->Store(), second.Value()->Store());
}

TEST(StoreLoaderTest, UnknownBuiltinNameFailsWithInvalidParam)
{
    auto loaded = StoreLoader::Load("no-such-store");
    ASSERT_FALSE(loaded);
    EXPECT_EQ(loaded.Error(), Status::InvalidParam());
    EXPECT_NE(loaded.Error().ToString().find("no-such-store"), std::string::npos);
}

TEST(StoreLoaderTest, MissingPathFailsWithOsApiError)
{
    auto loaded = StoreLoader::Load("/nonexistent/no_such_store.so");
    ASSERT_FALSE(loaded);
    EXPECT_EQ(loaded.Error(), Status::OsApiError());
}

TEST(LoadedStoreTest, AdoptDestroysStoreOnDestruction)
{
    bool destroyed = false;
    {
        auto loaded = LoadedStore::Adopt(std::make_unique<FakeStore>(&destroyed));
        ASSERT_NE(loaded, nullptr);
        EXPECT_NE(loaded->Store(), nullptr);
    }
    EXPECT_TRUE(destroyed);
}

TEST(LoadedStoreTest, MoveConstructTransfersOwnership)
{
    bool destroyed = false;
    auto loaded = LoadedStore::Adopt(std::make_unique<FakeStore>(&destroyed));
    ASSERT_NE(loaded, nullptr);
    {
        LoadedStore moved(std::move(*loaded));
        EXPECT_EQ(loaded->Store(), nullptr);
        EXPECT_NE(moved.Store(), nullptr);
        EXPECT_FALSE(destroyed);
    }
    EXPECT_TRUE(destroyed);
}

TEST(LoadedStoreTest, MoveAssignmentReleasesPreviousInstance)
{
    bool sourceDestroyed = false;
    bool targetDestroyed = false;
    auto target = LoadedStore::Adopt(std::make_unique<FakeStore>(&targetDestroyed));
    ASSERT_NE(target, nullptr);
    {
        auto source = LoadedStore::Adopt(std::make_unique<FakeStore>(&sourceDestroyed));
        ASSERT_NE(source, nullptr);
        *target = std::move(*source);
        EXPECT_EQ(source->Store(), nullptr);
    }
    EXPECT_FALSE(sourceDestroyed);
    EXPECT_TRUE(targetDestroyed);
    target.reset();
    EXPECT_TRUE(sourceDestroyed);
}

TEST(LoadedStoreTest, SelfMoveAssignmentKeepsInstance)
{
    bool destroyed = false;
    auto loaded = LoadedStore::Adopt(std::make_unique<FakeStore>(&destroyed));
    ASSERT_NE(loaded, nullptr);
    auto* store = loaded->Store();
    LoadedStore* alias = loaded.get();
    *loaded = std::move(*alias);
    EXPECT_FALSE(destroyed);
    EXPECT_EQ(loaded->Store(), store);
}

}  // namespace
}  // namespace UC::Proxy
