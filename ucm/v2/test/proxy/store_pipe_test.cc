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
#include "store_pipe.h"
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "fixtures/fake_store.h"

namespace UC::Proxy {
namespace {

class LoggingStore : public Store::Store {
public:
    LoggingStore(std::vector<std::string>* log, std::string name)
        : log_(log), name_(std::move(name))
    {
    }

    ~LoggingStore() override { log_->push_back(name_); }

    Status Setup(const Dictionary& /*config*/) override { return Status::Ok(); }

    Expected<std::vector<std::uint8_t>> Lookup(const std::vector<BlockId>& /*blockIds*/) override
    {
        return std::vector<std::uint8_t>{};
    }

    Status Load(std::size_t /*jobId*/, JobDescriptor /*job*/, AddressedToken /*token*/) override
    {
        return Status::Ok();
    }

    Status Dump(std::size_t /*jobId*/, JobDescriptor /*job*/, AddressedToken /*token*/) override
    {
        return Status::Ok();
    }

    std::vector<std::pair<std::size_t, bool>> GetFinished() override { return {}; }

    std::vector<std::pair<std::size_t, bool>> Wait(
        const std::vector<std::size_t>& /*jobIds*/) override
    {
        return {};
    }

private:
    std::vector<std::string>* log_;
    std::string name_;
};

std::unique_ptr<LoadedStore> AdoptLogging(std::vector<std::string>* log, const std::string& name)
{
    return LoadedStore::Adopt(std::unique_ptr<Store::Store>(new LoggingStore(log, name)));
}

TEST(StorePipeTest, TearsDownFromHeadToTails)
{
    std::vector<std::string> log;
    {
        std::vector<std::unique_ptr<LoadedStore>> stores;
        stores.push_back(AdoptLogging(&log, "head"));
        stores.push_back(AdoptLogging(&log, "t1"));
        stores.push_back(AdoptLogging(&log, "t2"));
        StorePipe pipe(std::move(stores));
    }
    EXPECT_EQ(log, (std::vector<std::string>{"head", "t1", "t2"}));
}

TEST(StorePipeTest, ServesAsStoreEntry)
{
    std::vector<std::string> log;
    const std::vector<BlockId> keys = MakeKeys(1, 2);
    auto* fake = new FakeStore(true, true);
    {
        std::vector<std::unique_ptr<LoadedStore>> stores;
        stores.push_back(LoadedStore::Adopt(std::unique_ptr<Store::Store>(fake)));
        stores.push_back(AdoptLogging(&log, "t1"));
        StorePipe pipe(std::move(stores));
        EXPECT_TRUE(pipe.OnNewRequest(MakeContext("r")).Success());
        auto hits = pipe.Lookup("r", keys);
        ASSERT_TRUE(hits);
        ASSERT_TRUE(hits.Value().has_value());
        auto token = pipe.PrepareGet("r", keys);
        ASSERT_TRUE(token);
        EXPECT_EQ(token.Value().keys, keys);
        pipe.CompleteGet("r");
        ASSERT_EQ(fake->pins.size(), 1);
        EXPECT_EQ(fake->pins[0], keys);
        ASSERT_EQ(fake->unpins.size(), 1);
        pipe.OnRequestFinished("r");
    }
    EXPECT_EQ(log, (std::vector<std::string>{"t1"}));
}

TEST(StorePipeTest, AcceptsSingleStoreChain)
{
    std::vector<std::string> log;
    {
        std::vector<std::unique_ptr<LoadedStore>> stores;
        stores.push_back(AdoptLogging(&log, "head"));
        StorePipe pipe(std::move(stores));
    }
    EXPECT_EQ(log, (std::vector<std::string>{"head"}));
}

}  // namespace
}  // namespace UC::Proxy
