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
#include <gtest/gtest.h>
#include <list>
#include "detail/random.h"
#include "trans/share_buffer.h"

class UCPCStoreShareBufferTest : public testing::Test {
protected:
    UC::Test::Detail::Random rd;
    std::shared_ptr<UC::ShareBuffer::Reader> MakeBufferWithRandomBlockId(
        UC::ShareBuffer& shareBuffer)
    {
        auto blockId = rd.RandomString(16);
        auto path = "/tmp/block/" + blockId;
        return shareBuffer.MakeReader(blockId, path);
    }
};

TEST_F(UCPCStoreShareBufferTest, ShareBufferUsedOut)
{
    auto uniqueId = "uc-pcstore-test-" + rd.RandomString(10);
    constexpr size_t blockSize = 4096;
    constexpr size_t blockNumber = 4;
    constexpr size_t localBlockNumber = 1;
    UC::ShareBuffer shareBuffer;
    ASSERT_TRUE(shareBuffer.Setup(blockSize, blockNumber, false, uniqueId).Success());
    std::vector<std::shared_ptr<UC::ShareBuffer::Reader>> readers;
    for (size_t i = 0; i < blockNumber + localBlockNumber; i++) {
        auto reader = MakeBufferWithRandomBlockId(shareBuffer);
        ASSERT_NE(reader, nullptr);
        if (i < blockNumber) {
            ASSERT_TRUE(reader->Shared());
        } else {
            ASSERT_FALSE(reader->Shared());
        }
        readers.push_back(std::move(reader));
    }
    std::for_each(readers.begin(), readers.end(),
                  [](auto& reader) { ASSERT_NE(reader->GetData(), 0u); });
}

TEST_F(UCPCStoreShareBufferTest, ShareBufferReuse)
{
    auto uniqueId = "uc-pcstore-test-" + rd.RandomString(10);
    constexpr size_t blockSize = 4096;
    constexpr size_t blockNumber = 4;
    constexpr size_t reuseNumber = 2;
    UC::ShareBuffer shareBuffer;
    ASSERT_TRUE(shareBuffer.Setup(blockSize, blockNumber, false, uniqueId).Success());
    std::vector<std::shared_ptr<UC::ShareBuffer::Reader>> readers;
    for (size_t i = 0; i < blockNumber; ++i) {
        auto reader = MakeBufferWithRandomBlockId(shareBuffer);
        ASSERT_NE(reader, nullptr);
        ASSERT_TRUE(reader->Shared());
        readers.push_back(std::move(reader));
    }
    for (size_t i = 0; i < reuseNumber; ++i) {
        readers.pop_back();
        auto reader = MakeBufferWithRandomBlockId(shareBuffer);
        ASSERT_NE(reader, nullptr);
        ASSERT_TRUE(reader->Shared());
        readers.push_back(std::move(reader));
    }
}

TEST_F(UCPCStoreShareBufferTest, InsertShareBufferToReadTaskList)
{
    struct ReadTask {
        std::string blockId;
        std::shared_ptr<UC::ShareBuffer::Reader> reader;
    };
    auto uniqueId = "uc-pcstore-test-" + rd.RandomString(10);
    constexpr size_t blockNumber = 4;
    constexpr size_t localBlockNumber = 2;
    std::list<ReadTask> totalReadTasks;
    std::list<ReadTask> readTasks;
    UC::ShareBuffer shareBuffer;
    ASSERT_TRUE(shareBuffer.Setup(4096, blockNumber, false, uniqueId).Success());
    for (size_t i = 0; i < blockNumber + localBlockNumber; i++) {
        ReadTask task;
        task.blockId = rd.RandomString(16);
        task.reader = shareBuffer.MakeReader(task.blockId, "/tmp/block/" + task.blockId);
        ASSERT_NE(task.reader, nullptr);
        readTasks.push_back(std::move(task));
    }
    totalReadTasks.splice(totalReadTasks.end(), readTasks);
    ASSERT_EQ(totalReadTasks.size(), blockNumber + localBlockNumber);
    std::for_each(totalReadTasks.begin(), totalReadTasks.end(),
                  [](auto& task) { ASSERT_NE(task.reader->GetData(), 0u); });
    totalReadTasks.clear();
}
