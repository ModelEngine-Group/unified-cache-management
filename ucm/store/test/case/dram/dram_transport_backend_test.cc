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
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include "transport_executor.h"

namespace UC::Dram {
namespace {

class MockTransportBackend final : public ITransportBackend {
public:
    Expected<MemoryHandle> RegisterMemory(void* address, std::size_t length,
                                          MemoryRegionType) override
    {
        if (address == nullptr || length == 0) {
            return Status::InvalidParam("invalid memory registration");
        }
        auto handle = nextRegistration_.fetch_add(1, std::memory_order_relaxed);
        if (handle == 0) { return Status::Error("memory handle exhausted"); }
        return handle;
    }

    Status UnregisterMemory(MemoryHandle handle) override
    {
        return handle == 0 ? Status::InvalidParam("invalid memory registration handle")
                           : Status::OK();
    }

    TransmitCompleted Transmit(const ::UC::Dram::Transmit& command) noexcept override
    {
        return TransmitCompleted{
            command.token, Status::Error("no production DramStore transport backend is installed")};
    }

    Status Connect(const ::UC::Dram::Connect& command) noexcept override
    {
        return command.transportManagerId.empty()
                   ? Status::InvalidParam("remote TransportManager id is missing")
                   : Status::OK();
    }
    Status Fence(const ::UC::Dram::FenceEpoch&) noexcept override { return Status::OK(); }
    Status Stop() noexcept override { return Status::OK(); }

private:
    std::atomic<MemoryHandle> nextRegistration_{1};
};

std::shared_ptr<ITransportBackend> CreateMockTransportBackend()
{
    return std::make_shared<MockTransportBackend>();
}

std::string TestManagerId(std::uint16_t port) { return "127.0.0.1:" + std::to_string(port); }

TEST(TransportBackendTest, ConnectionRequiresTransportManagerId)
{
    auto backend = CreateMockTransportBackend();
    Connect command{1, kDefaultLaneId, 1, TestManagerId(1234)};
    EXPECT_TRUE(backend->Connect(command).Success());
    command.transportManagerId.clear();
    EXPECT_EQ(backend->Connect(command), Status::InvalidParam());
}

TEST(TransportBackendTest, RegistrationReturnsOpaqueHandle)
{
    auto backend = CreateMockTransportBackend();
    std::uint32_t memory = 0;
    auto registration = backend->RegisterMemory(&memory, sizeof(memory), MemoryRegionType::HOST);
    ASSERT_TRUE(registration);
    EXPECT_NE(registration.Value(), MemoryHandle{0});
    EXPECT_TRUE(backend->UnregisterMemory(registration.Value()).Success());
}

TEST(TransportExecutorTest, PublisherIsConstructorDependency)
{
    TransportExecutor invalid(
        TransportExecutor::Options{1, 1, 1, CreateMockTransportBackend(), {}});
    EXPECT_TRUE(invalid.Start().Failure());

    TransportExecutor executor(TransportExecutor::Options{1, 1, 1, CreateMockTransportBackend(),
                                                          [](NodeId, NodeEvent) {}});
    ASSERT_TRUE(executor.Start().Success());
    EXPECT_TRUE(executor.Shutdown().Success());
}

TEST(TransportExecutorTest, ShutdownStopsNewCommands)
{
    TransportExecutor executor(TransportExecutor::Options{1, 1, 1, CreateMockTransportBackend(),
                                                          [](NodeId, NodeEvent) {}});
    ASSERT_TRUE(executor.Start().Success());
    ASSERT_TRUE(executor.Shutdown().Success());

    TransportCommand command{
        Connect{1, kDefaultLaneId, 1, TestManagerId(1234)}
    };
    EXPECT_TRUE(executor.TryPost(command).Failure());
}

TEST(TransportExecutorTest, EventPublisherExceptionIsFatal)
{
    EXPECT_DEATH(
        {
            TransportExecutor executor(TransportExecutor::Options{
                1, 1, 1, CreateMockTransportBackend(),
                [](NodeId, NodeEvent) { throw std::runtime_error("event receiver failed"); }});
            (void)executor.Start();
            TransportCommand command(Connect{1, kDefaultLaneId, 1, TestManagerId(1234)});
            (void)executor.TryPost(command);
            (void)executor.Shutdown();
        },
        "");
}

TEST(TransportExecutorTest, RequestLimitsDeriveSeparateCommandAndFenceBudgets)
{
    std::mutex mutex;
    std::condition_variable changed;
    bool publisherEntered = false;
    bool releasePublisher = false;
    TransportExecutor executor(
        TransportExecutor::Options{1, 1, 1, CreateMockTransportBackend(), [&](NodeId, NodeEvent) {
                                       std::unique_lock lock(mutex);
                                       publisherEntered = true;
                                       changed.notify_all();
                                       changed.wait(lock, [&] { return releasePublisher; });
                                   }});
    ASSERT_TRUE(executor.Start().Success());

    TransportCommand executing{
        Connect{1, kDefaultLaneId, 1, TestManagerId(1234)}
    };
    ASSERT_TRUE(executor.TryPost(executing).Success());
    bool entered = false;
    {
        std::unique_lock lock(mutex);
        entered = changed.wait_for(lock, std::chrono::seconds{1}, [&] { return publisherEntered; });
    }

    TransportCommand firstCommand{
        Connect{1, kDefaultLaneId, 2, TestManagerId(1234)}
    };
    TransportCommand secondCommand{
        Connect{1, kDefaultLaneId, 3, TestManagerId(1234)}
    };
    TransportCommand commandOverflow{
        Connect{1, kDefaultLaneId, 4, TestManagerId(1234)}
    };
    TransportCommand fence{
        FenceEpoch{1, kDefaultLaneId, 1}
    };
    TransportCommand fenceOverflow{
        FenceEpoch{1, kDefaultLaneId, 2}
    };
    const auto firstStatus = executor.TryPost(firstCommand);
    const auto secondStatus = executor.TryPost(secondCommand);
    const auto commandOverflowStatus = executor.TryPost(commandOverflow);
    const auto fenceStatus = executor.TryPost(fence);
    const auto fenceOverflowStatus = executor.TryPost(fenceOverflow);

    {
        std::lock_guard lock(mutex);
        releasePublisher = true;
    }
    changed.notify_all();

    EXPECT_TRUE(entered);
    EXPECT_TRUE(firstStatus.Success());
    EXPECT_TRUE(secondStatus.Success());
    EXPECT_EQ(commandOverflowStatus, Status::Error());
    EXPECT_TRUE(fenceStatus.Success());
    EXPECT_EQ(fenceOverflowStatus, Status::Error());
    EXPECT_TRUE(executor.Shutdown().Success());
}

}  // namespace
}  // namespace UC::Dram
