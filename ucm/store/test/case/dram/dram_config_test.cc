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
#include <string>
#include <vector>
#include "config.h"

namespace UC::Dram {
namespace {

Detail::Dictionary BaseConfig(bool includeTensorSizes = true)
{
    Detail::Dictionary config;
    config.Set("local_control_endpoint", std::string{"127.0.0.1:6000"});
    config.Set("local_host", std::string{"127.0.0.1"});
    config.Set("local_transport_manager_id", std::string{"127.0.0.1:6100"});
    config.Set("node_control_endpoints",
               std::vector<std::string>{"127.0.0.1:7000", "127.0.0.1:9000"});
    config.Set("node_transport_manager_ids",
               std::vector<std::string>{"127.0.0.1:7100", "127.0.0.1:9100"});
    if (includeTensorSizes) { config.Set("tensor_size_list", std::vector<ssize_t>{4096}); }
    return config;
}

TEST(DramConfigTest, KeepsOnlyGlobalIoEntryBudget)
{
    auto input = BaseConfig();
    input.SetNumber("max_io_entries", 23);
    auto parsed = DramConfig::Parse(input);
    ASSERT_TRUE(parsed);
    EXPECT_EQ(parsed.Value().maxIoEntries, std::size_t{23});
}

TEST(DramConfigTest, DerivesNodeAndReplyCapacitiesFromGlobalEntryBudget)
{
    auto input = BaseConfig();
    input.SetNumber("max_io_entries", 7);
    auto parsed = DramConfig::Parse(input);
    ASSERT_TRUE(parsed);
    const auto& config = parsed.Value();
    EXPECT_EQ(config.nodeScheduler.limits.maxInflightRequests, std::size_t{7});
    EXPECT_EQ(config.nodeScheduler.limits.maxBatchEntries, std::size_t{7});
    EXPECT_EQ(config.replySlotCount, std::size_t{14});
    EXPECT_EQ(config.replySlotSize, std::uint32_t{13});
}

TEST(DramConfigTest, CapsDerivedNodeAndReplyCapacities)
{
    auto input = BaseConfig();
    auto parsed = DramConfig::Parse(input);
    ASSERT_TRUE(parsed);
    const auto& config = parsed.Value();
    EXPECT_EQ(config.nodeScheduler.limits.maxInflightRequests, std::size_t{128});
    EXPECT_EQ(config.nodeScheduler.limits.maxBatchEntries, std::size_t{128});
    EXPECT_EQ(config.replySlotCount, std::size_t{256});
    EXPECT_EQ(config.replySlotSize, std::uint32_t{73});
}

TEST(DramConfigTest, OverridesWorkerCountIndependently)
{
    auto input = BaseConfig();
    input.SetNumber("transport_worker_count", 7);
    auto parsed = DramConfig::Parse(input);
    ASSERT_TRUE(parsed);
    EXPECT_EQ(parsed.Value().transportRuntime.workerCount, std::size_t{7});
}

TEST(DramConfigTest, OverridesRunnerCountIndependently)
{
    auto input = BaseConfig();
    input.SetNumber("node_runner_count", 3);
    auto parsed = DramConfig::Parse(input);
    ASSERT_TRUE(parsed);
    EXPECT_EQ(parsed.Value().nodeScheduler.runnerCount, std::size_t{3});
}

TEST(DramConfigTest, ParsesFixedReconnectInterval)
{
    auto input = BaseConfig();
    input.SetNumber("reconnect_interval_ms", 37);
    auto parsed = DramConfig::Parse(input);
    ASSERT_TRUE(parsed);
    EXPECT_EQ(parsed.Value().nodeScheduler.reconnectInterval.count(), 37);
}

TEST(DramConfigTest, RequiresTensorSizes) { EXPECT_FALSE(DramConfig::Parse(BaseConfig(false))); }

TEST(DramConfigTest, ParsesRouterTypeIntoStrongConfiguration)
{
    auto input = BaseConfig();
    input.Set("router_type", std::string{"maglev"});
    auto parsed = DramConfig::Parse(input);
    ASSERT_TRUE(parsed);
    EXPECT_EQ(parsed.Value().routerType, UC::Router::RouterType::MAGLEV_FULL_SPREAD);

    input.Set("router_type", std::string{"unsupported"});
    EXPECT_FALSE(DramConfig::Parse(input));
}

TEST(DramConfigTest, RejectsMalformedControlEndpointsAndEmptyManagerIds)
{
    auto local = BaseConfig();
    local.Set("local_control_endpoint", std::string{"client"});
    EXPECT_FALSE(DramConfig::Parse(local));

    auto manager = BaseConfig();
    manager.Set("local_transport_manager_id", std::string{});
    EXPECT_FALSE(DramConfig::Parse(manager));

    auto remote = BaseConfig();
    remote.Set("node_transport_manager_ids", std::vector<std::string>{"", "127.0.0.1:9100"});
    EXPECT_FALSE(DramConfig::Parse(remote));
}

TEST(DramConfigTest, RejectsInvalidSchedulerBoundaries)
{
    auto zeroBudget = BaseConfig();
    zeroBudget.SetNumber("max_io_entries", 0);
    EXPECT_FALSE(DramConfig::Parse(zeroBudget));

    auto mismatchedNodes = BaseConfig();
    mismatchedNodes.Set("node_control_endpoints", std::vector<std::string>{"127.0.0.1:7000"});
    EXPECT_FALSE(DramConfig::Parse(mismatchedNodes));

    auto emptyNodes = BaseConfig();
    emptyNodes.Set("node_control_endpoints", std::vector<std::string>{});
    emptyNodes.Set("node_transport_manager_ids", std::vector<std::string>{});
    EXPECT_FALSE(DramConfig::Parse(emptyNodes));
}

TEST(DramConfigTest, ParsesControlEndpointsAndStoresManagerIds)
{
    auto input = BaseConfig();
    input.Set("role", std::string{"scheduler"});
    input.Set("local_control_endpoint", std::string{"127.0.0.1:06000"});
    auto parsed = DramConfig::Parse(input);
    ASSERT_TRUE(parsed);
    const auto& config = parsed.Value();
    EXPECT_EQ(config.localControlHost, "127.0.0.1");
    EXPECT_EQ(config.localControlPort, std::uint16_t{6000});
    EXPECT_EQ(config.localTransportManagerId, "127.0.0.1:6100");
    ASSERT_EQ(config.nodeScheduler.nodes.size(), std::size_t{2});
    EXPECT_EQ(config.nodeScheduler.nodes[0].nodeId, NodeId{0});
    EXPECT_EQ(config.nodeScheduler.nodes[0].controlHost, "127.0.0.1");
    EXPECT_EQ(config.nodeScheduler.nodes[0].controlPort, std::uint16_t{7000});
    EXPECT_EQ(config.nodeScheduler.nodes[0].transportManagerId, "127.0.0.1:7100");
    EXPECT_EQ(config.nodeScheduler.nodes[1].nodeId, NodeId{1});
}

TEST(DramConfigTest, UsesConfiguredPortsForScheduler)
{
    auto input = BaseConfig();
    input.Set("role", std::string{"scheduler"});
    input.SetNumber("device_id", 3);
    input.SetNumber("hixl_listen_port", 36666);
    input.Set("enable_hixl_cs", true);
    auto parsed = DramConfig::Parse(input);
    ASSERT_TRUE(parsed);
    EXPECT_EQ(parsed.Value().localControlPort, std::uint16_t{6000});
    EXPECT_EQ(parsed.Value().localTransportManagerId, "127.0.0.1:6100");
    EXPECT_EQ(parsed.Value().hixlListenPort, std::uint16_t{36666});
    EXPECT_TRUE(parsed.Value().enableHixlCs);
}

TEST(DramConfigTest, OffsetsWorkerPortsByDeviceId)
{
    auto input = BaseConfig();
    input.Set("role", std::string{"worker"});
    input.SetNumber("device_id", 3);
    auto parsed = DramConfig::Parse(input);
    ASSERT_TRUE(parsed);
    EXPECT_EQ(parsed.Value().localControlPort, std::uint16_t{6004});
    EXPECT_EQ(parsed.Value().localTransportManagerId, "127.0.0.1:6104");
    EXPECT_EQ(parsed.Value().hixlListenPort, std::uint16_t{36667});
}

TEST(DramConfigTest, RejectsInvalidRoleAndWorkerPortOverflow)
{
    auto invalidRole = BaseConfig();
    invalidRole.Set("role", std::string{"server"});
    EXPECT_FALSE(DramConfig::Parse(invalidRole));

    auto overflow = BaseConfig();
    overflow.Set("role", std::string{"worker"});
    overflow.Set("local_control_endpoint", std::string{"127.0.0.1:65535"});
    EXPECT_FALSE(DramConfig::Parse(overflow));

    auto hixlOverflow = BaseConfig();
    hixlOverflow.SetNumber("hixl_listen_port", 65535);
    EXPECT_FALSE(DramConfig::Parse(hixlOverflow));
}

TEST(DramConfigTest, ParsesKvCacheRegistrationConfig)
{
    auto input = BaseConfig();
    input.Set("gpu_kv_buffer_addrs", std::vector<ssize_t>{0x1000, 0x2000});
    input.Set("gpu_kv_buffer_sizes", std::vector<ssize_t>{1024, 2048});
    auto parsed = DramConfig::Parse(input);
    ASSERT_TRUE(parsed);
    EXPECT_EQ(parsed.Value().gpuKvBufferAddrs, (std::vector<uintptr_t>{0x1000, 0x2000}));
    EXPECT_EQ(parsed.Value().gpuKvBufferSizes, (std::vector<std::size_t>{1024, 2048}));
}

TEST(DramConfigTest, RejectsInvalidKvCacheRegistrationConfig)
{
    for (const auto value : {ssize_t{0}, ssize_t{-1}}) {
        auto input = BaseConfig();
        input.Set("gpu_kv_buffer_addrs", std::vector<ssize_t>{value});
        input.Set("gpu_kv_buffer_sizes", std::vector<ssize_t>{1024});
        EXPECT_FALSE(DramConfig::Parse(input));
        input.Set("gpu_kv_buffer_addrs", std::vector<ssize_t>{0x1000});
        input.Set("gpu_kv_buffer_sizes", std::vector<ssize_t>{value});
        EXPECT_FALSE(DramConfig::Parse(input));
    }
    auto input = BaseConfig();
    input.Set("gpu_kv_buffer_addrs", std::vector<ssize_t>{0x1000});
    EXPECT_FALSE(DramConfig::Parse(input));
    input = BaseConfig();
    input.Set("gpu_kv_buffer_sizes", std::vector<ssize_t>{1024});
    EXPECT_FALSE(DramConfig::Parse(input));
}

}  // namespace
}  // namespace UC::Dram
