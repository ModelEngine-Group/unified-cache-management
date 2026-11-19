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
#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <mutex>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include "view/m800_view_master_server.h"
#include "view/m800_view_slave_server.h"


namespace kv::test {
namespace {

struct FakeAgentState {
    mutable std::mutex mutex;
    std::uint64_t epoch{0};
    std::map<std::string, std::vector<M800AgentNodeView>> namespaces;
    std::map<std::string, Status> namespaceStatus;
    Status initializeStatus{Status::OK()};
    Status startStatus{Status::OK()};
    Status epochStatus{Status::OK()};
    Status namesStatus{Status::OK()};
    std::atomic<int> stopCount{0};
    std::atomic<bool> blockStop{false};
    std::atomic<bool> stopEntered{false};
    std::atomic<int> getInDelayMs{0};
    std::atomic<bool> blockGetIn{false};
    std::atomic<bool> getInEntered{false};
};

M800AgentNodeView MakeNode(std::int32_t osdId, const std::string& address, bool up = true,
                           bool in = true)
{
    M800AgentNodeView node;
    node.osdId = osdId;
    node.up = up;
    node.in = in;
    node.addrs.emplace_back(address);
    return node;
}

NodeInfo MakeNodeInfo(const std::string& ip)
{
    NodeEndpoint endpoint;
    endpoint.ip = ip;
    endpoint.port = 6000;
    endpoint.protocol = Protocol::UB;
    endpoint.attrs["protocol"] = "uboe";
    return NodeInfo{{endpoint}};
}

class FakeAgent final : public M800ViewAgentClient {
public:
    explicit FakeAgent(std::shared_ptr<FakeAgentState> state) : state_(std::move(state)) {}

    Status Initialize() override
    {
        std::lock_guard<std::mutex> guard(state_->mutex);
        return state_->initializeStatus;
    }
    Status Start() override
    {
        std::lock_guard<std::mutex> guard(state_->mutex);
        return state_->startStatus;
    }
    void Stop() override
    {
        ++state_->stopCount;
        state_->blockGetIn.store(false);
        state_->stopEntered.store(true);
        while (state_->blockStop.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    Status GetEpoch(std::uint64_t& epoch) const override
    {
        std::lock_guard<std::mutex> guard(state_->mutex);
        if (!state_->epochStatus.ok()) { return state_->epochStatus; }
        epoch = state_->epoch;
        return Status::OK();
    }

    Status GetAllNsNames(std::vector<std::string>& names) const override
    {
        std::lock_guard<std::mutex> guard(state_->mutex);
        if (!state_->namesStatus.ok()) { return state_->namesStatus; }
        names.clear();
        for (const auto& [name, unused] : state_->namespaces) {
            (void)unused;
            names.emplace_back(name);
        }
        return Status::OK();
    }

    Status GetViewByNs(const std::string& name, bool up, bool in,
                       std::vector<M800AgentNodeView>& views) const override
    {
        auto status = GetInByNs(name, views);
        if (!status.ok()) { return status; }
        views.erase(
            std::remove_if(views.begin(), views.end(),
                           [up, in](const auto& node) { return node.up != up || node.in != in; }),
            views.end());
        return Status::OK();
    }

    Status GetInByNs(const std::string& name, std::vector<M800AgentNodeView>& views) const override
    {
        if (state_->blockGetIn.load()) {
            state_->getInEntered.store(true);
            while (state_->blockGetIn.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        const auto delay = state_->getInDelayMs.load();
        if (delay > 0) { std::this_thread::sleep_for(std::chrono::milliseconds(delay)); }
        std::lock_guard<std::mutex> guard(state_->mutex);
        const auto failure = state_->namespaceStatus.find(name);
        if (failure != state_->namespaceStatus.end() && !failure->second.ok()) {
            return failure->second;
        }
        const auto iter = state_->namespaces.find(name);
        if (iter == state_->namespaces.end()) {
            return Status::Error(StatusCode::NOT_FOUND, "namespace not found");
        }
        views = iter->second;
        return Status::OK();
    }

private:
    std::shared_ptr<FakeAgentState> state_;
};

class FakeAgentFactory final : public M800ViewAgentClientFactory {
public:
    explicit FakeAgentFactory(std::shared_ptr<FakeAgentState> state) : state_(std::move(state)) {}

    Status Create(const M800ViewServerConfig& config,
                  std::unique_ptr<M800ViewAgentClient>& client) const override
    {
        (void)config;
        client = std::make_unique<FakeAgent>(state_);
        return Status::OK();
    }

private:
    std::shared_ptr<FakeAgentState> state_;
};

std::uint16_t AllocatePort()
{
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    EXPECT_GE(fd, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    EXPECT_EQ(bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)), 0);
    socklen_t size = sizeof(address);
    EXPECT_EQ(getsockname(fd, reinterpret_cast<sockaddr*>(&address), &size), 0);
    const auto port = ntohs(address.sin_port);
    close(fd);
    return port;
}

M800ViewServerConfig MasterConfig(std::uint16_t port)
{
    M800ViewServerConfig config;
    config.role = M800ViewRole::MASTER;
    config.listenAddr = "127.0.0.1:" + std::to_string(port);
    config.agentFetchTimeoutMs = 600;
    config.httpTimeoutMs = 500;
    config.endpointProtocol = Protocol::UB;
    config.endpointPort = 6000;
    return config;
}

std::string MakeViewTextEpoch(std::uint64_t epoch)
{
    const auto value = std::to_string(epoch);
    return "globalEpoch=" + value + "\nnsCount=1\nns=1\nviewEpoch=" + value +
           "\nviewId=1\nasuIds=1\n"
           "asuInfo.1=protocol=uboe,port=6000,local.comm_id=10.0.0.1\n";
}

std::unique_ptr<ViewHttpServer> StartTextServer(std::uint16_t port,
                                                std::shared_ptr<std::string> body,
                                                std::shared_ptr<std::mutex> bodyMutex,
                                                std::atomic<int>* requestCount = nullptr)
{
    auto server = CreateViewHttpServer(300);
    EXPECT_TRUE(server
                    ->Start("127.0.0.1:" + std::to_string(port),
                            [body, bodyMutex, requestCount](const ViewHttpRequest&) {
                                if (requestCount != nullptr) { ++(*requestCount); }
                                std::lock_guard<std::mutex> guard(*bodyMutex);
                                return ViewHttpReply{ViewHttpReplyKind::FULL_VIEW, *body};
                            })
                    .ok());
    return server;
}

std::shared_ptr<M800ViewMasterServer> MakeMaster(const M800ViewServerConfig& config,
                                                 const std::shared_ptr<FakeAgentState>& state)
{ return CreateM800ViewMasterServer(config, std::make_unique<FakeAgentFactory>(state)); }

void SetNamespaces(const std::shared_ptr<FakeAgentState>& state, std::uint64_t epoch,
                   const std::map<std::string, std::vector<M800AgentNodeView>>& namespaces)
{
    std::lock_guard<std::mutex> guard(state->mutex);
    state->epoch = epoch;
    state->namespaces = namespaces;
    state->namespaceStatus.clear();
}

class SlowPublishUniverse final : public ViewUniverse {
public:
    explicit SlowPublishUniverse(std::chrono::milliseconds delay)
        : inner_(CreateViewUniverse()), delay_(delay)
    {
    }

    Status PublishAll(const ViewUniverse& next) override
    {
        std::this_thread::sleep_for(delay_);
        return inner_->PublishAll(next);
    }
    Status ReplaceAll(const ViewUniverse& next) override { return inner_->ReplaceAll(next); }
    Status GetView(KvNsId nsId, GlobalView& view) const override
    { return inner_->GetView(nsId, view); }
    void GetSnapshot(std::unordered_map<KvNsId, GlobalView>& views,
                     std::uint64_t& globalEpoch) const override
    { inner_->GetSnapshot(views, globalEpoch); }
    std::unordered_map<KvNsId, GlobalView> GetAllViews() const override
    { return inner_->GetAllViews(); }
    std::vector<KvNsId> GetAllNsIds() const override { return inner_->GetAllNsIds(); }
    std::uint64_t GetGlobalEpoch() const override { return inner_->GetGlobalEpoch(); }
    bool Empty() const override { return inner_->Empty(); }
    void Clear() override { inner_->Clear(); }

private:
    std::shared_ptr<ViewUniverse> inner_;
    std::chrono::milliseconds delay_;
};

// 测试场景：主节点启动时 Agent 尚无命名空间数据，另一线程延迟 300ms 发布 epoch 10 的 pool-a OSD
// 视图。 预期结果：Start 等待初始 OSD Map 就绪后成功返回，耗时至少 200ms，不将空数据视为启动成功。
TEST(M800ViewMasterTest, StartWaitsForInitialOsdMap)
{
    auto state = std::make_shared<FakeAgentState>();
    std::thread publisher([state] {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        SetNamespaces(state, 10,
                      {
                          {"pool-a", {MakeNode(1, "10.0.0.1")}}
        });
    });

    auto masterConfig = MasterConfig(AllocatePort());
    auto master = MakeMaster(masterConfig, state);
    ASSERT_NE(master, nullptr);
    const auto begin = std::chrono::steady_clock::now();
    const auto status = master->Start();
    publisher.join();
    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_GE(std::chrono::steady_clock::now() - begin, std::chrono::milliseconds(200));
    master->Stop();
}

// 测试场景：Agent 始终无初始命名空间数据且获取超时配置为 5000ms，主节点启动等待 250ms 后并发调用
// Stop。 预期结果：Stop 与启动线程汇合在 1500ms 内完成，Start 返回 CANCELED，且错误消息包含
// canceled。
TEST(M800ViewMasterTest, StopInterruptsInitialReadinessWait)
{
    auto state = std::make_shared<FakeAgentState>();
    auto config = MasterConfig(AllocatePort());
    config.agentFetchTimeoutMs = 5000;
    auto master = MakeMaster(config, state);
    Status startStatus;
    std::thread starter([&] { startStatus = master->Start(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    const auto begin = std::chrono::steady_clock::now();
    master->Stop();
    starter.join();
    EXPECT_LT(std::chrono::steady_clock::now() - begin, std::chrono::milliseconds(1500));
    EXPECT_EQ(startStatus.code, StatusCode::CANCELED);
    EXPECT_NE(startStatus.message.find("canceled"), std::string::npos);
}

// 测试场景：主节点启动读取已有 pool-a 时 Agent 的 GetInByNs 被持续阻塞，确认进入查询后并发调用
// Stop。 预期结果：Stop 与启动线程汇合在 1 秒内完成，被阻塞的查询解除，Start 以 CANCELED 返回。
TEST(M800ViewMasterTest, StopInterruptsBlockedNamespaceQuery)
{
    auto state = std::make_shared<FakeAgentState>();
    SetNamespaces(state, 10,
                  {
                      {"pool-a", {MakeNode(1, "10.0.0.1")}}
    });
    state->blockGetIn.store(true);
    auto master = MakeMaster(MasterConfig(AllocatePort()), state);
    Status startStatus;
    std::thread starter([&] { startStatus = master->Start(); });
    for (int index = 0; index < 2000 && !state->getInEntered.load(); ++index) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(state->getInEntered.load());

    const auto begin = std::chrono::steady_clock::now();
    master->Stop();
    starter.join();
    EXPECT_LT(std::chrono::steady_clock::now() - begin, std::chrono::seconds(1));
    EXPECT_EQ(startStatus.code, StatusCode::CANCELED);
}

// 测试场景：对已有 pool-a 视图的主节点连续 20 轮并发执行 Start 与 Stop，并交替让出 CPU 或延迟 1ms
// 改变竞争时序。 预期结果：每轮均能正常汇合且 Start 仅返回 OK 或 CANCELED，随后重复 Stop
// 不发生自死锁。
TEST(M800ViewMasterTest, ConcurrentStartStopDoesNotSelfDeadlock)
{
    for (int iteration = 0; iteration < 20; ++iteration) {
        auto state = std::make_shared<FakeAgentState>();
        SetNamespaces(state, 10,
                      {
                          {"pool-a", {MakeNode(1, "10.0.0.1")}}
        });
        auto master = MakeMaster(MasterConfig(AllocatePort()), state);
        Status startStatus;
        std::thread starter([&] { startStatus = master->Start(); });
        if (iteration % 2 == 0) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        master->Stop();
        starter.join();
        EXPECT_TRUE(startStatus.ok() || startStatus.code == StatusCode::CANCELED)
            << "concurrent Stop must produce OK or CANCELED, got: " << startStatus.message;
        master->Stop();
    }
}

// 测试场景：主节点已启动且 HTTP 连接只发送部分请求，停止过程同时被阻塞在 Agent::Stop，期间再次调用
// Start。 预期结果：停止中的并发 Start 返回 CANCELED；关闭不完整请求的 HTTP 连接并解除 Agent
// 停止阻塞后，可完成停止与清理。
TEST(M800ViewMasterTest, StartWhileHttpStopIsBlockedReturnsCanceled)
{
    auto state = std::make_shared<FakeAgentState>();
    SetNamespaces(state, 10,
                  {
                      {"pool-a", {MakeNode(1, "10.0.0.1")}}
    });
    const auto port = AllocatePort();
    auto config = MasterConfig(port);
    config.httpTimeoutMs = 2000;
    auto master = MakeMaster(config, state);
    ASSERT_TRUE(master->Start().ok());

    const int client = socket(AF_INET, SOCK_STREAM, 0);
    ASSERT_GE(client, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(port);
    ASSERT_EQ(connect(client, reinterpret_cast<sockaddr*>(&address), sizeof(address)), 0);
    const std::string partialRequest = "GET /v1/views";
    ASSERT_GT(send(client, partialRequest.data(), partialRequest.size(), MSG_NOSIGNAL), 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    state->blockStop.store(true);
    state->stopEntered.store(false);
    std::thread stopping([&] { master->Stop(); });
    for (int index = 0; index < 2000 && !state->stopEntered.load(); ++index) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!state->stopEntered.load()) {
        state->blockStop.store(false);
        shutdown(client, SHUT_RDWR);
        close(client);
        stopping.join();
        FAIL() << "timed out waiting for agent Stop";
        return;
    }
    EXPECT_EQ(master->Start().code, StatusCode::CANCELED);
    shutdown(client, SHUT_RDWR);
    close(client);
    state->blockStop.store(false);
    stopping.join();
    master->Stop();
}

// 测试场景：主节点已成功加载 epoch 10 的 pool-a 初始视图并保持运行，再次调用 Start。
// 预期结果：首次 Start 成功，第二次 Start 返回 RESOURCE_BUSY。
TEST(M800ViewMasterTest, DuplicateStartStillReturnsResourceBusy)
{
    auto state = std::make_shared<FakeAgentState>();
    SetNamespaces(state, 10,
                  {
                      {"pool-a", {MakeNode(1, "10.0.0.1")}}
    });
    auto master = MakeMaster(MasterConfig(AllocatePort()), state);
    ASSERT_TRUE(master->Start().ok());
    EXPECT_EQ(master->Start().code, StatusCode::RESOURCE_BUSY);
    master->Stop();
}

// 测试场景：初始 pool-a 节点的地址为含嵌入 NUL 及尾随 junk 的字符串 10.0.0.1\0junk。
// 预期结果：主节点拒绝该非法地址，Start 返回 NOT_FOUND，而不是按 NUL 前缀截断后启动。
TEST(M800ViewMasterTest, EmbeddedNulAddressIsRejected)
{
    auto state = std::make_shared<FakeAgentState>();
    const auto invalidAddress = std::string("10.0.0.1\0junk", 13);
    SetNamespaces(state, 10,
                  {
                      {"pool-a", {MakeNode(1, invalidAddress)}}
    });
    auto master = MakeMaster(MasterConfig(AllocatePort()), state);
    EXPECT_EQ(master->Start().code, StatusCode::NOT_FOUND);
    master->Stop();
}

// 测试场景：初始 pool-a 节点地址显式指定端口 0（10.0.0.1:0）。
// 预期结果：主节点不使用默认端口兜底，Start 返回 NOT_FOUND。
TEST(M800ViewMasterTest, ExplicitZeroPortAddressIsRejected)
{
    auto state = std::make_shared<FakeAgentState>();
    SetNamespaces(state, 10,
                  {
                      {"pool-a", {MakeNode(1, "10.0.0.1:0")}}
    });
    auto master = MakeMaster(MasterConfig(AllocatePort()), state);
    EXPECT_EQ(master->Start().code, StatusCode::NOT_FOUND);
    master->Stop();
}

// 测试场景：初始 Agent 数据同时包含哈希映射到同一 KvNsId 的 pool-69670 与
// pool-261455，获取超时配置为 5000ms。 预期结果：Start 不进入长时间重试，并在 1 秒内返回
// INTERNAL_ERROR。
TEST(M800ViewMasterTest, InitialHashCollisionReturnsInternalErrorImmediately)
{
    auto state = std::make_shared<FakeAgentState>();
    SetNamespaces(
        state, 10,
        {
            {"pool-69670",  {MakeNode(1, "10.0.0.1")}},
            {"pool-261455", {MakeNode(2, "10.0.0.2")}}
    });
    auto config = MasterConfig(AllocatePort());
    config.agentFetchTimeoutMs = 5000;
    auto master = MakeMaster(config, state);
    const auto begin = std::chrono::steady_clock::now();
    const auto status = master->Start();
    EXPECT_EQ(status.code, StatusCode::INTERNAL_ERROR);
    EXPECT_LT(std::chrono::steady_clock::now() - begin, std::chrono::seconds(1));
    master->Stop();
}

// 测试场景：主节点停止被阻塞在 Agent::Stop 时并发发起 Start，20ms 后解除停止阻塞；若重启遭遇
// RESOURCE_BUSY 或 CANCELED 则再启动一次。 预期结果：最终 Start 成功且新 Agent
// 未丢失，可读取全量视图并保持全局 epoch 为 10。
TEST(M800ViewMasterTest, RestartDuringStopNeverLosesNewAgent)
{
    auto state = std::make_shared<FakeAgentState>();
    SetNamespaces(state, 10,
                  {
                      {"pool-a", {MakeNode(1, "10.0.0.1")}}
    });
    auto master = MakeMaster(MasterConfig(AllocatePort()), state);
    ASSERT_TRUE(master->Start().ok());

    state->blockStop.store(true);
    state->stopEntered.store(false);
    std::thread stopping([&] { master->Stop(); });
    for (int index = 0; index < 2000 && !state->stopEntered.load(); ++index) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!state->stopEntered.load()) {
        state->blockStop.store(false);
        stopping.join();
        FAIL() << "timed out waiting for agent Stop";
        return;
    }

    Status restartStatus;
    std::thread restarting([&] { restartStatus = master->Start(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    state->blockStop.store(false);
    stopping.join();
    restarting.join();
    if (restartStatus.code == StatusCode::RESOURCE_BUSY ||
        restartStatus.code == StatusCode::CANCELED) {
        restartStatus = master->Start();
    }
    ASSERT_TRUE(restartStatus.ok()) << restartStatus.message;
    auto universe = CreateViewUniverse();
    EXPECT_TRUE(master->GetAllViews(*universe).ok());
    EXPECT_EQ(universe->GetGlobalEpoch(), 10);
    master->Stop();
}

// 测试场景：Agent 始终没有命名空间数据，主节点初始获取截止时间为 40ms，启动会经历空数据重试。
// 预期结果：Start 返回 CONNECTION_ERROR，且总耗时低于 180ms，不因重试间隔显著越过配置截止时间。
TEST(M800ViewMasterTest, InitialRetryWaitDoesNotExceedConfiguredDeadline)
{
    auto state = std::make_shared<FakeAgentState>();
    auto config = MasterConfig(AllocatePort());
    config.agentFetchTimeoutMs = 40;
    auto master = MakeMaster(config, state);
    const auto begin = std::chrono::steady_clock::now();
    const auto status = master->Start();
    EXPECT_EQ(status.code, StatusCode::CONNECTION_ERROR);
    EXPECT_LT(std::chrono::steady_clock::now() - begin, std::chrono::milliseconds(180));
    master->Stop();
}

// 测试场景：初始 pool-a 视图在 Agent 延迟 80ms 后可用，但主节点获取截止时间为
// 30ms，随后用从节点读取并在取消延迟后重启主节点。 预期结果：首次 Start 返回
// CONNECTION_ERROR，超时后的迟到结果不发布、从节点拉取也返回 CONNECTION_ERROR；重启后视图成功发布且
// epoch 为 10。
TEST(M800ViewMasterTest, SlowInitialSuccessAfterDeadlineIsNotPublished)
{
    auto state = std::make_shared<FakeAgentState>();
    SetNamespaces(state, 10,
                  {
                      {"pool-a", {MakeNode(1, "10.0.0.1")}}
    });
    state->getInDelayMs.store(80);
    auto config = MasterConfig(AllocatePort());
    config.agentFetchTimeoutMs = 30;
    const auto port = AllocatePort();
    config.listenAddr = "127.0.0.1:" + std::to_string(port);
    auto master = MakeMaster(config, state);
    EXPECT_EQ(master->Start().code, StatusCode::CONNECTION_ERROR);

    M800ViewServerConfig workerConfig;
    workerConfig.role = M800ViewRole::SLAVE;
    workerConfig.masterAddrs = {config.listenAddr};
    workerConfig.httpTimeoutMs = 100;
    auto worker = CreateM800ViewSlaveServer(workerConfig);
    auto universe = CreateViewUniverse();
    EXPECT_EQ(worker->GetAllViews(*universe).code, StatusCode::CONNECTION_ERROR);

    state->getInDelayMs.store(0);
    ASSERT_TRUE(master->Start().ok());
    EXPECT_TRUE(master->GetAllViews(*universe).ok());
    EXPECT_EQ(universe->GetGlobalEpoch(), 10);
    worker->Stop();
    master->Stop();
}

// 测试场景：Agent 始终不提供初始命名空间数据，主节点 Agent 获取超时配置为 250ms。
// 预期结果：Start 等待至超时窗口后返回 CONNECTION_ERROR，实际耗时至少 200ms。
TEST(M800ViewMasterTest, InitialTimeoutReturnsConnectionError)
{
    auto state = std::make_shared<FakeAgentState>();
    auto config = MasterConfig(AllocatePort());
    config.agentFetchTimeoutMs = 250;
    auto master = MakeMaster(config, state);
    const auto begin = std::chrono::steady_clock::now();
    const auto status = master->Start();
    EXPECT_EQ(status.code, StatusCode::CONNECTION_ERROR);
    EXPECT_GE(std::chrono::steady_clock::now() - begin, std::chrono::milliseconds(200));
    master->Stop();
}

// 测试场景：主从先同步两个命名空间，再将 pool-b 置空、将全部命名空间置空，并在 Master 停止后调用
// legacy GetGlobalView。 预期结果：任一 namespace 失败都使本轮返回 NOT_FOUND 且不发布候选，Worker
// 始终保留 epoch 123 的完整旧视图；GetGlobalView 固定返回 UNSUPPORTED、输出不变且不触发网络或 Agent
// 查询。
TEST(M800ViewMasterSlaveTest, SynchronizesMultipleNamespacesAndRejectsPartialFailure)
{
    auto state = std::make_shared<FakeAgentState>();
    SetNamespaces(
        state, 123,
        {
            {"pool-a",
             {MakeNode(10, "10.0.0.10", true, true), MakeNode(11, "10.0.0.11", false, true)}},
            {"pool-b", {MakeNode(20, "10.0.0.20:7000")}                                     }
    });
    const auto port = AllocatePort();
    auto master = MakeMaster(MasterConfig(port), state);
    ASSERT_TRUE(master->Start().ok());

    M800ViewServerConfig workerConfig;
    workerConfig.role = M800ViewRole::SLAVE;
    workerConfig.masterAddrs = {"127.0.0.1:" + std::to_string(port)};
    workerConfig.httpTimeoutMs = 500;
    auto worker = CreateM800ViewSlaveServer(workerConfig);
    auto universe = CreateViewUniverse();
    ASSERT_TRUE(worker->GetAllViews(*universe).ok());
    EXPECT_EQ(universe->GetGlobalEpoch(), 123);
    EXPECT_EQ(universe->GetAllNsIds().size(), 2);

    const auto nsA = M800PoolNameToKvNsId("pool-a");
    GlobalView viewA;
    ASSERT_TRUE(universe->GetView(nsA, viewA).ok());
    EXPECT_EQ(viewA.viewId, nsA);
    EXPECT_EQ(viewA.viewEpoch, 123);
    EXPECT_EQ(viewA.asuMap.size(), 2);
    EXPECT_EQ(viewA.asuMap.at(10).endpoints.front().protocol, Protocol::UB);
    EXPECT_EQ(viewA.asuMap.at(10).endpoints.front().port, 6000);
    EXPECT_EQ(viewA.asuMap.at(11).endpoints.front().port, 6000);

    ASSERT_TRUE(worker->GetAllViews(*universe).ok());
    EXPECT_EQ(universe->GetAllNsIds().size(), 2);

    SetNamespaces(state, 124,
                  {
                      {"pool-a", {MakeNode(12, "10.0.0.12")}},
                      {"pool-b", {}                         }
    });
    auto refreshed = CreateViewUniverse();
    EXPECT_EQ(master->GetAllViews(*refreshed).code, StatusCode::NOT_FOUND);
    EXPECT_TRUE(refreshed->Empty());
    ASSERT_TRUE(worker->GetAllViews(*universe).ok());
    EXPECT_EQ(universe->GetGlobalEpoch(), 123);
    EXPECT_EQ(universe->GetAllNsIds().size(), 2);
    ASSERT_TRUE(universe->GetView(nsA, viewA).ok());
    EXPECT_EQ(viewA.asuMap.count(12), 0);
    EXPECT_EQ(viewA.asuMap.count(10), 1);
    GlobalView viewB;
    ASSERT_TRUE(universe->GetView(M800PoolNameToKvNsId("pool-b"), viewB).ok());
    EXPECT_EQ(viewB.asuMap.count(20), 1);
    EXPECT_EQ(viewB.viewEpoch, 123);

    SetNamespaces(state, 125,
                  {
                      {"pool-a", {}},
                      {"pool-b", {}}
    });
    EXPECT_EQ(master->GetAllViews(*refreshed).code, StatusCode::NOT_FOUND);
    ASSERT_TRUE(worker->GetAllViews(*universe).ok());
    EXPECT_EQ(universe->GetGlobalEpoch(), 123);
    EXPECT_EQ(universe->GetAllNsIds().size(), 2);

    master->Stop();
    GlobalView legacyMasterView;
    legacyMasterView.viewId = 999;
    GlobalView legacyWorkerView;
    legacyWorkerView.viewId = 999;
    EXPECT_EQ(master->GetGlobalView(legacyMasterView).code, StatusCode::UNSUPPORTED);
    EXPECT_EQ(worker->GetGlobalView(legacyWorkerView).code, StatusCode::UNSUPPORTED);
    EXPECT_EQ(legacyMasterView.viewId, 999);
    EXPECT_EQ(legacyWorkerView.viewId, 999);
    worker->Stop();
}

// 测试场景：主节点视图先从 pool-a/pool-b 变为仅 pool-a，再新增 pool-c，最后同时下发哈希冲突的
// pool-69670 与 pool-261455。 预期结果：删除和新增按 Agent 当前权威集合生效，命名空间数依次为
// 2、1、2；冲突刷新返回 INTERNAL_ERROR 且全局 epoch 保持 102，不部分覆盖旧视图。
TEST(M800ViewMasterSlaveTest, NamespaceDeletionAndHashCollisionPreserveAuthorityRules)
{
    auto state = std::make_shared<FakeAgentState>();
    SetNamespaces(
        state, 100,
        {
            {"pool-a", {MakeNode(1, "10.0.0.1")}},
            {"pool-b", {MakeNode(2, "10.0.0.2")}}
    });
    auto master = MakeMaster(MasterConfig(AllocatePort()), state);
    ASSERT_TRUE(master->Start().ok());
    auto universe = CreateViewUniverse();
    ASSERT_TRUE(master->GetAllViews(*universe).ok());
    EXPECT_EQ(universe->GetAllNsIds().size(), 2);

    SetNamespaces(state, 101,
                  {
                      {"pool-a", {MakeNode(3, "10.0.0.3")}}
    });
    ASSERT_TRUE(master->GetAllViews(*universe).ok());
    EXPECT_EQ(universe->GetAllNsIds().size(), 1);
    EXPECT_EQ(universe->GetAllNsIds().front(), M800PoolNameToKvNsId("pool-a"));

    SetNamespaces(
        state, 102,
        {
            {"pool-a", {MakeNode(3, "10.0.0.3")}},
            {"pool-c", {MakeNode(6, "10.0.0.6")}}
    });
    ASSERT_TRUE(master->GetAllViews(*universe).ok());
    EXPECT_EQ(universe->GetAllNsIds().size(), 2);
    GlobalView viewC;
    EXPECT_EQ(universe->GetView(M800PoolNameToKvNsId("pool-c"), viewC).code, StatusCode::OK);

    ASSERT_EQ(M800PoolNameToKvNsId("pool-69670"), M800PoolNameToKvNsId("pool-261455"));
    SetNamespaces(
        state, 103,
        {
            {"pool-69670",  {MakeNode(4, "10.0.0.4")}},
            {"pool-261455", {MakeNode(5, "10.0.0.5")}}
    });
    const auto previousEpoch = universe->GetGlobalEpoch();
    EXPECT_EQ(master->GetAllViews(*universe).code, StatusCode::INTERNAL_ERROR);
    EXPECT_EQ(universe->GetGlobalEpoch(), previousEpoch);
    master->Stop();
}

// 测试场景：从节点以 300ms HTTP 超时拉取 epoch 10 的完整视图，但目标 ViewUniverse::PublishAll
// 额外阻塞 400ms。 预期结果：发布耗时超过 HTTP 超时不导致失败，GetAllViews 返回 OK 且全局 epoch 为
// 10。
TEST(M800ViewSlaveTest, SuccessfulPublishAfterDeadlineReturnsSuccess)
{
    auto body = std::make_shared<std::string>(MakeViewTextEpoch(10));
    auto bodyMutex = std::make_shared<std::mutex>();
    const auto port = AllocatePort();
    auto server = StartTextServer(port, body, bodyMutex);
    M800ViewServerConfig config;
    config.role = M800ViewRole::SLAVE;
    config.masterAddrs = {"127.0.0.1:" + std::to_string(port)};
    config.httpTimeoutMs = 300;
    auto worker = CreateM800ViewSlaveServer(config);
    SlowPublishUniverse universe{std::chrono::milliseconds{400}};

    EXPECT_TRUE(worker->GetAllViews(universe).ok());
    EXPECT_EQ(universe.GetGlobalEpoch(), 10);
    worker->Stop();
    server->Stop();
}

// 测试场景：从节点已缓存 epoch 10 后，主节点依次返回 epoch 9 全量文本、非法文本、空全量响应及空 304
// NOT_MODIFIED。 预期结果：lower epoch 返回 INVALID_ARGUMENT，非法或空全量响应返回
// INTERNAL_ERROR，304 返回 OK，所有情况下现有全局 epoch 均保持 10。
TEST(M800ViewSlaveTest, EmptyMalformedAndOlderResponsesPreserveNewerView)
{
    std::mutex responseMutex;
    std::string responseBody = MakeViewTextEpoch(10);
    ViewHttpReplyKind responseKind = ViewHttpReplyKind::FULL_VIEW;
    const auto port = AllocatePort();
    auto server = CreateViewHttpServer(300);
    ASSERT_TRUE(server
                    ->Start("127.0.0.1:" + std::to_string(port),
                            [&](const ViewHttpRequest&) {
                                std::lock_guard<std::mutex> guard(responseMutex);
                                return ViewHttpReply{responseKind, responseBody};
                            })
                    .ok());

    M800ViewServerConfig config;
    config.role = M800ViewRole::SLAVE;
    config.masterAddrs = {"127.0.0.1:" + std::to_string(port)};
    config.httpTimeoutMs = 300;
    auto worker = CreateM800ViewSlaveServer(config);
    auto universe = CreateViewUniverse();
    ASSERT_TRUE(worker->GetAllViews(*universe).ok());
    EXPECT_EQ(universe->GetGlobalEpoch(), 10);

    const auto setResponse = [&](ViewHttpReplyKind kind, const std::string& body) {
        std::lock_guard<std::mutex> guard(responseMutex);
        responseKind = kind;
        responseBody = body;
    };
    setResponse(ViewHttpReplyKind::FULL_VIEW, MakeViewTextEpoch(9));
    EXPECT_EQ(worker->GetAllViews(*universe).code, StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(universe->GetGlobalEpoch(), 10);
    setResponse(ViewHttpReplyKind::FULL_VIEW, "not-a-view");
    EXPECT_EQ(worker->GetAllViews(*universe).code, StatusCode::INTERNAL_ERROR);
    EXPECT_EQ(universe->GetGlobalEpoch(), 10);
    setResponse(ViewHttpReplyKind::FULL_VIEW, "");
    EXPECT_EQ(worker->GetAllViews(*universe).code, StatusCode::INTERNAL_ERROR);
    EXPECT_EQ(universe->GetGlobalEpoch(), 10);
    setResponse(ViewHttpReplyKind::NOT_MODIFIED, "");
    EXPECT_TRUE(worker->GetAllViews(*universe).ok());
    EXPECT_EQ(universe->GetGlobalEpoch(), 10);

    worker->Stop();
    server->Stop();
}

// 测试场景：从节点已保存 epoch 10、节点地址 10.0.0.1 的视图，主节点随后返回同为 epoch 10
// 但地址改为 10.0.0.99 的全量视图。 预期结果：相同 epoch 不覆盖已发布内容，拉取返回
// OK，节点地址仍为 10.0.0.1 且全局 epoch 保持 10。
TEST(M800ViewSlaveTest, EqualEpochDoesNotReplaceUniverse)
{
    auto body = std::make_shared<std::string>(MakeViewTextEpoch(10));
    auto bodyMutex = std::make_shared<std::mutex>();
    const auto port = AllocatePort();
    auto server = StartTextServer(port, body, bodyMutex);
    M800ViewServerConfig config;
    config.role = M800ViewRole::SLAVE;
    config.masterAddrs = {"127.0.0.1:" + std::to_string(port)};
    config.httpTimeoutMs = 300;
    auto worker = CreateM800ViewSlaveServer(config);
    auto universe = CreateViewUniverse();
    ASSERT_TRUE(worker->GetAllViews(*universe).ok());

    {
        std::lock_guard<std::mutex> guard(*bodyMutex);
        *body = MakeViewTextEpoch(10);
        const auto pos = body->find("10.0.0.1");
        ASSERT_NE(pos, std::string::npos);
        body->replace(pos, std::string{"10.0.0.1"}.size(), "10.0.0.99");
    }
    ASSERT_TRUE(worker->GetAllViews(*universe).ok());
    GlobalView view;
    ASSERT_TRUE(universe->GetView(1, view).ok());
    EXPECT_EQ(view.asuMap.at(1).endpoints.front().ip, "10.0.0.1");
    EXPECT_EQ(universe->GetGlobalEpoch(), 10);
    worker->Stop();
    server->Stop();
}

// 测试场景：从节点先向空输出发布 epoch 7，再换用另一个空 ViewUniverse 拉取，最后再次使用已含 epoch
// 7 的原输出；服务器仅对 sinceEpoch=0 返回全量视图。 预期结果：第二个空输出也被填为 epoch
// 7，三次请求的 sinceEpoch 依次为 0、0、7，第三次可接受 304。
TEST(M800ViewSlaveTest, EmptyOutputForcesFullDownloadAfterPreviousFetch)
{
    const auto port = AllocatePort();
    auto server = CreateViewHttpServer(300);
    std::mutex requestMutex;
    std::vector<std::uint64_t> requestedEpochs;
    ASSERT_TRUE(server
                    ->Start("127.0.0.1:" + std::to_string(port),
                            [&](const ViewHttpRequest& request) {
                                {
                                    std::lock_guard<std::mutex> lock(requestMutex);
                                    requestedEpochs.emplace_back(request.sinceEpoch);
                                }
                                if (request.sinceEpoch == 0) {
                                    return ViewHttpReply{ViewHttpReplyKind::FULL_VIEW,
                                                         MakeViewTextEpoch(7)};
                                }
                                return ViewHttpReply{ViewHttpReplyKind::NOT_MODIFIED, {}};
                            })
                    .ok());

    M800ViewServerConfig config;
    config.role = M800ViewRole::SLAVE;
    config.masterAddrs = {"127.0.0.1:" + std::to_string(port)};
    config.httpTimeoutMs = 300;
    auto worker = CreateM800ViewSlaveServer(config);
    auto first = CreateViewUniverse();
    ASSERT_TRUE(worker->GetAllViews(*first).ok());
    ASSERT_EQ(first->GetGlobalEpoch(), 7);

    auto emptyOutput = CreateViewUniverse();
    ASSERT_TRUE(worker->GetAllViews(*emptyOutput).ok());
    EXPECT_FALSE(emptyOutput->Empty());
    EXPECT_EQ(emptyOutput->GetGlobalEpoch(), 7);
    ASSERT_TRUE(worker->GetAllViews(*first).ok());

    {
        std::lock_guard<std::mutex> lock(requestMutex);
        ASSERT_EQ(requestedEpochs.size(), 3);
        EXPECT_EQ(requestedEpochs[0], 0);
        EXPECT_EQ(requestedEpochs[1], 0);
        EXPECT_EQ(requestedEpochs[2], 7);
    }
    worker->Stop();
    server->Stop();
}

// 测试场景：空从节点依次配置两个主节点，首个对 sinceEpoch=0 错误返回 304
// NOT_MODIFIED，备用主节点提供 epoch 10 全量视图。 预期结果：零 epoch 的 304
// 被视为无效并切换备用主节点，最终拉取成功、全局 epoch 为 10 且视图非空。
TEST(M800ViewSlaveTest, ZeroEpoch304IsRejectedAndFallsBackToFullView)
{
    const auto portA = AllocatePort();
    auto serverA = CreateViewHttpServer(300);
    ASSERT_TRUE(serverA
                    ->Start("127.0.0.1:" + std::to_string(portA),
                            [](const ViewHttpRequest&) {
                                return ViewHttpReply{ViewHttpReplyKind::NOT_MODIFIED, {}};
                            })
                    .ok());
    auto bodyB = std::make_shared<std::string>(MakeViewTextEpoch(10));
    auto mutexB = std::make_shared<std::mutex>();
    const auto portB = AllocatePort();
    auto serverB = StartTextServer(portB, bodyB, mutexB);

    M800ViewServerConfig config;
    config.role = M800ViewRole::SLAVE;
    config.masterAddrs = {"127.0.0.1:" + std::to_string(portA),
                          "127.0.0.1:" + std::to_string(portB)};
    config.httpTimeoutMs = 300;
    auto worker = CreateM800ViewSlaveServer(config);
    auto universe = CreateViewUniverse();
    ASSERT_TRUE(worker->GetAllViews(*universe).ok());
    EXPECT_EQ(universe->GetGlobalEpoch(), 10);
    EXPECT_FALSE(universe->Empty());
    worker->Stop();
    serverA->Stop();
    serverB->Stop();
}

// 测试场景：从节点已从首选主节点获得 epoch 10，首选与备用主节点随后都改为 epoch
// 9，并统计备用节点请求数。 预期结果：两个 lower epoch 都被 PublishAll 拒绝并返回
// INVALID_ARGUMENT；备用主节点被请求一次，原 epoch 10 和旧视图保持不变。
TEST(M800ViewSlaveTest, StaleResponsesFailWithoutChangingUniverseAndTryBackup)
{
    auto bodyA = std::make_shared<std::string>(MakeViewTextEpoch(10));
    auto bodyB = std::make_shared<std::string>(MakeViewTextEpoch(10));
    auto mutexA = std::make_shared<std::mutex>();
    auto mutexB = std::make_shared<std::mutex>();
    std::atomic<int> requestsB{0};
    const auto portA = AllocatePort();
    const auto portB = AllocatePort();
    auto serverA = StartTextServer(portA, bodyA, mutexA);
    auto serverB = StartTextServer(portB, bodyB, mutexB, &requestsB);

    M800ViewServerConfig config;
    config.role = M800ViewRole::SLAVE;
    config.masterAddrs = {"127.0.0.1:" + std::to_string(portA),
                          "127.0.0.1:" + std::to_string(portB)};
    config.httpTimeoutMs = 300;
    auto worker = CreateM800ViewSlaveServer(config);
    auto universe = CreateViewUniverse();
    ASSERT_TRUE(worker->GetAllViews(*universe).ok());
    EXPECT_EQ(universe->GetGlobalEpoch(), 10);
    EXPECT_EQ(requestsB.load(), 0);

    {
        std::lock_guard<std::mutex> guardA(*mutexA);
        std::lock_guard<std::mutex> guardB(*mutexB);
        *bodyA = MakeViewTextEpoch(9);
        *bodyB = MakeViewTextEpoch(9);
    }
    EXPECT_EQ(worker->GetAllViews(*universe).code, StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(universe->GetGlobalEpoch(), 10);
    EXPECT_EQ(requestsB.load(), 1);
    worker->Stop();
    serverA->Stop();
    serverB->Stop();
}

// 测试场景：从节点已从首选主节点获得 epoch 10，备用地址无服务，首选主节点随后返回 epoch 9
// 的旧快照。 预期结果：lower epoch 被 PublishAll 拒绝后继续尝试备用地址，最终返回备用连接错误，原
// epoch 10 和旧视图保持不变。
TEST(M800ViewSlaveTest, StaleResponseFailsAndTriesUnavailableBackup)
{
    auto body = std::make_shared<std::string>(MakeViewTextEpoch(10));
    auto bodyMutex = std::make_shared<std::mutex>();
    const auto port = AllocatePort();
    auto server = StartTextServer(port, body, bodyMutex);
    const auto unavailablePort = AllocatePort();
    M800ViewServerConfig config;
    config.role = M800ViewRole::SLAVE;
    config.masterAddrs = {"127.0.0.1:" + std::to_string(port),
                          "127.0.0.1:" + std::to_string(unavailablePort)};
    config.httpTimeoutMs = 150;
    auto worker = CreateM800ViewSlaveServer(config);
    auto universe = CreateViewUniverse();
    ASSERT_TRUE(worker->GetAllViews(*universe).ok());
    EXPECT_EQ(universe->GetGlobalEpoch(), 10);

    {
        std::lock_guard<std::mutex> guard(*bodyMutex);
        *body = MakeViewTextEpoch(9);
    }
    EXPECT_EQ(worker->GetAllViews(*universe).code, StatusCode::CONNECTION_ERROR);
    EXPECT_EQ(universe->GetGlobalEpoch(), 10);
    worker->Stop();
    server->Stop();
}

// 测试场景：首选主节点先提供 epoch 10、备用主节点已准备 epoch 11；缓存建立后首选改发 epoch
// 9，并统计备用请求数。 预期结果：首选 lower epoch 被拒绝后回退到备用主节点，备用 epoch 11
// 成功发布且请求数为 1。
TEST(M800ViewSlaveTest, StaleFirstMasterFallsThroughToNewerBackup)
{
    auto bodyA = std::make_shared<std::string>(MakeViewTextEpoch(10));
    auto bodyB = std::make_shared<std::string>(MakeViewTextEpoch(11));
    auto mutexA = std::make_shared<std::mutex>();
    auto mutexB = std::make_shared<std::mutex>();
    std::atomic<int> requestsB{0};
    const auto portA = AllocatePort();
    const auto portB = AllocatePort();
    auto serverA = StartTextServer(portA, bodyA, mutexA);
    auto serverB = StartTextServer(portB, bodyB, mutexB, &requestsB);

    M800ViewServerConfig config;
    config.role = M800ViewRole::SLAVE;
    config.masterAddrs = {"127.0.0.1:" + std::to_string(portA),
                          "127.0.0.1:" + std::to_string(portB)};
    config.httpTimeoutMs = 300;
    auto worker = CreateM800ViewSlaveServer(config);
    auto universe = CreateViewUniverse();
    ASSERT_TRUE(worker->GetAllViews(*universe).ok());
    EXPECT_EQ(universe->GetGlobalEpoch(), 10);
    EXPECT_EQ(requestsB.load(), 0);

    {
        std::lock_guard<std::mutex> guard(*mutexA);
        *bodyA = MakeViewTextEpoch(9);
    }
    ASSERT_TRUE(worker->GetAllViews(*universe).ok());
    EXPECT_EQ(universe->GetGlobalEpoch(), 11);
    EXPECT_EQ(requestsB.load(), 1);
    worker->Stop();
    serverA->Stop();
    serverB->Stop();
}

// 测试场景：输出 ViewUniverse 已含 viewId=999、全局 epoch 20 的视图，首选主节点仅提供 epoch
// 15，备用主节点提供 epoch 25。 预期结果：首选 lower epoch 被 PublishAll 拒绝后请求备用主节点，备用
// epoch 25 成功发布且请求数为 1。
TEST(M800ViewSlaveTest, OlderMasterSnapshotFailsAndFallsThroughToNewerBackup)
{
    auto bodyA = std::make_shared<std::string>(MakeViewTextEpoch(15));
    auto bodyB = std::make_shared<std::string>(MakeViewTextEpoch(25));
    auto mutexA = std::make_shared<std::mutex>();
    auto mutexB = std::make_shared<std::mutex>();
    std::atomic<int> requestsB{0};
    const auto portA = AllocatePort();
    const auto portB = AllocatePort();
    auto serverA = StartTextServer(portA, bodyA, mutexA);
    auto serverB = StartTextServer(portB, bodyB, mutexB, &requestsB);

    GlobalView existing;
    existing.viewEpoch = 20;
    existing.viewId = 999;
    existing.asuMap.emplace(1, MakeNodeInfo("10.0.0.9"));
    auto universe = CreateViewUniverse(
        {
            {999, existing}
    },
        20);
    M800ViewServerConfig config;
    config.role = M800ViewRole::SLAVE;
    config.masterAddrs = {"127.0.0.1:" + std::to_string(portA),
                          "127.0.0.1:" + std::to_string(portB)};
    config.httpTimeoutMs = 300;
    auto worker = CreateM800ViewSlaveServer(config);
    ASSERT_TRUE(worker->GetAllViews(*universe).ok());
    EXPECT_EQ(universe->GetGlobalEpoch(), 25);
    EXPECT_EQ(requestsB.load(), 1);
    worker->Stop();
    serverA->Stop();
    serverB->Stop();
}

// 测试场景：主节点已有 epoch 100 的 pool-a/pool-b，Agent 在 epoch 110 将 pool-a 改为节点 3，并把
// pool-b 节点 2 的地址改为非法 not-an-ip。 预期结果：pool-a 虽可生成新视图，pool-b
// 的永久非法数据仍使整轮返回 NOT_FOUND；输出和 HTTP 旧快照完整保持 epoch 100
// 与旧节点，不发生部分更新或回填。
TEST(M800ViewMasterTest, InvalidNamespaceRejectsWholeRefreshAndRetainsOldView)
{
    auto state = std::make_shared<FakeAgentState>();
    SetNamespaces(
        state, 100,
        {
            {"pool-a", {MakeNode(1, "10.0.0.1")}},
            {"pool-b", {MakeNode(2, "10.0.0.2")}}
    });
    auto master = MakeMaster(MasterConfig(AllocatePort()), state);
    ASSERT_TRUE(master->Start().ok());
    auto universe = CreateViewUniverse();
    ASSERT_TRUE(master->GetAllViews(*universe).ok());

    SetNamespaces(
        state, 110,
        {
            {"pool-a", {MakeNode(3, "10.0.0.3")} },
            {"pool-b", {MakeNode(2, "not-an-ip")}}
    });
    EXPECT_EQ(master->GetAllViews(*universe).code, StatusCode::NOT_FOUND);
    EXPECT_EQ(universe->GetGlobalEpoch(), 100);
    GlobalView viewA;
    GlobalView viewB;
    ASSERT_TRUE(universe->GetView(M800PoolNameToKvNsId("pool-a"), viewA).ok());
    ASSERT_TRUE(universe->GetView(M800PoolNameToKvNsId("pool-b"), viewB).ok());
    EXPECT_EQ(viewA.asuMap.count(1), 1);
    EXPECT_EQ(viewA.asuMap.count(3), 0);
    EXPECT_EQ(viewB.asuMap.count(2), 1);
    EXPECT_EQ(viewB.viewEpoch, 100);
    master->Stop();
}

// 测试场景：基于 epoch 100 旧视图，epoch 110 刷新时 pool-b 查询超时、pool-a 更新为节点 3；epoch 111
// 再让 pool-a 先返回 TIMEOUT、pool-b 返回 INTERNAL_ERROR。
// 预期结果：任一命名空间查询失败都使整轮失败，输出和 HTTP 快照始终保留 epoch 100
// 的完整旧视图；两者均失败时返回按名称排序首先遇到的 TIMEOUT。
TEST(M800ViewMasterSlaveTest, NamespaceQueryErrorsRejectWholeRefreshAndReturnFirstFailure)
{
    auto state = std::make_shared<FakeAgentState>();
    SetNamespaces(
        state, 100,
        {
            {"pool-a", {MakeNode(1, "10.0.0.1")}},
            {"pool-b", {MakeNode(2, "10.0.0.2")}}
    });
    auto master = MakeMaster(MasterConfig(AllocatePort()), state);
    ASSERT_TRUE(master->Start().ok());
    auto universe = CreateViewUniverse();
    ASSERT_TRUE(master->GetAllViews(*universe).ok());

    SetNamespaces(
        state, 110,
        {
            {"pool-a", {MakeNode(3, "10.0.0.3")}},
            {"pool-b", {MakeNode(2, "10.0.0.2")}}
    });
    {
        std::lock_guard<std::mutex> guard(state->mutex);
        state->namespaceStatus["pool-b"] = Status::Error(StatusCode::TIMEOUT, "pool-b timed out");
    }
    EXPECT_EQ(master->GetAllViews(*universe).code, StatusCode::TIMEOUT);
    EXPECT_EQ(universe->GetGlobalEpoch(), 100);
    GlobalView viewA;
    GlobalView viewB;
    ASSERT_TRUE(universe->GetView(M800PoolNameToKvNsId("pool-a"), viewA).ok());
    ASSERT_TRUE(universe->GetView(M800PoolNameToKvNsId("pool-b"), viewB).ok());
    EXPECT_EQ(viewA.asuMap.count(1), 1);
    EXPECT_EQ(viewA.asuMap.count(3), 0);
    EXPECT_EQ(viewB.asuMap.count(2), 1);
    EXPECT_EQ(viewB.viewEpoch, 100);

    SetNamespaces(
        state, 111,
        {
            {"pool-a", {MakeNode(4, "10.0.0.4")}},
            {"pool-b", {MakeNode(5, "10.0.0.5")}}
    });
    {
        std::lock_guard<std::mutex> guard(state->mutex);
        state->namespaceStatus["pool-a"] = Status::Error(StatusCode::TIMEOUT, "pool-a timed out");
        state->namespaceStatus["pool-b"] =
            Status::Error(StatusCode::INTERNAL_ERROR, "pool-b internal error");
    }
    EXPECT_EQ(master->GetAllViews(*universe).code, StatusCode::TIMEOUT);
    EXPECT_EQ(universe->GetGlobalEpoch(), 100);
    EXPECT_EQ(viewB.asuMap.count(2), 1);
    master->Stop();
}

// 测试场景：主节点 Agent 仅持有 epoch 100，但传给 GetAllViews 的输出 ViewUniverse 已含 epoch
// 200、viewId=999 的更新视图。 预期结果：Agent 的 lower epoch 被 PublishAll 拒绝，调用方输出仍保持
// epoch 200；HTTP 不提供伪更新，Worker 获得 Master 首图 epoch 100 和原 pool-a 节点。
TEST(M800ViewMasterSlaveTest, LowerAgentEpochRejectsPublishAndPreservesHttpSnapshot)
{
    auto state = std::make_shared<FakeAgentState>();
    SetNamespaces(state, 100,
                  {
                      {"pool-a", {MakeNode(1, "10.0.0.1")}}
    });
    const auto port = AllocatePort();
    auto master = MakeMaster(MasterConfig(port), state);
    ASSERT_TRUE(master->Start().ok());

    GlobalView newerView;
    newerView.viewEpoch = 200;
    newerView.viewId = 999;
    newerView.asuMap.emplace(1, MakeNodeInfo("10.0.0.9"));
    auto newerUniverse = CreateViewUniverse(
        {
            {999, newerView}
    },
        200);
    EXPECT_EQ(master->GetAllViews(*newerUniverse).code, StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(newerUniverse->GetGlobalEpoch(), 200);

    M800ViewServerConfig workerConfig;
    workerConfig.role = M800ViewRole::SLAVE;
    workerConfig.masterAddrs = {"127.0.0.1:" + std::to_string(port)};
    workerConfig.httpTimeoutMs = 300;
    auto worker = CreateM800ViewSlaveServer(workerConfig);
    auto fetched = CreateViewUniverse();
    ASSERT_TRUE(worker->GetAllViews(*fetched).ok());
    EXPECT_EQ(fetched->GetGlobalEpoch(), 100);
    GlobalView actualView;
    const auto nsA = M800PoolNameToKvNsId("pool-a");
    ASSERT_TRUE(fetched->GetView(nsA, actualView).ok());
    EXPECT_EQ(actualView.viewEpoch, 100);
    EXPECT_EQ(actualView.viewId, nsA);
    ASSERT_EQ(actualView.asuMap.size(), 1);
    EXPECT_EQ(actualView.asuMap.at(1).endpoints.front().ip, "10.0.0.1");
    EXPECT_EQ(fetched->GetView(999, actualView).code, StatusCode::NOT_FOUND);
    worker->Stop();
    master->Stop();
}

// 测试场景：从节点未配置任何主节点地址，直接调用 GetAllViews。
// 预期结果：返回 INVALID_ARGUMENT，错误消息同时提示新键 view.master_addrs 和兼容键
// view.master_addr。
TEST(M800ViewSlaveTest, MissingMasterAddressMessageMentionsBothKeys)
{
    M800ViewServerConfig config;
    config.role = M800ViewRole::SLAVE;
    auto worker = CreateM800ViewSlaveServer(config);
    auto universe = CreateViewUniverse();
    const auto status = worker->GetAllViews(*universe);
    EXPECT_EQ(status.code, StatusCode::INVALID_ARGUMENT);
    EXPECT_NE(status.message.find("view.master_addrs"), std::string::npos);
    EXPECT_NE(status.message.find("view.master_addr"), std::string::npos);
    worker->Stop();
}

// 测试场景：从节点分别使用非数字主机名 localhost:9100、含嵌入 NUL 的 IPv4 地址 127.0.0.1\0junk 和
// IPv6 地址 [::1]\0junk。 预期结果：三种地址均返回 INVALID_ARGUMENT，且输出 ViewUniverse
// 始终为空，不按 NUL 前缀截断或解析主机名。
TEST(M800ViewSlaveTest, NonNumericAndEmbeddedNulMasterAddressesAreRejected)
{
    const std::vector<std::string> invalidAddresses = {
        "localhost:9100",
        std::string("127.0.0.1\0junk", 14),
        std::string("[::1]\0junk", 10),
    };
    for (const auto& address : invalidAddresses) {
        M800ViewServerConfig config;
        config.role = M800ViewRole::SLAVE;
        config.masterAddrs = {address};
        auto worker = CreateM800ViewSlaveServer(config);
        auto universe = CreateViewUniverse();
        EXPECT_EQ(worker->GetAllViews(*universe).code, StatusCode::INVALID_ARGUMENT);
        EXPECT_TRUE(universe->Empty());
    }
}

// 测试场景：从节点启动时主节点先提供 epoch 10 全量视图，首次 Start 成功后关闭服务器，再次调用
// Start。 预期结果：Start 会执行初始拉取并透传失败，首次返回 OK，服务器关闭后的第二次 Start 返回
// CONNECTION_ERROR。
TEST(M800ViewSlaveTest, StartPerformsInitialFetchAndReturnsFailures)
{
    auto body = std::make_shared<std::string>(MakeViewTextEpoch(10));
    auto bodyMutex = std::make_shared<std::mutex>();
    const auto port = AllocatePort();
    auto server = StartTextServer(port, body, bodyMutex);
    M800ViewServerConfig config;
    config.role = M800ViewRole::SLAVE;
    config.masterAddrs = {"127.0.0.1:" + std::to_string(port)};
    config.httpTimeoutMs = 300;
    auto worker = CreateM800ViewSlaveServer(config);
    ASSERT_TRUE(worker->Start().ok());
    server->Stop();
    EXPECT_EQ(worker->Start().code, StatusCode::CONNECTION_ERROR);
    worker->Stop();
}

// 测试场景：从节点先于主节点启动，配置的主地址无监听服务，输出 ViewUniverse 预置全局 epoch 8 和一个
// viewId=9 的视图。 预期结果：拉取返回 CONNECTION_ERROR，已有输出不被清空或改写，全局 epoch 仍为 8
// 且命名空间数量仍为 1。
TEST(M800ViewSlaveTest, WorkerBeforeMasterKeepsOutputUnchanged)
{
    M800ViewServerConfig config;
    config.role = M800ViewRole::SLAVE;
    config.masterAddrs = {"127.0.0.1:" + std::to_string(AllocatePort())};
    config.httpTimeoutMs = 150;
    auto worker = CreateM800ViewSlaveServer(config);
    GlobalView initial;
    initial.viewEpoch = 8;
    initial.viewId = 9;
    auto universe = CreateViewUniverse(
        {
            {9, initial}
    },
        8);
    const auto status = worker->GetAllViews(*universe);
    EXPECT_EQ(status.code, StatusCode::CONNECTION_ERROR);
    EXPECT_EQ(universe->GetGlobalEpoch(), 8);
    EXPECT_EQ(universe->GetAllNsIds().size(), 1);
    worker->Stop();
}

}  // namespace
}  // namespace kv::test
