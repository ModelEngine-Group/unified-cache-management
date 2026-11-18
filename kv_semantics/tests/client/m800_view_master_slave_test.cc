#include <gtest/gtest.h>
#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <sys/socket.h>
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
        views.erase(std::remove_if(views.begin(), views.end(),
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
    EXPECT_TRUE(server->Start("127.0.0.1:" + std::to_string(port),
                              [body, bodyMutex, requestCount](const ViewHttpRequest&) {
                                  if (requestCount != nullptr) { ++(*requestCount); }
                                  std::lock_guard<std::mutex> guard(*bodyMutex);
                                  return ViewHttpReply{ViewHttpReplyKind::FULL_VIEW, *body};
                              })
                    .ok());
    return server;
}

std::shared_ptr<M800ViewMasterServer> MakeMaster(
    const M800ViewServerConfig& config, const std::shared_ptr<FakeAgentState>& state)
{
    return CreateM800ViewMasterServer(config, std::make_unique<FakeAgentFactory>(state));
}

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
    Status GetView(KvNsId nsId, GlobalView& view) const override { return inner_->GetView(nsId, view); }
    void GetSnapshot(std::unordered_map<KvNsId, GlobalView>& views,
                     std::uint64_t& globalEpoch) const override
    {
        inner_->GetSnapshot(views, globalEpoch);
    }
    std::unordered_map<KvNsId, GlobalView> GetAllViews() const override { return inner_->GetAllViews(); }
    std::vector<KvNsId> GetAllNsIds() const override { return inner_->GetAllNsIds(); }
    std::uint64_t GetGlobalEpoch() const override { return inner_->GetGlobalEpoch(); }
    bool Empty() const override { return inner_->Empty(); }
    void Clear() override { inner_->Clear(); }

private:
    std::shared_ptr<ViewUniverse> inner_;
    std::chrono::milliseconds delay_;
};

TEST(M800ViewMasterTest, StartWaitsForInitialOsdMap)
{
    auto state = std::make_shared<FakeAgentState>();
    std::thread publisher([state] {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        SetNamespaces(state, 10, {{"pool-a", {MakeNode(1, "10.0.0.1")}}});
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

TEST(M800ViewMasterTest, StopInterruptsBlockedNamespaceQuery)
{
    auto state = std::make_shared<FakeAgentState>();
    SetNamespaces(state, 10, {{"pool-a", {MakeNode(1, "10.0.0.1")}}});
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

TEST(M800ViewMasterTest, ConcurrentStartStopDoesNotSelfDeadlock)
{
    for (int iteration = 0; iteration < 20; ++iteration) {
        auto state = std::make_shared<FakeAgentState>();
        SetNamespaces(state, 10, {{"pool-a", {MakeNode(1, "10.0.0.1")}}});
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

TEST(M800ViewMasterTest, StartWhileHttpStopIsBlockedReturnsCanceled)
{
    auto state = std::make_shared<FakeAgentState>();
    SetNamespaces(state, 10, {{"pool-a", {MakeNode(1, "10.0.0.1")}}});
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

TEST(M800ViewMasterTest, DuplicateStartStillReturnsResourceBusy)
{
    auto state = std::make_shared<FakeAgentState>();
    SetNamespaces(state, 10, {{"pool-a", {MakeNode(1, "10.0.0.1")}}});
    auto master = MakeMaster(MasterConfig(AllocatePort()), state);
    ASSERT_TRUE(master->Start().ok());
    EXPECT_EQ(master->Start().code, StatusCode::RESOURCE_BUSY);
    master->Stop();
}

TEST(M800ViewMasterTest, EmbeddedNulAddressIsRejected)
{
    auto state = std::make_shared<FakeAgentState>();
    const auto invalidAddress = std::string("10.0.0.1\0junk", 13);
    SetNamespaces(state, 10, {{"pool-a", {MakeNode(1, invalidAddress)}}});
    auto master = MakeMaster(MasterConfig(AllocatePort()), state);
    EXPECT_EQ(master->Start().code, StatusCode::NOT_FOUND);
    master->Stop();
}

TEST(M800ViewMasterTest, ExplicitZeroPortAddressIsRejected)
{
    auto state = std::make_shared<FakeAgentState>();
    SetNamespaces(state, 10, {{"pool-a", {MakeNode(1, "10.0.0.1:0")}}});
    auto master = MakeMaster(MasterConfig(AllocatePort()), state);
    EXPECT_EQ(master->Start().code, StatusCode::NOT_FOUND);
    master->Stop();
}

TEST(M800ViewMasterTest, InitialHashCollisionReturnsInternalErrorImmediately)
{
    auto state = std::make_shared<FakeAgentState>();
    SetNamespaces(state, 10,
                  {{"pool-69670", {MakeNode(1, "10.0.0.1")}},
                   {"pool-261455", {MakeNode(2, "10.0.0.2")}}});
    auto config = MasterConfig(AllocatePort());
    config.agentFetchTimeoutMs = 5000;
    auto master = MakeMaster(config, state);
    const auto begin = std::chrono::steady_clock::now();
    const auto status = master->Start();
    EXPECT_EQ(status.code, StatusCode::INTERNAL_ERROR);
    EXPECT_LT(std::chrono::steady_clock::now() - begin, std::chrono::seconds(1));
    master->Stop();
}

TEST(M800ViewMasterTest, RestartDuringStopNeverLosesNewAgent)
{
    auto state = std::make_shared<FakeAgentState>();
    SetNamespaces(state, 10, {{"pool-a", {MakeNode(1, "10.0.0.1")}}});
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

TEST(M800ViewMasterTest, SlowInitialSuccessAfterDeadlineIsNotPublished)
{
    auto state = std::make_shared<FakeAgentState>();
    SetNamespaces(state, 10, {{"pool-a", {MakeNode(1, "10.0.0.1")}}});
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

TEST(M800ViewMasterSlaveTest, SynchronizesMultipleNamespacesAndPreservesFailedNamespace)
{
    auto state = std::make_shared<FakeAgentState>();
    SetNamespaces(state, 123,
                  {{"pool-a", {MakeNode(10, "10.0.0.10", true, true),
                               MakeNode(11, "10.0.0.11", false, true)}},
                   {"pool-b", {MakeNode(20, "10.0.0.20:7000")}}});
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
                  {{"pool-a", {MakeNode(12, "10.0.0.12")}}, {"pool-b", {}}});
    auto refreshed = CreateViewUniverse();
    ASSERT_TRUE(master->GetAllViews(*refreshed).ok());
    ASSERT_TRUE(worker->GetAllViews(*universe).ok());
    EXPECT_EQ(universe->GetGlobalEpoch(), 124);
    EXPECT_EQ(universe->GetAllNsIds().size(), 2);
    ASSERT_TRUE(universe->GetView(nsA, viewA).ok());
    EXPECT_EQ(viewA.asuMap.count(12), 1);
    EXPECT_EQ(viewA.asuMap.count(10), 0);
    GlobalView viewB;
    ASSERT_TRUE(universe->GetView(M800PoolNameToKvNsId("pool-b"), viewB).ok());
    EXPECT_EQ(viewB.asuMap.count(20), 1);
    EXPECT_EQ(viewB.viewEpoch, 124);

    GlobalView legacyMasterView;
    GlobalView legacyWorkerView;
    ASSERT_TRUE(master->GetGlobalView(legacyMasterView).ok());
    ASSERT_TRUE(worker->GetGlobalView(legacyWorkerView).ok());
    EXPECT_EQ(legacyMasterView.viewId, universe->GetAllNsIds().front());
    EXPECT_EQ(legacyWorkerView.viewId, universe->GetAllNsIds().front());

    SetNamespaces(state, 125, {{"pool-a", {}}, {"pool-b", {}}});
    EXPECT_EQ(master->GetAllViews(*refreshed).code, StatusCode::NOT_FOUND);
    ASSERT_TRUE(worker->GetAllViews(*universe).ok());
    EXPECT_EQ(universe->GetGlobalEpoch(), 124);
    EXPECT_EQ(universe->GetAllNsIds().size(), 2);

    worker->Stop();
    master->Stop();
}

TEST(M800ViewMasterSlaveTest, NamespaceDeletionAndHashCollisionPreserveAuthorityRules)
{
    auto state = std::make_shared<FakeAgentState>();
    SetNamespaces(state, 100,
                  {{"pool-a", {MakeNode(1, "10.0.0.1")}},
                   {"pool-b", {MakeNode(2, "10.0.0.2")}}});
    auto master = MakeMaster(MasterConfig(AllocatePort()), state);
    ASSERT_TRUE(master->Start().ok());
    auto universe = CreateViewUniverse();
    ASSERT_TRUE(master->GetAllViews(*universe).ok());
    EXPECT_EQ(universe->GetAllNsIds().size(), 2);

    SetNamespaces(state, 101, {{"pool-a", {MakeNode(3, "10.0.0.3")}}});
    ASSERT_TRUE(master->GetAllViews(*universe).ok());
    EXPECT_EQ(universe->GetAllNsIds().size(), 1);
    EXPECT_EQ(universe->GetAllNsIds().front(), M800PoolNameToKvNsId("pool-a"));

    SetNamespaces(state, 102,
                  {{"pool-a", {MakeNode(3, "10.0.0.3")}},
                   {"pool-c", {MakeNode(6, "10.0.0.6")}}});
    ASSERT_TRUE(master->GetAllViews(*universe).ok());
    EXPECT_EQ(universe->GetAllNsIds().size(), 2);
    GlobalView viewC;
    EXPECT_EQ(universe->GetView(M800PoolNameToKvNsId("pool-c"), viewC).code, StatusCode::OK);

    ASSERT_EQ(M800PoolNameToKvNsId("pool-69670"), M800PoolNameToKvNsId("pool-261455"));
    SetNamespaces(state, 103,
                  {{"pool-69670", {MakeNode(4, "10.0.0.4")}},
                   {"pool-261455", {MakeNode(5, "10.0.0.5")}}});
    const auto previousEpoch = universe->GetGlobalEpoch();
    EXPECT_EQ(master->GetAllViews(*universe).code, StatusCode::INTERNAL_ERROR);
    EXPECT_EQ(universe->GetGlobalEpoch(), previousEpoch);
    master->Stop();
}

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

TEST(M800ViewSlaveTest, EmptyMalformedAndOlderResponsesPreserveNewerView)
{
    std::mutex responseMutex;
    std::string responseBody = MakeViewTextEpoch(10);
    ViewHttpReplyKind responseKind = ViewHttpReplyKind::FULL_VIEW;
    const auto port = AllocatePort();
    auto server = CreateViewHttpServer(300);
    ASSERT_TRUE(server->Start("127.0.0.1:" + std::to_string(port),
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
    EXPECT_TRUE(worker->GetAllViews(*universe).ok());
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

TEST(M800ViewSlaveTest, EmptyOutputForcesFullDownloadAfterPreviousFetch)
{
    const auto port = AllocatePort();
    auto server = CreateViewHttpServer(300);
    std::mutex requestMutex;
    std::vector<std::uint64_t> requestedEpochs;
    ASSERT_TRUE(server->Start("127.0.0.1:" + std::to_string(port),
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

TEST(M800ViewSlaveTest, ZeroEpoch304IsRejectedAndFallsBackToFullView)
{
    const auto portA = AllocatePort();
    auto serverA = CreateViewHttpServer(300);
    ASSERT_TRUE(serverA->Start("127.0.0.1:" + std::to_string(portA),
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

TEST(M800ViewSlaveTest, StaleResponseIsNoOpInsidePublishAllWithoutSecondMaster)
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
    EXPECT_TRUE(worker->GetAllViews(*universe).ok());
    EXPECT_EQ(universe->GetGlobalEpoch(), 10);
    EXPECT_EQ(requestsB.load(), 0);
    worker->Stop();
    serverA->Stop();
    serverB->Stop();
}

TEST(M800ViewSlaveTest, StaleResponseReturnsOkWithoutTryingFailedBackup)
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
    EXPECT_TRUE(worker->GetAllViews(*universe).ok());
    EXPECT_EQ(universe->GetGlobalEpoch(), 10);
    worker->Stop();
    server->Stop();
}

TEST(M800ViewSlaveTest, StaleFirstMasterDoesNotFallThroughToNewerBackup)
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
    EXPECT_EQ(universe->GetGlobalEpoch(), 10);
    EXPECT_EQ(requestsB.load(), 0);
    worker->Stop();
    serverA->Stop();
    serverB->Stop();
}

TEST(M800ViewSlaveTest, PublishAllAloneRejectsOlderMasterSnapshot)
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
    auto universe = CreateViewUniverse({{999, existing}}, 20);
    M800ViewServerConfig config;
    config.role = M800ViewRole::SLAVE;
    config.masterAddrs = {"127.0.0.1:" + std::to_string(portA),
                          "127.0.0.1:" + std::to_string(portB)};
    config.httpTimeoutMs = 300;
    auto worker = CreateM800ViewSlaveServer(config);
    ASSERT_TRUE(worker->GetAllViews(*universe).ok());
    EXPECT_EQ(universe->GetGlobalEpoch(), 20);
    EXPECT_EQ(requestsB.load(), 0);
    worker->Stop();
    serverA->Stop();
    serverB->Stop();
}

TEST(M800ViewMasterSlaveTest, NamespaceQueryErrorsRetainOldViewAndReturnFirstFailure)
{
    auto state = std::make_shared<FakeAgentState>();
    SetNamespaces(state, 100,
                  {{"pool-a", {MakeNode(1, "10.0.0.1")}},
                   {"pool-b", {MakeNode(2, "10.0.0.2")}}});
    auto master = MakeMaster(MasterConfig(AllocatePort()), state);
    ASSERT_TRUE(master->Start().ok());

    SetNamespaces(state, 110,
                  {{"pool-a", {MakeNode(3, "10.0.0.3")}},
                   {"pool-b", {MakeNode(2, "10.0.0.2")}}});
    {
        std::lock_guard<std::mutex> guard(state->mutex);
        state->namespaceStatus["pool-b"] = Status::Error(StatusCode::TIMEOUT, "pool-b timed out");
    }
    auto universe = CreateViewUniverse();
    ASSERT_TRUE(master->GetAllViews(*universe).ok());
    GlobalView viewA;
    GlobalView viewB;
    ASSERT_TRUE(universe->GetView(M800PoolNameToKvNsId("pool-a"), viewA).ok());
    ASSERT_TRUE(universe->GetView(M800PoolNameToKvNsId("pool-b"), viewB).ok());
    EXPECT_EQ(viewA.asuMap.count(3), 1);
    EXPECT_EQ(viewB.asuMap.count(2), 1);
    EXPECT_EQ(viewB.viewEpoch, 110);

    SetNamespaces(state, 111,
                  {{"pool-a", {MakeNode(4, "10.0.0.4")}},
                   {"pool-b", {MakeNode(5, "10.0.0.5")}}});
    {
        std::lock_guard<std::mutex> guard(state->mutex);
        state->namespaceStatus["pool-a"] = Status::Error(StatusCode::TIMEOUT, "pool-a timed out");
        state->namespaceStatus["pool-b"] =
            Status::Error(StatusCode::INTERNAL_ERROR, "pool-b internal error");
    }
    EXPECT_EQ(master->GetAllViews(*universe).code, StatusCode::TIMEOUT);
    EXPECT_EQ(universe->GetGlobalEpoch(), 110);
    EXPECT_EQ(viewB.asuMap.count(2), 1);
    master->Stop();
}

TEST(M800ViewMasterSlaveTest, HttpSnapshotUsesActualPostPublishState)
{
    auto state = std::make_shared<FakeAgentState>();
    SetNamespaces(state, 100, {{"pool-a", {MakeNode(1, "10.0.0.1")}}});
    const auto port = AllocatePort();
    auto master = MakeMaster(MasterConfig(port), state);
    ASSERT_TRUE(master->Start().ok());

    GlobalView newerView;
    newerView.viewEpoch = 200;
    newerView.viewId = 999;
    newerView.asuMap.emplace(1, MakeNodeInfo("10.0.0.9"));
    auto newerUniverse = CreateViewUniverse({{999, newerView}}, 200);
    ASSERT_TRUE(master->GetAllViews(*newerUniverse).ok());
    EXPECT_EQ(newerUniverse->GetGlobalEpoch(), 200);

    M800ViewServerConfig workerConfig;
    workerConfig.role = M800ViewRole::SLAVE;
    workerConfig.masterAddrs = {"127.0.0.1:" + std::to_string(port)};
    workerConfig.httpTimeoutMs = 300;
    auto worker = CreateM800ViewSlaveServer(workerConfig);
    auto fetched = CreateViewUniverse();
    ASSERT_TRUE(worker->GetAllViews(*fetched).ok());
    EXPECT_EQ(fetched->GetGlobalEpoch(), 200);
    GlobalView actualView;
    ASSERT_TRUE(fetched->GetView(999, actualView).ok());
    EXPECT_EQ(actualView.viewEpoch, 200);
    EXPECT_EQ(actualView.viewId, 999);
    ASSERT_EQ(actualView.asuMap.size(), 1);
    EXPECT_EQ(actualView.asuMap.at(1).endpoints.front().ip, "10.0.0.9");
    worker->Stop();
    master->Stop();
}

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
    auto universe = CreateViewUniverse({{9, initial}}, 8);
    const auto status = worker->GetAllViews(*universe);
    EXPECT_EQ(status.code, StatusCode::CONNECTION_ERROR);
    EXPECT_EQ(universe->GetGlobalEpoch(), 8);
    EXPECT_EQ(universe->GetAllNsIds().size(), 1);
    worker->Stop();
}

}  // namespace
}  // namespace kv::test
