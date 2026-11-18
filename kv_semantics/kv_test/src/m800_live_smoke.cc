#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include "view/m800_view_master_server.h"
#include "view/m800_view_slave_server.h"

namespace {

std::string GetArg(int argc, const char** argv, const std::string& name, const std::string& fallback = {})
{
    for (int index = 1; index + 1 < argc; ++index) {
        if (name == argv[index]) { return argv[index + 1]; }
    }
    return fallback;
}

int Fail(const std::string& message, const kv::Status* status = nullptr)
{
    std::cerr << "[FAIL] " << message;
    if (status != nullptr) {
        std::cerr << ", code=" << static_cast<int>(status->code) << ", message=" << status->message;
    }
    std::cerr << std::endl;
    return 1;
}

struct StopGuard {
    std::shared_ptr<kv::M800ViewMasterServer> master;
    std::shared_ptr<kv::M800ViewSlaveServer> worker;
    ~StopGuard()
    {
        if (worker) { worker->Stop(); }
        if (master) { master->Stop(); }
    }
};

int Run(int argc, const char** argv)
{
    const auto monHost = GetArg(argc, argv, "--mon-host");
    const auto clientName = GetArg(argc, argv, "--client-name", "client.admin");
    const auto clientKey = GetArg(argc, argv, "--client-key");
    const auto agentLib = GetArg(argc, argv, "--agent-lib", "/usr/lib/libkv_agent.so");
    const auto listenAddr = GetArg(argc, argv, "--listen-addr", "127.0.0.1:19100");
    const auto requiredPool = GetArg(argc, argv, "--require-pool");
    const auto forbiddenPool = GetArg(argc, argv, "--forbid-pool");
    const auto expectedAddPool = GetArg(argc, argv, "--expect-add");
    const auto expectedRemovePool = GetArg(argc, argv, "--expect-remove");
    const auto watchSecondsArg = GetArg(argc, argv, "--watch-seconds", "0");
    const auto refreshIntervalMsArg = GetArg(argc, argv, "--refresh-interval-ms", "30000");
    const auto watchSeconds = static_cast<unsigned>(std::strtoul(watchSecondsArg.c_str(), nullptr, 10));
    if (monHost.empty() || clientKey.empty()) {
        std::cerr << "usage: m800-live-smoke --mon-host <host:port> --client-key <key> "
                     "[--client-name client.admin] [--agent-lib path] [--listen-addr 127.0.0.1:19100] "
                     "[--require-pool name] [--forbid-pool name] [--expect-add name] [--expect-remove name] "
                     "[--watch-seconds N] [--refresh-interval-ms N]\n";
        return 2;
    }

    kv::M800ViewServerConfig masterConfig;
    masterConfig.role = kv::M800ViewRole::MASTER;
    masterConfig.listenAddr = listenAddr;
    masterConfig.monHost = monHost;
    masterConfig.clientName = clientName;
    masterConfig.clientKey = clientKey;
    masterConfig.agentLibPath = agentLib;
    masterConfig.agentFetchTimeoutMs = 15000;
    masterConfig.refreshIntervalMs = std::strtoull(refreshIntervalMsArg.c_str(), nullptr, 10);
    masterConfig.httpTimeoutMs = 3000;

    kv::M800ViewServerConfig workerConfig;
    workerConfig.role = kv::M800ViewRole::SLAVE;
    workerConfig.masterAddrs = {"127.0.0.1:1", listenAddr};
    workerConfig.httpTimeoutMs = 3000;

    auto worker = kv::CreateM800ViewSlaveServer(workerConfig);
    if (!worker) { return Fail("failed to create worker"); }
    auto master = kv::CreateM800ViewMasterServer(masterConfig);
    if (!master) { return Fail("failed to create master"); }
    StopGuard guard{master, worker};

    auto universe = kv::CreateViewUniverse();
    const auto beforeStart = worker->GetAllViews(*universe);
    if (beforeStart.code != kv::StatusCode::CONNECTION_ERROR) {
        return Fail("worker unexpectedly succeeded before Master started", &beforeStart);
    }
    std::cout << "[OK] worker-before-master returns CONNECTION_ERROR as expected\n";

    const auto startStatus = master->Start();
    if (!startStatus.ok()) { return Fail("Master Start failed", &startStatus); }
    std::cout << "[OK] Master started with real Agent\n";

    const auto firstFetch = worker->GetAllViews(*universe);
    if (!firstFetch.ok()) { return Fail("Slave first full fetch failed", &firstFetch); }
    if (universe->Empty()) { return Fail("first universe is empty"); }
    const auto firstEpoch = universe->GetGlobalEpoch();
    if (firstEpoch == 0) { return Fail("first globalEpoch is zero"); }

    const auto nsIds = universe->GetAllNsIds();
    for (const auto nsId : nsIds) {
        kv::GlobalView view;
        const auto viewStatus = universe->GetView(nsId, view);
        if (!viewStatus.ok()) { return Fail("GetView failed for namespace", &viewStatus); }
        if (view.viewEpoch != firstEpoch || view.viewId != nsId) {
            return Fail("viewEpoch/viewId does not match unified contract");
        }
        if (view.asuMap.empty()) { return Fail("namespace has no routable ASU"); }
        for (const auto& [nodeId, nodeInfo] : view.asuMap) {
            if (nodeInfo.localKvNsId != 0) { return Fail("M800 localKvNsId must be zero"); }
            if (nodeInfo.endpoints.empty()) { return Fail("ASU has no endpoint"); }
            const auto& endpoint = nodeInfo.endpoints.front();
            if (endpoint.ip.empty() || endpoint.protocol != kv::Protocol::UB || endpoint.port != 6000) {
                return Fail("unexpected endpoint protocol/port");
            }
            std::cout << "[VIEW] ns=" << nsId << ", node=" << nodeId << ", ip=" << endpoint.ip
                      << ", endpoints=" << nodeInfo.endpoints.size() << '\n';
        }
    }

    if (!requiredPool.empty()) {
        const auto requiredId = kv::M800PoolNameToKvNsId(requiredPool);
        kv::GlobalView requiredView;
        const auto requiredStatus = universe->GetView(requiredId, requiredView);
        if (!requiredStatus.ok()) {
            return Fail("required pool is missing: " + requiredPool, &requiredStatus);
        }
        std::cout << "[OK] required pool is present: " << requiredPool << ", ns=" << requiredId << '\n';
    }

    if (!forbiddenPool.empty()) {
        const auto forbiddenId = kv::M800PoolNameToKvNsId(forbiddenPool);
        kv::GlobalView forbiddenView;
        const auto forbiddenStatus = universe->GetView(forbiddenId, forbiddenView);
        if (forbiddenStatus.code != kv::StatusCode::NOT_FOUND) {
            return Fail("forbidden pool is unexpectedly present: " + forbiddenPool, &forbiddenStatus);
        }
        std::cout << "[OK] forbidden pool is absent: " << forbiddenPool << '\n';
    }

    const auto secondFetch = worker->GetAllViews(*universe);
    if (!secondFetch.ok()) { return Fail("Slave second fetch/304 failed", &secondFetch); }
    if (universe->GetGlobalEpoch() != firstEpoch) {
        return Fail("epoch changed unexpectedly during immediate second fetch");
    }
    std::cout << "[OK] second fetch is an OK/no-op at epoch=" << firstEpoch << '\n';
    std::cout << "[OK] bad master address failover succeeded\n";

    if (watchSeconds > 0) {
        bool addSeen = expectedAddPool.empty();
        bool removeSeen = expectedRemovePool.empty();
        const auto expectedAddId = expectedAddPool.empty() ? kv::KvNsId{0}
                                                            : kv::M800PoolNameToKvNsId(expectedAddPool);
        const auto expectedRemoveId = expectedRemovePool.empty()
                                          ? kv::KvNsId{0}
                                          : kv::M800PoolNameToKvNsId(expectedRemovePool);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(watchSeconds);
        while (std::chrono::steady_clock::now() < deadline && !(addSeen && removeSeen)) {
            const auto masterStatus = master->GetAllViews(*universe);
            if (!masterStatus.ok()) { return Fail("Master refresh failed during watch", &masterStatus); }
            const auto workerStatus = worker->GetAllViews(*universe);
            if (!workerStatus.ok()) { return Fail("Slave refresh failed during watch", &workerStatus); }
            std::cout << "[WATCH] epoch=" << universe->GetGlobalEpoch()
                      << " nsCount=" << universe->GetAllNsIds().size() << '\n';
            if (!addSeen) {
                kv::GlobalView addedView;
                if (universe->GetView(expectedAddId, addedView).ok()) {
                    addSeen = true;
                    std::cout << "[OK] expected namespace appeared: " << expectedAddPool << '\n';
                }
            }
            if (!removeSeen && (expectedAddPool.empty() || addSeen)) {
                kv::GlobalView removedView;
                if (universe->GetView(expectedRemoveId, removedView).code == kv::StatusCode::NOT_FOUND) {
                    removeSeen = true;
                    std::cout << "[OK] expected namespace disappeared: " << expectedRemovePool << '\n';
                }
            }
            if (!(addSeen && removeSeen)) { std::this_thread::sleep_for(std::chrono::seconds(1)); }
        }
        if (!addSeen || !removeSeen) { return Fail("dynamic namespace expectation timed out"); }
        std::cout << "[OK] dynamic namespace add/remove validated\n";
    }

    std::cout << "[OK] M800 live Master->HTTP->Slave view validation completed, namespaces="
              << universe->GetAllNsIds().size() << ", epoch=" << universe->GetGlobalEpoch() << '\n';
    return 0;
}

}  // namespace

int main(int argc, const char** argv)
{
    try {
        return Run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] unexpected exception: " << error.what() << std::endl;
        return 3;
    } catch (...) {
        std::cerr << "[FAIL] unexpected unknown exception" << std::endl;
        return 3;
    }
}
