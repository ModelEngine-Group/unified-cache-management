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
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "m800_live_smoke_watch.h"
#include "view/m800_view_master_server.h"
#include "view/m800_view_slave_server.h"

namespace {

std::string GetArg(int argc, const char** argv, const std::string& name,
                   const std::string& fallback = {})
{
    for (int index = 1; index + 1 < argc; ++index) {
        if (name == argv[index]) { return argv[index + 1]; }
    }
    return fallback;
}

std::vector<std::size_t> ParseCounts(const std::string& text)
{
    std::vector<std::size_t> counts;
    std::size_t begin{0};
    while (begin <= text.size()) {
        const auto end = text.find(',', begin);
        const auto token =
            text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        if (!token.empty()) { counts.emplace_back(std::strtoull(token.c_str(), nullptr, 10)); }
        if (end == std::string::npos) { break; }
        begin = end + 1;
    }
    return counts;
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

std::uint64_t MaxViewEpoch(const std::unordered_map<kv::KvNsId, kv::GlobalView>& views)
{
    std::uint64_t epoch{0};
    for (const auto& item : views) { epoch = std::max(epoch, item.second.viewEpoch); }
    return epoch;
}

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
    const auto watchAsuPool = GetArg(argc, argv, "--watch-asu-pool");
    const auto expectedAsuCounts = ParseCounts(GetArg(argc, argv, "--expect-asu-counts"));
    const auto watchSecondsArg = GetArg(argc, argv, "--watch-seconds", "0");
    const auto refreshIntervalMsArg = GetArg(argc, argv, "--refresh-interval-ms", "30000");
    const auto watchSeconds =
        static_cast<unsigned>(std::strtoul(watchSecondsArg.c_str(), nullptr, 10));
    if (monHost.empty() || clientKey.empty()) {
        std::cerr
            << "usage: m800-live-smoke --mon-host <host:port> --client-key <key> "
               "[--client-name client.admin] [--agent-lib path] [--listen-addr 127.0.0.1:19100] "
               "[--require-pool name] [--forbid-pool name] [--expect-add name] [--expect-remove "
               "name] "
               "[--watch-seconds N] [--refresh-interval-ms N] "
               "[--watch-asu-pool name --expect-asu-counts 3,5,4,5]\n";
        return 2;
    }
    if (!expectedAsuCounts.empty() && (watchAsuPool.empty() || watchSeconds == 0)) {
        std::cerr << "--expect-asu-counts requires --watch-asu-pool and --watch-seconds\n";
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

    std::unordered_map<kv::KvNsId, kv::GlobalView> universe;
    const auto beforeStart = worker->GetAllViews(universe);
    if (beforeStart.code != kv::StatusCode::CONNECTION_ERROR) {
        return Fail("worker unexpectedly succeeded before Master started", &beforeStart);
    }
    std::cout << "[OK] worker-before-master returns CONNECTION_ERROR as expected\n";

    const auto startStatus = master->Start();
    if (!startStatus.ok()) { return Fail("Master Start failed", &startStatus); }
    std::cout << "[OK] Master started with real Agent\n";

    const auto firstFetch = worker->GetAllViews(universe);
    if (!firstFetch.ok()) { return Fail("Slave first full fetch failed", &firstFetch); }
    if (universe.empty()) { return Fail("first universe is empty"); }
    const auto firstEpoch = MaxViewEpoch(universe);
    if (firstEpoch == 0) { return Fail("first globalEpoch is zero"); }

    for (const auto& [nsId, view] : universe) {
        if (view.viewEpoch != firstEpoch || view.viewId != nsId) {
            return Fail("viewEpoch/viewId does not match unified contract");
        }
        if (view.asuMap.empty()) { return Fail("namespace has no routable ASU"); }
        for (const auto& [nodeId, nodeInfo] : view.asuMap) {
            if (nodeInfo.localKvNsId != 0) { return Fail("M800 localKvNsId must be zero"); }
            if (nodeInfo.endpoints.empty()) { return Fail("ASU has no endpoint"); }
            const auto& endpoint = nodeInfo.endpoints.front();
            if (endpoint.ip.empty() || endpoint.protocol != kv::Protocol::UB ||
                endpoint.port != 6000) {
                return Fail("unexpected endpoint protocol/port");
            }
            std::cout << "[VIEW] ns=" << nsId << ", node=" << nodeId << ", ip=" << endpoint.ip
                      << ", endpoints=" << nodeInfo.endpoints.size() << '\n';
        }
    }

    if (!requiredPool.empty()) {
        const auto requiredId = kv::M800PoolNameToKvNsId(requiredPool);
        kv::GlobalView requiredView;
        const auto requiredIter = universe.find(requiredId);
        if (requiredIter == universe.end()) {
            return Fail("required pool is missing: " + requiredPool);
        }
        requiredView = requiredIter->second;
        std::cout << "[OK] required pool is present: " << requiredPool << ", ns=" << requiredId
                  << '\n';
    }

    if (!forbiddenPool.empty()) {
        const auto forbiddenId = kv::M800PoolNameToKvNsId(forbiddenPool);
        kv::GlobalView forbiddenView;
        if (universe.find(forbiddenId) != universe.end()) {
            return Fail("forbidden pool is unexpectedly present: " + forbiddenPool);
        }
        std::cout << "[OK] forbidden pool is absent: " << forbiddenPool << '\n';
    }

    const auto secondFetch = worker->GetAllViews(universe);
    if (!secondFetch.ok()) { return Fail("Slave second fetch/304 failed", &secondFetch); }
    if (MaxViewEpoch(universe) != firstEpoch) {
        return Fail("epoch changed unexpectedly during immediate second fetch");
    }
    std::cout << "[OK] second fetch is an OK/no-op at epoch=" << firstEpoch << '\n';
    std::cout << "[OK] bad master address failover succeeded\n";

    if (watchSeconds > 0) {
        const kv::test::M800LiveSmokeWatchOptions watchOptions{watchSeconds};
        bool addSeen = false;
        bool removeSeen = false;
        const auto expectedAddId =
            expectedAddPool.empty() ? kv::KvNsId{0} : kv::M800PoolNameToKvNsId(expectedAddPool);
        const auto expectedRemoveId = expectedRemovePool.empty()
                                          ? kv::KvNsId{0}
                                          : kv::M800PoolNameToKvNsId(expectedRemovePool);
        const auto watchAsuId =
            watchAsuPool.empty() ? kv::KvNsId{0} : kv::M800PoolNameToKvNsId(watchAsuPool);
        std::size_t asuSequenceIndex{0};
        bool asuSequenceDone = expectedAsuCounts.empty();
        bool hasLastAsuCount{false};
        std::size_t lastAsuCount{0};
        std::uint64_t lastAsuChangeEpoch{0};
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(watchSeconds);
        std::cout << "[OK] watch mode keeps the Master HTTP listener alive for " << watchSeconds
                  << " seconds\n";
        while (
            !kv::test::WatchShouldStop(std::chrono::steady_clock::now(), deadline, watchOptions)) {
            const auto masterStatus = master->GetAllViews(universe);
            if (!masterStatus.ok()) {
                return Fail("Master refresh failed during watch", &masterStatus);
            }
            const auto workerStatus = worker->GetAllViews(universe);
            if (!workerStatus.ok()) {
                return Fail("Slave refresh failed during watch", &workerStatus);
            }

            std::size_t currentAsuCount{0};
            if (!watchAsuPool.empty()) {
                kv::GlobalView asuView;
                const auto watchIter = universe.find(watchAsuId);
                if (watchIter != universe.end()) {
                    currentAsuCount = watchIter->second.asuMap.size();
                }
            }
            const auto currentEpoch = MaxViewEpoch(universe);
            std::cout << "[WATCH] epoch=" << currentEpoch << " nsCount=" << universe.size();
            if (!watchAsuPool.empty()) { std::cout << " asuCount=" << currentAsuCount; }
            std::cout << '\n';

            if (!watchAsuPool.empty()) {
                if (hasLastAsuCount && currentAsuCount != lastAsuCount) {
                    if (currentEpoch <= lastAsuChangeEpoch) {
                        return Fail("ASU count changed without a newer globalEpoch");
                    }
                    std::cout << "[OK] ASU count changed " << lastAsuCount << "->"
                              << currentAsuCount << " at epoch " << lastAsuChangeEpoch << "->"
                              << currentEpoch << '\n';
                    lastAsuCount = currentAsuCount;
                    lastAsuChangeEpoch = currentEpoch;
                } else if (!hasLastAsuCount) {
                    lastAsuCount = currentAsuCount;
                    lastAsuChangeEpoch = currentEpoch;
                    hasLastAsuCount = true;
                }
            }
            if (!asuSequenceDone && currentAsuCount == expectedAsuCounts[asuSequenceIndex]) {
                std::cout << "[OK] expected ASU count observed: " << currentAsuCount << '\n';
                ++asuSequenceIndex;
                asuSequenceDone = asuSequenceIndex == expectedAsuCounts.size();
            }
            if (!addSeen && !expectedAddPool.empty()) {
                kv::GlobalView addedView;
                if (universe.find(expectedAddId) != universe.end()) {
                    addSeen = true;
                    std::cout << "[OK] expected namespace appeared: " << expectedAddPool << '\n';
                }
            }
            if (!removeSeen && !expectedRemovePool.empty() &&
                (expectedAddPool.empty() || addSeen)) {
                kv::GlobalView removedView;
                if (universe.find(expectedRemoveId) == universe.end()) {
                    removeSeen = true;
                    std::cout << "[OK] expected namespace disappeared: " << expectedRemovePool
                              << '\n';
                }
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        const bool addMissing = !expectedAddPool.empty() && !addSeen;
        const bool removeMissing = !expectedRemovePool.empty() && !removeSeen;
        const bool asuSequenceMissing = !expectedAsuCounts.empty() && !asuSequenceDone;
        if (addMissing || removeMissing || asuSequenceMissing) {
            return Fail("dynamic namespace/ASU expectation timed out");
        }
        std::cout << "[OK] dynamic namespace/ASU transitions validated\n";
    }

    std::cout << "[OK] M800 live Master->HTTP->Slave view validation completed, namespaces="
              << universe.size() << ", epoch=" << MaxViewEpoch(universe) << '\n';
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
