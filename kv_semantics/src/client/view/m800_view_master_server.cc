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
#include "view/m800_view_master_server.h"
#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <map>
#include <unordered_map>
#include <utility>
#include "logger.h"
#include "view/m800_view_codec.h"
#include "view/view_universe.h"

namespace kv {
namespace {

constexpr std::size_t kMaxNamespaces = 64;
constexpr std::size_t kMaxAsusPerNamespace = 4096;
constexpr std::size_t kMaxEndpointsPerAsu = 4;
constexpr auto kInitialRetryInterval = std::chrono::milliseconds{200};
constexpr const char* kHashCollisionMessage = "M800 pool name hash collision";
constexpr const char* kNoUsableEndpointMessage = "namespace has no usable endpoint:";

Status MasterError(StatusCode code, const std::string& message)
{
    return Status::Error(code, message);
}

bool ParsePort(std::string_view text, std::uint16_t& port)
{
    if (text.empty()) { return false; }
    unsigned int value{0};
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto result = std::from_chars(begin, end, value, 10);
    if (result.ec != std::errc{} || result.ptr != end || value == 0 || value > 65535) {
        return false;
    }
    port = static_cast<std::uint16_t>(value);
    return true;
}

bool ValidIpAddress(const std::string& ip)
{
    if (ip.find('\0') != std::string::npos) { return false; }
    in_addr ipv4{};
    in6_addr ipv6{};
    return inet_pton(AF_INET, ip.c_str(), &ipv4) == 1 ||
           inet_pton(AF_INET6, ip.c_str(), &ipv6) == 1;
}

std::uint64_t ViewsGlobalEpoch(const std::unordered_map<KvNsId, GlobalView>& views)
{
    std::uint64_t epoch{0};
    for (const auto& item : views) { epoch = std::max(epoch, item.second.viewEpoch); }
    return epoch;
}

bool ParseNodeEndpoint(const std::string& address, const M800ViewServerConfig& config,
                       NodeEndpoint& endpoint)
{
    if (address.empty()) { return false; }
    std::string ip;
    std::uint16_t port{0};
    if (address.front() == '[') {
        const auto close = address.find(']');
        if (close == std::string::npos || close == 1) { return false; }
        ip = address.substr(1, close - 1);
        if (close + 1 < address.size()) {
            if (address[close + 1] != ':' ||
                !ParsePort(std::string_view{address}.substr(close + 2), port)) {
                return false;
            }
        }
    } else {
        const auto colonCount =
            static_cast<std::size_t>(std::count(address.begin(), address.end(), ':'));
        if (colonCount == 0) {
            ip = address;
        } else if (colonCount == 1) {
            const auto colon = address.find(':');
            ip = address.substr(0, colon);
            if (!ParsePort(std::string_view{address}.substr(colon + 1), port)) { return false; }
        } else {
            ip = address;
        }
    }
    if (!ValidIpAddress(ip)) { return false; }
    endpoint.ip = std::move(ip);
    endpoint.port = port == 0 ? config.endpointPort : port;
    endpoint.protocol = config.endpointProtocol;
    switch (endpoint.protocol) {
        case Protocol::UB: endpoint.attrs["protocol"] = "uboe"; break;
        case Protocol::ROCE: endpoint.attrs["protocol"] = "roce"; break;
        case Protocol::TCP: endpoint.attrs["protocol"] = "tcp"; break;
    }
    return true;
}

class M800ViewMasterServerImpl final : public M800ViewMasterServer {
public:
    M800ViewMasterServerImpl(M800ViewServerConfig config,
                             std::unique_ptr<M800ViewAgentClientFactory> agentFactory)
        : config_(std::move(config)), agentFactory_(std::move(agentFactory))
    {
        if (!agentFactory_) { agentFactory_ = CreateM800ViewAgentClientFactory(); }
        KV_INFO("M800ViewServer created, mon_host={}, agent={}", config_.monHost,
                config_.agentLibPath);
        atomic_store(&lastFetched_, std::shared_ptr<const ViewUniverse>{});
    }

    ~M800ViewMasterServerImpl() override { Stop(); }

    Status Start() override
    {
        std::lock_guard<std::mutex> fetchGuard(fetchMutex_);
        {
            std::lock_guard<std::mutex> guard(mutex_);
            if (started_) {
                return MasterError(StatusCode::RESOURCE_BUSY, "M800 master is already started");
            }
            if (stopping_) {
                return MasterError(StatusCode::CANCELED,
                                   "M800 master start was canceled by a concurrent stop");
            }
            running_.store(true);
        }

        auto status = InitializeAgent();
        if (!status.ok()) {
            running_.store(false);
            CleanupAgent();
            return status;
        }

        const auto configuredTimeout =
            config_.agentFetchTimeoutMs == 0 ? std::uint64_t{10000} : config_.agentFetchTimeoutMs;
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(configuredTimeout);
        std::shared_ptr<ViewUniverse> initial;
        Status lastFailure =
            MasterError(StatusCode::CONNECTION_ERROR, "M800 initial view unavailable");
        while (running_.load()) {
            auto candidate = CreateViewUniverse();
            lastFailure = FetchAndPublish(*candidate, deadline);
            if (lastFailure.ok()) {
                initial = std::move(candidate);
                break;
            }
            if ((lastFailure.code == StatusCode::INTERNAL_ERROR &&
                 lastFailure.message == kHashCollisionMessage) ||
                (lastFailure.code == StatusCode::NOT_FOUND &&
                 lastFailure.message.rfind(kNoUsableEndpointMessage, 0) == 0)) {
                running_.store(false);
                CleanupAgent();
                return lastFailure;
            }
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) { break; }
            const auto remaining =
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
            const auto waitTime = std::min(kInitialRetryInterval, remaining);
            if (waitTime <= std::chrono::milliseconds::zero()) { break; }
            std::unique_lock<std::mutex> waitLock(waitMutex_);
            waitCv_.wait_for(waitLock, waitTime, [this] { return !running_.load(); });
        }
        if (!initial) {
            const bool canceled = !running_.load();
            running_.store(false);
            CleanupAgent();
            if (canceled) {
                return MasterError(StatusCode::CANCELED, "M800 master start was canceled");
            }
            return MasterError(
                StatusCode::CONNECTION_ERROR,
                "M800 initial view was not ready before timeout: " + lastFailure.message);
        }

        auto httpServer = CreateViewHttpServer(config_.httpTimeoutMs);
        if (!httpServer) {
            running_.store(false);
            CleanupAgent();
            return MasterError(StatusCode::INTERNAL_ERROR, "failed to allocate M800 HTTP server");
        }
        bool canceled{false};
        {
            std::lock_guard<std::mutex> guard(mutex_);
            canceled = !running_.load();
            if (!canceled) {
                status = httpServer->Start(
                    config_.listenAddr,
                    [this](const ViewHttpRequest& request) { return HandleHttpRequest(request); });
                if (status.ok()) {
                    httpServer_ = std::move(httpServer);
                    started_ = true;
                }
            }
        }
        if (canceled) {
            CleanupAgent();
            return MasterError(StatusCode::CANCELED, "M800 master start was canceled");
        }
        if (!status.ok()) {
            running_.store(false);
            CleanupAgent();
            return status;
        }
        if (!running_.load()) {
            std::shared_ptr<ViewHttpServer> startedServer;
            {
                std::lock_guard<std::mutex> guard(mutex_);
                startedServer = std::move(httpServer_);
                httpServer_.reset();
                started_ = false;
            }
            if (startedServer) { startedServer->Stop(); }
            CleanupAgent();
            return MasterError(StatusCode::CANCELED, "M800 master start was canceled");
        }
        return Status::OK();
    }

    void Stop() override
    {
        std::lock_guard<std::mutex> stopGuard(stopMutex_);

        std::shared_ptr<ViewHttpServer> httpServer;
        {
            std::lock_guard<std::mutex> guard(mutex_);
            stopping_ = true;
            running_.store(false);
            started_ = false;
            httpServer = httpServer_;
        }
        waitCv_.notify_all();
        if (httpServer) { httpServer->Stop(); }

        // Ask the agent to cancel an in-flight read before waiting for the
        // master fetch mutex. Cleanup then waits for the query to finish and
        // releases the handle/library in their owning order.
        const auto agent = GetAgent();
        if (agent) { agent->Stop(); }
        std::lock_guard<std::mutex> fetchGuard(fetchMutex_);
        CleanupAgent();
        std::lock_guard<std::mutex> guard(mutex_);
        httpServer_.reset();
        stopping_ = false;
    }

    Status GetAllViews(std::unordered_map<KvNsId, GlobalView>& views) override
    {
        std::lock_guard<std::mutex> fetchGuard(fetchMutex_);
        if (!running_.load()) {
            return MasterError(StatusCode::NOT_INITIALIZED, "M800 master is not running");
        }
        auto output = CreateViewUniverse(views, ViewsGlobalEpoch(views));
        if (!output) {
            return MasterError(StatusCode::INTERNAL_ERROR,
                               "failed to allocate M800 output ViewUniverse");
        }
        const auto status = FetchAndPublish(*output);
        if (!status.ok()) { return status; }

        std::unordered_map<KvNsId, GlobalView> nextViews;
        std::uint64_t nextEpoch{0};
        output->GetSnapshot(nextViews, nextEpoch);
        views = std::move(nextViews);
        return Status::OK();
    }

    Status GetGlobalViewByKvnsId(KvNsId nsId, GlobalView& view) override
    {
        std::unordered_map<KvNsId, GlobalView> views;
        auto status = GetAllViews(views);
        if (!status.ok()) { return status; }

        const auto iter = views.find(nsId);
        if (iter == views.end()) {
            return MasterError(StatusCode::NOT_FOUND,
                               "M800 view namespace not found, nsId=" + std::to_string(nsId));
        }
        view = iter->second;
        return Status::OK();
    }

    Status GetGlobalView(GlobalView& view) override
    {
        (void)view;
        return MasterError(StatusCode::UNSUPPORTED,
                           "M800 view master serves multiple namespaces; use GetAllViews");
    }

private:
    Status InitializeAgent()
    {
        if (!agentFactory_) {
            return MasterError(StatusCode::INTERNAL_ERROR, "M800 agent factory is unavailable");
        }
        std::unique_ptr<M800ViewAgentClient> created;
        auto status = agentFactory_->Create(config_, created);
        if (!status.ok()) { return status; }
        if (!created) {
            return MasterError(StatusCode::INTERNAL_ERROR, "agent factory returned a null client");
        }
        status = created->Initialize();
        if (!status.ok()) { return status; }
        status = created->Start();
        if (!status.ok()) {
            created->Stop();
            return status;
        }
        std::lock_guard<std::mutex> guard(mutex_);
        agent_ = std::shared_ptr<M800ViewAgentClient>(std::move(created));
        return Status::OK();
    }

    void CleanupAgent()
    {
        std::shared_ptr<M800ViewAgentClient> agent;
        {
            std::lock_guard<std::mutex> guard(mutex_);
            agent = std::move(agent_);
            agent_.reset();
        }
        if (agent) {
            agent->Stop();
            KV_INFO("M800ViewServer shutdown, agent closed");
        }
    }

    std::shared_ptr<M800ViewAgentClient> GetAgent()
    {
        std::lock_guard<std::mutex> guard(mutex_);
        return agent_;
    }

    Status FetchAndPublish(ViewUniverse& universe, std::chrono::steady_clock::time_point deadline =
                                                       std::chrono::steady_clock::time_point::max())
    {
        if (!running_.load()) {
            return MasterError(StatusCode::CANCELED, "M800 view publish was canceled");
        }
        const auto isExpired = [&] {
            return deadline != std::chrono::steady_clock::time_point::max() &&
                   std::chrono::steady_clock::now() >= deadline;
        };
        if (isExpired()) {
            return MasterError(StatusCode::CONNECTION_ERROR, "M800 initial view deadline expired");
        }
        auto next = CreateViewUniverse();
        auto status = FetchUniverse(*next);
        if (!status.ok()) { return status; }
        if (!running_.load()) {
            return MasterError(StatusCode::CANCELED, "M800 view publish was canceled");
        }
        if (isExpired()) {
            return MasterError(StatusCode::CONNECTION_ERROR,
                               "M800 initial view arrived after its deadline");
        }
        status = universe.PublishAll(*next);
        if (!status.ok()) { return status; }

        // PublishAll returns OK only for a real publish or an equal-epoch
        // no-op. Empty or lower-epoch snapshots fail above and preserve the
        // last-known-good snapshot. Read back the caller's actual state instead
        // of exposing the candidate as the authoritative HTTP snapshot.
        std::unordered_map<KvNsId, GlobalView> publishedViews;
        std::uint64_t publishedEpoch{0};
        universe.GetSnapshot(publishedViews, publishedEpoch);
        auto published = CreateViewUniverse(std::move(publishedViews), publishedEpoch);
        atomic_store(&lastFetched_, std::shared_ptr<const ViewUniverse>(std::move(published)));
        return Status::OK();
    }

    Status FetchUniverse(ViewUniverse& universe)
    {
        auto agent = GetAgent();
        if (!agent) {
            return MasterError(StatusCode::NOT_INITIALIZED, "M800 agent is unavailable");
        }

        Status lastStatus = MasterError(StatusCode::INTERNAL_ERROR, "M800 view fetch failed");
        for (int attempt = 0; attempt < 2; ++attempt) {
            if (!running_.load()) {
                return MasterError(StatusCode::CANCELED, "M800 view fetch canceled");
            }
            std::uint64_t epochBefore{0};
            lastStatus = agent->GetEpoch(epochBefore);
            if (!lastStatus.ok()) { return lastStatus; }
            if (epochBefore == 0) {
                lastStatus = MasterError(StatusCode::NOT_FOUND, "agent OSDMap epoch is not ready");
                continue;
            }

            std::vector<std::string> nsNames;
            lastStatus = agent->GetAllNsNames(nsNames);
            if (!lastStatus.ok()) { return lastStatus; }
            if (nsNames.empty()) {
                lastStatus = MasterError(StatusCode::NOT_FOUND, "agent returned no namespaces");
                continue;
            }
            std::sort(nsNames.begin(), nsNames.end());
            nsNames.erase(std::unique(nsNames.begin(), nsNames.end()), nsNames.end());

            std::unordered_map<KvNsId, std::string> poolById;
            bool collision{false};
            for (const auto& name : nsNames) {
                const auto nsId = M800PoolNameToKvNsId(name);
                const auto [iter, inserted] = poolById.emplace(nsId, name);
                if (!inserted && iter->second != name) {
                    KV_ERROR("M800 KvNsId collision: pool '{}' and '{}' both map to {}",
                             iter->second, name, nsId);
                    collision = true;
                }
            }
            if (collision) {
                return MasterError(StatusCode::INTERNAL_ERROR, kHashCollisionMessage);
            }

            std::unordered_map<KvNsId, GlobalView> nextViews;

            for (const auto& name : nsNames) {
                if (!running_.load()) {
                    return MasterError(StatusCode::CANCELED, "M800 view fetch canceled");
                }
                const auto nsId = M800PoolNameToKvNsId(name);
                std::vector<M800AgentNodeView> nodes;
                auto status = agent->GetInByNs(name, nodes);
                if (!status.ok() || nodes.empty()) {
                    if (status.ok()) {
                        status = MasterError(StatusCode::NOT_FOUND,
                                             "namespace has no in-state OSDs: " + name);
                    }
                    KV_WARN("M800 namespace '{}' fetch failed: {}", name, status.message);
                    return status;
                }

                GlobalView view;
                view.viewEpoch = epochBefore;
                view.viewId = nsId;
                for (const auto& node : nodes) {
                    if (node.osdId < 0) {
                        KV_WARN("M800 namespace '{}' contains invalid OSD id {}", name, node.osdId);
                        continue;
                    }
                    NodeInfo info;
                    info.localKvNsId = 0;
                    for (const auto& address : node.addrs) {
                        NodeEndpoint endpoint;
                        if (!ParseNodeEndpoint(address, config_, endpoint)) {
                            KV_WARN("M800 namespace '{}' OSD {} has invalid address '{}'", name,
                                    node.osdId, address);
                            continue;
                        }
                        info.endpoints.emplace_back(std::move(endpoint));
                        if (info.endpoints.size() == kMaxEndpointsPerAsu) { break; }
                    }
                    if (node.addrs.size() > kMaxEndpointsPerAsu) {
                        KV_WARN("M800 ASU {} has {} addresses, truncating to {}", node.osdId,
                                node.addrs.size(), kMaxEndpointsPerAsu);
                    }
                    if (info.endpoints.empty()) { continue; }
                    view.asuMap.emplace(static_cast<NodeId>(node.osdId), std::move(info));
                    if (view.asuMap.size() > kMaxAsusPerNamespace) {
                        const auto maxElement = std::max_element(
                            view.asuMap.begin(), view.asuMap.end(),
                            [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
                        view.asuMap.erase(maxElement);
                    }
                }
                if (nodes.size() > kMaxAsusPerNamespace) {
                    KV_WARN("M800 namespace '{}' has {} ASUs, truncating to {}", name, nodes.size(),
                            kMaxAsusPerNamespace);
                }
                if (view.asuMap.empty()) {
                    KV_WARN("ns={} view empty, no ASU endpoints", name);
                    return MasterError(StatusCode::NOT_FOUND,
                                       "namespace has no usable endpoint: " + name);
                }
                KV_DEBUG("ns={} view built, epoch={}, asu_count={}", name, epochBefore,
                         view.asuMap.size());
                nextViews.emplace(nsId, std::move(view));
            }

            std::uint64_t epochAfter{0};
            lastStatus = agent->GetEpoch(epochAfter);
            if (!lastStatus.ok()) { return lastStatus; }
            if (epochAfter != epochBefore) {
                lastStatus = MasterError(StatusCode::IN_PROGRESS,
                                         "OSDMap epoch changed during M800 view fetch");
                continue;
            }
            if (nextViews.size() > kMaxNamespaces) {
                std::vector<KvNsId> ids;
                ids.reserve(nextViews.size());
                for (const auto& [nsId, unused] : nextViews) {
                    (void)unused;
                    ids.emplace_back(nsId);
                }
                std::sort(ids.begin(), ids.end());
                KV_WARN("M800 universe has {} namespaces, truncating to {}", ids.size(),
                        kMaxNamespaces);
                for (std::size_t index = kMaxNamespaces; index < ids.size(); ++index) {
                    nextViews.erase(ids[index]);
                }
            }
            auto next = CreateViewUniverse(std::move(nextViews), epochBefore);
            return universe.ReplaceAll(*next);
        }
        return lastStatus;
    }

    ViewHttpReply HandleHttpRequest(const ViewHttpRequest& request)
    {
        const auto snapshot = atomic_load(&lastFetched_);
        if (!snapshot || snapshot->Empty()) { return {ViewHttpReplyKind::INTERNAL_ERROR, {}}; }
        if (request.sinceEpoch != 0 && request.sinceEpoch == snapshot->GetGlobalEpoch()) {
            return {ViewHttpReplyKind::NOT_MODIFIED, {}};
        }
        std::string body;
        const auto status = M800ViewCodec::Encode(*snapshot, body);
        if (!status.ok() || body.size() > 4 * 1024 * 1024) {
            return {ViewHttpReplyKind::INTERNAL_ERROR, {}};
        }
        return {ViewHttpReplyKind::FULL_VIEW, std::move(body)};
    }

    M800ViewServerConfig config_;
    std::unique_ptr<M800ViewAgentClientFactory> agentFactory_;
    std::shared_ptr<M800ViewAgentClient> agent_;
    std::shared_ptr<ViewHttpServer> httpServer_;
    std::shared_ptr<const ViewUniverse> lastFetched_;

    std::mutex fetchMutex_;
    std::mutex mutex_;
    std::mutex stopMutex_;
    std::mutex waitMutex_;
    std::condition_variable waitCv_;
    std::atomic<bool> running_{false};
    bool started_{false};
    bool stopping_{false};
};

}  // namespace

std::shared_ptr<M800ViewMasterServer> CreateM800ViewMasterServer(
    const M800ViewServerConfig& config, std::unique_ptr<M800ViewAgentClientFactory> agentFactory)
{
    try {
        return std::make_shared<M800ViewMasterServerImpl>(config, std::move(agentFactory));
    } catch (...) {
        return nullptr;
    }
}

}  // namespace kv
