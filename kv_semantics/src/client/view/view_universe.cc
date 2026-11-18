/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include "view_universe.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "logger.h"

namespace kv {
namespace {

struct ViewUniverseState {
    ViewUniverseState(std::unordered_map<KvNsId, GlobalView> inputViews,
                      std::uint64_t inputGlobalEpoch)
        : views(std::move(inputViews)), globalEpoch(inputGlobalEpoch)
    {
    }

    const std::unordered_map<KvNsId, GlobalView> views;
    const std::uint64_t globalEpoch;
};

using ViewUniverseStatePtr = std::shared_ptr<const ViewUniverseState>;

ViewUniverseStatePtr MakeState(std::unordered_map<KvNsId, GlobalView> views,
                               std::uint64_t globalEpoch)
{
    return std::make_shared<ViewUniverseState>(std::move(views), globalEpoch);
}

bool EndpointEquals(const NodeEndpoint& lhs, const NodeEndpoint& rhs)
{
    return lhs.ip == rhs.ip && lhs.port == rhs.port && lhs.protocol == rhs.protocol &&
           lhs.numaNode == rhs.numaNode && lhs.hcaName == rhs.hcaName &&
           lhs.hcaPort == rhs.hcaPort && lhs.attrs == rhs.attrs;
}

}  // namespace

namespace detail {

bool NodeInfoEquals(const NodeInfo& lhs, const NodeInfo& rhs)
{
    if (lhs.localKvNsId != rhs.localKvNsId || lhs.endpoints.size() != rhs.endpoints.size()) {
        return false;
    }
    for (std::size_t index = 0; index < lhs.endpoints.size(); ++index) {
        if (!EndpointEquals(lhs.endpoints[index], rhs.endpoints[index])) { return false; }
    }
    return true;
}

std::string BuildViewPublishedLog(KvNsId nsId, std::uint64_t previousEpoch,
                                  const GlobalView& nextView)
{
    return "ns=" + std::to_string(nsId) + " view published, epoch=" +
           std::to_string(previousEpoch) + "->" + std::to_string(nextView.viewEpoch) +
           ", asu_count=" + std::to_string(nextView.asuMap.size());
}

std::string BuildViewNotPublishedLog(KvNsId nsId, std::uint64_t currentEpoch)
{
    return "ns=" + std::to_string(nsId) + " view not published, epoch unchanged=" +
           std::to_string(currentEpoch);
}

std::string BuildViewPublishedSummaryLog(std::size_t nsChanged, std::size_t nsTotal,
                                         std::size_t changedNs)
{
    return "ViewUniverse published, ns_changed=" + std::to_string(nsChanged) +
           ", ns_total=" + std::to_string(nsTotal) +
           ", changed_ns=" + std::to_string(changedNs);
}

}  // namespace detail

namespace {

bool GlobalViewEquals(const GlobalView& lhs, const GlobalView& rhs)
{
    if (lhs.viewEpoch != rhs.viewEpoch || lhs.viewId != rhs.viewId ||
        lhs.createTimeMs != rhs.createTimeMs || lhs.expireTimeMs != rhs.expireTimeMs ||
        lhs.asuMap.size() != rhs.asuMap.size()) {
        return false;
    }
    for (const auto& item : lhs.asuMap) {
        const auto rhsIt = rhs.asuMap.find(item.first);
        if (rhsIt == rhs.asuMap.end() || !detail::NodeInfoEquals(item.second, rhsIt->second)) {
            return false;
        }
    }
    return true;
}

void LogViewsNotPublished(const std::unordered_map<KvNsId, GlobalView>& views,
                          std::uint64_t currentEpoch)
{
    std::vector<KvNsId> ids;
    ids.reserve(views.size());
    for (const auto& item : views) { ids.emplace_back(item.first); }
    std::sort(ids.begin(), ids.end());
    for (const auto nsId : ids) {
        KV_INFO_UNLIMITED("{}", detail::BuildViewNotPublishedLog(nsId, currentEpoch));
    }
}

void LogViewsPublished(const std::unordered_map<KvNsId, GlobalView>& currentViews,
                       const std::unordered_map<KvNsId, GlobalView>& nextViews)
{
    std::vector<KvNsId> ids;
    ids.reserve(nextViews.size());
    for (const auto& item : nextViews) { ids.emplace_back(item.first); }
    std::sort(ids.begin(), ids.end());
    for (const auto nsId : ids) {
        const auto currentIt = currentViews.find(nsId);
        const auto previousEpoch = currentIt == currentViews.end() ? 0 : currentIt->second.viewEpoch;
        const auto& nextView = nextViews.at(nsId);
        KV_INFO_UNLIMITED("{}", detail::BuildViewPublishedLog(nsId, previousEpoch, nextView));
    }
}

class ViewUniverseImpl final : public ViewUniverse {
public:
    ViewUniverseImpl()
        : state_(MakeState(std::unordered_map<KvNsId, GlobalView>{}, 0))
    {
    }

    ViewUniverseImpl(std::unordered_map<KvNsId, GlobalView> views, std::uint64_t globalEpoch)
        : state_(MakeState(std::move(views), globalEpoch))
    {
    }

    Status PublishAll(const ViewUniverse& nextUniverse) override
    {
        std::unordered_map<KvNsId, GlobalView> nextViews;
        std::uint64_t nextEpoch{0};
        nextUniverse.GetSnapshot(nextViews, nextEpoch);
        if (nextViews.empty()) {
            KV_DEBUG("ViewUniverse publish no-op: empty snapshot, globalEpoch={}", nextEpoch);
            return Status::OK();
        }
        // Publish decisions and stores are serialized. Readers do not acquire
        // writeMutex_ and always acquire one complete immutable State.
        std::lock_guard<std::mutex> lock{writeMutex_};
        const auto current = LoadState();
        if (current->globalEpoch == 0 && !current->views.empty() && nextEpoch == 0) {
            LogViewsNotPublished(nextViews, current->globalEpoch);
            KV_DEBUG("ViewUniverse publish no-op: equal zero globalEpoch with existing views");
            return Status::OK();
        }
        if (current->globalEpoch != 0) {
            if (nextEpoch == current->globalEpoch) {
                LogViewsNotPublished(nextViews, current->globalEpoch);
                KV_DEBUG("ViewUniverse publish no-op: equal globalEpoch={}", nextEpoch);
                return Status::OK();
            }
            if (nextEpoch == 0 || nextEpoch < current->globalEpoch) {
                LogViewsNotPublished(nextViews, current->globalEpoch);
                KV_DEBUG("ViewUniverse publish no-op: stale globalEpoch={}, currentGlobalEpoch={}",
                         nextEpoch, current->globalEpoch);
                return Status::OK();
            }
        }

        const auto next = MakeState(std::move(nextViews), nextEpoch);
        std::vector<KvNsId> addedNs;
        std::vector<KvNsId> removedNs;
        std::vector<KvNsId> changedNs;
        for (const auto& item : next->views) {
            const auto currentIt = current->views.find(item.first);
            if (currentIt == current->views.end()) {
                addedNs.emplace_back(item.first);
            } else if (!GlobalViewEquals(currentIt->second, item.second)) {
                changedNs.emplace_back(item.first);
            }
        }
        for (const auto& item : current->views) {
            if (next->views.find(item.first) == next->views.end()) {
                removedNs.emplace_back(item.first);
            }
        }
        std::sort(addedNs.begin(), addedNs.end());
        std::sort(removedNs.begin(), removedNs.end());
        std::sort(changedNs.begin(), changedNs.end());

        StoreState(next);
        LogViewsPublished(current->views, next->views);
        for (const auto nsId : removedNs) {
            KV_INFO("ViewUniverse removed namespace: nsId={}, globalEpoch={}", nsId, nextEpoch);
        }
        KV_INFO_UNLIMITED("{}", detail::BuildViewPublishedSummaryLog(
                                    addedNs.size() + removedNs.size(), next->views.size(),
                                    changedNs.size()));
        return Status::OK();
    }

    Status ReplaceAll(const ViewUniverse& nextUniverse) override
    {
        // Take the source snapshot before serializing writers so self-replacement
        // and cross-universe replacement cannot recursively acquire a write lock.
        std::unordered_map<KvNsId, GlobalView> nextViews;
        std::uint64_t nextEpoch{0};
        nextUniverse.GetSnapshot(nextViews, nextEpoch);
        auto next = MakeState(std::move(nextViews), nextEpoch);

        std::lock_guard<std::mutex> lock{writeMutex_};
        StoreState(std::move(next));
        return Status::OK();
    }

    Status GetView(KvNsId nsId, GlobalView& view) const override
    {
        const auto state = LoadState();
        const auto it = state->views.find(nsId);
        if (it == state->views.end()) {
            return Status::Error(StatusCode::NOT_FOUND,
                                 "ViewUniverse namespace not found, nsId=" + std::to_string(nsId));
        }
        view = it->second;
        return Status::OK();
    }

    void GetSnapshot(std::unordered_map<KvNsId, GlobalView>& views,
                     std::uint64_t& globalEpoch) const override
    {
        const auto state = LoadState();
        views = state->views;
        globalEpoch = state->globalEpoch;
    }

    std::unordered_map<KvNsId, GlobalView> GetAllViews() const override
    {
        const auto state = LoadState();
        return state->views;
    }

    std::vector<KvNsId> GetAllNsIds() const override
    {
        const auto state = LoadState();
        std::vector<KvNsId> nsIds;
        nsIds.reserve(state->views.size());
        for (const auto& item : state->views) { nsIds.emplace_back(item.first); }
        std::sort(nsIds.begin(), nsIds.end());
        return nsIds;
    }

    std::uint64_t GetGlobalEpoch() const override { return LoadState()->globalEpoch; }

    bool Empty() const override { return LoadState()->views.empty(); }

    void Clear() override
    {
        std::lock_guard<std::mutex> lock{writeMutex_};
        StoreState(MakeState(std::unordered_map<KvNsId, GlobalView>{}, 0));
    }

private:
    ViewUniverseStatePtr LoadState() const
    {
        return std::atomic_load_explicit(&state_, std::memory_order_acquire);
    }

    void StoreState(ViewUniverseStatePtr next)
    {
        std::atomic_store_explicit(&state_, std::move(next), std::memory_order_release);
    }

    ViewUniverseStatePtr state_;
    std::mutex writeMutex_;
};

}  // namespace

void ViewUniverse::GetSnapshot(std::unordered_map<KvNsId, GlobalView>& views,
                               std::uint64_t& globalEpoch) const
{
    views = GetAllViews();
    globalEpoch = GetGlobalEpoch();
}

std::shared_ptr<ViewUniverse> CreateViewUniverse()
{
    return std::make_shared<ViewUniverseImpl>();
}

std::shared_ptr<ViewUniverse> CreateViewUniverse(
    std::unordered_map<KvNsId, GlobalView> views, std::uint64_t globalEpoch)
{
    return std::make_shared<ViewUniverseImpl>(std::move(views), globalEpoch);
}

}  // namespace kv
