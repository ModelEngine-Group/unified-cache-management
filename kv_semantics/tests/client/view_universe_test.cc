#include "view/view_universe.h"
#include "view/m800_view_codec.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kv::test {
namespace {

GlobalView MakeView(std::uint64_t viewEpoch, std::uint64_t viewId, NodeId nodeId)
{
    GlobalView view;
    view.viewEpoch = viewEpoch;
    view.viewId = viewId;
    view.createTimeMs = viewId + 1000;
    view.expireTimeMs = viewId + 2000;

    NodeEndpoint endpoint;
    endpoint.ip = "192.0.2." + std::to_string(nodeId);
    endpoint.port = static_cast<std::uint16_t>(6000 + nodeId % 1000);
    endpoint.protocol = Protocol::UB;
    endpoint.attrs["placement"] = "device";
    NodeInfo node;
    node.endpoints.emplace_back(std::move(endpoint));
    view.asuMap.emplace(nodeId, std::move(node));
    return view;
}

void ExpectView(const GlobalView& view, std::uint64_t viewEpoch, std::uint64_t viewId,
                NodeId nodeId)
{
    EXPECT_EQ(view.viewEpoch, viewEpoch);
    EXPECT_EQ(view.viewId, viewId);
    EXPECT_EQ(view.createTimeMs, viewId + 1000);
    EXPECT_EQ(view.expireTimeMs, viewId + 2000);
    ASSERT_EQ(view.asuMap.size(), std::size_t{1});
    const auto it = view.asuMap.find(nodeId);
    ASSERT_NE(it, view.asuMap.end());
    ASSERT_EQ(it->second.endpoints.size(), std::size_t{1});
    EXPECT_EQ(it->second.endpoints.front().ip, "192.0.2." + std::to_string(nodeId));
    EXPECT_EQ(it->second.endpoints.front().protocol, Protocol::UB);
    EXPECT_EQ(it->second.endpoints.front().attrs.at("placement"), "device");
}

std::shared_ptr<ViewUniverse> MakeUniverse(
    std::unordered_map<KvNsId, GlobalView> views, std::uint64_t globalEpoch)
{
    return CreateViewUniverse(std::move(views), globalEpoch);
}

TEST(ViewUniverseTest, FactoriesReturnIndependentSnapshots)
{
    std::unordered_map<KvNsId, GlobalView> source{{1, MakeView(7, 70, 10)}};
    auto universe = CreateViewUniverse(source, 10);
    source[1].viewId = 999;

    ASSERT_NE(universe, nullptr);
    EXPECT_FALSE(universe->Empty());
    EXPECT_EQ(universe->GetGlobalEpoch(), std::uint64_t{10});

    auto nsIds = universe->GetAllNsIds();
    std::sort(nsIds.begin(), nsIds.end());
    EXPECT_EQ(nsIds, std::vector<KvNsId>{1});

    GlobalView view;
    ASSERT_TRUE(universe->GetView(1, view).ok());
    ExpectView(view, 7, 70, 10);

    auto snapshot = universe->GetAllViews();
    snapshot[1].viewId = 1000;
    snapshot[1].asuMap.clear();
    ASSERT_TRUE(universe->GetView(1, view).ok());
    ExpectView(view, 7, 70, 10);

    auto empty = CreateViewUniverse();
    ASSERT_NE(empty, nullptr);
    EXPECT_TRUE(empty->Empty());
    EXPECT_EQ(empty->GetGlobalEpoch(), std::uint64_t{0});
    EXPECT_TRUE(empty->GetAllViews().empty());
    EXPECT_TRUE(empty->GetAllNsIds().empty());
}

TEST(ViewUniverseTest, MissingViewReturnsNotFoundWithoutChangingOutput)
{
    auto universe = MakeUniverse({{1, MakeView(1, 10, 10)}}, 1);
    GlobalView output = MakeView(88, 99, 77);

    const auto status = universe->GetView(2, output);

    EXPECT_EQ(status.code, StatusCode::NOT_FOUND);
    ExpectView(output, 88, 99, 77);
}

TEST(ViewUniverseTest, PublishAllEmptyZeroAndNonIncreasingEpochsAreWholeUniverseNoOps)
{
    auto universe = MakeUniverse({{1, MakeView(10, 100, 10)}}, 100);

    EXPECT_TRUE(universe->PublishAll(*CreateViewUniverse()).ok());
    EXPECT_TRUE(universe->PublishAll(*MakeUniverse({}, 200)).ok());
    for (const auto epoch : {std::uint64_t{0}, std::uint64_t{99}, std::uint64_t{100}}) {
        auto next = MakeUniverse({{2, MakeView(200, 200, 20)}}, epoch);
        EXPECT_TRUE(universe->PublishAll(*next).ok()) << "next epoch=" << epoch;
    }

    EXPECT_EQ(universe->GetGlobalEpoch(), std::uint64_t{100});
    EXPECT_EQ(universe->GetAllViews().size(), std::size_t{1});
    GlobalView view;
    ASSERT_TRUE(universe->GetView(1, view).ok());
    ExpectView(view, 10, 100, 10);
    EXPECT_EQ(universe->GetView(2, view).code, StatusCode::NOT_FOUND);
}

TEST(ViewUniverseTest, PublishAllAcceptsInitialNonEmptyZeroEpochSnapshot)
{
    auto universe = CreateViewUniverse();
    auto next = MakeUniverse({{7, MakeView(0, 70, 70)}}, 0);

    ASSERT_TRUE(universe->PublishAll(*next).ok());

    EXPECT_FALSE(universe->Empty());
    EXPECT_EQ(universe->GetGlobalEpoch(), std::uint64_t{0});
    GlobalView view;
    ASSERT_TRUE(universe->GetView(7, view).ok());
    ExpectView(view, 0, 70, 70);
}

TEST(ViewUniverseTest, PublishAllHigherGlobalEpochReplacesEverythingWithoutPerNsDecisions)
{
    auto universe = MakeUniverse(
        {{1, MakeView(100, 1000, 10)}, {2, MakeView(100, 2000, 20)}}, 100);
    auto next = MakeUniverse(
        {{1, MakeView(1, 1001, 11)}, {3, MakeView(0, 3000, 30)}}, 101);

    ASSERT_TRUE(universe->PublishAll(*next).ok());

    EXPECT_EQ(universe->GetGlobalEpoch(), std::uint64_t{101});
    auto views = universe->GetAllViews();
    ASSERT_EQ(views.size(), std::size_t{2});
    EXPECT_EQ(views.count(2), std::size_t{0});
    ExpectView(views.at(1), 1, 1001, 11);
    ExpectView(views.at(3), 0, 3000, 30);
}

TEST(ViewUniverseTest, ReplaceAllAndClearReplaceEpochAndContentsUnconditionally)
{
    auto universe = MakeUniverse({{1, MakeView(100, 1000, 10)}}, 100);
    auto older = MakeUniverse({{2, MakeView(1, 20, 20)}}, 2);

    ASSERT_TRUE(universe->ReplaceAll(*older).ok());
    EXPECT_EQ(universe->GetGlobalEpoch(), std::uint64_t{2});
    EXPECT_EQ(universe->GetAllNsIds(), std::vector<KvNsId>{2});

    auto emptyWithEpoch = MakeUniverse({}, 1);
    ASSERT_TRUE(universe->ReplaceAll(*emptyWithEpoch).ok());
    EXPECT_TRUE(universe->Empty());
    EXPECT_EQ(universe->GetGlobalEpoch(), std::uint64_t{1});

    universe->Clear();
    EXPECT_TRUE(universe->Empty());
    EXPECT_EQ(universe->GetGlobalEpoch(), std::uint64_t{0});
}

TEST(ViewUniverseTest, SelfPublishAndSelfReplaceDoNotDeadlockOrChangeState)
{
    auto universe = MakeUniverse({{1, MakeView(1, 10, 10)}}, 1);

    EXPECT_TRUE(universe->PublishAll(*universe).ok());
    EXPECT_TRUE(universe->ReplaceAll(*universe).ok());

    EXPECT_EQ(universe->GetGlobalEpoch(), std::uint64_t{1});
    GlobalView view;
    ASSERT_TRUE(universe->GetView(1, view).ok());
    ExpectView(view, 1, 10, 10);
}

TEST(ViewUniverseTest, ConcurrentReadersNeverObserveMixedStates)
{
    std::unordered_map<KvNsId, GlobalView> stateA{{1, MakeView(1, 10, 10)},
                                                   {2, MakeView(1, 11, 11)}};
    std::unordered_map<KvNsId, GlobalView> stateB{{1, MakeView(2, 20, 20)},
                                                   {2, MakeView(2, 21, 21)}};
    auto universe = MakeUniverse(stateA, 0);

    std::atomic<bool> done{false};
    std::atomic<bool> failed{false};
    std::vector<std::thread> readers;
    for (int reader = 0; reader < 2; ++reader) {
        readers.emplace_back([&] {
            while (!done.load(std::memory_order_acquire)) {
                const auto views = universe->GetAllViews();
                bool valid = views.size() == 2;
                for (const auto& item : views) {
                    const auto epoch = item.second.viewEpoch;
                    const auto firstEpoch = views.begin()->second.viewEpoch;
                    valid = valid && epoch == firstEpoch &&
                            ((epoch == 1 && (item.second.viewId == 10 || item.second.viewId == 11)) ||
                             (epoch == 2 && (item.second.viewId == 20 || item.second.viewId == 21)));
                }
                if (!valid) {
                    failed.store(true, std::memory_order_release);
                    break;
                }
            }
        });
    }

    for (std::uint64_t iteration = 1; iteration <= 1000; ++iteration) {
        auto next = MakeUniverse(iteration % 2 == 0 ? stateA : stateB, iteration);
        EXPECT_TRUE(universe->ReplaceAll(*next).ok());
    }
    done.store(true, std::memory_order_release);
    for (auto& reader : readers) { reader.join(); }

    EXPECT_FALSE(failed.load(std::memory_order_acquire));
}

TEST(ViewUniverseTest, CompoundReadsUseOneAtomicSnapshot)
{
    const std::unordered_map<KvNsId, GlobalView> stateA{{1, MakeView(10, 1, 10)}};
    const std::unordered_map<KvNsId, GlobalView> stateB{{2, MakeView(11, 2, 20)}};
    auto source = MakeUniverse(stateA, 10);
    auto replaced = CreateViewUniverse();
    auto published = CreateViewUniverse();
    std::atomic<bool> done{false};
    std::atomic<bool> failed{false};
    auto checkCoherent = [&](const ViewUniverse& universe) {
        std::unordered_map<KvNsId, GlobalView> views;
        std::uint64_t epoch{0};
        universe.GetSnapshot(views, epoch);
        for (const auto& [nsId, view] : views) {
            (void)nsId;
            if (view.viewEpoch != epoch) { return false; }
        }
        return true;
    };

    std::thread writer([&] {
        for (int index = 0; index < 200 && !done.load(); ++index) {
            auto next = MakeUniverse(index % 2 == 0 ? stateB : stateA, index % 2 == 0 ? 11 : 10);
            source->ReplaceAll(*next);
        }
    });
    for (int index = 0; index < 200; ++index) {
        if (!checkCoherent(*source)) { failed.store(true); }
        replaced->ReplaceAll(*source);
        if (!checkCoherent(*replaced)) { failed.store(true); }
        published->PublishAll(*source);
        if (!checkCoherent(*published)) { failed.store(true); }
        std::string encoded;
        if (!M800ViewCodec::Encode(*source, encoded).ok()) {
            failed.store(true);
        } else {
            auto decoded = CreateViewUniverse();
            if (!M800ViewCodec::Decode(encoded, *decoded).ok() || !checkCoherent(*decoded)) {
                failed.store(true);
            }
        }
    }
    done.store(true);
    writer.join();
    EXPECT_FALSE(failed.load());
}

}  // namespace
}  // namespace kv::test
