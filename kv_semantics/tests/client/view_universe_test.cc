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
#include "view/view_universe.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include "view/m800_view_codec.h"

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

std::shared_ptr<ViewUniverse> MakeUniverse(std::unordered_map<KvNsId, GlobalView> views,
                                           std::uint64_t globalEpoch)
{
    return CreateViewUniverse(std::move(views), globalEpoch);
}

// 测试场景：由现有视图集合和默认参数分别创建 ViewUniverse，并修改源集合及 GetAllViews 返回的快照。
// 预期结果：实例保存独立且完整的视图与全局 epoch，外部修改不影响内部状态，默认实例为空且 epoch 为
// 0。
TEST(ViewUniverseTest, FactoriesReturnIndependentSnapshots)
{
    std::unordered_map<KvNsId, GlobalView> source{
        {1, MakeView(7, 70, 10)}
    };
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

// 测试场景：在已有命名空间 1 的 ViewUniverse 中查询不存在的命名空间 2，输出参数预先填入其他视图。
// 预期结果：GetView 返回 NOT_FOUND，且不修改输出参数中的原有视图内容。
TEST(ViewUniverseTest, MissingViewReturnsNotFoundWithoutChangingOutput)
{
    auto universe = MakeUniverse(
        {
            {1, MakeView(1, 10, 10)}
    },
        1);
    GlobalView output = MakeView(88, 99, 77);

    const auto status = universe->GetView(2, output);

    EXPECT_EQ(status.code, StatusCode::NOT_FOUND);
    ExpectView(output, 88, 99, 77);
}

// 测试场景：向全局 epoch 为 100 且已有视图的实例发布空快照、epoch 0/99 的旧快照，以及 epoch 100
// 的相等快照。 预期结果：空 map 和 lower epoch 均返回 INVALID_ARGUMENT 并保留旧状态；equal epoch
// 返回 OK/no-op，不加入 namespace 2。
TEST(ViewUniverseTest, PublishAllRejectsEmptyAndLowerEpochsButEqualIsNoOp)
{
    auto universe = MakeUniverse(
        {
            {1, MakeView(10, 100, 10)}
    },
        100);

    EXPECT_EQ(universe->PublishAll(*CreateViewUniverse()).code, StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(universe->PublishAll(*MakeUniverse({}, 200)).code, StatusCode::INVALID_ARGUMENT);
    for (const auto epoch : {std::uint64_t{0}, std::uint64_t{99}}) {
        auto next = MakeUniverse(
            {
                {2, MakeView(200, 200, 20)}
        },
            epoch);
        EXPECT_EQ(universe->PublishAll(*next).code, StatusCode::INVALID_ARGUMENT)
            << "next epoch=" << epoch;
    }
    auto equal = MakeUniverse(
        {
            {2, MakeView(200, 200, 20)}
    },
        100);
    EXPECT_TRUE(universe->PublishAll(*equal).ok());

    EXPECT_EQ(universe->GetGlobalEpoch(), std::uint64_t{100});
    EXPECT_EQ(universe->GetAllViews().size(), std::size_t{1});
    GlobalView view;
    ASSERT_TRUE(universe->GetView(1, view).ok());
    ExpectView(view, 10, 100, 10);
    EXPECT_EQ(universe->GetView(2, view).code, StatusCode::NOT_FOUND);
}

// 测试场景：默认空实例首次发布一个全局 epoch 为 0、包含命名空间 7 的非空快照。
// 预期结果：nextEpoch 与当前 epoch 相等，因此返回 OK/no-op；实例仍为空且 epoch 为 0，不发布
// namespace 7。
TEST(ViewUniverseTest, PublishAllInitialZeroEpochIsEqualNoOp)
{
    auto universe = CreateViewUniverse();
    auto next = MakeUniverse(
        {
            {7, MakeView(0, 70, 70)}
    },
        0);

    ASSERT_TRUE(universe->PublishAll(*next).ok());

    EXPECT_TRUE(universe->Empty());
    EXPECT_EQ(universe->GetGlobalEpoch(), std::uint64_t{0});
    GlobalView view;
    EXPECT_EQ(universe->GetView(7, view).code, StatusCode::NOT_FOUND);
}

// 测试场景：用更高 globalEpoch 的快照修改 ns1、删除 ns2、新增 ns3，视图自身 epoch 更低。
// 预期结果：PublishAll 整体替换内容，不逐 ns 比较 epoch，最终仅保留 ns1 和 ns3。
TEST(ViewUniverseTest, PublishAllHigherGlobalEpochReplacesEverythingWithoutPerNsDecisions)
{
    auto universe = MakeUniverse(
        {
            {1, MakeView(100, 1000, 10)},
            {2, MakeView(100, 2000, 20)}
    },
        100);
    auto next = MakeUniverse(
        {
            {1, MakeView(1, 1001, 11)},
            {3, MakeView(0, 3000, 30)}
    },
        101);

    ASSERT_TRUE(universe->PublishAll(*next).ok());

    EXPECT_EQ(universe->GetGlobalEpoch(), std::uint64_t{101});
    auto views = universe->GetAllViews();
    ASSERT_EQ(views.size(), std::size_t{2});
    EXPECT_EQ(views.count(2), std::size_t{0});
    ExpectView(views.at(1), 1, 1001, 11);
    ExpectView(views.at(3), 0, 3000, 30);
}

// 测试场景：依次用全局 epoch 更低的非空快照和空快照调用 ReplaceAll，随后调用 Clear。
// 预期结果：ReplaceAll 无条件替换 epoch 与全部内容，空快照也会生效；Clear 清空所有视图并将全局
// epoch 重置为 0。
TEST(ViewUniverseTest, ReplaceAllAndClearReplaceEpochAndContentsUnconditionally)
{
    auto universe = MakeUniverse(
        {
            {1, MakeView(100, 1000, 10)}
    },
        100);
    auto older = MakeUniverse(
        {
            {2, MakeView(1, 20, 20)}
    },
        2);

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

// 测试场景：在同一个 ViewUniverse 实例上分别以自身为参数调用 PublishAll 和 ReplaceAll。
// 预期结果：自发布和自替换均成功且不会死锁，全局 epoch 与原有视图内容保持不变。
TEST(ViewUniverseTest, SelfPublishAndSelfReplaceDoNotDeadlockOrChangeState)
{
    auto universe = MakeUniverse(
        {
            {1, MakeView(1, 10, 10)}
    },
        1);

    EXPECT_TRUE(universe->PublishAll(*universe).ok());
    EXPECT_TRUE(universe->ReplaceAll(*universe).ok());

    EXPECT_EQ(universe->GetGlobalEpoch(), std::uint64_t{1});
    GlobalView view;
    ASSERT_TRUE(universe->GetView(1, view).ok());
    ExpectView(view, 1, 10, 10);
}

// 测试场景：两个读线程反复调用 GetAllViews，同时写线程在两组完整视图状态之间执行 1000 次
// ReplaceAll。
// 预期结果：每次读取都得到同一状态的完整原子快照，不会出现不同状态混合的视图，最终一致性检查无失败。
TEST(ViewUniverseTest, ConcurrentReadersNeverObserveMixedStates)
{
    std::unordered_map<KvNsId, GlobalView> stateA{
        {1, MakeView(1, 10, 10)},
        {2, MakeView(1, 11, 11)}
    };
    std::unordered_map<KvNsId, GlobalView> stateB{
        {1, MakeView(2, 20, 20)},
        {2, MakeView(2, 21, 21)}
    };
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
                    valid =
                        valid && epoch == firstEpoch &&
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

// 测试场景：写线程持续替换源实例，并发执行 GetSnapshot、ReplaceAll、PublishAll 和编解码读取。
// 预期结果：每次读取的视图 epoch 都与同次 globalEpoch 一致，不出现混合状态。
TEST(ViewUniverseTest, CompoundReadsUseOneAtomicSnapshot)
{
    const std::unordered_map<KvNsId, GlobalView> stateA{
        {1, MakeView(10, 1, 10)}
    };
    const std::unordered_map<KvNsId, GlobalView> stateB{
        {2, MakeView(11, 2, 20)}
    };
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

// 测试场景：以 20、10、5、30、1 的乱序向无序视图集合插入命名空间后调用 GetAllNsIds。
// 预期结果：返回的命名空间 ID 按升序排列为 1、5、10、20、30。
TEST(ViewUniverseTest, GetAllNsIdsAreSorted)
{
    std::unordered_map<KvNsId, GlobalView> views;
    for (const auto nsId : {KvNsId{20}, KvNsId{10}, KvNsId{5}, KvNsId{30}, KvNsId{1}}) {
        views.emplace(nsId, MakeView(1, nsId, nsId));
    }
    auto universe = CreateViewUniverse(std::move(views), 1);
    EXPECT_EQ(universe->GetAllNsIds(), (std::vector<KvNsId>{1, 5, 10, 20, 30}));
}

// 测试场景：分别构造单命名空间发布成功、未发布以及多命名空间发布汇总的日志消息。
// 预期结果：三类日志严格符合约定格式，并正确包含命名空间、epoch 变化、ASU 数量及变更统计。
TEST(ViewUniverseTest, PublishLogMessagesMatchContract)
{
    GlobalView next = MakeView(12, 7, 1);
    EXPECT_EQ(detail::BuildViewPublishedLog(7, 10, next),
              "ns=7 view published, epoch=10->12, asu_count=1");
    EXPECT_EQ(detail::BuildViewNotPublishedLog(7, 12),
              "ns=7 view not published, epoch unchanged=12");
    EXPECT_EQ(detail::BuildViewPublishedSummaryLog(2, 5, 1),
              "ViewUniverse published, ns_changed=2, ns_total=5, changed_ns=1");
}

// 测试场景：已有 ns1 且 globalEpoch=0 的实例，发布同 epoch 但含 ns2 的快照，再自发布。
// 预期结果：两次 equal epoch 发布均 OK/no-op，实例始终仅保留 ns1。
TEST(ViewUniverseTest, EqualZeroEpochWithExistingViewsIsNoOp)
{
    auto universe = MakeUniverse(
        {
            {1, MakeView(0, 1, 10)}
    },
        0);
    auto next = MakeUniverse(
        {
            {2, MakeView(0, 2, 20)}
    },
        0);

    ASSERT_TRUE(universe->PublishAll(*next).ok());
    EXPECT_EQ(universe->GetGlobalEpoch(), 0);
    EXPECT_EQ(universe->GetAllNsIds().size(), 1);
    EXPECT_EQ(universe->GetAllNsIds().front(), 1);
    EXPECT_TRUE(universe->PublishAll(*universe).ok());
    EXPECT_EQ(universe->GetAllNsIds().front(), 1);
}

// 测试场景：复制同一个 NodeInfo，仅将其中一份的 localKvNsId 从 1 改为 2，再分别与原节点比较。
// 预期结果：未修改的节点被判为相等，仅 localKvNsId 不同的节点被判为不相等，说明比较逻辑覆盖该字段。
TEST(ViewUniverseTest, LocalKvNsIdIsTheOnlyDifferenceDetectedAsChanged)
{
    auto currentView = MakeView(100, 1, 10);
    currentView.asuMap.begin()->second.localKvNsId = 1;
    auto unchangedView = currentView;
    auto changedView = currentView;
    changedView.asuMap.begin()->second.localKvNsId = 2;

    EXPECT_TRUE(detail::NodeInfoEquals(currentView.asuMap.begin()->second,
                                       unchangedView.asuMap.begin()->second));
    EXPECT_FALSE(detail::NodeInfoEquals(currentView.asuMap.begin()->second,
                                        changedView.asuMap.begin()->second));
}

}  // namespace
}  // namespace kv::test
