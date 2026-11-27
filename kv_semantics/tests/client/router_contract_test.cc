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
#include <gtest/gtest.h>
#include <vector>
#include "router/router.h"

namespace kv::test {
namespace {

class FixedNodeRouter final : public Router {
public:
    FixedNodeRouter() : Router([](const Key&) { return NodeId{7}; }) {}

private:
    NodeId RouteKey(const Key&) const override { return 7; }
};

// 测试场景：通过 Shard 1 的 unavailableNodes 重载路由两个 key，且不可用集合包含固定目标节点。
// 预期结果：默认实现保持旧行为并委托 RouteKeys(keys)，节点 7 仍返回索引 0 和 1，不提前实现 Shard 7
// fallback。
TEST(RouterContractTest, UnavailableNodesOverloadPreservesLegacyRouting)
{
    FixedNodeRouter router;
    const std::vector<Key> keys{"alpha", "beta"};
    const auto expected = router.RouteKeys(keys);

    const auto actual = router.RouteKeys(keys, UnavailableNodeSet{7});

    EXPECT_EQ(actual, expected);
    ASSERT_EQ(actual.count(7), 1);
    EXPECT_EQ(actual.at(7), (std::vector<Router::EntryIndex>{0, 1}));
}

}  // namespace
}  // namespace kv::test
