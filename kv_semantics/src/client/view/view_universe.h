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
#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>
#include "kv_types.h"
#include "view/view_server.h"

namespace kv {

// ViewUniverse defines the update/query contract for the complete
// multi-namespace cluster view. Shard 5 provides the concrete implementation.
class ViewUniverse {
public:
    virtual ~ViewUniverse() = default;

    // Atomically publishes a non-empty complete universe. Once the current
    // globalEpoch is non-zero, a zero or non-increasing next globalEpoch is a
    // whole-universe no-op. A newer snapshot replaces every namespace; there
    // are no per-namespace epoch decisions.
    virtual Status PublishAll(const ViewUniverse& nextUniverse) = 0;
    // Unconditionally replaces the complete contents and globalEpoch, including
    // with an empty or older snapshot.
    virtual Status ReplaceAll(const ViewUniverse& nextUniverse) = 0;

    virtual Status GetView(KvNsId nsId, GlobalView& view) const = 0;
    // Returns the namespace map and global epoch from one immutable snapshot.
    // The default implementation preserves compatibility for external custom
    // universes; the built-in implementation overrides it atomically.
    virtual void GetSnapshot(std::unordered_map<KvNsId, GlobalView>& views,
                             std::uint64_t& globalEpoch) const;
    virtual std::unordered_map<KvNsId, GlobalView> GetAllViews() const = 0;
    virtual std::vector<KvNsId> GetAllNsIds() const = 0;
    virtual std::uint64_t GetGlobalEpoch() const = 0;
    virtual bool Empty() const = 0;
    virtual void Clear() = 0;
};

std::shared_ptr<ViewUniverse> CreateViewUniverse();
std::shared_ptr<ViewUniverse> CreateViewUniverse(
    std::unordered_map<KvNsId, GlobalView> views, std::uint64_t globalEpoch);

}  // namespace kv
