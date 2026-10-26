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
#include <cstddef>
#include <list>
#include <optional>
#include <vector>
#include "replacement_policy.h"
#include "replacement_policy_registry.h"

namespace UC::Store {

/**
 * @brief LRU discipline: evicts the least recently used slot; both
 *        admission and Touch() count as use.
 */
class LruPolicy : public ReplacementPolicy {
protected:
    Status OnSetup(std::size_t capacity) override
    {
        recency_.clear();
        pos_.assign(capacity, {});
        return Status::Ok();
    }
    void OnSlotAdmitted(std::size_t slot) override
    {
        pos_[slot] = recency_.insert(recency_.begin(), slot);
    }
    void OnSlotTouched(std::size_t slot) override
    {
        recency_.splice(recency_.begin(), recency_, pos_[slot]);
    }
    void OnSlotReleased(std::size_t slot) override { recency_.erase(pos_[slot]); }
    std::optional<std::size_t> SelectVictimSlot() override
    {
        for (auto it = recency_.rbegin(); it != recency_.rend(); ++it) {
            if (!IsSlotPinned(*it)) { return *it; }
        }
        return std::nullopt;
    }

private:
    std::list<std::size_t> recency_;
    std::vector<std::list<std::size_t>::iterator> pos_;
};

UC_STORE_REGISTER_REPLACEMENT_POLICY(LruPolicy, "lru")

}  // namespace UC::Store
