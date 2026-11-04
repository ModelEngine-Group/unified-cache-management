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
#include <cstdint>
#include <optional>
#include <vector>
#include "replacement_policy.h"
#include "replacement_policy_registry.h"

namespace UC::Store {

/**
 * @brief Second-chance (clock) discipline: a hand sweeps the ring of
 *        slots and referenced slots get a second chance; both
 *        admission and Touch() set the reference bit.
 */
class ClockPolicy : public ReplacementPolicy {
protected:
    Status OnSetup(std::size_t capacity) override
    {
        capacity_ = capacity;
        hand_ = 0;
        referenced_.assign(capacity, 0);
        return Status::Ok();
    }
    void OnSlotAdmitted(std::size_t slot) override { referenced_[slot] = 1; }
    void OnSlotTouched(std::size_t slot) override { referenced_[slot] = 1; }
    void OnSlotReleased(std::size_t slot) override { referenced_[slot] = 0; }
    std::optional<std::size_t> SelectVictimSlot() override
    {
        for (std::size_t i = 0; i < 2 * capacity_; i++) {
            const std::size_t slot = hand_;
            hand_ = (hand_ + 1) % capacity_;
            if (!IsSlotCommitted(slot) || IsSlotPinned(slot)) { continue; }
            if (referenced_[slot]) {
                referenced_[slot] = 0;
                continue;
            }
            return slot;
        }
        return std::nullopt;
    }

private:
    std::size_t capacity_{0};
    std::size_t hand_{0};
    std::vector<std::uint8_t> referenced_;
};

UC_STORE_REGISTER_REPLACEMENT_POLICY(ClockPolicy, "clock")

}  // namespace UC::Store
