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

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "asu_transport/types.h"
#include "thread/index_pool.h"

namespace UC::ASU {

template <typename Context, typename State>
class SlotTaskManagerBase {
public:
    static constexpr std::size_t kMinSlotCount = 1024;
    static constexpr std::size_t kDefaultMaxInflightTasks = 4096;

    static std::size_t RecommendSlotCount(std::size_t maxInflightTasks)
    {
        // Reserve extra slots for the configured maximum inflight workload.
        // For example: 4096 inflight tasks -> 8192 slots.
        const auto required = std::max<std::size_t>(kMinSlotCount, maxInflightTasks * 2);
        return NormalizeSlotCount(required);
    }

    explicit SlotTaskManagerBase(State initialState, std::string taskName,
                                 std::size_t maxInflightTasks = kDefaultMaxInflightTasks)
        : initialState_(initialState),
          taskName_(std::move(taskName)),
          slotIndexBits_(ComputeSlotIndexBits(RecommendSlotCount(maxInflightTasks))),
          slots_(RecommendSlotCount(maxInflightTasks)),
          slotMask_(slots_.size() - 1)
    {
        freeList_.Setup(static_cast<IndexPool::Index>(slots_.size()));
    }

    Status Submit(std::unique_ptr<Context> ctx, TaskId& taskId)
    {
        if (!ctx) {
            taskId = kInvalidTaskId;
            return Status::Error(StatusCode::INVALID_ARGUMENT, taskName_ + " task context is null");
        }

        const auto acquiredIndex = freeList_.Acquire();
        if (acquiredIndex == IndexPool::npos) {
            taskId = kInvalidTaskId;
            return Status::Error(StatusCode::INTERNAL_ERROR, taskName_ + " task table is full");
        }

        const auto slotIndex = static_cast<std::size_t>(acquiredIndex);
        auto& slot = slots_[slotIndex];

        std::uint8_t expected = SlotState::EMPTY;
        if (!slot.state.compare_exchange_strong(expected, SlotState::WRITING,
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire)) {
            freeList_.Release(acquiredIndex);
            taskId = kInvalidTaskId;
            return Status::Error(StatusCode::INTERNAL_ERROR, taskName_ + " task slot not empty");
        }

        const auto generation = slot.generation.fetch_add(1, std::memory_order_relaxed) + 1;
        if (!CanEncodeGeneration(generation)) {
            AtomicStoreCtx(slot, std::shared_ptr<Context>{}, std::memory_order_release);
            slot.taskId.store(kInvalidTaskId, std::memory_order_release);
            slot.state.store(SlotState::EMPTY, std::memory_order_release);
            freeList_.Release(acquiredIndex);
            taskId = kInvalidTaskId;
            return Status::Error(StatusCode::INTERNAL_ERROR,
                                 taskName_ + " task id generation overflow");
        }

        const auto newTaskId = MakeTaskId(slotIndex, generation);
        ctx->state.store(initialState_, std::memory_order_release);
        ctx->taskId = newTaskId;
        auto sharedCtx = std::shared_ptr<Context>(std::move(ctx));

        AtomicStoreCtx(slot, sharedCtx, std::memory_order_release);
        slot.taskId.store(newTaskId, std::memory_order_release);
        slot.state.store(SlotState::READY, std::memory_order_release);

        taskId = newTaskId;
        return Status::OK();
    }

    std::shared_ptr<Context> Get(TaskId taskId)
    {
        if (taskId == kInvalidTaskId) { return nullptr; }

        const auto slotIndex = static_cast<std::size_t>(ToTaskIdUInt(taskId) & slotMask_);
        auto& slot = slots_[slotIndex];

        const auto state1 = slot.state.load(std::memory_order_acquire);
        if (state1 != SlotState::READY) { return nullptr; }

        const auto id1 = slot.taskId.load(std::memory_order_acquire);
        if (id1 != taskId) { return nullptr; }

        auto ptr = AtomicLoadCtx(slot, std::memory_order_acquire);
        if (!ptr) { return nullptr; }

        const auto id2 = slot.taskId.load(std::memory_order_acquire);
        const auto state2 = slot.state.load(std::memory_order_acquire);
        if (state2 == SlotState::READY && id2 == taskId && ptr->taskId == taskId) { return ptr; }

        return nullptr;
    }

    std::vector<std::shared_ptr<Context>> GetAll()
    {
        std::vector<std::shared_ptr<Context>> tasks;
        for (const auto& slot : slots_) {
            if (slot.state.load(std::memory_order_acquire) != SlotState::READY) { continue; }

            auto ctx = AtomicLoadCtx(slot, std::memory_order_acquire);
            if (!ctx) { continue; }

            const auto taskId = slot.taskId.load(std::memory_order_acquire);
            const auto state = slot.state.load(std::memory_order_acquire);
            if (state == SlotState::READY && taskId == ctx->taskId) {
                tasks.emplace_back(std::move(ctx));
            }
        }
        return tasks;
    }

    Status Remove(TaskId taskId)
    {
        if (taskId == kInvalidTaskId) {
            return Status::Error(StatusCode::TASK_NOT_FOUND, taskName_ + " task not found");
        }

        const auto slotIndex = static_cast<std::size_t>(ToTaskIdUInt(taskId) & slotMask_);
        auto& slot = slots_[slotIndex];

        if (slot.state.load(std::memory_order_acquire) != SlotState::READY) {
            return Status::Error(StatusCode::TASK_NOT_FOUND, taskName_ + " task not found");
        }

        auto expectedTaskId = taskId;
        if (!slot.taskId.compare_exchange_strong(expectedTaskId, kInvalidTaskId,
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_acquire)) {
            return Status::Error(StatusCode::TASK_NOT_FOUND, taskName_ + " task not found");
        }

        slot.state.store(SlotState::REMOVING, std::memory_order_release);
        AtomicStoreCtx(slot, std::shared_ptr<Context>{}, std::memory_order_release);
        slot.state.store(SlotState::EMPTY, std::memory_order_release);

        freeList_.Release(static_cast<IndexPool::Index>(slotIndex));
        return Status::OK();
    }

private:
    using TaskIdUInt = std::make_unsigned_t<TaskId>;

    struct SlotState {
        static constexpr std::uint8_t EMPTY = 0;
        static constexpr std::uint8_t WRITING = 1;
        static constexpr std::uint8_t READY = 2;
        static constexpr std::uint8_t REMOVING = 3;
    };

    struct alignas(64) Slot {
        std::atomic<std::uint8_t> state{SlotState::EMPTY};
        std::atomic<TaskIdUInt> generation{0};
        std::atomic<TaskId> taskId{kInvalidTaskId};
        std::shared_ptr<Context> ctx;
    };

private:
    static TaskIdUInt ToTaskIdUInt(TaskId taskId) { return static_cast<TaskIdUInt>(taskId); }

    static std::size_t NormalizeSlotCount(std::size_t n)
    {
        n = std::max<std::size_t>(n, kMinSlotCount);

        std::size_t power = 1;
        while (power < n) {
            if (power > (std::numeric_limits<std::size_t>::max() >> 1)) { return power; }
            power <<= 1;
        }

        return power;
    }

    static std::size_t ComputeSlotIndexBits(std::size_t slotCount)
    {
        std::size_t bits = 0;
        for (auto s = slotCount; s > 1; s >>= 1) { ++bits; }
        return bits;
    }

    bool CanEncodeGeneration(TaskIdUInt generation) const
    {
        constexpr std::size_t kTotalBits =
            static_cast<std::size_t>(std::numeric_limits<TaskIdUInt>::digits);

        if (slotIndexBits_ >= kTotalBits) { return false; }

        if (generation == 0) { return false; }

        const auto generationBits = kTotalBits - slotIndexBits_;
        if (generationBits >= kTotalBits) { return true; }

        const auto maxGeneration = (static_cast<TaskIdUInt>(1) << generationBits) - 1;
        return generation <= maxGeneration;
    }

    TaskId MakeTaskId(std::size_t slotIndex, TaskIdUInt generation) const
    {
        const auto raw =
            (generation << slotIndexBits_) | static_cast<TaskIdUInt>(slotIndex & slotMask_);
        return static_cast<TaskId>(raw);
    }

    static std::shared_ptr<Context> AtomicLoadCtx(const Slot& slot, std::memory_order order)
    {
        return std::atomic_load_explicit(&slot.ctx, order);
    }

    static void AtomicStoreCtx(Slot& slot, std::shared_ptr<Context> ptr, std::memory_order order)
    {
        std::atomic_store_explicit(&slot.ctx, std::move(ptr), order);
    }

    State initialState_;
    std::string taskName_;
    std::size_t slotIndexBits_{0};

    std::vector<Slot> slots_;
    std::size_t slotMask_{0};
    IndexPool freeList_;
};

}  // namespace UC::ASU
