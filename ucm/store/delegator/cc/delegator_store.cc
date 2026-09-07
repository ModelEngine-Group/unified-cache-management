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
 */
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>
#include "delegator_executor.h"
#include "logger/logger.h"
#include "trans/event.h"
#include "ucmstore_v1.h"

extern "C" UC::StoreV1* MakeAsuStore();

namespace UC::Delegator {

class DelegatorStore final : public StoreV1 {
public:
    ~DelegatorStore() override;

    Status Setup(const Detail::Dictionary& config) override;
    std::string Readme() const override;
    Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId* blocks, size_t num) override;
    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, size_t num) override;
    void Prefetch(const Detail::BlockId* blocks, size_t num) override;
    Expected<Detail::TaskHandle> Load(Detail::TaskDesc task) override;
    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task) override;
    Expected<bool> Check(Detail::TaskHandle task) override;
    Status Wait(Detail::TaskHandle task) override;

private:
    std::shared_ptr<StoreV1> backend_;
    std::unique_ptr<Executor> executor_;
    bool setup_{false};
};

namespace {

struct Config {
    std::string backendType;
    std::string role;
    std::vector<std::size_t> tensorSizes;
    std::int32_t deviceId{-1};
    std::size_t bufferNumber{0};
    std::size_t streamNumber{Executor::kDefaultStreamNumber};
};

Expected<Config> ParseConfig(const Detail::Dictionary& input)
{
    Config config;
    input.Get("delegator_backend", config.backendType);
    input.Get("role", config.role);
    ssize_t deviceId = -1;
    ssize_t bufferNumber = 0;
    ssize_t streamNumber = static_cast<ssize_t>(Executor::kDefaultStreamNumber);
    input.GetNumber("device_id", deviceId);
    input.GetNumber("delegator_buffer_number", bufferNumber);
    input.GetNumber("delegator_stream_number", streamNumber);
    if (deviceId < std::numeric_limits<std::int32_t>::min() ||
        deviceId > std::numeric_limits<std::int32_t>::max()) {
        return Status::InvalidParam("invalid delegator device_id");
    }
    config.deviceId = static_cast<std::int32_t>(deviceId);
    if (bufferNumber > 0) { config.bufferNumber = static_cast<std::size_t>(bufferNumber); }
    if (streamNumber > 0) { config.streamNumber = static_cast<std::size_t>(streamNumber); }

    if (config.backendType != "ASU") {
        return Status::InvalidParam("unsupported delegator backend({})", config.backendType);
    }
    if (config.role != "scheduler" && config.role != "worker") {
        return Status::InvalidParam("invalid delegator role({})", config.role);
    }
    if (config.role == "scheduler") { return config; }

    std::size_t tensorSize = 0;
    input.GetNumber("tensor_size", tensorSize);
    if (tensorSize != 0) {
        std::size_t shardSize = 0;
        input.GetNumber("shard_size", shardSize);
        if (shardSize == 0 || shardSize % tensorSize != 0) {
            return Status::InvalidParam("invalid delegator tensor_size/shard_size");
        }
        config.tensorSizes.assign(shardSize / tensorSize, tensorSize);
    } else {
        input.GetNumbers("tensor_size_list", config.tensorSizes);
    }

    // Delegator config
    if (config.deviceId < 0 || bufferNumber <= 0 || streamNumber <= 0 ||
        config.tensorSizes.empty()) {
        return Status::InvalidParam("invalid delegator worker config");
    }

    return config;
}

}  // namespace

DelegatorStore::~DelegatorStore()
{
    executor_.reset();
    backend_.reset();
}

Status DelegatorStore::Setup(const Detail::Dictionary& input)
{
    if (setup_ || executor_) { return Status::InvalidParam("delegator store already setup"); }

    auto parsed = ParseConfig(input);
    if (!parsed) { return parsed.Error(); }
    auto config = std::move(parsed).Value();

    // Init backend
    if (!backend_) {
        try {
            backend_ = std::shared_ptr<StoreV1>(MakeAsuStore());
        } catch (const std::bad_alloc&) {
            return Status::OutOfMemory();
        }
    }

    if (config.role == "scheduler") {
        auto status = backend_->Setup(input);
        if (status.Failure()) {
            backend_.reset();
            return status;
        }
        setup_ = true;
        UC_INFO("DelegatorStore setup in scheduler query-only mode with {}.", backend_->Readme());
        return Status::OK();
    }

    auto created = Executor::Create(backend_, std::move(config.tensorSizes), config.deviceId,
                                    config.bufferNumber, config.streamNumber, input);
    if (!created) {
        backend_.reset();
        return created.Error();
    }
    executor_ = std::move(created).Value();
    setup_ = true;
    UC_INFO("DelegatorStore setup with {}.", backend_->Readme());
    return Status::OK();
}

std::string DelegatorStore::Readme() const
{
    return backend_ ? "DelegatorStore(" + backend_->Readme() + ")" : "DelegatorStore";
}

Expected<std::vector<uint8_t>> DelegatorStore::Lookup(const Detail::BlockId* blocks, size_t num)
{
    if (!backend_) { return Status::Error("delegator store is not setup"); }
    return backend_->Lookup(blocks, num);
}

Expected<ssize_t> DelegatorStore::LookupOnPrefix(const Detail::BlockId* blocks, size_t num)
{
    if (!backend_) { return Status::Error("delegator store is not setup"); }
    return backend_->LookupOnPrefix(blocks, num);
}

void DelegatorStore::Prefetch(const Detail::BlockId* blocks, size_t num)
{
    if (backend_) { backend_->Prefetch(blocks, num); }
}

Expected<Detail::TaskHandle> DelegatorStore::Load(Detail::TaskDesc task)
{
    if (!executor_) { return Status::Unsupported(); }
    return executor_->Submit(std::move(task), Operation::LOAD);
}

Expected<Detail::TaskHandle> DelegatorStore::Dump(Detail::TaskDesc task)
{
    if (!executor_) { return Status::Unsupported(); }
    const auto status = Trans::Event{task.prerequisiteHandle}.Synchronize();
    if (status.Failure()) {
        UC_ERROR("Delegator wait prerequisite event failed, status={}.", status);
        return status;
    }
    task.prerequisiteHandle = 0;
    return executor_->Submit(std::move(task), Operation::DUMP);
}

Expected<bool> DelegatorStore::Check(Detail::TaskHandle task)
{
    if (!executor_) { return Status::Unsupported(); }
    return executor_->Check(task);
}

Status DelegatorStore::Wait(Detail::TaskHandle task)
{
    if (!executor_) { return Status::Unsupported(); }
    return executor_->Wait(task);
}

}  // namespace UC::Delegator

extern "C" UC::StoreV1* MakeDelegatorStore() { return new UC::Delegator::DelegatorStore(); }
