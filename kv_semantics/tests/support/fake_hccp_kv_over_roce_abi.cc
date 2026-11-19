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
// Fake implementation of the HCCP KV-over-RoCE C ABI used by the provider unit
// tests. The provider dlopens this library, so the test drives behavior through
// the HccpFake* control exports below. Three variants are built from this file:
//   * default: every required and optional symbol is exported;
//   * HCCP_FAKE_MINIMAL: optional symbols are missing (fallback coverage);
//   * HCCP_FAKE_MISSING_DELETE: DeleteConnections is missing (load failure).
#include "fake_hccp_kv_over_roce_abi.h"
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>
#include "hccp_kv_over_roce/hccp_kv_over_roce_abi.h"

namespace {

constexpr std::uint32_t kUseQpNum = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint32_t kManagerBase = 0x1000;
constexpr std::uint32_t kRkeyBase = 1000;
constexpr std::uint32_t kAccessFlags = 7;
constexpr std::uint8_t kQpcFillBase = 0x40;

struct RegisterDesc {
    int32_t memoryType{0};
    std::uint64_t addr{0};
    std::uint64_t size{0};
    std::uint64_t localAddr{0};
};

struct FakeState {
    std::mutex mutex;

    // Programmable behavior.
    std::int32_t createResult{0};
    std::uint32_t createReportedCount{kUseQpNum};
    bool createManagerNull{false};
    std::int32_t deleteResult{0};
    std::int32_t registerResult{0};
    std::int32_t registerZeroRkeyIndex{-1};
    std::int32_t unregisterResult{0};
    std::int32_t sendResult{0};
    std::int32_t sendBatchErrorIndex{-1};
    std::uint32_t abiVersion{HCCP_KV_ROCE_ABI_VERSION};

    // Recorded calls and arguments.
    std::uint32_t createCalls{0};
    std::uint32_t deleteCalls{0};
    std::uint32_t registerCalls{0};
    std::uint32_t unregisterCalls{0};
    std::uint32_t sendCalls{0};
    std::uint32_t capabilitiesCalls{0};
    bool handlesPrefilled{false};
    std::vector<std::uint8_t> qpcFillBytes;
    std::vector<void*> sendSlotAddresses;
    std::vector<void*> deleteManagers;
    std::vector<RegisterDesc> registerDescs;
    std::vector<HccpKvRoceMemoryInfo> unregisterInfos;
    void* lastManager{nullptr};
    void* lastCapabilitiesManager{nullptr};
};

std::unique_ptr<FakeState>& State()
{
    static std::unique_ptr<FakeState> state = std::make_unique<FakeState>();
    return state;
}

std::vector<std::uint8_t> MakeQpcFill(std::uint32_t qpNum)
{
    std::vector<std::uint8_t> fill;
    fill.reserve(qpNum);
    for (std::uint32_t index = 0; index < qpNum; ++index) {
        fill.push_back(static_cast<std::uint8_t>(kQpcFillBase + (index % 0x40U)));
    }
    return fill;
}

}  // namespace

// ─── Required ABI ───

extern "C" int32_t CreateConnection(const char* localIp, const char* remoteIp, uint32_t port,
                                    uint32_t qpNum, uint32_t timeout, void** connectionHandles,
                                    uint32_t* connectionHandleNum, void** connectionManager)
{
    (void)localIp;
    (void)remoteIp;
    (void)port;
    (void)timeout;
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    ++state.createCalls;

    if (connectionHandles == nullptr || connectionHandleNum == nullptr ||
        connectionManager == nullptr) {
        return HCCP_KV_ROCE_INVALID_ARGUMENT;
    }

    state.qpcFillBytes = MakeQpcFill(qpNum);
    bool prefilled = true;
    for (std::uint32_t index = 0; index < qpNum; ++index) {
        if (connectionHandles[index] == nullptr) {
            prefilled = false;
            continue;
        }
        auto* qpc = static_cast<HccpKvRoceQpc*>(connectionHandles[index]);
        std::memset(qpc->data, state.qpcFillBytes[index], sizeof(qpc->data));
    }
    state.handlesPrefilled = prefilled;

    if (state.createResult != 0) {
        *connectionHandleNum = 0;
        *connectionManager = nullptr;
        return state.createResult;
    }

    // One manager per call so the provider record and the delete call can be
    // matched by identity.
    const auto manager =
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(kManagerBase + state.createCalls));
    state.lastManager = manager;
    *connectionManager = state.createManagerNull ? nullptr : manager;
    *connectionHandleNum =
        state.createReportedCount == kUseQpNum ? qpNum : state.createReportedCount;
    return HCCP_KV_ROCE_OK;
}

#ifndef HCCP_FAKE_MISSING_DELETE
extern "C" int32_t DeleteConnections(void* const* connectionManagers, uint32_t managerNum)
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    ++state.deleteCalls;
    state.deleteManagers.clear();
    if (connectionManagers != nullptr) {
        for (std::uint32_t index = 0; index < managerNum; ++index) {
            state.deleteManagers.push_back(connectionManagers[index]);
        }
    }
    return state.deleteResult;
}
#endif

extern "C" int32_t RegisterMemory(const HccpKvRoceRegisterMemoryDesc* descs, uint32_t descNum,
                                  HccpKvRoceMemoryInfo* memoryInfos)
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    ++state.registerCalls;
    state.registerDescs.clear();
    if (descs != nullptr) {
        for (std::uint32_t index = 0; index < descNum; ++index) {
            state.registerDescs.push_back({descs[index].memoryType, descs[index].addr,
                                           descs[index].size, descs[index].localAddr});
        }
    }
    if (state.registerResult != 0) { return state.registerResult; }
    if (memoryInfos == nullptr || descs == nullptr) { return HCCP_KV_ROCE_INVALID_ARGUMENT; }

    for (std::uint32_t index = 0; index < descNum; ++index) {
        memoryInfos[index] = {};
        memoryInfos[index].addr = descs[index].addr;
        memoryInfos[index].size = descs[index].size;
        memoryInfos[index].access = kAccessFlags;
        memoryInfos[index].rkey =
            static_cast<std::int32_t>(index) == state.registerZeroRkeyIndex ? 0 : kRkeyBase + index;
    }
    return HCCP_KV_ROCE_OK;
}

// ─── Optional ABI ───

#ifndef HCCP_FAKE_MINIMAL
extern "C" int32_t UnregisterMemory(const HccpKvRoceMemoryInfo* memoryInfos, uint32_t infoNum)
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    ++state.unregisterCalls;
    state.unregisterInfos.clear();
    if (memoryInfos != nullptr) {
        for (std::uint32_t index = 0; index < infoNum; ++index) {
            state.unregisterInfos.push_back(memoryInfos[index]);
        }
    }
    return state.unregisterResult;
}

extern "C" int32_t Send(const HccpKvRoceSendIoBatch* batches, uint32_t batchNum,
                        uint32_t kernelCount, uint32_t quietCount, int32_t* statuses)
{
    (void)kernelCount;
    (void)quietCount;
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    ++state.sendCalls;
    state.sendSlotAddresses.clear();
    if (batches != nullptr) {
        for (std::uint32_t index = 0; index < batchNum; ++index) {
            state.sendSlotAddresses.push_back(batches[index].slotAddress);
        }
    }
    if (statuses != nullptr) {
        for (std::uint32_t index = 0; index < batchNum; ++index) {
            statuses[index] = static_cast<std::int32_t>(index) == state.sendBatchErrorIndex
                                  ? HCCP_KV_ROCE_IO_ERROR
                                  : 0;
        }
    }
    return state.sendResult;
}

extern "C" int32_t GetServerCapabilities(void* connectionManager,
                                         HccpKvRoceServerCapabilities* capabilities)
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    ++state.capabilitiesCalls;
    state.lastCapabilitiesManager = connectionManager;
    if (capabilities != nullptr) {
        *capabilities = {};
        capabilities->queueNum = 8;
        capabilities->ioQueueDepth = 32;
        capabilities->ioQueueKeyConcurrency = 3;
        capabilities->connectionKeyConcurrency = 4;
        capabilities->singleValueMaxBytes = 4096;
        capabilities->batchValueMaxBytes = 8192;
        capabilities->batchStoreKeys = 16;
        capabilities->batchLoadKeys = 17;
        capabilities->deleteKeys = 18;
        capabilities->queryKeys = 19;
        capabilities->keyLength = 64;
        capabilities->kvCapabilities = 5;
    }
    return HCCP_KV_ROCE_OK;
}

extern "C" uint32_t GetAbiVersion(void)
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.abiVersion;
}
#endif  // HCCP_FAKE_MINIMAL

// ─── Test control surface ───

extern "C" void HccpFakeReset() { State() = std::make_unique<FakeState>(); }

extern "C" void HccpFakeSetCreateResult(std::int32_t result)
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.createResult = result;
}

extern "C" void HccpFakeSetCreateReportedCount(std::uint32_t count, std::int32_t useQpNum)
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.createReportedCount = useQpNum != 0 ? kUseQpNum : count;
}

extern "C" void HccpFakeSetCreateManagerNull(std::int32_t managerNull)
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.createManagerNull = managerNull != 0;
}

extern "C" void HccpFakeSetDeleteResult(std::int32_t result)
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.deleteResult = result;
}

extern "C" void HccpFakeSetRegisterResult(std::int32_t result)
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.registerResult = result;
}

extern "C" void HccpFakeSetRegisterZeroRkeyIndex(std::int32_t index)
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.registerZeroRkeyIndex = index;
}

extern "C" void HccpFakeSetUnregisterResult(std::int32_t result)
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.unregisterResult = result;
}

extern "C" void HccpFakeSetSendResult(std::int32_t result)
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.sendResult = result;
}

extern "C" void HccpFakeSetSendBatchErrorIndex(std::int32_t index)
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.sendBatchErrorIndex = index;
}

extern "C" void HccpFakeSetAbiVersion(std::uint32_t version)
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.abiVersion = version;
}

extern "C" std::uint32_t HccpFakeCreateCalls()
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.createCalls;
}

extern "C" std::uint32_t HccpFakeDeleteCalls()
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.deleteCalls;
}

extern "C" std::uint32_t HccpFakeRegisterCalls()
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.registerCalls;
}

extern "C" std::uint32_t HccpFakeUnregisterCalls()
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.unregisterCalls;
}

extern "C" std::uint32_t HccpFakeSendCalls()
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.sendCalls;
}

extern "C" std::uint32_t HccpFakeCapabilitiesCalls()
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.capabilitiesCalls;
}

extern "C" std::int32_t HccpFakeHandlesPrefilled()
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.handlesPrefilled ? 1 : 0;
}

extern "C" std::uint32_t HccpFakeQpcCount()
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    return static_cast<std::uint32_t>(state.qpcFillBytes.size());
}

extern "C" std::uint32_t HccpFakeQpcByte(std::uint32_t qpIndex)
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (qpIndex >= state.qpcFillBytes.size()) { return 0; }
    return state.qpcFillBytes[qpIndex];
}

extern "C" std::uint32_t HccpFakeSendSlotCount()
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    return static_cast<std::uint32_t>(state.sendSlotAddresses.size());
}

extern "C" void* HccpFakeSendSlot(std::uint32_t index)
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (index >= state.sendSlotAddresses.size()) { return nullptr; }
    return state.sendSlotAddresses[index];
}

extern "C" std::uint32_t HccpFakeDeleteManagerCount()
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    return static_cast<std::uint32_t>(state.deleteManagers.size());
}

extern "C" void* HccpFakeDeleteManager(std::uint32_t index)
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (index >= state.deleteManagers.size()) { return nullptr; }
    return state.deleteManagers[index];
}

extern "C" void* HccpFakeLastManager()
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.lastManager;
}

extern "C" void* HccpFakeCapabilitiesManager()
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.lastCapabilitiesManager;
}

extern "C" std::uint32_t HccpFakeRegisterDescCount()
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    return static_cast<std::uint32_t>(state.registerDescs.size());
}

extern "C" std::int32_t HccpFakeRegisterDescMemoryType(std::uint32_t index)
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (index >= state.registerDescs.size()) { return -1; }
    return state.registerDescs[index].memoryType;
}

extern "C" std::uint64_t HccpFakeRegisterDescAddr(std::uint32_t index)
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (index >= state.registerDescs.size()) { return 0; }
    return state.registerDescs[index].addr;
}

extern "C" std::uint64_t HccpFakeRegisterDescSize(std::uint32_t index)
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (index >= state.registerDescs.size()) { return 0; }
    return state.registerDescs[index].size;
}

extern "C" std::uint64_t HccpFakeRegisterDescLocalAddr(std::uint32_t index)
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (index >= state.registerDescs.size()) { return 0; }
    return state.registerDescs[index].localAddr;
}

extern "C" std::uint32_t HccpFakeUnregisterInfoCount()
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    return static_cast<std::uint32_t>(state.unregisterInfos.size());
}

extern "C" std::uint32_t HccpFakeUnregisterInfoRkey(std::uint32_t index)
{
    auto& state = *State();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (index >= state.unregisterInfos.size()) { return 0; }
    return state.unregisterInfos[index].rkey;
}
