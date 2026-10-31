/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */
#ifndef UCM_PREFETCH_TEST_MOCK_STORE_V1_H
#define UCM_PREFETCH_TEST_MOCK_STORE_V1_H

#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include "status/status.h"
#include "type/dictionary.h"
#include "type/types.h"
#include "ucmstore_v1.h"

namespace ucmprefetch_test {

/*
 * Minimal, configurable mock of UC::StoreV1.  It records every call and lets
 * tests inject failures / canned host-buffer payloads so the prefetch code
 * paths that depend on mStoreV1_ (LastChunkMove, UCMTensorMerge, indexer load,
 * UCMKVPrefetch) can be driven deterministically without a real store.
 *
 * After the ddr-load refactor (post 717d416) the UC::StoreV1 interface dropped
 * the LoadToDevice / LoadSSDToHost / WaitSSDToHost / LoadToHostBuffer /
 * ReleaseHostBuffer methods (and the HostBufferResult / HostBufferView types).
 * The prefetch code now drives loads through Load + Wait, with shards carrying
 * caller-provided DDR host addresses; the mock's Load honours that contract by
 * writing a deterministic per-shard payload into each shard's host address.
 */
class MockStoreV1 : public UC::StoreV1 {
public:
    struct CallRecord {
        std::string name;
        UC::Detail::TaskDesc task;
        UC::Detail::TaskHandle handle{0};
        size_t blockCount{0};
    };

    /* Failure injection switches. */
    /* Legacy switches kept as inert data members so older tests that still set
     * them compile; they have no effect now that the corresponding virtuals are
     * gone from the UC::StoreV1 interface. */
    bool loadToHostBufferFails{false};
    bool loadToDeviceFails{false};
    bool loadSSDToHostFails{false};
    bool waitSSDToHostFails{false};
    bool releaseHostBufferFails{false};
    bool mismatchedViewCount{false};
    bool nullHostAddr{false};

    /* Live switches for the surviving Load / Wait / Lookup interface. */
    bool lookupFails{false};
    bool waitFails{false};
    bool checkFails{false};
    bool loadFails{false};
    bool dumpFails{false};

    /* Configuration of canned per-shard payloads written into DDR addresses. */
    void SetHostBufferPayload(std::size_t bytesPerBlock, std::size_t kvLayerBytes)
    {
        std::lock_guard<std::mutex> lock(mu_);
        bytesPerBlock_ = bytesPerBlock;
        kvLayerBytes_ = kvLayerBytes;
    }

    std::vector<CallRecord> calls;
    std::mutex mu_;

    MockStoreV1() { SetHostBufferPayload(4096, 1024); }

    UC::Status Setup(const UC::Detail::Dictionary&) override { return UC::Status::OK(); }
    std::string Readme() const override { return "MockStoreV1"; }

    UC::Expected<std::vector<uint8_t>> Lookup(const UC::Detail::BlockId* blocks,
                                              size_t num) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        CallRecord r{"Lookup"};
        r.blockCount = num;
        calls.push_back(std::move(r));
        if (lookupFails) { return UC::Status::Error("lookup-failed-injected"); }
        return std::vector<uint8_t>(num, 1u);
    }

    UC::Expected<ssize_t> LookupOnPrefix(const UC::Detail::BlockId*, size_t num) override
    {
        return static_cast<ssize_t>(num) - 1;
    }
    UC::Expected<ssize_t> LookupOnReverse(const UC::Detail::BlockId*, size_t num) override
    {
        return num == 0 ? ssize_t(-1) : ssize_t(num - 1);
    }

    void Prefetch(const UC::Detail::BlockId*, size_t) override { /* no-op */ }

    UC::Expected<UC::Detail::TaskHandle> Load(UC::Detail::TaskDesc task) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        CallRecord r{"Load"};
        r.task = task;
        r.handle = NextHandle();
        r.blockCount = task.size();
        calls.push_back(std::move(r));
        if (loadFails) { return UC::Status::Error("load-failed-injected"); }
        /* DDR contract: for shards carrying a host address, write a deterministic
         * per-shard payload (shard index + 1, mod 256) into the address so tests
         * can verify the downstream copy pipeline. This mirrors how the real
         * store fills the caller-provided DDR staging slots. */
        for (std::size_t i = 0; i < task.size(); ++i) {
            void* addr = nullptr;
            for (const auto& a : task[i].addrs) {
                addr = a;
                break;
            }
            if (addr != nullptr && bytesPerBlock_ > 0) {
                std::memset(addr, static_cast<int>((i + 1) & 0xFF), bytesPerBlock_);
            }
        }
        return UC::Detail::TaskHandle{r.handle};
    }

    UC::Expected<UC::Detail::TaskHandle> Dump(UC::Detail::TaskDesc task) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        CallRecord r{"Dump"};
        r.task = task;
        r.handle = NextHandle();
        calls.push_back(std::move(r));
        if (dumpFails) { return UC::Status::Error("dump-failed-injected"); }
        return UC::Detail::TaskHandle{r.handle};
    }

    UC::Expected<bool> Check(UC::Detail::TaskHandle) override
    {
        if (checkFails) { return UC::Status::Error("check-failed-injected"); }
        return true;
    }

    UC::Status Wait(UC::Detail::TaskHandle handle) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        CallRecord r{"Wait"};
        r.handle = handle;
        calls.push_back(std::move(r));
        if (waitFails) { return UC::Status::Error("wait-failed-injected"); }
        return UC::Status::OK();
    }

    ~MockStoreV1() override = default;

private:
    UC::Detail::TaskHandle NextHandle() { return ++nextHandle_; }
    UC::Detail::TaskHandle nextHandle_{0};
    std::size_t bytesPerBlock_{4096};
    std::size_t kvLayerBytes_{1024};
};

}  // namespace ucmprefetch_test

#endif  // UCM_PREFETCH_TEST_MOCK_STORE_V1_H
