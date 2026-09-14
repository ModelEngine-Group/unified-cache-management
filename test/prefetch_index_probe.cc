// CPU index probe. Real native buffer/locks; fake L2 availability and no NPU/SSD I/O.
#include <iostream>
#include <cstring>
#include <unistd.h>
#include "cache/cc/reserved_prefetch.h"
#include "detail/mock_store.h"

class ProbeBackend : public UC::Test::Detail::MockStore {
public:
    size_t lookups{0};
    UC::Expected<ssize_t> LookupOnPrefix(const UC::Detail::BlockId*, size_t n) override
    { ++lookups; return static_cast<ssize_t>(n) - 1; }
    UC::Expected<std::shared_ptr<UC::ReadLease>> AcquireReadLease(
        const UC::Detail::BlockId*, size_t) override
    { return std::make_shared<UC::ReadLease>(); }
};

int main()
{
    for (bool shared : {false, true}) {
        ProbeBackend backend;
        UC::CacheStore::Config c;
        c.uniqueId = "reserve_probe_" + std::to_string(getpid()) + (shared ? "_s" : "_p");
        c.deviceId = 0;
        c.shardSize = 64;
        c.blockSize = c.shardSize * 78;
        c.tensorSizes = {64};
        c.bufferCapacity = c.blockSize * 1024;
        c.loadExclusiveBufferNumber = 0;
        c.shareBufferEnable = shared;
        c.ioDirect = false;
        c.storeBackend = &backend;
        UC::CacheStore::BufferManager manager;
        if (manager.Setup(c).Failure()) { return 1; }
        UC::CacheStore::ReservedPrefetch early(&manager, &backend, c.blockSize, c.shardSize, 1000);
        std::vector<UC::Detail::BlockId> blocks(512);
        for (size_t i = 0; i < blocks.size(); ++i) { std::memcpy(blocks[i].data(), &i, sizeof(i)); }
        auto measure = [&](const char* label) {
            backend.lookups = 0;
            UC::StopWatch timer;
            auto r = early.Reserve(blocks.data(), blocks.size());
            if (!r || r.Value().totalHitBlocks != blocks.size()) { std::abort(); }
            std::cout << "PROBE shared=" << shared << " case=" << label
                      << " ms=" << timer.Elapsed().count() * 1000
                      << " l2_calls=" << backend.lookups << '\n';
            return r.Value().handle;
        };
        auto cold = measure("cold_l2");
        early.Release(cold);
        for (const auto& block : blocks) {
            for (size_t shard = 0; shard < 78; ++shard) {
                auto pin = manager.GetTransBuffer()->Get(block, shard);
                pin.MarkReady();
            }
        }
        auto held = measure("first_hot");
        for (size_t i = 0; i < 4; ++i) { early.Release(measure("live_hot")); }
        UC::Detail::TaskDesc task;
        std::vector<uint64_t> reservations;
        for (size_t i = 0; i < 16; ++i) {
            auto r = early.Reserve(blocks.data() + i * 32, 32);
            if (!r) { return 2; }
            reservations.push_back(r.Value().handle);
        }
        task.loadReservations = reservations;
        for (const auto& block : blocks) {
            for (size_t shard = 0; shard < 78; ++shard) { task.push_back({block, shard, {nullptr}}); }
        }
        for (size_t i = 0; i < 4; ++i) {
            UC::StopWatch timer;
            auto handles = early.LoadHandles(task);
            if (handles.size() != task.size()) { return 3; }
            std::cout << "PROBE shared=" << shared << " case=load_handles ms="
                      << timer.Elapsed().count() * 1000 << '\n';
        }
        for (auto r : reservations) { early.Release(r); }
        early.Release(held);
    }
}
