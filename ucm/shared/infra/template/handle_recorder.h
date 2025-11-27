#ifndef UC_INFRA_HANDLE_POOL_H
#define UC_INFRA_HANDLE_POOL_H

#include <functional>
#include "status/status.h"
#include "hashmap.h"

namespace UC {

template <typename KeyType, typename HandleType>
class HandlePool {
private:
    struct PoolEntry {
        HandleType handle;
        uint64_t refCount;
    };
    using PoolMap = HashMap<KeyType, PoolEntry, std::hash<KeyType>, 10>;
    PoolMap pool_;

public:
    HandlePool() = default;
    HandlePool(const HandlePool&) = delete;
    HandlePool& operator=(const HandlePool&) = delete;

    static HandlePool& Instance()
    {
        static HandlePool instance;
        return instance;
    }

    Status Get(const KeyType& key, HandleType& handle,
               std::function<Status(HandleType&)> instantiate)
    {
        auto result = pool_.GetOrCreate(key, [&instantiate](PoolEntry& entry) -> bool {
            HandleType h{};

            auto status = instantiate(h);
            if (status.Failure()) {
                return false;
            }

            entry.handle = h;
            entry.refCount = 1;
            return true;
        });

        if (!result.has_value()) {
            return Status::Error();
        }

        auto& entry = result.value().get();
        entry.refCount++;
        handle = entry.handle;
        return Status::OK();
    }

    void Put(const KeyType& key,
             std::function<void(HandleType)> cleanup)
    {
        pool_.Upsert(key, [&cleanup](PoolEntry& entry) -> bool {
            entry.refCount--;
            if (entry.refCount > 0) {
                return false;
            }
            cleanup(entry.handle);
            return true;
        });
    }

    void ClearAll(std::function<void(HandleType)> cleanup)
    {
        pool_.ForEach([&cleanup](const KeyType& key, PoolEntry& entry) {
            (void)key;
            cleanup(entry.handle);
        });
        pool_.Clear();
    }
};

} // namespace UC

#endif
