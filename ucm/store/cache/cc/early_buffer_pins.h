#ifndef UCM_CACHE_EARLY_BUFFER_PINS_H
#define UCM_CACHE_EARLY_BUFFER_PINS_H
#include <vector>
#include "trans_buffer.h"

namespace UC::CacheStore {
// Access the existing buffer only. No buffer, index, lock or shared identity is owned here.
class EarlyBufferPins {
public:
    struct Batch {
        std::vector<TransBuffer::Handle> pins;
        size_t bucketLocks{0};
        bool capacityConflict{false};
    };
    static TransBuffer::Handle PinOne(TransBuffer& buffer, const Detail::BlockId& block,
                                     size_t shard, bool* capacityConflict = nullptr,
                                     bool includeLoading = false);
    static Batch PinBlocks(TransBuffer& buffer, const Detail::BlockId* blocks,
                           size_t count, size_t shardsPerBlock);
    static void Touch(TransBuffer& buffer, const std::vector<TransBuffer::Handle>& pins);
private:
    static TransBuffer::Handle PinNode(TransBuffer& buffer, size_t node,
                                      bool* capacityConflict, bool includeLoading);
};
}  // namespace UC::CacheStore
#endif
