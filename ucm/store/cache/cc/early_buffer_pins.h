#ifndef UCM_CACHE_EARLY_BUFFER_PINS_H
#define UCM_CACHE_EARLY_BUFFER_PINS_H
#include <vector>
#include "trans_buffer.h"
#include "early_domain.h"

namespace UC::CacheStore {
// Access the existing buffer only. No buffer, index, lock or shared identity is owned here.
class EarlyBufferPins {
public:
    struct Batch {
        std::vector<TransBuffer::Handle> pins;
        size_t bucketLocks{0};
        bool capacityConflict{false};
    };
    static Batch PinBlocks(TransBuffer& buffer, const Detail::BlockId* blocks,
                           size_t count, size_t shardsPerBlock);
    static void Touch(TransBuffer& buffer, const std::vector<TransBuffer::Handle>& pins);
    static std::vector<size_t> Locations(const std::vector<TransBuffer::Handle>& pins);
    static std::vector<TransBuffer::Handle> Recall(TransBuffer& buffer,
        const Detail::BlockId& block, const std::vector<size_t>& locations, size_t& checks);
    static size_t HintBlockCapacity(TransBuffer& buffer, size_t shardsPerBlock);
    static size_t ReservableSlots(TransBuffer& buffer);
    static Early::Domain DescribeDomain(TransBuffer& buffer);
private:
    static TransBuffer::Handle PinNode(TransBuffer& buffer, size_t node,
                                      bool& capacityConflict);
};
}  // namespace UC::CacheStore
#endif
