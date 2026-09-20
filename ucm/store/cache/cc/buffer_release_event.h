#ifndef UCM_BUFFER_RELEASE_EVENT_H
#define UCM_BUFFER_RELEASE_EVENT_H
#include <atomic>
#include <climits>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
namespace UC::CacheStore {
// Lives in the native buffer mapping. The epoch is an opportunity to retry,
// never an allocation credit. FUTEX_WAIT compares under the kernel wait lock.
struct BufferReleaseEvent {
    std::atomic<uint32_t> epoch{0};
    std::atomic<uint32_t> waiters{0};
    void Notify() noexcept
    {
        epoch.fetch_add(1, std::memory_order_seq_cst);
        if (waiters.load(std::memory_order_seq_cst)) {
            syscall(SYS_futex, &epoch, FUTEX_WAKE, INT_MAX, nullptr, nullptr, 0);
        }
    }
    uint32_t Wait(uint32_t observed, unsigned milliseconds) noexcept
    {
        waiters.fetch_add(1, std::memory_order_seq_cst);
        const timespec timeout{milliseconds / 1000, long(milliseconds % 1000) * 1000000};
        if (epoch.load(std::memory_order_seq_cst) == observed) {
            syscall(SYS_futex, &epoch, FUTEX_WAIT, observed, &timeout, nullptr, 0);
        }
        waiters.fetch_sub(1, std::memory_order_seq_cst);
        return epoch.load(std::memory_order_seq_cst);
    }
};
static_assert(std::atomic<uint32_t>::is_always_lock_free);
}  // namespace UC::CacheStore
#endif
