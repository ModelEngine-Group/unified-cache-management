#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <new>
#include <vector>
#include "logger/logger.h"

namespace UC::Compressor {

class MemoryPool {
private:
    void* pool{nullptr};
    size_t blockSize;
    size_t poolSize;
    std::vector<void*> freeBlocks;
    std::mutex mutex_;

public:
    MemoryPool(size_t blockSize, size_t poolSize) : blockSize(blockSize), poolSize(poolSize)
    {
        this->blockSize = (blockSize + 4095) & ~static_cast<size_t>(4095);
        size_t totalSize = this->blockSize * poolSize;

        void* rawPool = nullptr;
        if (posix_memalign(&rawPool, 4096, totalSize) != 0) { throw std::bad_alloc(); }
        std::unique_ptr<void, decltype(&free)> poolGuard(rawPool, &free);

        freeBlocks.reserve(poolSize);
        for (size_t i = 0; i < poolSize; ++i) {
            freeBlocks.push_back(static_cast<char*>(rawPool) + i * this->blockSize);
        }

        pool = poolGuard.release();
    }

    ~MemoryPool()
    {
        if (pool) { free(pool); }
        UC_DEBUG("free all pool.");
    }

    void* allocate()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (freeBlocks.empty()) { throw std::bad_alloc(); }
        void* block = freeBlocks.back();
        freeBlocks.pop_back();
        return block;
    }

    void deallocate(const std::vector<void*>& blocks)
    {
        if (blocks.empty()) { return; }
        std::lock_guard<std::mutex> lock(mutex_);
        freeBlocks.insert(freeBlocks.end(), blocks.begin(), blocks.end());
        UC_DEBUG("deallocate blocks count: {}", blocks.size());
    }
};

}  // namespace UC::Compressor

#endif  // MEMORY_POOL_H