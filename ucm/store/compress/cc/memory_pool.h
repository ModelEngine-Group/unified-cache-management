#include <vector>
#include <cstddef>
#include <cstdlib>
#include <mutex>

#include "logger/logger.h"

namespace UC::Compressor {

class MemoryPool
{
private:
    void* pool{nullptr};
    size_t blockSize;
    size_t poolSize;
    std::vector<void*> freeBlocks;
    std::mutex mutex_;
public:
    MemoryPool(size_t blockSize, size_t poolSize) : blockSize(blockSize), poolSize(poolSize) {
        this->blockSize = (blockSize + 4095) & ~static_cast<size_t>(4095);
        size_t totalSize = this->blockSize * poolSize;

        if (posix_memalign(&pool, 4096, totalSize) != 0) {

        }

        freeBlocks.reserve(poolSize);
        for (size_t i = 0; i < poolSize; ++i) {
            freeBlocks.push_back(static_cast<char*>(pool) + i * this->blockSize);
        }
        
    }
    
    ~MemoryPool() {
        if (pool) free(pool);
        UC_DEBUG("free all pool.");
    }

    void* allocate() {
        std::lock_guard<std::mutex> lock(mutex_);
        void* block = freeBlocks.back();
        freeBlocks.pop_back();
        return block;
    }

    void deallocate(void* block) {
        std::lock_guard<std::mutex> lock(mutex_);
        freeBlocks.push_back(block);
    }
};

}