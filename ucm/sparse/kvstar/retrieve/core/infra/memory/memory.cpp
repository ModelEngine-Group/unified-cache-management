#include "memory.h"
#include <cstdlib>

namespace KVStar {

std::shared_ptr<void> MakePtr(void *ptr) {
    if (!ptr) { return nullptr; }
    return std::shared_ptr<void>(ptr, [](void *ptr) { free(ptr); });
}

std::shared_ptr<void> Memory::Alloc(const size_t size) { return MakePtr(malloc(size)); }

std::shared_ptr<void> Memory::AllocAlign(const size_t size) {
    void *ptr = nullptr;
    auto ret = posix_memalign(&ptr, _alignment, size);
    if (ret != 0) { return nullptr; }
    return MakePtr(ptr);
}
}