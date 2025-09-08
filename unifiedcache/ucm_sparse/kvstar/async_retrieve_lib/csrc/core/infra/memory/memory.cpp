#include "memory.h"
#include <cstdlib>

namespace KVStar {

// 分配的内存被智能指针自动管理，当指针离开作用域时，内存会被自动释放, 避免忘记free
std::shared_ptr<void> MakePtr(void *ptr) {
    if (!ptr) { return nullptr; }
    return std::shared_ptr<void>(ptr, [](void *ptr) { free(ptr); }); // 设置自定义删除器
}

std::shared_ptr<void> Memory::Alloc(const size_t size) { return MakePtr(malloc(size)); }

std::shared_ptr<void> Memory::AllocAlign(const size_t size) {
    void *ptr = nullptr;
    auto ret = posix_memalign(&ptr, _alignment, size);
    if (ret != 0) { return nullptr; }
    return MakePtr(ptr);
}
}