#ifndef UCM_SPARSE_KVSTAR_RETRIEVE_MEMORY_H
#define UCM_SPARSE_KVSTAR_RETRIEVE_MEMORY_H

#include <memory>
#include <cstddef>

namespace KVStar {

class Memory {
public:
    static bool Aligned(const size_t size) { return size % _alignment == 0;}
    static size_t Align(const size_t size) { return (size + _alignment - 1) / _alignment * _alignment; }
    static std::shared_ptr<void> Alloc(const size_t size);
    static std::shared_ptr<void> AllocAlign(const size_t size);

private:
    static constexpr size_t _alignment{4096};
};
}


#endif //UCM_SPARSE_KVSTAR_RETRIEVE_MEMORY_H