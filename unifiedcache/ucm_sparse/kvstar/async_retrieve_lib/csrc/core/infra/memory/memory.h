#ifndef UCM_SPARSE_KVSTAR_RETRIEVE_MEMORY_H
#define UCM_SPARSE_KVSTAR_RETRIEVE_MEMORY_H

#include <memory>
#include <cstddef>

namespace KVStar {

// 内存分配相关API, 4k对齐
// 静态工具类，所有成员都是 static，不需要创建 Memory 类的实例，直接通过类名来调用方法，e.g. Memory::Alloc(1024)
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