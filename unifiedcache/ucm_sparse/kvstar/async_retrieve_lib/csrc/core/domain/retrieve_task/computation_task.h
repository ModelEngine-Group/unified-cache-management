#ifndef KVSTAR_RETRIEVE_CLIB_COMPUTATION_TASK_H
#define KVSTAR_RETRIEVE_CLIB_COMPUTATION_TASK_H

#include <vector>
#include <cstdint>
#include <optional>

namespace KVStar {

// 纯C++的Tensor元数据结构, 不含任何Python对象
struct PlainTensor {
    void* data = nullptr;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
};


}



#endif