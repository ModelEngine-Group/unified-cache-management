#ifndef KVSTAR_RETRIEVE_CLIB_COMPUTATION_TASK_H
#define KVSTAR_RETRIEVE_CLIB_COMPUTATION_TASK_H

#include <vector>
#include <cstdint>
#include <optional>

namespace KVStar {

struct PlainTensor {
    void* data = nullptr;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
};


}



#endif