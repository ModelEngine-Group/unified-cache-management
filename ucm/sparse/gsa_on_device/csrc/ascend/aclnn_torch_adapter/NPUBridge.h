// from vllm-ascend, see
// https://github.com/vllm-project/vllm-ascend/blob/main/csrc/aclnn_torch_adapter/NPUBridge.h

#pragma once
#include <c10/core/StorageImpl.h>
#include "NPUStorageImpl.h"

namespace vllm_ascend
{

    class NPUBridge
    {
    public:
        // at::tensor to NPUStorageImpl
        static NPUStorageImpl *GetNpuStorageImpl(const at::Tensor &tensor);

        // c10::StorageImpl to NPUStorageImpl
        static NPUStorageImpl *GetNpuStorageImpl(c10::StorageImpl *storageImpl);

        // c10::Storage to NPUStorageImpl
        static NPUStorageImpl *GetNpuStorageImpl(c10::Storage &&storage);

        // tensor to NPUStorageDesc
        static NPUStorageDesc &GetNpuStorageImplDesc(const at::Tensor &tensor);
    };
}
