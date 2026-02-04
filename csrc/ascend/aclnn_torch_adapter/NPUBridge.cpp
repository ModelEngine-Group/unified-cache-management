// from vllm-ascend, see
// https://github.com/vllm-project/vllm-ascend/blob/main/csrc/aclnn_torch_adapter/NPUBridge.cpp

#include "NPUBridge.h"

namespace vllm_ascend
{
    NPUStorageImpl *NPUBridge::GetNpuStorageImpl(c10::StorageImpl *storageImpl)
    {
        return static_cast<NPUStorageImpl *>(storageImpl);
    }

    NPUStorageImpl *NPUBridge::GetNpuStorageImpl(c10::Storage &&storage)
    {
        return static_cast<NPUStorageImpl *>(storage.unsafeGetStorageImpl());
    }

    NPUStorageImpl *NPUBridge::GetNpuStorageImpl(const at::Tensor &tensor)
    {
        return static_cast<NPUStorageImpl *>(tensor.storage().unsafeGetStorageImpl());
    }

    NPUStorageDesc &NPUBridge::GetNpuStorageImplDesc(const at::Tensor &tensor)
    {
        return static_cast<NPUStorageImpl *>(tensor.storage().unsafeGetStorageImpl())->npu_desc_;
    }
}
