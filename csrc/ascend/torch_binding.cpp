#include <torch/extension.h>
#include <torch/library.h>
#include <torch/version.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include <torch_npu/csrc/npu/Module.h>
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "utils.h"
#include "aclnn_torch_adapter/op_api_common.h"

#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>

namespace ucm {

at::Tensor convert_hamming_dist_top_k_output(const at::Tensor &hashq, 
                                             const at::Tensor &hashkCache,
                                             const c10::optional<at::Tensor>& indices) {
    if (indices.has_value()) {
        return indices.value();
    }

    auto n_bs = hashq.size(0);
    auto n_kv_heads = hashkCache.size(1); 
    auto n_max_kv = 512; // 设置和hamming_dist_top_k算子实现一致
    at::Tensor res = at::empty({n_bs, n_kv_heads, n_max_kv}, torch::TensorOptions().dtype(torch::kInt32).device(hashq.device()));
    return res;
}

at::Tensor npu_hamming_dist_top_k(const at::Tensor &hashq, 
                                       const at::Tensor &hashkCache,
                                       const at::Tensor& hashkCacheRope,
                                       const at::Tensor &topN,
                                       const at::Tensor &seqLen, 
                                       const c10::optional<at::Tensor> &chunkSize,
                                       const c10::optional<int64_t> maxSeqLen, 
                                       const c10::optional<int64_t> sink, 
                                       const c10::optional<int64_t> recent, 
                                       const c10::optional<int64_t> supportOffload,
                                       const c10::optional<at::Tensor> &blockTable,
                                       const c10::optional<at::Tensor> &mask,
                                       const c10::optional<at::Tensor>& indices) {

    auto&& maxSeqLen_ = maxSeqLen.value_or(0);
    auto&& sink_ = sink.value_or(0);
    auto&& recent_ = recent.value_or(0);
    auto&& supportOffload_ = supportOffload.value_or(0);

    at::Tensor out = convert_hamming_dist_top_k_output(hashq, hashkCache, indices);
    // 调用aclnn接口计算
    EXEC_NPU_CMD(aclnnHammingDistTopK, hashq, hashkCache, topN, seqLen, chunkSize, blockTable, indices, hashkCacheRope, mask, maxSeqLen_, sink_, recent_, supportOffload_, out);
    return out;
}

// 为NPU设备注册前向实现
at::Tensor npu_reshape_and_cache_bnsd(const at::Tensor& hashq,
                                           const at::Tensor& hashkCache,
                                           const at::Tensor& slotMapping,
                                           const at::Tensor& seqLen,
                                           const at::Tensor& hashkCacheOut) {
    // 调用aclnn接口计算
    EXEC_NPU_CMD(aclnnReshapeAndCacheBnsd, hashq, hashkCache, slotMapping, seqLen, hashkCacheOut);
    return hashkCacheOut;
}

} // namespace ucm

TORCH_LIBRARY_EXPAND(CONCAT(_C, _ucm), ops)
{

    // ucm custom ops
    ops.def(
        "npu_hamming_dist_top_k(Tensor q, Tensor k_comp, Tensor k_comp_rope, Tensor k,"
        "                      Tensor seq_len, Tensor? chunk_size=None,"
        "                      int? max_seq_len=None, int? sink=None, int? recent=None, int? support_offload=None,"
        "                      Tensor? key_block_table=None, Tensor? mask=None, Tensor? indices=None) -> Tensor"
    );
    ops.impl("npu_hamming_dist_top_k", torch::kPrivateUse1, &ucm::npu_hamming_dist_top_k);
    
    ops.def(
        "npu_reshape_and_cache_bnsd(Tensor q, Tensor k_comp, Tensor slot_mapping, Tensor seq_len, Tensor k_out) -> Tensor"
    );
    ops.impl("npu_reshape_and_cache_bnsd", torch::kPrivateUse1, &ucm::npu_reshape_and_cache_bnsd);
}
