#include <vector>
#include <stdexcept>
#include <queue>
#include <cmath>
#include <numeric>
#include <algorithm>

#include "simd_compute_kernel.h"
#include "kvstar_retrieve/kvstar_retrieve.h"
#include "logger/logger.h"

#if defined(__ARM_NEON)       // 在ARM平台上
    #include <arm_neon.h>
    typedef __fp16 DataType;

    float fp16_to_fp32(DataType fp16) {
        return static_cast<float>(fp16);
    }
#else                         // 在 x86 平台上，使用uint16_t存储半精度值，并通过F16C库进行转换
    #include <cstdint>
    #include <immintrin.h>
    typedef uint16_t DataType;

    float fp16_to_fp32(DataType fp16) {
        __m128i h = _mm_cvtsi32_si128((uint16_t)fp16);  // 将 uint16_t 加载到 XMM 寄存器
        __m128  f = _mm_cvtph_ps(h);                    // 使用 F16C 指令转换
        return _mm_cvtss_f32(f);                        // 提取第一个 float
    }
#endif

namespace KVStar{

void Execute(const RetrieveTask &task, TaskResult &result) {
    // 1. 维度解析
    const auto& q_shape = task.queryGroup.shape; // (x, H, d_orig)
    const auto& k_shape = task.blkRepre.shape; // (n, M, h, d_pruned) // <-- 多一个块内多向量表征维度

    if (q_shape.size() != 3) throw std::runtime_error("Query shape must be 3D (x, H, d).");
    const int64_t num_tokens = q_shape[0];
    const int64_t num_q_heads = q_shape[1];
    const int64_t d_orig = q_shape[2];

    if (k_shape.size() != 4) throw std::runtime_error("BlockRep shape must be 4D (n, M, h, d).");
    const int64_t num_blocks = k_shape[0];
    const int64_t M = k_shape[1]; // 块内多向量表征维度
    const int64_t num_kv_heads = k_shape[2];
    const int64_t d_pruned = k_shape[3];

    if (num_q_heads % num_kv_heads != 0) throw std::runtime_error("Num_q_heads must be a divisible by num_kv_heads.");
    const int64_t g = num_q_heads / num_kv_heads;

    // 2. 裁剪query通道
    const DataType* q_ptr_for_computation; // 指针指向最终用于计算的Query数据
    std::vector<DataType> pruned_q_vec; // 需要裁剪时, 把结果填充到该vector, 在if外声明保证生命周期

    const DataType* q_orig_ptr = static_cast<const DataType*>(task.queryGroup.data);

    if (task.dPrunedIndex.has_value()) {
        // 如果有裁剪索引, 就裁
        const auto& pruned_spec = task.dPrunedIndex.value();
        if (pruned_spec.shape.size() != 2 || pruned_spec.shape[0] != num_kv_heads || pruned_spec.shape[1] != d_pruned) {
            throw std::runtime_error("dPrunedIndex shape is inconsistent with K's shape.");
        }
        const int64_t* pruned_indices_ptr = static_cast<const int64_t *>(pruned_spec.data);

        // 分配内存存放裁剪后的query
        pruned_q_vec.resize(num_tokens * num_q_heads * d_pruned);

        for (int64_t x = 0; x < num_tokens; ++x) {
            for (int64_t h = 0; h < num_kv_heads; ++h) {
                // 获取当前key头的对应裁剪索引
                const int64_t* current_pruned_indices = pruned_indices_ptr + h * d_pruned;
                // 组内所有query共享相同裁剪索引
                for (int64_t gg = 0; gg < g; ++gg) {
                    int64_t H = h * g + gg; // 计算query head index
                    // 填充query head的d_pruned个维度
                    for (int64_t d_p = 0; d_p < d_pruned; ++d_p) {
                        // 从原始索引列表中找到需要抓取的原始维度索引
                        int64_t d_o = current_pruned_indices[d_p];
                        // 源地址: q_orig_ptr[x, H, d_o]
                        // 目标地址: pruned_q_vec[x, H, d_p]
                        pruned_q_vec[(x * num_q_heads + H) * d_pruned + d_p] = q_orig_ptr[(x * num_q_heads + H) * d_orig + d_o];

                    }

                }

            }

        }

        // 让计算指针指向新创建的、已裁剪的Query数据
        q_ptr_for_computation = pruned_q_vec.data();
    } else {
        // 没有维度裁剪索引, 则Q、Key维度应该一致
        if (d_orig != d_pruned) {
            throw std::runtime_error("Dimension mismatch: No dPrunedIndex, but Q and K head dims differ.");
        }
        // 计算指针直接指向原始、未改变的Query数据
        q_ptr_for_computation = q_orig_ptr;
    }

    const int64_t S = num_blocks * M;

    // Key指针始终指向传入的、已是正确维度的blkRepre数据
    const DataType* k_ptr = static_cast<const DataType*>(task.blkRepre.data);

    // 3. 计算 'xhgd, shd->xhgs', 结果张量是4D (x,h,g,s)
    std::vector<float> scires_xhgs(num_tokens * num_kv_heads * g * S);
    for (int64_t x = 0; x < num_tokens; ++x) {
        for (int64_t h = 0; h < num_kv_heads; ++h) {
            for (int64_t gg = 0; gg < g; ++gg) {
                int64_t H = h * g + gg; // q_token's head index
                // 指向 q_token[x, H, :]
                const DataType* q_vec = q_ptr_for_computation + (x * num_q_heads + H) * d_pruned;

                // 遍历所有 S = n * M 个 key vectors
                for (int64_t s = 0; s < S; ++s) {
                    // 指向 k_cache_flattened[s, h, :]
                    // k_ptr的原始布局是 (n,M,h,d)。拉平为(S,h,d)后，(s,h,d)的地址等价于原先的地址。
                    const DataType* k_vec = k_ptr + (s * num_kv_heads + h) * d_pruned;
                    float score = 0.0f;
                    for (int64_t d = 0; d < d_pruned; ++d) {
                        score += fp16_to_fp32(q_vec[d]) * fp16_to_fp32(k_vec[d]);
                    }
                    // 结果写入4D的扁平化数组
                    scires_xhgs[(((x * num_kv_heads + h) * g + gg) * S + s)] = score;
                }
            }
        }
    }

    // 4. 在 S 维度上计算 softmax, 总共有 num_tokens * num_q_heads 个 S 维向量需要做softmax, Note: num_q_heads = num_kv_heads * g
    for (int64_t i = 0; i < num_tokens * num_q_heads; ++i) {
        // 每个 S 维向量的起始地址
        float* current_scores = &scires_xhgs[i * S];

        // Softmax on S-dimension vector
        float max_val = current_scores[0];
        for (int64_t s = 1; s < S; ++s) {
            if (current_scores[s] > max_val) {
                max_val = current_scores[s];
            }
        }

        float sum_exp = 0.0f;
        for (int64_t s = 0; s < S; ++s) {
            current_scores[s] = expf(current_scores[s] - max_val);
            sum_exp += current_scores[s];
        }

        // Handle sum_exp being zero to avoid division by zero
        if (sum_exp > 1e-9) {
            for (int64_t s = 0; s < S; ++s) {
                current_scores[s] /= sum_exp;
            }
        }
    }

    // 5. 聚合 'xhgs->n', 这一步等价于 python 的 reshape + einsum
    std::vector<float> final_scores_n(num_blocks, 0.0f);
    for (int64_t xhgi = 0; xhgi < num_tokens * num_kv_heads * g; ++xhgi) {
        for (int64_t s = 0; s < S; ++s) {
            // 通过 s 计算出它属于哪个 block n
            int64_t n = s / M;
            final_scores_n[n] += scires_xhgs[xhgi * S + s];
        }
    }

    // 6. TopK on 'n'
    using ScoreIndexPair = std::pair<float, int64_t>;
    std::priority_queue<ScoreIndexPair, std::vector<ScoreIndexPair>, std::greater<ScoreIndexPair>> top_k_heap;

    for (int64_t n = 0; n < num_blocks; ++n) {
        if (top_k_heap.size() < task.topK) {
            top_k_heap.push({final_scores_n[n], n});
        } else if (final_scores_n[n] > top_k_heap.top().first) {
            top_k_heap.pop();
            top_k_heap.push({final_scores_n[n], n});
        }
    }

    std::vector<int64_t> topk_indices(top_k_heap.size());
    int index_pos = top_k_heap.size() - 1;
    while (!top_k_heap.empty()) {
        topk_indices[index_pos--] = top_k_heap.top().second;
        top_k_heap.pop();
    }

    // 7. 填充结果
    {
        std::lock_guard<std::mutex> lock(result.mtx);
        result.topkIndices = std::move(topk_indices);
        result.status.store(TaskStatus::SUCCESS, std::memory_order_release);
    }

}


}
