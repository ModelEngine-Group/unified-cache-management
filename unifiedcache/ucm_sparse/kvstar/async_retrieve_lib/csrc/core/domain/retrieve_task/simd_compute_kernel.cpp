#include <vector>
#include <stdexcept>
#include <queue>
#include <cmath>
#include <numeric>
#include <algorithm>

#include "simd_compute_kernel.h"
#include "kvstar_retrieve/kvstar_retrieve.h"
#include "logger/logger.h"

// TODO: 适配多平台SIMD, 当前适配arm_neon
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace KVStar{

void Execute(const RetrieveTask &task, TaskResult &result) {
    // 1. 维度解析
    using DataType = __fp16;

    const auto& q_shape = task.queryGroup.shape; // (x, H, d_orig)
    const auto& k_shape = task.blkRepre.shape; // (n, h, d_pruned)

    const int64_t num_tokens = q_shape[0];
    const int64_t num_q_heads = q_shape[1];
    const int64_t d_orig = q_shape[2];

    const int64_t num_blocks = k_shape[0];
    const int64_t num_kv_heads = k_shape[1];
    const int64_t d_pruned = k_shape[2];

    if (q_shape.size() != 3) throw std::runtime_error("Query shape must be 3D (x, H, d).");
    if (k_shape.size() != 3) throw std::runtime_error("BlockRep shape must be 3D (n, h, d).");
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

    // Key指针始终指向传入的、已是正确维度的blkRepre数据
    const DataType* k_ptr = static_cast<const DataType*>(task.blkRepre.data);

    // 3. 计算 'xhgd, nhd->xhgn'
    std::vector<float> scires_xhgn(num_tokens * num_kv_heads * g * num_blocks);
    for (int64_t x = 0; x < num_tokens; ++x) {
        for (int64_t h = 0; h < num_kv_heads; ++h) {
            for (int64_t gg = 0; gg < g; ++gg) {
                int64_t H = h * g + gg;
                const DataType* q_vec = q_ptr_for_computation + (x * num_q_heads + H) * d_pruned;
                for (int64_t n = 0; n < num_blocks; ++n) {
                    const DataType* k_vec = k_ptr + (n * num_kv_heads + h) *d_pruned;
                    float score = 0.0f;
                    for (int64_t d = 0; d < d_pruned; ++d) {
                        score += static_cast<float>(q_vec[d]) * static_cast<float>(k_vec[d]);
                    }
                    scires_xhgn[((x * num_kv_heads + h) * g +gg) * num_blocks + n] = score;
                }
            }
        }
    }

    // 4. 计算softmax
    for (int64_t i = 0; i < num_tokens * num_kv_heads * g; ++i) {
        float* current_scores = &scires_xhgn[i * num_blocks];

        float max_val = current_scores[0];
        for (int64_t n = 1; n < num_blocks; ++n) {
            if (current_scores[n] > max_val) {
                max_val = current_scores[n];
            }
        }

        float sum_exp = 0.0f;
        for (int64_t n = 0; n < num_blocks; ++n) {
            current_scores[n] = expf(current_scores[n] - max_val);
            sum_exp += current_scores[n];
        }

        for (int64_t n = 0; n < num_blocks; ++n) {
            current_scores[n] /= sum_exp;
        }

    }

    // 5. 聚合 'xhgn->n'
    std::vector<float> final_scores_n(num_blocks, 0.0f);
    for (int64_t i = 0; i < num_tokens * num_kv_heads * g; ++i) {
        for (int64_t n = 0; n < num_blocks; ++n) {
            final_scores_n[n] += scires_xhgn[i * num_blocks + n];

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
