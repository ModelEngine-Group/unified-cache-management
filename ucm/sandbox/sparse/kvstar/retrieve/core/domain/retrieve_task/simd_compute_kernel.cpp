#include <vector>
#include <stdexcept>
#include <queue>
#include <cmath>
#include <numeric>
#include <algorithm>

#include "simd_compute_kernel.h"
#include "kvstar_retrieve/kvstar_retrieve.h"
#include "logger/logger.h"

#if defined(__ARM_NEON)
    #include <arm_neon.h>
    typedef __fp16 DataType;

    float fp16_to_fp32(DataType fp16) {
        return static_cast<float>(fp16);
    }
#else
    #include <cstdint>
    #include <immintrin.h>
    typedef uint16_t DataType;

    float fp16_to_fp32(DataType fp16) {
        __m128i h = _mm_cvtsi32_si128((uint16_t)fp16);
        __m128  f = _mm_cvtph_ps(h);
        return _mm_cvtss_f32(f);
    }
#endif

namespace KVStar{

void Execute(const RetrieveTask &task, TaskResult &result) {
    const auto& q_shape = task.queryGroup.shape; // (x, H, d_orig)
    const auto& k_shape = task.blkRepre.shape; // (n, M, h, d_pruned)

    if (q_shape.size() != 3) throw std::runtime_error("Query shape must be 3D (x, H, d).");
    const int64_t num_tokens = q_shape[0];
    const int64_t num_q_heads = q_shape[1];
    const int64_t d_orig = q_shape[2];

    if (k_shape.size() != 4) throw std::runtime_error("BlockRep shape must be 4D (n, M, h, d).");
    const int64_t num_blocks = k_shape[0];
    const int64_t M = k_shape[1];
    const int64_t num_kv_heads = k_shape[2];
    const int64_t d_pruned = k_shape[3];

    if (num_q_heads % num_kv_heads != 0) throw std::runtime_error("Num_q_heads must be a divisible by num_kv_heads.");
    const int64_t g = num_q_heads / num_kv_heads;

    const DataType* q_ptr_for_computation;
    std::vector<DataType> pruned_q_vec;

    const DataType* q_orig_ptr = static_cast<const DataType*>(task.queryGroup.data);

    if (task.dPrunedIndex.has_value()) {
        const auto& pruned_spec = task.dPrunedIndex.value();
        if (pruned_spec.shape.size() != 2 || pruned_spec.shape[0] != num_kv_heads || pruned_spec.shape[1] != d_pruned) {
            throw std::runtime_error("dPrunedIndex shape is inconsistent with K's shape.");
        }
        const int64_t* pruned_indices_ptr = static_cast<const int64_t *>(pruned_spec.data);

        pruned_q_vec.resize(num_tokens * num_q_heads * d_pruned);

        for (int64_t x = 0; x < num_tokens; ++x) {
            for (int64_t h = 0; h < num_kv_heads; ++h) {
                const int64_t* current_pruned_indices = pruned_indices_ptr + h * d_pruned;
                for (int64_t gg = 0; gg < g; ++gg) {
                    int64_t H = h * g + gg;
                    for (int64_t d_p = 0; d_p < d_pruned; ++d_p) {
                        int64_t d_o = current_pruned_indices[d_p];
                        pruned_q_vec[(x * num_q_heads + H) * d_pruned + d_p] = q_orig_ptr[(x * num_q_heads + H) * d_orig + d_o];

                    }

                }

            }

        }

        q_ptr_for_computation = pruned_q_vec.data();
    } else {
        if (d_orig != d_pruned) {
            throw std::runtime_error("Dimension mismatch: No dPrunedIndex, but Q and K head dims differ.");
        }

        q_ptr_for_computation = q_orig_ptr;
    }

    const int64_t S = num_blocks * M;

    const DataType* k_ptr = static_cast<const DataType*>(task.blkRepre.data);

    std::vector<float> scires_xhgs(num_tokens * num_kv_heads * g * S);
    for (int64_t x = 0; x < num_tokens; ++x) {
        for (int64_t h = 0; h < num_kv_heads; ++h) {
            for (int64_t gg = 0; gg < g; ++gg) {
                int64_t H = h * g + gg; // q_token's head index
                const DataType* q_vec = q_ptr_for_computation + (x * num_q_heads + H) * d_pruned;

                for (int64_t s = 0; s < S; ++s) {
                    const DataType* k_vec = k_ptr + (s * num_kv_heads + h) * d_pruned;
                    float score = 0.0f;
                    for (int64_t d = 0; d < d_pruned; ++d) {
                        score += fp16_to_fp32(q_vec[d]) * fp16_to_fp32(k_vec[d]);
                    }
                    scires_xhgs[(((x * num_kv_heads + h) * g + gg) * S + s)] = score;
                }
            }
        }
    }

    for (int64_t i = 0; i < num_tokens * num_q_heads; ++i) {
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

    std::vector<float> final_scores_n(num_blocks, 0.0f);
    for (int64_t xhgi = 0; xhgi < num_tokens * num_kv_heads * g; ++xhgi) {
        for (int64_t s = 0; s < S; ++s) {
            int64_t n = s / M;
            final_scores_n[n] += scires_xhgs[xhgi * S + s];
        }
    }

    // TopK on 'n'
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

    {
        std::lock_guard<std::mutex> lock(result.mtx);
        result.topkIndices = std::move(topk_indices);
        result.status.store(TaskStatus::SUCCESS, std::memory_order_release);
    }

}


}
