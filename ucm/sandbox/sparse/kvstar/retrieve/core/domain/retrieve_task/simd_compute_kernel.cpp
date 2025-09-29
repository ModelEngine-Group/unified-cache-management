#include <vector>
#include <stdexcept>
#include <queue>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include "simd_compute_kernel.h"
#include "kvstar_retrieve/kvstar_retrieve.h"
#include "logger/logger.h"
#include <iomanip>


namespace KVStar
{
#if defined(__ARM_NEON)
#include <arm_neon.h>
    namespace neon_impl
    {
        __attribute__((always_inline)) inline static __fp16 hmax_f16(const __fp16 *__restrict x, int n) noexcept
        {
            int i = 0;
            float16x8_t vmax8 = vdupq_n_f16((__fp16)(-1.0f / 0.0f));
            for (; i + 8 <= n; i += 8)
            {
                float16x8_t v = vld1q_f16(x + i);
                vmax8 = vmaxq_f16(vmax8, v);
            }
            float16x4_t lo = vget_low_f16(vmax8);
            float16x4_t hi = vget_high_f16(vmax8);
            float16x4_t m4 = vmax_f16(lo, hi);
            float16x4_t m2 = vpmax_f16(m4, m4);
            float16x4_t m1 = vpmax_f16(m2, m2);
            __fp16 m = vget_lane_f16(m1, 0);
            for (; i < n; ++i)
                if (x[i] > m)
                    m = x[i];
            return m;
        }

        __attribute__((always_inline)) inline static float hmax_f32(const float *__restrict x, int n) noexcept
        {
            int i = 0;
            float32x4_t vmax4 = vdupq_n_f32(-1.0f / 0.0f);
            for (; i + 4 <= n; i += 4)
            {
                float32x4_t v = vld1q_f32(x + i);
                vmax4 = vmaxq_f32(vmax4, v);
            }

            float32x2_t max2 = vpmax_f32(vget_low_f32(vmax4), vget_high_f32(vmax4));
            float32x2_t max1 = vpmax_f32(max2, max2);
            float m = vget_lane_f32(max1, 0);
            for (; i < n; ++i)
                if (x[i] > m)
                    m = x[i];
            return m;
        }

        __attribute__((always_inline)) inline static float hsum_f16x8(float16x8_t v) noexcept
        {
            float16x4_t lo = vget_low_f16(v);
            float16x4_t hi = vget_high_f16(v);
            float16x4_t s4 = vadd_f16(lo, hi);
            float16x4_t p2 = vpadd_f16(s4, s4);
            float16x4_t s = vadd_f16(vdup_lane_f16(p2, 0), vdup_lane_f16(p2, 1));
            return vget_lane_f16(s, 0);
        }

        __attribute__((always_inline)) inline static float hsum_f32x4(float32x4_t v) noexcept
        {
            float32x2_t lo = vget_low_f32(v);   // v0 v1
            float32x2_t hi = vget_high_f32(v);  // v2 v3
            float32x2_t s2 = vadd_f32(lo, hi);  // v0+v2, v1+v3
            float32x2_t s1 = vpadd_f32(s2, s2); // (v0+v2)+(v1+v3), ...
            return vget_lane_f32(s1, 0);
        }

        __attribute__((always_inline)) inline static void _neon_gemv_fp32_fp16(const __fp16 *__restrict A,
                                                                               const __fp16 *__restrict x,
                                                                               float *__restrict y,
                                                                               int K, int N, int stride) noexcept
        {
            int j = 0;
            for (; j + 32 <= K; j += 32)
            {
                const float16x8_t x0 = vld1q_f16(x + j + 0 * 8);
                const float16x8_t x1 = vld1q_f16(x + j + 1 * 8);
                const float16x8_t x2 = vld1q_f16(x + j + 2 * 8);
                const float16x8_t x3 = vld1q_f16(x + j + 3 * 8);

                for (int i = 0; i < N; i++)
                {
                    const __fp16 *Ai = A + (int64_t)i * stride;
                    float16x8_t acc = vdupq_n_f16((__fp16)0);
                    acc = vfmaq_f16(acc, vld1q_f16(Ai + j + 0 * 8), x0);
                    acc = vfmaq_f16(acc, vld1q_f16(Ai + j + 1 * 8), x1);
                    acc = vfmaq_f16(acc, vld1q_f16(Ai + j + 2 * 8), x2);
                    acc = vfmaq_f16(acc, vld1q_f16(Ai + j + 3 * 8), x3);
                    y[i] += hsum_f16x8(acc);
                }
            }
            for (; j + 16 <= K; j += 16)
            {
                const float16x8_t x0 = vld1q_f16(x + j + 0 * 8);
                const float16x8_t x1 = vld1q_f16(x + j + 1 * 8);

                for (int i = 0; i < N; i++)
                {
                    const __fp16 *Ai = A + (int64_t)i * stride;
                    float16x8_t acc = vdupq_n_f16((__fp16)0);
                    acc = vfmaq_f16(acc, vld1q_f16(Ai + j + 0 * 8), x0);
                    acc = vfmaq_f16(acc, vld1q_f16(Ai + j + 1 * 8), x1);
                    y[i] += hsum_f16x8(acc);
                }
            }

            for (; j + 8 <= K; j += 8)
            {
                const float16x8_t x0 = vld1q_f16(x + j + 0 * 8);
                for (int i = 0; i < N; i++)
                {
                    const __fp16 *Ai = A + (int64_t)i * stride;
                    float16x8_t acc = vmulq_f16(vld1q_f16(Ai + j), x0);
                    y[i] += hsum_f16x8(acc);
                }
            }

            if (j < K)
            {
                for (int i = 0; i < N; i++)
                {
                    const __fp16 *Ai = A + (int64_t)i * stride;
                    for (int jj = j; jj < K; jj++)
                        y[i] += Ai[jj] * (x + j)[jj];
                }
            }
            return;
        }
        __attribute__((always_inline)) inline static float32x4_t _neon_exp_approx_f32(float32x4_t x) noexcept
        {
            const float32x4_t log2e = vdupq_n_f32(1.44269504088896341f);

            float32x4_t y = vmulq_f32(x, log2e); // y = x * log2(e)

            float32x4_t n_f = vrndaq_f32(y); // n = round(y)

            float32x4_t r = vsubq_f32(y, n_f); // r = (y - n)

            // Taylor series
            const float32x4_t c1 = vdupq_n_f32(1.0f);
            const float32x4_t c2 = vdupq_n_f32(1.0f);
            const float32x4_t c3 = vdupq_n_f32(1.0f / 2.0f);
            const float32x4_t c4 = vdupq_n_f32(1.0f / 6.0f);
            const float32x4_t c5 = vdupq_n_f32(1.0f / 24.0f);
            const float32x4_t c6 = vdupq_n_f32(1.0f / 120.0f);
            const float32x4_t c7 = vdupq_n_f32(1.0f / 720.0f);

            // Horner's method
            float32x4_t p = vmlaq_f32(c6, c7, r); // c6 + c7*r
            p = vmlaq_f32(c5, p, r);              // c5 + r*p
            p = vmlaq_f32(c4, p, r);              // c4 + r*p
            p = vmlaq_f32(c3, p, r);              // c3 + r*p
            p = vmlaq_f32(c2, p, r);              // c2 + r*p
            p = vmlaq_f32(c1, p, r);              // c1 + r*p  ≈ exp(r)

            int32x4_t n_i = vmaxq_s32(vcvtq_s32_f32(n_f), vdupq_n_s32(-127));
            // n_i = vminq_s32(n_i, vdupq_n_s32(128));
            int32x4_t e_i = vaddq_s32(n_i, vdupq_n_s32(127));
            e_i = vshlq_n_s32(e_i, 23);

            float32x4_t s = vreinterpretq_f32_s32(e_i);

            return vmulq_f32(p, s);
        }

        __attribute__((always_inline)) inline static float32x4_t _avx2_exp_approx_f32(float32x4_t x) noexcept
        {
            const float32x4_t exp_hi = vdupq_n_f32(88.3762626647949f);
            const float32x4_t exp_lo = vdupq_n_f32(-88.3762626647949f);

            x = vminq_f32(x, exp_hi);
            x = vmaxq_f32(x, exp_lo);

            const float32x4_t LOG2EF = vdupq_n_f32(1.44269504088896341f);

            float32x4_t fx = vrndaq_f32(vmulq_f32(x, LOG2EF));

            float32x4_t C1 = vdupq_n_f32(0.693359375);
            float32x4_t C2 = vdupq_n_f32(-2.12194440e-4);
            x = vsubq_f32(x, vmulq_f32(fx, C1));
            x = vsubq_f32(x, vmulq_f32(fx, C2));

            float32x4_t f5 = vdupq_n_f32(1.9875691500E-4);
            float32x4_t f4 = vdupq_n_f32(1.3981999507E-3);
            float32x4_t f3 = vdupq_n_f32(8.3334519073E-3);
            float32x4_t f2 = vdupq_n_f32(4.1665795894E-2);
            float32x4_t f1 = vdupq_n_f32(1.6666665459E-1);
            float32x4_t f0 = vdupq_n_f32(5.0000001201E-1);
            // Horner's method
            float32x4_t y = f5;
            y = vmlaq_f32(f4, y, x);
            y = vmlaq_f32(f3, y, x);
            y = vmlaq_f32(f2, y, x);
            y = vmlaq_f32(f1, y, x);
            y = vmlaq_f32(f0, y, x);
            y = vmlaq_f32(vaddq_f32(x, vdupq_n_f32(1.0f)), y, vmulq_f32(x, x));

            int32x4_t e_i = vaddq_s32(vcvtq_s32_f32(fx), vdupq_n_s32(127));
            e_i = vshlq_n_s32(e_i, 23);

            float32x4_t s = vreinterpretq_f32_s32(e_i);

            return vmulq_f32(y, s);
        }

        __attribute__((always_inline)) inline static void _neon_softmax_fp32_inplace(float *__restrict x, int N) noexcept
        {
            if (N <= 0)
                return;

            const float m = hmax_f32(x, N);

            float32x4_t maxv = vdupq_n_f32(m);

            int i = 0;
            for (; i + 4 <= N; i += 4)
            {
                float32x4_t v = vld1q_f32(x + i);
                v = vsubq_f32(v, maxv);
                vst1q_f32(x + i, v);
            }
            for (; i < N; ++i)
                x[i] = (x[i] - m);


            float32x4_t acc = vdupq_n_f32(0.0f);
            i = 0;
            for (; i + 4 <= N; i += 4)
            {
                float32x4_t eh = _avx2_exp_approx_f32(vld1q_f32(x + i));
                vst1q_f32(x + i, eh);
                acc = vaddq_f32(acc, eh);
            }
            float sum = hsum_f32x4(acc);
            for (; i < N; ++i)
            {
                float ef = expf(x[i]);
                x[i] = ef;
                sum += ef;
            }

            // 4) x <- x / sum
            const float inv_sum_h = (1.0f / sum);
            const float32x4_t v_inv_sum_h = vdupq_n_f32(inv_sum_h);
            i = 0;
            for (; i + 4 <= N; i += 4)
            {
                float32x4_t v = vld1q_f32(x + i);
                v = vmulq_f32(v, v_inv_sum_h);
                vst1q_f32(x + i, v);
            }
            for (; i < N; ++i)
                x[i] = (x[i] * inv_sum_h);
            return;
            // return ret;
        }

        __attribute__((always_inline)) inline static void _neon_accumulate_fp32_inplace(float *__restrict x, float32x4_t &y, float &r, int N) noexcept
        {
            while (N >= 16)
            {
                float32x4_t a0 = vld1q_f32(x + 0);
                float32x4_t a1 = vld1q_f32(x + 1 * 4);
                float32x4_t a2 = vld1q_f32(x + 2 * 4);
                float32x4_t a3 = vld1q_f32(x + 3 * 4);

                y = vaddq_f32(y, a0);
                y = vaddq_f32(y, a1);
                y = vaddq_f32(y, a2);
                y = vaddq_f32(y, a3);

                N -= 16;
                x += 16;
            }

            while (N >= 4)
            {
                y = vaddq_f32(y, vld1q_f32(x));
                N -= 4;
                x += 4;
            }

            while (N > 0)
            {
                r += *x;
                x++;
                N--;
            }

            return;
        }

        __attribute__((always_inline)) inline static int64_t execute_impl(const RetrieveTask &task, TaskResult &result)
        {
            int64_t ret = 0;
            using DataType = __fp16;

            const auto &q_shape = task.queryGroup.shape; // (x, H, d_orig)
            const auto &k_shape = task.blkRepre.shape;   // (n, M, h, d_pruned)

            if (q_shape.size() != 3)
                throw std::runtime_error("Query shape must be 3D (x, H, d).");
            const int64_t num_tokens = q_shape[0];
            const int64_t num_q_heads = q_shape[1];
            const int64_t d_orig = q_shape[2];

            if (k_shape.size() != 4)
                throw std::runtime_error("BlockRep shape must be 4D (n, M, h, d).");
            const int64_t num_blocks = k_shape[0];
            const int64_t M = k_shape[1];
            const int64_t num_kv_heads = k_shape[2];
            const int64_t d_pruned = k_shape[3];

            if (num_q_heads % num_kv_heads != 0)
                throw std::runtime_error("Num_q_heads must be a divisible by num_kv_heads.");
            const int64_t g = num_q_heads / num_kv_heads;

            const DataType *q_ptr_for_computation;
            std::vector<DataType> pruned_q_vec;

            const DataType *q_orig_ptr = static_cast<const DataType *>(task.queryGroup.data);

            if (task.dPrunedIndex.has_value())
            {
                const auto &pruned_spec = task.dPrunedIndex.value();
                if (pruned_spec.shape.size() != 2 || pruned_spec.shape[0] != num_kv_heads || pruned_spec.shape[1] != d_pruned)
                {
                    throw std::runtime_error("dPrunedIndex shape is inconsistent with K's shape.");
                }
                const int64_t *pruned_indices_ptr = static_cast<const int64_t *>(pruned_spec.data);

                pruned_q_vec.resize(num_tokens * num_q_heads * d_pruned);

                for (int64_t x = 0; x < num_tokens; ++x)
                {
                    for (int64_t h = 0; h < num_kv_heads; ++h)
                    {
                        const int64_t *current_pruned_indices = pruned_indices_ptr + h * d_pruned;
                        for (int64_t gg = 0; gg < g; ++gg)
                        {
                            int64_t H = h * g + gg;
                            for (int64_t d_p = 0; d_p < d_pruned; ++d_p)
                            {
                                int64_t d_o = current_pruned_indices[d_p];
                                pruned_q_vec[(x * num_q_heads + H) * d_pruned + d_p] = q_orig_ptr[(x * num_q_heads + H) * d_orig + d_o];
                            }
                        }
                    }
                }
                q_ptr_for_computation = pruned_q_vec.data();
            }
            else
            {
                if (d_orig != d_pruned)
                {
                    throw std::runtime_error("Dimension mismatch: No dPrunedIndex, but Q and K head dims differ.");
                }
                q_ptr_for_computation = q_orig_ptr;
            }

            const int64_t S = num_blocks * M;
            const DataType *k_ptr = static_cast<const DataType *>(task.blkRepre.data);

            std::vector<float> scires_xhgs(num_tokens * num_kv_heads * g * S, 0.0f);

            for (int64_t h = 0; h < num_kv_heads; ++h)
            {
                const DataType *X = k_ptr + h * d_pruned;
                for (int64_t x = 0; x < num_tokens; ++x)
                {
                    for (int64_t gg = 0; gg < g; ++gg)
                    {
                        int64_t H = h * g + gg;
                        const DataType *q_vec = q_ptr_for_computation + (x * num_q_heads + H) * d_pruned;
                        auto y = scires_xhgs.data() + (x * num_kv_heads * g + H) * S;
                        _neon_gemv_fp32_fp16(X, q_vec, y, d_pruned, S, num_kv_heads * d_pruned);
                    }
                }
            }

            for (int64_t i = 0; i < num_tokens * num_q_heads; ++i)
            {
                _neon_softmax_fp32_inplace(&scires_xhgs[i * S], S);
            }

            std::vector<std::pair<float, int64_t>> final_scores_n;
            final_scores_n.reserve(num_blocks);
            for (int64_t i = 0; i < num_blocks; ++i)
                final_scores_n.emplace_back(0.0f, i);

            for (int64_t n = 0; n < num_blocks; n++)
            {
                float32x4_t acc = vdupq_n_f32(0.0f);
                float r = 0.0f;
                for (int64_t xhgi = 0; xhgi < num_tokens * num_kv_heads * g; ++xhgi)
                {
                    _neon_accumulate_fp32_inplace(&scires_xhgs[xhgi * S + n * M], acc, r, M);
                }
                final_scores_n[n].first -= hsum_f32x4(acc) + r;
            }

            std::nth_element(final_scores_n.begin(), final_scores_n.begin() + task.topK - 1, final_scores_n.end());

            std::vector<int64_t> topk_indices(task.topK);
            for (int i = 0; i < task.topK; i++)
            {
                topk_indices[i] = final_scores_n[i].second;
            }

            {
                std::lock_guard<std::mutex> lock(result.mtx);
                result.topkIndices = std::move(topk_indices);
                result.status.store(TaskStatus::SUCCESS, std::memory_order_release);
            }
            return ret;
        }

    } // neon_impl
#elif defined(__AVX2__)
#include <immintrin.h>
    namespace avx2_impl
    {
        using DataType = uint16_t;

        __attribute__((always_inline)) inline static float _avx2_hsum_m256(const __m256 v) noexcept
        {
            __m128 vlow = _mm256_castps256_ps128(v);
            __m128 vhigh = _mm256_extractf128_ps(v, 1);
            vlow = _mm_add_ps(vlow, vhigh);
            __m128 shuf = _mm_movehdup_ps(vlow);
            __m128 sums = _mm_add_ps(vlow, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            return _mm_cvtss_f32(_mm_add_ss(sums, shuf));
        }

        __attribute__((always_inline)) inline float fp16_to_fp32(DataType x) noexcept
        {
            __m128i h = _mm_cvtsi32_si128((uint16_t)x);
            __m128 f = _mm_cvtph_ps(h);
            return _mm_cvtss_f32(f);
        }

        __attribute__((always_inline)) inline static void _avx2_matvec_fp32_fp16(const DataType *__restrict A,
                                                                                 const DataType *__restrict x,
                                                                                 float *__restrict y,
                                                                                 int K, int N, int stride) noexcept
        {
            int j = 0;
            for (; j + 32 <= K; j += 32)
            {
                const __m128i x0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x + j + 0 * 8));
                const __m128i x1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x + j + 1 * 8));
                const __m128i x2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x + j + 2 * 8));
                const __m128i x3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x + j + 3 * 8));

                for (int i = 0; i < N; i++)
                {
                    const __m128i Ai0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(A + (int64_t)i * stride + j + 0 * 8));
                    const __m128i Ai1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(A + (int64_t)i * stride + j + 1 * 8));
                    const __m128i Ai2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(A + (int64_t)i * stride + j + 2 * 8));
                    const __m128i Ai3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(A + (int64_t)i * stride + j + 3 * 8));

                    __m256 acc = _mm256_setzero_ps();
                    acc = _mm256_fmadd_ps(_mm256_cvtph_ps(x0), _mm256_cvtph_ps(Ai0), acc);
                    acc = _mm256_fmadd_ps(_mm256_cvtph_ps(x1), _mm256_cvtph_ps(Ai1), acc);
                    acc = _mm256_fmadd_ps(_mm256_cvtph_ps(x2), _mm256_cvtph_ps(Ai2), acc);
                    acc = _mm256_fmadd_ps(_mm256_cvtph_ps(x3), _mm256_cvtph_ps(Ai3), acc);

                    y[i] += _avx2_hsum_m256(acc);
                }
            }
            for (; j + 16 <= K; j += 16)
            {
                const __m128i x0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x + j + 0 * 8));
                const __m128i x1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x + j + 1 * 8));

                for (int i = 0; i < N; i++)
                {
                    const __m128i Ai0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(A + (int64_t)i * stride + j + 0 * 8));
                    const __m128i Ai1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(A + (int64_t)i * stride + j + 1 * 8));

                    __m256 acc = _mm256_setzero_ps();
                    acc = _mm256_fmadd_ps(_mm256_cvtph_ps(x0), _mm256_cvtph_ps(Ai0), acc);
                    acc = _mm256_fmadd_ps(_mm256_cvtph_ps(x1), _mm256_cvtph_ps(Ai1), acc);

                    y[i] += _avx2_hsum_m256(acc);
                }
            }

            for (; j + 8 <= K; j += 8)
            {
                const __m128i x0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x + j + 0 * 8));
                for (int i = 0; i < N; i++)
                {
                    const __m128i Ai = _mm_loadu_si128(reinterpret_cast<const __m128i *>(A + (int64_t)i * stride + j));
                    __m256 acc = _mm256_mul_ps(_mm256_cvtph_ps(x0), _mm256_cvtph_ps(Ai));
                    y[i] += _avx2_hsum_m256(acc);
                }
            }

            if (j < K)
            {
                for (int i = 0; i < N; i++)
                {
                    const DataType *Ai = A + (int64_t)i * stride;
                    for (int jj = j; jj < K; jj++)
                        y[i] += fp16_to_fp32(Ai[jj]) * fp16_to_fp32((x + j)[jj]);
                }
            }
            return;
        }

        __attribute__((always_inline)) inline static float _avx2_hmax_f32(const float *__restrict x, int n) noexcept
        {
            int i = 0;
            __m256 vmax8 = _mm256_set1_ps(-1.0f / 0.0f);
            for (; i + 8 <= n; i += 8)
            {
                vmax8 = _mm256_max_ps(vmax8, _mm256_loadu_ps(x + i));
            }
            __m256 max1 = _mm256_max_ps(vmax8, _mm256_permute2f128_ps(vmax8, vmax8, 0x01));
            __m256 max2 = _mm256_max_ps(max1, _mm256_shuffle_ps(max1, max1, _MM_SHUFFLE(2, 3, 0, 1)));
            __m256 max3 = _mm256_max_ps(max2, _mm256_shuffle_ps(max2, max2, _MM_SHUFFLE(1, 0, 3, 2)));
            float m = _mm256_cvtss_f32(max3);
            for (; i < n; ++i)
                if (x[i] > m)
                    m = x[i];
            return m;
        }

        __attribute__((always_inline)) inline static __m256 _avx2_exp_approx_f32(__m256 x) noexcept
        {
            const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
            const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);

            x = _mm256_min_ps(x, exp_hi);
            x = _mm256_max_ps(x, exp_lo);

            const __m256 LOG2EF = _mm256_set1_ps(1.44269504088896341f);
            __m256 fx = _mm256_fmadd_ps(x, LOG2EF, _mm256_set1_ps(0.5f));

            __m256 tmp = _mm256_floor_ps(fx);
            __m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
            mask = _mm256_and_ps(mask, _mm256_set1_ps(1.0f));
            fx = _mm256_sub_ps(tmp, mask);

            __m256 cephes_exp_C1 = _mm256_set1_ps(0.693359375);
            __m256 cephes_exp_C2 = _mm256_set1_ps(-2.12194440e-4);
            x = _mm256_sub_ps(x, _mm256_mul_ps(fx, cephes_exp_C1));
            x = _mm256_sub_ps(x, _mm256_mul_ps(fx, cephes_exp_C2));

            //  Chebyshev polynomials, y = f(x) = 1 + r + r^2 * (f0 + f1*x + f2*x^2 + f3*x^3 +f4*x^4 + f5x^5)
            //                                  = 1 + r + r^2 * (f0 + x * ( f1 + x * ( f2 + x ( f3 + x ( f4 + x f5 ))))) (Horner's method)
            __m256 f5 = _mm256_set1_ps(1.9875691500E-4);
            __m256 f4 = _mm256_set1_ps(1.3981999507E-3);
            __m256 f3 = _mm256_set1_ps(8.3334519073E-3);
            __m256 f2 = _mm256_set1_ps(4.1665795894E-2);
            __m256 f1 = _mm256_set1_ps(1.6666665459E-1);
            __m256 f0 = _mm256_set1_ps(5.0000001201E-1);
            // Horner's method
            __m256 y = f5;
            y = _mm256_fmadd_ps(y, x, f4);
            y = _mm256_fmadd_ps(y, x, f3);
            y = _mm256_fmadd_ps(y, x, f2);
            y = _mm256_fmadd_ps(y, x, f1);
            y = _mm256_fmadd_ps(y, x, f0);
            y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), _mm256_add_ps(x, _mm256_set1_ps(1.0f)));

            __m256i e_i = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(fx), _mm256_set1_epi32(127)), 23);
            return _mm256_mul_ps(y, _mm256_castsi256_ps(e_i));
        }

        __attribute__((always_inline)) inline static void _avx2_softmax_fp32_inplace(float *__restrict x, int N) noexcept
        {
            if (N <= 0)
                return;
            // int64_t ret = 0;

            const float m = _avx2_hmax_f32(x, N);

            __m256 maxv = _mm256_set1_ps(m);

            int i = 0;
            for (; i + 8 <= N; i += 8)
            {
                __m256 v = _mm256_loadu_ps(x + i);
                v = _mm256_sub_ps(v, maxv);
                _mm256_storeu_ps(x + i, v);
            }
            for (; i < N; ++i)
                x[i] = (x[i] - m);


            __m256 acc = _mm256_setzero_ps();
            i = 0;
            for (; i + 8 <= N; i += 8)
            {
                // __m256 eh = _avx2_exp_approx_f32(_mm256_loadu_ps(x + i));
                __m256 eh = _avx2_exp_approx_f32(_mm256_loadu_ps(x + i));

                _mm256_storeu_ps(x + i, eh);
                acc = _mm256_add_ps(acc, eh);
            }
            float sum = _avx2_hsum_m256(acc);
            for (; i < N; ++i)
            {
                float ef = expf(x[i]);
                x[i] = ef;
                sum += ef;
            }

            // 4) x <- x / sum
            const float inv_sum_h = (1.0f / sum);
            const __m256 v_inv_sum_h = _mm256_set1_ps(inv_sum_h);
            i = 0;
            for (; i + 8 <= N; i += 8)
            {
                __m256 v = _mm256_loadu_ps(x + i);
                v = _mm256_mul_ps(v, v_inv_sum_h);
                _mm256_storeu_ps(x + i, v);
            }
            for (; i < N; ++i)
                x[i] = (x[i] * inv_sum_h);
            return;
            // return ret;
        }

        __attribute__((always_inline)) inline static void _avx2_accumulate_fp32_inplace(float *__restrict x, __m256 &y, float &r, int N) noexcept
        {
            int i = 0;
            for (; i + 32 < N; i += 32)
            {
                __m256 a0 = _mm256_loadu_ps(x + i + 0 * 8);
                __m256 a1 = _mm256_loadu_ps(x + i + 1 * 8);
                __m256 a2 = _mm256_loadu_ps(x + i + 2 * 8);
                __m256 a3 = _mm256_loadu_ps(x + i + 3 * 8);
                y = _mm256_add_ps(y, a0);
                y = _mm256_add_ps(y, a1);
                y = _mm256_add_ps(y, a2);
                y = _mm256_add_ps(y, a3);
            }
            for (; i + 16 < N; i += 16)
            {
                __m256 a0 = _mm256_loadu_ps(x + i + 0 * 8);
                __m256 a1 = _mm256_loadu_ps(x + i + 1 * 8);
                y = _mm256_add_ps(y, a0);
                y = _mm256_add_ps(y, a1);
            }
            for (; i + 8 < N; i += 8)
            {
                y = _mm256_add_ps(y, _mm256_loadu_ps(x + i));
            }
            for (; i < N; i++)
            {
                r += *(x + i);
            }
            return;
        }

        __attribute__((always_inline)) inline static void execute_impl(const RetrieveTask &task, TaskResult &result)
        {
            const auto &q_shape = task.queryGroup.shape; // (x, H, d_orig)
            const auto &k_shape = task.blkRepre.shape;   // (n, M, h, d_pruned)

            if (q_shape.size() != 3)
                throw std::runtime_error("Query shape must be 3D (x, H, d).");
            const int64_t num_tokens = q_shape[0];
            const int64_t num_q_heads = q_shape[1];
            const int64_t d_orig = q_shape[2];

            if (k_shape.size() != 4)
                throw std::runtime_error("BlockRep shape must be 4D (n, M, h, d).");
            const int64_t num_blocks = k_shape[0];
            const int64_t M = k_shape[1];
            const int64_t num_kv_heads = k_shape[2];
            const int64_t d_pruned = k_shape[3];

            if (num_q_heads % num_kv_heads != 0)
                throw std::runtime_error("Num_q_heads must be a divisible by num_kv_heads.");
            const int64_t g = num_q_heads / num_kv_heads;


            const DataType *q_ptr_for_computation;
            std::vector<DataType> pruned_q_vec;

            const DataType *q_orig_ptr = static_cast<const DataType *>(task.queryGroup.data);

            if (task.dPrunedIndex.has_value())
            {
                const auto &pruned_spec = task.dPrunedIndex.value();
                if (pruned_spec.shape.size() != 2 || pruned_spec.shape[0] != num_kv_heads || pruned_spec.shape[1] != d_pruned)
                {
                    throw std::runtime_error("dPrunedIndex shape is inconsistent with K's shape.");
                }
                const int64_t *pruned_indices_ptr = static_cast<const int64_t *>(pruned_spec.data);

                pruned_q_vec.resize(num_tokens * num_q_heads * d_pruned);

                for (int64_t x = 0; x < num_tokens; ++x)
                {
                    for (int64_t h = 0; h < num_kv_heads; ++h)
                    {
                        const int64_t *current_pruned_indices = pruned_indices_ptr + h * d_pruned;
                        for (int64_t gg = 0; gg < g; ++gg)
                        {
                            int64_t H = h * g + gg;
                            for (int64_t d_p = 0; d_p < d_pruned; ++d_p)
                            {
                                int64_t d_o = current_pruned_indices[d_p];
                                pruned_q_vec[(x * num_q_heads + H) * d_pruned + d_p] = q_orig_ptr[(x * num_q_heads + H) * d_orig + d_o];
                            }
                        }
                    }
                }
                q_ptr_for_computation = pruned_q_vec.data();
            }
            else
            {
                if (d_orig != d_pruned)
                {
                    throw std::runtime_error("Dimension mismatch: No dPrunedIndex, but Q and K head dims differ.");
                }
                q_ptr_for_computation = q_orig_ptr;
            }

            const int64_t S = num_blocks * M;
            const DataType *k_ptr = static_cast<const DataType *>(task.blkRepre.data);

            std::vector<float> scires_xhgs(num_tokens * num_kv_heads * g * S, 0.0f);
            // std::cout << S << "," << g << std::endl;
            // auto a = std::chrono::high_resolution_clock::now();

            for (int64_t h = 0; h < num_kv_heads; ++h)
            {
                const DataType *X = k_ptr + h * d_pruned;
                for (int64_t x = 0; x < num_tokens; ++x)
                {
                    // std::cout << h * g << ", " << x * num_q_heads << ", " << num_tokens << std::endl;
                    int64_t q_pos = h * g + x * num_q_heads;
                    for (int64_t gg = 0; gg < g; ++gg)
                    {
                        const DataType *q_vec = q_ptr_for_computation + (q_pos + gg) * d_pruned;
                        auto y = scires_xhgs.data() + (q_pos + gg) * S;
                        _avx2_matvec_fp32_fp16(X, q_vec, y, d_pruned, S, num_kv_heads * d_pruned);
                    }
                }
            }

            // auto b = std::chrono::high_resolution_clock::now();
            // float duration = std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
            // std::cout << duration / 1000.0f << std::endl;

            for (int64_t i = 0; i < num_tokens * num_q_heads; ++i)
            {
                _avx2_softmax_fp32_inplace(&scires_xhgs[i * S], S);
            }

            std::vector<std::pair<float, int64_t>> final_scores_n;
            final_scores_n.reserve(num_blocks);
            for (int64_t i = 0; i < num_blocks; ++i)
                final_scores_n.emplace_back(0.0f, i);

            for (int64_t n = 0; n < num_blocks; n++)
            {
                __m256 acc = _mm256_setzero_ps();
                float r = 0.0f;
                for (int64_t xhgi = 0; xhgi < num_tokens * num_kv_heads * g; ++xhgi)
                {
                    _avx2_accumulate_fp32_inplace(&scires_xhgs[xhgi * S + n * M], acc, r, M);
                }
                final_scores_n[n].first -= _avx2_hsum_m256(acc) + r;
            }

            // 6. TopK on 'n'
            std::nth_element(final_scores_n.begin(), final_scores_n.begin() + task.topK - 1, final_scores_n.end());

            std::vector<int64_t> topk_indices(task.topK);
            for (int i = 0; i < task.topK; i++)
            {
                topk_indices[i] = final_scores_n[i].second;
            }

            {
                std::lock_guard<std::mutex> lock(result.mtx);
                result.topkIndices = std::move(topk_indices);
                result.status.store(TaskStatus::SUCCESS, std::memory_order_release);
            }
            return;
        }
    } // avx2_impl

#else
    namespace scalar_impl
    {
#if defined(__aarch64__) || defined(__arm__)
        using DataType = __fp16;
#else
        using DataType = uint16_t;
        __attribute__((always_inline)) static float fp16_to_float(DataType h) noexcept
        {
            uint32_t s = (h & 0x8000u) << 16; // sign -> bit31
            uint32_t e = (h & 0x7C00u) >> 10; // 5-bit exponent
            uint32_t m = (h & 0x03FFu);       // 10-bit mantissa

            uint32_t bits;
            if (e == 0)
            {
                if (m == 0)
                {
                    bits = s; // exponent=0, mantissa=0
                }
                else
                {
                    int shift = 0;
                    while ((m & 0x400u) == 0u)
                    {
                        m <<= 1;
                        ++shift;
                    }
                    m &= 0x3FFu;
                    int32_t ef = 113 - shift;
                    bits = s | (static_cast<uint32_t>(ef) << 23) | (m << 13);
                }
            }
            else if (e == 0x1Fu)
            {
                bits = s | 0x7F800000u | (m ? (m << 13) : 0u);
            }
            else
            {
                uint32_t ef = e + 112u;
                bits = s | (ef << 23) | (m << 13);
            }

            float f;
            std::memcpy(&f, &bits, sizeof(f));
            return f;
        }

#endif

        __attribute__((always_inline)) inline static void execute_impl(const RetrieveTask &task, TaskResult &result)
        {

            const auto &q_shape = task.queryGroup.shape; // (x, H, d_orig)
            const auto &k_shape = task.blkRepre.shape;   // (n, M, h, d_pruned)

            if (q_shape.size() != 3)
                throw std::runtime_error("Query shape must be 3D (x, H, d).");
            const int64_t num_tokens = q_shape[0];
            const int64_t num_q_heads = q_shape[1];
            const int64_t d_orig = q_shape[2];

            if (k_shape.size() != 4)
                throw std::runtime_error("BlockRep shape must be 4D (n, M, h, d).");
            const int64_t num_blocks = k_shape[0];
            const int64_t M = k_shape[1];
            const int64_t num_kv_heads = k_shape[2];
            const int64_t d_pruned = k_shape[3];

            if (num_q_heads % num_kv_heads != 0)
                throw std::runtime_error("Num_q_heads must be a divisible by num_kv_heads.");
            const int64_t g = num_q_heads / num_kv_heads;

            const DataType *q_ptr_for_computation;
            std::vector<DataType> pruned_q_vec;

            const DataType *q_orig_ptr = static_cast<const DataType *>(task.queryGroup.data);

            if (task.dPrunedIndex.has_value())
            {
                const auto &pruned_spec = task.dPrunedIndex.value();
                if (pruned_spec.shape.size() != 2 || pruned_spec.shape[0] != num_kv_heads || pruned_spec.shape[1] != d_pruned)
                {
                    throw std::runtime_error("dPrunedIndex shape is inconsistent with K's shape.");
                }
                const int64_t *pruned_indices_ptr = static_cast<const int64_t *>(pruned_spec.data);

                pruned_q_vec.resize(num_tokens * num_q_heads * d_pruned);

                for (int64_t x = 0; x < num_tokens; ++x)
                {
                    for (int64_t h = 0; h < num_kv_heads; ++h)
                    {
                        const int64_t *current_pruned_indices = pruned_indices_ptr + h * d_pruned;
                        for (int64_t gg = 0; gg < g; ++gg)
                        {
                            int64_t H = h * g + gg;
                            for (int64_t d_p = 0; d_p < d_pruned; ++d_p)
                            {
                                int64_t d_o = current_pruned_indices[d_p];
                                pruned_q_vec[(x * num_q_heads + H) * d_pruned + d_p] = q_orig_ptr[(x * num_q_heads + H) * d_orig + d_o];
                            }
                        }
                    }
                }

                q_ptr_for_computation = pruned_q_vec.data();
            }
            else
            {
                if (d_orig != d_pruned)
                {
                    throw std::runtime_error("Dimension mismatch: No dPrunedIndex, but Q and K head dims differ.");
                }
                q_ptr_for_computation = q_orig_ptr;
            }

            const int64_t S = num_blocks * M;

            const DataType *k_ptr = static_cast<const DataType *>(task.blkRepre.data);
            std::vector<float> scires_xhgs(num_tokens * num_kv_heads * g * S);
            for (int64_t x = 0; x < num_tokens; ++x)
            {
                for (int64_t h = 0; h < num_kv_heads; ++h)
                {
                    for (int64_t gg = 0; gg < g; ++gg)
                    {
                        int64_t H = h * g + gg; // q_token's head index
                        const DataType *q_vec = q_ptr_for_computation + (x * num_q_heads + H) * d_pruned;

                        for (int64_t s = 0; s < S; ++s)
                        {
                            const DataType *k_vec = k_ptr + (s * num_kv_heads + h) * d_pruned;
                            float score = 0.0f;
                            for (int64_t d = 0; d < d_pruned; ++d)
                            {
#if defined(__aarch64__) || defined(__arm__)
                                score += static_cast<float>(q_vec[d]) * static_cast<float>(k_vec[d]);
#else
                                score += fp16_to_float(q_vec[d]) * fp16_to_float(k_vec[d]);
#endif
                            }
                            scires_xhgs[(((x * num_kv_heads + h) * g + gg) * S + s)] = score;
                        }
                    }
                }
            }

            for (int64_t i = 0; i < num_tokens * num_q_heads; ++i)
            {
                float *current_scores = &scires_xhgs[i * S];

                // Softmax on S-dimension vector
                float max_val = current_scores[0];
                for (int64_t s = 1; s < S; ++s)
                {
                    if (current_scores[s] > max_val)
                    {
                        max_val = current_scores[s];
                    }
                }

                float sum_exp = 0.0f;
                for (int64_t s = 0; s < S; ++s)
                {
                    current_scores[s] = expf(current_scores[s] - max_val);
                    sum_exp += current_scores[s];
                }

                // Handle sum_exp being zero to avoid division by zero
                if (sum_exp > 1e-9)
                {
                    for (int64_t s = 0; s < S; ++s)
                    {
                        current_scores[s] /= sum_exp;
                    }
                }
            }

            std::vector<float> final_scores_n(num_blocks, 0.0f);
            for (int64_t xhgi = 0; xhgi < num_tokens * num_kv_heads * g; ++xhgi)
            {
                for (int64_t s = 0; s < S; ++s)
                {
                    int64_t n = s / M;
                    final_scores_n[n] += scires_xhgs[xhgi * S + s];
                }
            }

            // 6. TopK on 'n'
            using ScoreIndexPair = std::pair<float, int64_t>;
            std::priority_queue<ScoreIndexPair, std::vector<ScoreIndexPair>, std::greater<ScoreIndexPair>> top_k_heap;

            for (int64_t n = 0; n < num_blocks; ++n)
            {
                if (top_k_heap.size() < task.topK)
                {
                    top_k_heap.push({final_scores_n[n], n});
                }
                else if (final_scores_n[n] > top_k_heap.top().first)
                {
                    top_k_heap.pop();
                    top_k_heap.push({final_scores_n[n], n});
                }
            }

            std::vector<int64_t> topk_indices(top_k_heap.size());
            int index_pos = top_k_heap.size() - 1;
            while (!top_k_heap.empty())
            {
                topk_indices[index_pos--] = top_k_heap.top().second;
                top_k_heap.pop();
            }

            {
                std::lock_guard<std::mutex> lock(result.mtx);
                result.topkIndices = std::move(topk_indices);
                result.status.store(TaskStatus::SUCCESS, std::memory_order_release);
            }
        }

    } // scalar_impl

#endif

    void Execute(const RetrieveTask &task, TaskResult &result)
    {
#if defined(__ARM_NEON)
        neon_impl::execute_impl(task, result);
#elif defined(__AVX2__)
        avx2_impl::execute_impl(task, result);
#else
        scalar_impl::execute_impl(task, result);
#endif
    }

}
