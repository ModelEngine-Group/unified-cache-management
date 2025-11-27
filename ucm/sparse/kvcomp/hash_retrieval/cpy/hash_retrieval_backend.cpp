// hash_retrieval_backend.cpp

#include <cstddef>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <iostream>
#include <random>
#include <climits>  // 用于UINT16_MAX
#include <omp.h>
#ifdef NUMA_ENABLED
#include <numaif.h>
#include <numa.h>
#endif
#include <sched.h>
#include <unistd.h>
#include <cstring>
#include <errno.h>


#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    #include <immintrin.h>   // SSE/AVX
    #include <nmmintrin.h>   // POPCNT (SSE4.2)
#endif

#define VEC_SIZE 16

#if defined(__ARM_NEON) || defined(__ARM_NEON__)

using vec16u = uint8x16_t;

static inline vec16u vec_loadu16(const uint8_t* p) {
    return vld1q_u8(p);
}

static inline vec16u vec_xor(vec16u a, vec16u b) {
    return veorq_u8(a, b);
}

static inline uint16_t vec_sum_u8(vec16u v) {
#if defined(__aarch64__) || defined(_M_ARM64)
    return vaddvq_u8(v);
#else
    uint16x8_t s16 = vpaddlq_u8(v);
    uint32x4_t s32 = vpaddlq_u16(s16);
    uint64x2_t s64 = vpaddlq_u32(s32);
    return (uint16_t)(vgetq_lane_u64(s64, 0) + vgetq_lane_u64(s64, 1));
#endif
}

static inline uint16_t vec_popcnt_xor_sum16(const uint8_t* a, const uint8_t* b) {
    vec16u va = vec_loadu16(a);
    vec16u vb = vec_loadu16(b);
    vec16u vx = vec_xor(va, vb);
    vec16u pc = vcntq_u8(vx);
    return vec_sum_u8(pc);
}

static inline uint16_t vec_popcnt_xor_sum16_vec(vec16u qa, const uint8_t* b) {
    vec16u vb = vec_loadu16(b);
    vec16u vx = vec_xor(qa, vb);
    vec16u pc = vcntq_u8(vx);
    return vec_sum_u8(pc);
}

void print_uint8x16(uint8x16_t vec) {
    uint8_t array[16];
    vst1q_u8(array, vec);
    for (int i = 0; i < 16; ++i) {
        std::cout << static_cast<int>(array[i]) << " ";
    }
    std::cout << std::endl;
}

#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)

using vec16u = __m128i;

static inline vec16u vec_loadu16(const uint8_t* p) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
}

static inline vec16u vec_xor(vec16u a, vec16u b) {
    return _mm_xor_si128(a, b);
}

static inline uint16_t vec_popcnt_xor_sum16(const uint8_t* a, const uint8_t* b) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
    __m128i vx = _mm_xor_si128(va, vb);

    uint64_t lo, hi;
#if defined(__SSE4_1__)
    lo = static_cast<uint64_t>(_mm_extract_epi64(vx, 0));
    hi = static_cast<uint64_t>(_mm_extract_epi64(vx, 1));
#else
    alignas(16) uint64_t tmp[2];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), vx);
    lo = tmp[0];
    hi = tmp[1];
#endif
    return (uint16_t)(__builtin_popcountll(lo) + __builtin_popcountll(hi));
}

static inline uint16_t vec_popcnt_xor_sum16_vec(vec16u qa, const uint8_t* b) {
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
    __m128i vx = _mm_xor_si128(qa, vb);

    uint64_t lo, hi;
#if defined(__SSE4_1__)
    lo = static_cast<uint64_t>(_mm_extract_epi64(vx, 0));
    hi = static_cast<uint64_t>(_mm_extract_epi64(vx, 1));
#else
    alignas(16) uint64_t tmp[2];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), vx);
    lo = tmp[0];
    hi = tmp[1];
#endif
    return (uint16_t)(__builtin_popcountll(lo) + __builtin_popcountll(hi));
}

#else   

static inline uint16_t vec_popcnt_xor_sum16(const uint8_t* a, const uint8_t* b) {
    uint16_t s = 0;
    for (int i = 0; i < 16; ++i)
        s += __builtin_popcount((unsigned)(a[i] ^ b[i]));
    return s;
}

#endif

namespace py = pybind11;

class HashRetrievalWorkerBackend {
public:
    HashRetrievalWorkerBackend(py::array_t<uint8_t> data, 
                               py::dict cpu_idx_tbl)
        : data_array_(data), stop_workers_(false), next_req_id_(0)
    {
        py::buffer_info info = data_array_.request();
        num_blocks_  = info.shape[0];
        block_size_  = info.shape[1];
        dim_         = info.shape[2];
        vec_per_dim_ = dim_ / VEC_SIZE;  // data_每个值类型uint8_t,组成8*16_t进行simd加速
        tail_dim_    = dim_ % VEC_SIZE;      
        tail_start_  = vec_per_dim_ * VEC_SIZE;
        data_        = static_cast<const uint8_t*>(info.ptr);

        // Start worker threads
        for (auto cpu_idx : cpu_idx_tbl) {
            py::list core_ids = cpu_idx.second.cast<py::list>();

            for (size_t i = 0; i < core_ids.size(); ++i) {
                int core_id = core_ids[i].cast<int>();
                worker_threads_.emplace_back(&HashRetrievalWorkerBackend::worker_loop, this);

                // 核心绑定代码
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(core_id, &cpuset);  // 绑定每个线程到指定的核心

                pthread_t thread = worker_threads_.back().native_handle();
                
                // 设置 CPU 亲和性
                int rc = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
                if (rc != 0) {
                    std::cerr << "Error binding thread " << i << " to CPU core " << core_id << std::endl;
                }

            #ifdef NUMA_ENABLED
                int numaId = cpu_idx.first.cast<int>();
                // 设置内存亲和性
                unsigned long nodeMask = 1UL << numaId;
                rc = set_mempolicy(MPOL_BIND, &nodeMask, sizeof(nodeMask) * 8);
                if (rc != 0) {
                    std::cerr << "Error binding memory to NUMA node " << numaId << std::endl;
                }
            #endif

            }

        }
    }

    ~HashRetrievalWorkerBackend() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_workers_ = true;
            cond_.notify_all();
        }
        for (auto& t: worker_threads_) t.join();
    }

    int submit(py::array_t<uint8_t> query, int topk, py::array_t<int> indexes) {
        py::buffer_info qinfo = query.request();
        py::buffer_info iinfo = indexes.request();
        if (qinfo.shape[1] != dim_)
            throw std::runtime_error("Query dim mismatch");
        if ((size_t)iinfo.shape[0] != (size_t)qinfo.shape[0])
            throw std::runtime_error("Query and indexes batch mismatch");

        int req_id = next_req_id_.fetch_add(1);

        auto q = std::vector<uint8_t>((uint8_t*)qinfo.ptr, (uint8_t*)qinfo.ptr + qinfo.shape[0] * dim_);

        // Parse indexes to vector<vector<int>>
        size_t n_requests = iinfo.shape[0], max_index_number = iinfo.shape[1];
        const int* idx_ptr = static_cast<const int*>(iinfo.ptr);
        std::vector<std::vector<int>> idxvec(n_requests);
        for (size_t i = 0; i < n_requests; ++i) {
            for (size_t j = 0; j < max_index_number; ++j) {
                int index = idx_ptr[i * max_index_number + j];
                if (index != -1) idxvec[i].push_back(index);
            }
        }

        auto status = std::make_shared<RequestStatus>();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            requests_.emplace(Request{req_id, std::move(q), n_requests, topk, std::move(idxvec)});
            request_status_[req_id] = status;
        }
        cond_.notify_one();
        return req_id;
    }

    bool poll(int req_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        return results_.find(req_id) != results_.end();
    }

    void wait(int req_id) {
        std::shared_ptr<RequestStatus> s;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = request_status_.find(req_id);
            if (it == request_status_.end()) throw std::runtime_error("Bad req_id");
            s = it->second;
        }
        std::unique_lock<std::mutex> lk2(s->m);
        s->cv.wait(lk2, [&] { return s->done; });
    }

    py::dict get_result(int req_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = results_.find(req_id);
        if (it == results_.end()) throw std::runtime_error("Result not ready");

        size_t batch_size = it->second.indices.size();
        size_t topk = batch_size > 0 ? it->second.indices[0].size() : 0;
        py::array_t<int> indices({batch_size, topk});
        py::array_t<int> scores({batch_size, topk});

        auto indices_ptr = static_cast<int*>(indices.request().ptr);
        auto scores_ptr = static_cast<int*>(scores.request().ptr);

        for (size_t i = 0; i < batch_size; ++i) {
            memcpy(indices_ptr + i * topk, it->second.indices[i].data(), topk * sizeof(int));
            memcpy(scores_ptr + i * topk, it->second.scores[i].data(), topk * sizeof(int));
        }
        py::dict result;
        result["indices"] = indices;
        result["scores"] = scores;
        results_.erase(it);
        return result;
    }

private:
    struct Request {
        int req_id;
        std::vector<uint8_t> query; // Flattened [batch, dim]
        size_t batch;
        int topk;
        std::vector<std::vector<int>> indexes; // Per-request index subset
    };
    struct Result {
        std::vector<std::vector<int>> indices;
        std::vector<std::vector<int>> scores;
    };

    struct RequestStatus {
        std::mutex m;
        std::condition_variable cv;
        bool done = false;
    };

    void worker_loop() {
        while (true) {
            Request req;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cond_.wait(lock, [&]{ return stop_workers_ || !requests_.empty(); });
                if (stop_workers_ && requests_.empty()) return;
                req = std::move(requests_.front());
                requests_.pop();
            }

            Result res;
            res.indices.resize(req.batch);
            res.scores.resize(req.batch);

            // #pragma omp parallel for schedule(dynamic)
            for (size_t b = 0; b < req.batch; ++b) {
                const uint8_t* q_ptr = req.query.data() + b * dim_;
                const auto& allowed = req.indexes[b];
                std::vector<std::pair<int, int>> heap;
                heap.reserve(allowed.size());

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || \
    defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
                // 1.预加载 query 向量
                vec16u q_vecs[vec_per_dim_]; // 存储query向量
                for (size_t v = 0; v < vec_per_dim_; ++v) {
                    q_vecs[v] = vec_loadu16(q_ptr + v * VEC_SIZE);
                }
#endif

                // 2.遍历允许的索引
                for (auto idx : allowed) {
                    const uint8_t* base_idx_ptr = data_ + idx * block_size_ * dim_;

                    int score = UINT16_MAX;    // 初始化为最大值

                    // 3.内层向量化计算
                    // #pragma omp parallel for
                    for (size_t t_idx = 0; t_idx < block_size_; ++t_idx) {
                        int sum = 0;
                        const uint8_t* k_base = base_idx_ptr + t_idx * dim_;

                        // 计算每个向量的相似度
#if defined(__ARM_NEON) || defined(__ARM_NEON__) || \
    defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
                        for (size_t v = 0; v < vec_per_dim_; ++v) {
                            sum += vec_popcnt_xor_sum16_vec(
                                q_vecs[v],
                                k_base + v * VEC_SIZE
                            );
                        }
#else
                        for (size_t v = 0; v < vec_per_dim_; ++v) {
                            sum += vec_popcnt_xor_sum16(
                                q_ptr  + v * VEC_SIZE,
                                k_base + v * VEC_SIZE
                            );
                        }
#endif
                        if (tail_dim_ != 0) {
                            for (size_t t = 0; t < tail_dim_; ++t) {
                                uint8_t x = q_ptr [tail_start_+t] ^ k_base[tail_start_+t];
                                sum += __builtin_popcount((unsigned)x);
                            }
                        }

                        // 如果得分为0，则跳出循环
                        if (sum < score) {
                            score = sum;
                            if (score == 0) {
                                break;
                            }
                        }
                    }

                    // 将结果加入堆中
                    heap.emplace_back(score, idx);
                }

                // 获取当前TopK
                int curr_topk = std::min((int)heap.size(), req.topk);

                // 对堆进行部分排序，获取TopK
                std::partial_sort(heap.begin(), heap.begin() + curr_topk, heap.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });

                // 保存TopK结果
                for (int k = 0; k < curr_topk; ++k) {
                    res.scores[b].push_back(heap[k].first);
                    res.indices[b].push_back(heap[k].second);
                }
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                results_[req.req_id] = std::move(res);
                auto s = request_status_[req.req_id];
                {
                    std::lock_guard<std::mutex> lk2(s->m);
                    s->done = true;
                }
                s->cv.notify_all();
            }
        }
    }

    py::array_t<uint8_t> data_array_;
    const uint8_t* data_ = nullptr;
    ssize_t dim_;
    size_t num_blocks_, block_size_, vec_per_dim_, tail_dim_, tail_start_;
    std::queue<Request> requests_;
    std::unordered_map<int, Result> results_;
    std::vector<std::thread> worker_threads_;
    std::mutex mutex_;
    std::condition_variable cond_;
    std::unordered_map<int, std::shared_ptr<RequestStatus>> request_status_;
    bool stop_workers_;
    std::atomic<int> next_req_id_;
};

PYBIND11_MODULE(hash_retrieval_backend, m) {
    py::class_<HashRetrievalWorkerBackend>(m, "HashRetrievalWorkerBackend")
        .def(py::init<py::array_t<uint8_t>, py::dict>())
        .def("submit", &HashRetrievalWorkerBackend::submit)
        .def("poll", &HashRetrievalWorkerBackend::poll)
        .def("get_result", &HashRetrievalWorkerBackend::get_result)
        .def("wait", &HashRetrievalWorkerBackend::wait);
}
