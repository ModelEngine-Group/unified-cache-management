// retrieval_backend.cpp

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
#include <random>

namespace py = pybind11;

class RetrievalWorkerBackend {
public:
    RetrievalWorkerBackend(py::array_t<float> data)
        : data_array_(data), stop_workers_(false), next_req_id_(0)
    {
        py::buffer_info info = data_array_.request();
        n_items_ = info.shape[0];
        dim_ = info.shape[1];
        data_ = static_cast<const float*>(info.ptr);

        // Start worker threads
        int n_workers = std::thread::hardware_concurrency();
        for (int i = 0; i < n_workers; ++i) {
            worker_threads_.emplace_back(&RetrievalWorkerBackend::worker_loop, this);
        }
    }

    ~RetrievalWorkerBackend() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_workers_ = true;
            cond_.notify_all();
        }
        for (auto& t: worker_threads_) t.join();
    }

    int submit(py::array_t<float> query, int topk, py::array_t<int> indexes) {
        py::buffer_info qinfo = query.request();
        py::buffer_info iinfo = indexes.request();
        if (qinfo.shape[1] != dim_)
            throw std::runtime_error("Query dim mismatch");
        if ((size_t)iinfo.shape[0] != (size_t)qinfo.shape[0])
            throw std::runtime_error("Query and indexes batch mismatch");

        int req_id = next_req_id_.fetch_add(1);

        auto q = std::vector<float>((float*)qinfo.ptr, (float*)qinfo.ptr + qinfo.shape[0] * dim_);

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
        py::array_t<float> scores({batch_size, topk});

        auto indices_ptr = static_cast<int*>(indices.request().ptr);
        auto scores_ptr = static_cast<float*>(scores.request().ptr);

        for (size_t i = 0; i < batch_size; ++i) {
            memcpy(indices_ptr + i * topk, it->second.indices[i].data(), topk * sizeof(int));
            memcpy(scores_ptr + i * topk, it->second.scores[i].data(), topk * sizeof(float));
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
        std::vector<float> query; // Flattened [batch, dim]
        size_t batch;
        int topk;
        std::vector<std::vector<int>> indexes; // Per-request index subset
    };
    struct Result {
        std::vector<std::vector<int>> indices;
        std::vector<std::vector<float>> scores;
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

            // for performance
            // std::mt19937 gen(42);
            // for (size_t b = 0; b < req.batch; ++b) {
            //     const float* q_ptr = req.query.data() + b * dim_;
            //     const auto& allowed = req.indexes[b];

            //     std::vector<int> index;
            //     int i = 0;
            //     for (auto &c: allowed) {
            //         index.push_back(i++);
            //     }
            //     std::shuffle(index.begin(), index.end(), gen);
            //     int curr_topk = std::min(static_cast<int>(allowed.size()), req.topk);
            //     for (int k = 0; k < curr_topk; ++k) {
            //         res.indices[b].push_back(allowed[index[k]]);
            //         res.scores[b].push_back(0.0f); // Dummy/fixed score
            //     }
            // }

            // for precision
            for (size_t b = 0; b < req.batch; ++b) {
                const float* q_ptr = req.query.data() + b * dim_;
                const auto& allowed = req.indexes[b];
                std::vector<std::pair<float, int>> heap;
                heap.reserve(allowed.size());
                for (auto idx : allowed) {
                    float score = 0.0f;
                    for (ssize_t d = 0; d < dim_; ++d) {
                        score += q_ptr[d] * data_[idx * dim_ + d];
                    }
                    heap.emplace_back(score, idx);
                }
                int curr_topk = std::min((int)heap.size(), req.topk);
                std::partial_sort(heap.begin(), heap.begin() + curr_topk, heap.end(),
                    [](const auto& a, const auto& b){ return a.first > b.first; });

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

    py::array_t<float> data_array_;
    const float* data_ = nullptr;
    ssize_t n_items_, dim_;
    std::queue<Request> requests_;
    std::unordered_map<int, Result> results_;
    std::vector<std::thread> worker_threads_;
    std::mutex mutex_;
    std::condition_variable cond_;
    std::unordered_map<int, std::shared_ptr<RequestStatus>> request_status_;
    bool stop_workers_;
    std::atomic<int> next_req_id_;
};

PYBIND11_MODULE(retrieval_backend, m) {
    py::class_<RetrievalWorkerBackend>(m, "RetrievalWorkerBackend")
        .def(py::init<py::array_t<float>>())
        .def("submit", &RetrievalWorkerBackend::submit)
        .def("poll", &RetrievalWorkerBackend::poll)
        .def("get_result", &RetrievalWorkerBackend::get_result)
        .def("wait", &RetrievalWorkerBackend::wait);
}
