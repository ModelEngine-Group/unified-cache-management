#include <stdexcept>
#include <string>
#include <vector>

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

extern "C" int esa_retrieval_launcher(torch::Tensor query, torch::Tensor repre_cache, torch::Tensor q_index, torch::Tensor repre_index, torch::Tensor repre_index_cpu,
        torch::Tensor batch_offset, torch::Tensor score, torch::Tensor score_cpu, torch::Tensor score_sorted_cpu, torch::Tensor index_sorted_cpu,
        int batch, int s);

extern "C" int esa_retrieval_poll(int handle);
extern "C" int esa_retrieval_cleanup(int handle);
extern "C" int esa_retrieval_pending();
extern "C" void esa_retrieval_shutdown();

extern "C" void esa_topk(torch::Tensor score, torch::Tensor index, torch::Tensor offsets, torch::Tensor score_out, torch::Tensor index_out, torch::Tensor workspace);

extern "C" void esa_repre(torch::Tensor key_cache, torch::Tensor repre_cache, torch::Tensor block_table, torch::Tensor repre_table);

extern "C" void esa_copy(torch::Tensor src, torch::Tensor dst, size_t size);

extern "C" void esa_copy_batch(torch::Tensor src_ptrs, torch::Tensor dst_ptrs, int size);

extern "C" void esa_scatter_copy(torch::Tensor src, torch::Tensor dst, torch::Tensor block_table_src, torch::Tensor block_table_dst);

struct RetrievalInputTensor{
    torch::Tensor query;
    torch::Tensor repre_cache;
    torch::Tensor q_index;
    torch::Tensor repre_index;
    torch::Tensor repre_index_cpu;
    torch::Tensor batch_offset;
    int batch;
    int s;
};

struct RetrievalOutputTensor{
    torch::Tensor score;
    // New CPU pinned outputs for async D2H + host callback argsort
    torch::Tensor score_cpu;          // 1D pinned CPU tensor [s], same dtype as score
    torch::Tensor score_sorted_cpu;   // 1D pinned CPU tensor [s], same dtype as score
    torch::Tensor index_sorted_cpu;   // 1D pinned CPU tensor [s], int32
};


int esa_retrieval(RetrievalInputTensor input, RetrievalOutputTensor output){
    auto query = input.query;
    auto repre_cache = input.repre_cache;
    auto q_index = input.q_index;
    auto repre_index = input.repre_index;
    auto repre_index_cpu = input.repre_index_cpu;
    auto batch_offset = input.batch_offset;

    auto score = output.score;
    // CPU pinned outputs
    auto score_cpu = output.score_cpu;
    auto score_sorted_cpu = output.score_sorted_cpu;
    auto index_sorted_cpu = output.index_sorted_cpu;

    return esa_retrieval_launcher(
        query, repre_cache, q_index, repre_index, repre_index_cpu,
        batch_offset, score,
        score_cpu, score_sorted_cpu, index_sorted_cpu,
        input.batch, input.s
    );
}


#define STRINGFY(func) #func
#define TORCH_BINDING_COMMON_EXTENSION(func) \
    m.def(STRINGFY(func), &func, STRINGFY(func))

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ESA cuda kernels for block feature extraction and block retrieval";
    py::class_<RetrievalInputTensor>(m, "RetrievalInputTensor")
        .def(py::init<>())
        .def_readwrite("query", &RetrievalInputTensor::query)
        .def_readwrite("repre_cache", &RetrievalInputTensor::repre_cache)
        .def_readwrite("q_index", &RetrievalInputTensor::q_index)
        .def_readwrite("repre_index", &RetrievalInputTensor::repre_index)
        .def_readwrite("repre_index_cpu", &RetrievalInputTensor::repre_index_cpu)
        .def_readwrite("batch_offset", &RetrievalInputTensor::batch_offset)
        .def_readwrite("batch", &RetrievalInputTensor::batch)
        .def_readwrite("s", &RetrievalInputTensor::s);

    py::class_<RetrievalOutputTensor>(m, "RetrievalOutputTensor")
        .def(py::init<>())
        .def_readwrite("score", &RetrievalOutputTensor::score)
        .def_readwrite("score_cpu", &RetrievalOutputTensor::score_cpu)
        .def_readwrite("score_sorted_cpu", &RetrievalOutputTensor::score_sorted_cpu)
        .def_readwrite("index_sorted_cpu", &RetrievalOutputTensor::index_sorted_cpu);

    TORCH_BINDING_COMMON_EXTENSION(esa_retrieval);
    TORCH_BINDING_COMMON_EXTENSION(esa_topk);
    TORCH_BINDING_COMMON_EXTENSION(esa_repre);
    TORCH_BINDING_COMMON_EXTENSION(esa_copy);
    TORCH_BINDING_COMMON_EXTENSION(esa_scatter_copy);
    TORCH_BINDING_COMMON_EXTENSION(esa_copy_batch);

    // Async retrieval helpers
    m.def("esa_retrieval_poll", &esa_retrieval_poll, "Poll whether CPU argsort finished (returns 0/1)");
    m.def("esa_retrieval_cleanup", &esa_retrieval_cleanup, "Cleanup a retrieval handle");
    m.def("esa_retrieval_pending", &esa_retrieval_pending, "Number of pending retrieval contexts");
    m.def("esa_retrieval_shutdown", &esa_retrieval_shutdown, "Shutdown retrieval worker/callback streams");
}
