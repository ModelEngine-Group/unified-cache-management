#include <stdexcept>
#include <string>
#include <vector>

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

extern "C" void esa_retrieval_launcher(torch::Tensor query, torch::Tensor repre_cache, torch::Tensor q_index, torch::Tensor repre_index,
        torch::Tensor batch_offset, torch::Tensor workspace, torch::Tensor score, torch::Tensor score_sorted, torch::Tensor index, torch::Tensor index_sorted, int batch, int s);

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
    torch::Tensor batch_offset;
    torch::Tensor workspace;
    int batch;
    int s;
};

struct RetrievalOutputTensor{
    torch::Tensor score;
    torch::Tensor index;
    torch::Tensor score_sorted;
    torch::Tensor index_sorted;
};


void esa_retrieval(RetrievalInputTensor input, RetrievalOutputTensor output){
    auto query = input.query;
    auto repre_cache = input.repre_cache;
    auto q_index = input.q_index;
    auto repre_index = input.repre_index;
    auto batch_offset = input.batch_offset;
    auto workspace = input.workspace;

    auto score = output.score;
    auto index = output.index;
    auto score_sorted = output.score_sorted;
    auto index_sorted = output.index_sorted;
    esa_retrieval_launcher(query, repre_cache, q_index, repre_index, batch_offset, workspace, score, score_sorted, index, index_sorted, input.batch, input.s);
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
        .def_readwrite("batch_offset", &RetrievalInputTensor::batch_offset)
        .def_readwrite("workspace", &RetrievalInputTensor::workspace)
        .def_readwrite("batch", &RetrievalInputTensor::batch)
        .def_readwrite("s", &RetrievalInputTensor::s);

    py::class_<RetrievalOutputTensor>(m, "RetrievalOutputTensor")
        .def(py::init<>())
        .def_readwrite("score", &RetrievalOutputTensor::score)
        .def_readwrite("score_sorted", &RetrievalOutputTensor::score_sorted)
        .def_readwrite("index", &RetrievalOutputTensor::index)
        .def_readwrite("index_sorted", &RetrievalOutputTensor::index_sorted);

    TORCH_BINDING_COMMON_EXTENSION(esa_retrieval);
    TORCH_BINDING_COMMON_EXTENSION(esa_topk);
    TORCH_BINDING_COMMON_EXTENSION(esa_repre);
    TORCH_BINDING_COMMON_EXTENSION(esa_copy);
    TORCH_BINDING_COMMON_EXTENSION(esa_scatter_copy);
    TORCH_BINDING_COMMON_EXTENSION(esa_copy_batch);
}
