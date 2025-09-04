#pragma GCC diagnostic push
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#pragma GCC diagnostic pop
#include "kvcache_pre.h"

namespace ucmprefetch{
    PYBIND11_MODULE(gsa_prefetch, m)
    {
        pybind11::class_<ucmprefetch::GSAPrefetchEngineC>(m, "GSAPrefetchEngineC")
            .def(pybind11::init<torch::Tensor &, torch::Tensor &,
            torch::Tensor &, torch::Tensor &, bool>())
            .def("set_blocks_map", &ucmprefetch::GSAPrefetchEngineC::SetBlocksMap)
            .def("add_blocks_map", &ucmprefetch::GSAPrefetchEngineC::AddBlocksMap)
            .def("del_blocks_map", &ucmprefetch::GSAPrefetchEngineC::DelBlocksMap)
            .def("run_async_prefetch_bs", &ucmprefetch::GSAPrefetchEngineC::RunAsyncPrefetchBs)
            .def("set_blocks_table_info", &ucmprefetch::GSAPrefetchEngineC::SetBlockTableInfo)
            .def("obtain_load_blocks", &ucmprefetch::GSAPrefetchEngineC::ObtainLoadBlocks)
            .def("obtain_miss_idxs", &ucmprefetch::GSAPrefetchEngineC::ObtainMissIdxs)
            .def("obtain_blocks_map", &ucmprefetch::GSAPrefetchEngineC::ObtainBlocksMap);
    }
}
