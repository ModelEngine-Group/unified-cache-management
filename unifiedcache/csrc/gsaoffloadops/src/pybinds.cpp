#pragma GCC diagnostic push
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#pragma GCC diagnostic pop
#include "cal_kpre_and_topk.h"

PYBIND11_MODULE(gsa_offload_ops, m)
{
    pybind11::class_<CalKpreAndTopk>(m, "CalKpreAndTopk")
    .def(pybind11::init<int, int, int, int, int>())
    .def_readwrite("k_cache", &CalKpreAndTopk::m_kCache)
    .def_readwrite("q_cache", &CalKpreAndTopk::m_qCache)
    .def("set_kpre_method_param", &CalKpreAndTopk::SetKpreMethodParam)
    .def("set_kpre_cache", &CalKpreAndTopk::SetKpreCache)
    .def("set_topk_cache", &CalKpreAndTopk::SetTopkCache)
    .def("set_common_param", &CalKpreAndTopk::SetCommonParam)
    .def("set_topk_param", &CalKpreAndTopk::SetTopkParam)
    .def("set_kpre_param", &CalKpreAndTopk::SetKpreParam)
    .def("set_kpre_data_ready", &CalKpreAndTopk::SetKpreDataReady)
    .def("set_topk_data_ready", &CalKpreAndTopk::SetTopkDataReady)
    .def("add_copy_req", &CalKpreAndTopk::AddCopyReq)
    .def("is_calculate_finish", &CalKpreAndTopk::IsCalculateFinish);
}



