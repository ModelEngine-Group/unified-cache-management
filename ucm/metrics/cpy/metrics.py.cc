// monitor_pybind.cc
#include <pybind11/pybind11.h>
#include "stats_monitor.h"

namespace py = pybind11;
namespace UC {

void bind_monitor(py::module_& m) {

    py::class_<UCMStatsMonitor>(m, "UCMStatsMonitor")
        .def_static("GetInstance", &UCMStatsMonitor::GetInstance,
                    py::return_value_policy::reference)
        .def("update_stats", &UCMStatsMonitor::update_stats)
        .def("get_stats_and_clear", &UCMStatsMonitor::get_stats_and_clear)
}

}  // namespace UC

// 在主绑定入口中注册
PYBIND11_MODULE(ucmmonitor, module) {
    m.attr("__version__") = "0.1.0";
    UC::bind_monitor(module);
}