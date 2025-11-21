// monitor_pybind.cc
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "stats_monitor.h"

namespace py = pybind11;
namespace UC {

void bind_monitor(py::module_& m) {
    py::class_<UCMStatsMonitor>(m, "UCMStatsMonitor")
        .def(py::init<>())                              // 用默认构造函数
        .def("update_stats", &UCMStatsMonitor::updateStats)
        .def("reset_all", &UCMStatsMonitor::resetAllStats)
        .def("get_stats", &UCMStatsMonitor::getStats);
}

}  // namespace UC

// 在主绑定入口中注册
PYBIND11_MODULE(ucmmonitor, module) {
    module.attr("__version__") = "0.1.0";
    UC::bind_monitor(module);
}