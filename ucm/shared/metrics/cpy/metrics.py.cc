// monitor_pybind.cc
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "stats_monitor.h"

namespace py = pybind11;
namespace UC::Metrics {

void bind_monitor(py::module_& m) {
    py::class_<StatsMonitor>(m, "StatsMonitor")
        .def_static("get_instance", &StatsMonitor::GetInstance,
                    py::return_value_policy::reference)
        .def("update_stats", &StatsMonitor::UpdateStats)
        .def("reset_all", &StatsMonitor::ResetAllStats)
        .def("get_stats", &StatsMonitor::GetStats)
        .def("get_stats_and_clear", &StatsMonitor::GetStatsAndClear);
}

}  // namespace UC

PYBIND11_MODULE(monitor, module) {
    module.attr("__version__") = "0.1.0";
    UC::Metrics::bind_monitor(module);
}