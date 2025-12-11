/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "stats/istats.h"
#include "stats_monitor_api.h"

namespace py = pybind11;
namespace UC::Metrics {

class PythonStatsWrapper : public IStats {
public:
    PythonStatsWrapper(py::object py_obj) : py_obj_(py::cast<py::object>(py_obj)) {}

    std::string Name() const override { return py_obj_.attr("Name")().cast<std::string>(); }

    void Update(const std::unordered_map<std::string, double>& params) override
    {
        py_obj_.attr("Update")(params);
    }

    void Reset() override { py_obj_.attr("Reset")(); }

    std::unordered_map<std::string, std::vector<double>> Data() override
    {
        return py_obj_.attr("Data")().cast<std::unordered_map<std::string, std::vector<double>>>();
    }

private:
    py::object py_obj_;
};

void bind_monitor(py::module_& m)
{
    py::class_<StatsResult>(m, "StatsResult")
        .def(py::init<>())
        .def_readonly("data", &StatsResult::data);
    m.def("create_stats", &CreateStats);
    m.def("update_stats", &UpdateStats);
    m.def("reset_stats", &ResetStats);
    m.def("reset_all", &ResetAllStats);
    m.def("get_stats", &GetStats);
    m.def("get_stats_and_clear", &GetStatsAndClear);
    m.def("get_all_stats_and_clear", &GetAllStatsAndClear);

    m.def(
        "register_stats",
        [](const std::string& name, py::object py_obj) {
            if (!py::hasattr(py_obj, "Name") || !py::hasattr(py_obj, "Update") ||
                !py::hasattr(py_obj, "Reset") || !py::hasattr(py_obj, "Data")) {
                throw std::runtime_error(
                    "Python object must implement Name/Update/Reset/Data methods");
            }

            RegistStats(name, [py_obj]() -> std::unique_ptr<IStats> {
                return std::make_unique<PythonStatsWrapper>(py_obj);
            });
        },
        py::arg("name"), py::arg("py_obj"));
}

} // namespace UC::Metrics

PYBIND11_MODULE(ucmmonitor, module)
{
    module.attr("project") = UCM_PROJECT_NAME;
    module.attr("version") = UCM_PROJECT_VERSION;
    module.attr("commit_id") = UCM_COMMIT_ID;
    module.attr("build_type") = UCM_BUILD_TYPE;
    UC::Metrics::bind_monitor(module);
}