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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "metrics_api.h"

namespace py = pybind11;
namespace UC::Metrics {

void bind_monitor(py::module_& m)
{
    m.def("set_up", &SetUp);
    m.def("create_stats", &CreateStats);
    m.def("update_stats", py::overload_cast<const std::string&, double>(&UpdateStats));
    m.def("update_stats",
          py::overload_cast<const std::unordered_map<std::string, double>&>(&UpdateStats));
    m.def("get_all_stats_and_clear", []() {
        py::gil_scoped_release releaseGil;
        return GetAllStatsAndClear();
    });
}

}  // namespace UC::Metrics

PYBIND11_MODULE(ucmmetrics, module)
{
    module.attr("project") = UCM_PROJECT_NAME;
    module.attr("version") = UCM_PROJECT_VERSION;
    module.attr("commit_id") = UCM_COMMIT_ID;
    module.attr("build_type") = UCM_BUILD_TYPE;
    UC::Metrics::bind_monitor(module);
}