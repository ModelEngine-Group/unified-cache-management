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
#include "fake_store.h"
#include "template/store_binder.h"

PYBIND11_MODULE(ucmfakestore, module)
{
    namespace py = pybind11;
    using namespace UC::FakeStore;
    using FakeStorePy = UC::Detail::StoreBinder<FakeStore, Config>;
    module.attr("project") = UCM_PROJECT_NAME;
    module.attr("version") = UCM_PROJECT_VERSION;
    module.attr("commit_id") = UCM_COMMIT_ID;
    module.attr("build_type") = UCM_BUILD_TYPE;
    auto store = py::class_<FakeStorePy, std::unique_ptr<FakeStorePy>>(module, "FakeStore");
    auto config = py::class_<Config>(store, "Config");
    config.def(py::init<>());
    config.def_readwrite("uniqueId", &Config::uniqueId);
    config.def_readwrite("bufferNumber", &Config::bufferNumber);
    config.def_readwrite("shareBufferEnable", &Config::shareBufferEnable);
    store.def(py::init<>());
    store.def("Self", &FakeStorePy::Self);
    store.def("Setup", &FakeStorePy::Setup);
    store.def("Lookup", &FakeStorePy::Lookup, py::arg("ids").noconvert());
    store.def("LookupOnPrefix", &FakeStorePy::LookupOnPrefix, py::arg("ids").noconvert());
    store.def("Prefetch", &FakeStorePy::Prefetch, py::arg("ids").noconvert());
    store.def("Load", &FakeStorePy::Load, py::arg("ids").noconvert(),
              py::arg("indexes").noconvert(), py::arg("addrs").noconvert());
    store.def("Dump", &FakeStorePy::Dump, py::arg("ids").noconvert(),
              py::arg("indexes").noconvert(), py::arg("addrs").noconvert());
    store.def("Check", &FakeStorePy::Check);
    store.def("Wait", &FakeStorePy::Wait);
}
