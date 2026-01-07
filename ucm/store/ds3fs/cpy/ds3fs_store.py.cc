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
#include "ds3fs_store.h"
#include "template/store_binder.h"

PYBIND11_MODULE(ucmds3fsstore, module)
{
    namespace py = pybind11;
    using namespace UC::Ds3fsStore;
    using Ds3fsStorePy = UC::Detail::StoreBinder<Ds3fsStore, Config>;
    module.attr("project") = UCM_PROJECT_NAME;
    module.attr("version") = UCM_PROJECT_VERSION;
    module.attr("commit_id") = UCM_COMMIT_ID;
    module.attr("build_type") = UCM_BUILD_TYPE;
    auto store = py::class_<Ds3fsStorePy, std::unique_ptr<Ds3fsStorePy>>(module, "Ds3fsStore");
    auto config = py::class_<Config>(store, "Config");
    config.def(py::init<>());
    config.def_readwrite("storageBackends", &Config::storageBackends);
    config.def_readwrite("deviceId", &Config::deviceId);
    config.def_readwrite("tensorSize", &Config::tensorSize);
    config.def_readwrite("shardSize", &Config::shardSize);
    config.def_readwrite("blockSize", &Config::blockSize);
    config.def_readwrite("ioDirect", &Config::ioDirect);
    config.def_readwrite("streamNumber", &Config::streamNumber);
    config.def_readwrite("timeoutMs", &Config::timeoutMs);
    config.def_readwrite("iorEntries", &Config::iorEntries);
    config.def_readwrite("iorDepth", &Config::iorDepth);
    config.def_readwrite("numaId", &Config::numaId);
    store.def(py::init<>());
    store.def("Self", &Ds3fsStorePy::Self);
    store.def("Setup", &Ds3fsStorePy::Setup);
    store.def("Lookup", &Ds3fsStorePy::Lookup, py::arg("ids").noconvert());
    store.def("Prefetch", &Ds3fsStorePy::Prefetch, py::arg("ids").noconvert());
    store.def("Load", &Ds3fsStorePy::Load, py::arg("ids").noconvert(),
              py::arg("indexes").noconvert(), py::arg("addrs").noconvert());
    store.def("Dump", &Ds3fsStorePy::Dump, py::arg("ids").noconvert(),
              py::arg("indexes").noconvert(), py::arg("addrs").noconvert());
    store.def("Check", &Ds3fsStorePy::Check);
    store.def("Wait", &Ds3fsStorePy::Wait);
}
