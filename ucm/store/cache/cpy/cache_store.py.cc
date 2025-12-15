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
#include "cache_store.h"
#include "template/store_binder.h"

PYBIND11_MODULE(ucmcachestore, module)
{
    namespace py = pybind11;
    using namespace UC::CacheStore;
    using CacheStorePy = UC::Detail::StoreBinder<CacheStore, Config>;
    module.attr("project") = UCM_PROJECT_NAME;
    module.attr("version") = UCM_PROJECT_VERSION;
    module.attr("commit_id") = UCM_COMMIT_ID;
    module.attr("build_type") = UCM_BUILD_TYPE;
    auto store = py::class_<CacheStorePy, std::unique_ptr<CacheStorePy>>(module, "CacheStore");
    auto config = py::class_<Config>(store, "Config");
    config.def(py::init<>());
    config.def_readwrite("backend", &Config::backend);
    config.def_readwrite("engineId", &Config::engineId);
    config.def_readwrite("deviceId", &Config::deviceId);
    config.def_readwrite("tensorSize", &Config::tensorSize);
    config.def_readwrite("shardSize", &Config::shardSize);
    config.def_readwrite("blockSize", &Config::blockSize);
    config.def_readwrite("bufferSize", &Config::bufferSize);
    config.def_readwrite("shareBufferEnable", &Config::shareBufferEnable);
    config.def_readwrite("waitingQueueDepth", &Config::waitingQueueDepth);
    config.def_readwrite("runningQueueDepth", &Config::runningQueueDepth);
    config.def_readwrite("transferTimeoutMs", &Config::transferTimeoutMs);
    store.def(py::init<>());
    store.def("Self", &CacheStorePy::Self);
    store.def("Setup", &CacheStorePy::Setup);
    store.def("Lookup", &CacheStorePy::Lookup, py::arg("ids").noconvert());
    store.def("Prefetch", &CacheStorePy::Prefetch, py::arg("ids").noconvert());
    store.def("Load", &CacheStorePy::Load, py::arg("ids").noconvert(),
              py::arg("indexes").noconvert(), py::arg("addrs").noconvert());
    store.def("Dump", &CacheStorePy::Dump, py::arg("ids").noconvert(),
              py::arg("indexes").noconvert(), py::arg("addrs").noconvert());
    store.def("Check", &CacheStorePy::Check);
    store.def("Wait", &CacheStorePy::Wait);
}
