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
#include "config_parser.h"
#include "library_loader.h"
#include "template/store_binder.h"
#include "ucmstore_v1.h"

namespace py = pybind11;

namespace UC::PipelineStore {

class PipelineStore {
    std::shared_ptr<StoreV1> back_{nullptr};

public:
    const std::shared_ptr<StoreV1>& GetBack() const { return back_; }
    void SetBack(std::shared_ptr<StoreV1> store) { back_ = std::move(store); }
    uintptr_t RawPtr() const { return (uintptr_t)(void*)back_.get(); }
    StoreV1* operator->() const { return back_.get(); }
    StoreV1& operator*() const { return *back_; }
};

class PipelineStorePy : public Detail::StoreBinder<PipelineStore> {
    using StoreLoader = LibraryLoader<StoreV1>;
    std::list<StoreLoader> loaders_;
    std::list<std::shared_ptr<StoreV1>> stores_;

public:
    void Stack(const std::string& name, const std::string& path, const py::dict& dict)
    {
        Detail::Dictionary config;
        ThrowIfFailed(ConfigParser::Parse(config, dict));
        config.Set<std::shared_ptr<StoreV1>>("store_backend", StoreBase().GetBack());
        StoreLoader loader{path, "Make" + name + "Store"};
        ThrowIfFailed(loader.LoadLibrary());
        auto store = loader.CreateObject();
        if (!store) { throw std::runtime_error{"failed to create store(" + name + ")"}; }
        ThrowIfFailed(store->Setup(config));
        loaders_.push_back(std::move(loader));
        stores_.push_back(store);
        StoreBase().SetBack(std::move(store));
    }
};

}  // namespace UC::PipelineStore

PYBIND11_MODULE(ucmpipelinestore, m)
{
    using namespace UC::PipelineStore;
    m.attr("project") = UCM_PROJECT_NAME;
    m.attr("version") = UCM_PROJECT_VERSION;
    m.attr("commit_id") = UCM_COMMIT_ID;
    m.attr("build_type") = UCM_BUILD_TYPE;
    auto s = py::class_<PipelineStorePy, std::unique_ptr<PipelineStorePy>>(m, "PipelineStore");
    s.def(py::init<>());
    s.def("Stack", &PipelineStorePy::Stack);
    s.def("Self", &PipelineStorePy::Self);
    s.def("Lookup", &PipelineStorePy::Lookup, py::arg("ids").noconvert());
    s.def("LookupOnPrefix", &PipelineStorePy::LookupOnPrefix, py::arg("ids").noconvert());
    s.def("Prefetch", &PipelineStorePy::Prefetch, py::arg("ids").noconvert());
    s.def("Load", &PipelineStorePy::Load, py::arg("ids").noconvert(),
          py::arg("indexes").noconvert(), py::arg("addrs").noconvert());
    s.def("Dump", &PipelineStorePy::Dump, py::arg("ids").noconvert(),
          py::arg("indexes").noconvert(), py::arg("addrs").noconvert());
    s.def("Check", &PipelineStorePy::Check);
    s.def("Wait", &PipelineStorePy::Wait);
}
