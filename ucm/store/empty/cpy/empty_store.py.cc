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
#include <atomic>
#include "template/store_binder.h"
#include "ucmstore_v1.h"

namespace UC::EmptyStore {

struct Config {
    void* padding;
};

class EmptyStore : public StoreV1 {
public:
    ~EmptyStore() override = default;
    Status Setup(const Config& config) { return Status::OK(); }
    std::string Readme() const override { return "EmptyStore"; }
    Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId* blocks, size_t num) override
    {
        return std::vector<uint8_t>(num, false);
    }
    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, size_t num) override
    {
        return -1;
    }
    void Prefetch(const Detail::BlockId* blocks, size_t num) override {}
    Expected<Detail::TaskHandle> Load(Detail::TaskDesc task) override { return NextId(); }
    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task) override { return NextId(); }
    Expected<bool> Check(Detail::TaskHandle taskId) override { return true; }
    Status Wait(Detail::TaskHandle taskId) override { return Status::OK(); }

private:
    static Detail::TaskHandle NextId() noexcept
    {
        static std::atomic<Detail::TaskHandle> id{1};
        return id.fetch_add(1, std::memory_order_relaxed);
    };
};

}  // namespace UC::EmptyStore

PYBIND11_MODULE(ucmemptystore, module)
{
    namespace py = pybind11;
    using namespace UC::EmptyStore;
    using EmptyStorePy = UC::Detail::StoreBinder<EmptyStore, Config>;
    module.attr("project") = UCM_PROJECT_NAME;
    module.attr("version") = UCM_PROJECT_VERSION;
    module.attr("commit_id") = UCM_COMMIT_ID;
    module.attr("build_type") = UCM_BUILD_TYPE;
    auto store = py::class_<EmptyStorePy, std::unique_ptr<EmptyStorePy>>(module, "EmptyStore");
    auto config = py::class_<Config>(store, "Config");
    config.def(py::init<>());
    store.def(py::init<>());
    store.def("Self", &EmptyStorePy::Self);
    store.def("Setup", &EmptyStorePy::Setup);
    store.def("Lookup", &EmptyStorePy::Lookup, py::arg("ids").noconvert());
    store.def("LookupOnPrefix", &EmptyStorePy::LookupOnPrefix, py::arg("ids").noconvert());
    store.def("Prefetch", &EmptyStorePy::Prefetch, py::arg("ids").noconvert());
    store.def("Load", &EmptyStorePy::Load, py::arg("ids").noconvert(),
              py::arg("indexes").noconvert(), py::arg("addrs").noconvert());
    store.def("Dump", &EmptyStorePy::Dump, py::arg("ids").noconvert(),
              py::arg("indexes").noconvert(), py::arg("addrs").noconvert());
    store.def("Check", &EmptyStorePy::Check);
    store.def("Wait", &EmptyStorePy::Wait);
}
