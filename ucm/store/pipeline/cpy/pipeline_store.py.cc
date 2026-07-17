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
#include <list>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "breaker_store.h"
#include "config_parser.h"
#include "library_loader.h"
#include "store_health_config.h"
#include "ucmstore_v1.h"

namespace py = pybind11;

namespace UC::PipelineStore {

class StoreNotFoundError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class StoreUnhealthyError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class PipelineStore {
    using StoreLoader = LibraryLoader<StoreV1>;
    template <typename T>
    struct BufferArrayView {
        const T* data;
        size_t num;
        BufferArrayView(const pybind11::buffer& buffer)
        {
            const auto info = buffer.request(false);
            data = static_cast<const T*>(info.ptr);
            const auto scale = sizeof(T) / info.itemsize;
            num = static_cast<size_t>(info.shape[0]) / scale;
        }
        const T* operator[](size_t i) const noexcept { return data + i; }
    };
    template <typename T>
    struct Buffer2DArrayView {
        const T* data;
        size_t rows, cols;
        Buffer2DArrayView(const pybind11::buffer& buffer)
        {
            const auto info = buffer.request(false);
            data = static_cast<const T*>(info.ptr);
            const auto scale = sizeof(T) / info.itemsize;
            rows = static_cast<size_t>(info.shape[0]) / scale;
            cols = static_cast<size_t>(info.shape[1]) / scale;
        }
        const T* operator[](size_t r) const noexcept { return data + r * cols; }
    };

    std::list<StoreLoader> loaders_;
    std::list<std::shared_ptr<StoreV1>> stores_;
    std::list<std::shared_ptr<BreakerStore>> breakers_;
    StoreV1* entry_{nullptr};
    StoreHealthConfig healthConfig_;

    StoreV1* StoreBack() const { return entry_; }
    [[noreturn]] static void ThrowError(const Status& s)
    {
        if (s == Status::NotFound()) { throw StoreNotFoundError{s.ToString()}; }
        if (s == Status::StoreUnhealthy()) { throw StoreUnhealthyError{s.ToString()}; }
        throw std::runtime_error{s.ToString()};
    }
    static void ThrowIfFailed(const Status& s)
    {
        if (s.Success()) { return; }
        ThrowError(s);
    }
    static StoreHealthConfig ParseHealthConfig(const py::dict& config)
    {
        StoreHealthConfig result;
        auto readSeconds = [&config](const char* name, auto defaultValue) {
            if (!config.contains(name)) { return defaultValue; }
            auto seconds = py::cast<double>(config[name]);
            return std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::duration<double>(seconds));
        };
        if (config.contains("enabled")) { result.enabled = py::cast<bool>(config["enabled"]); }
        result.healthCheckInterval =
            readSeconds("health_check_interval_s", result.healthCheckInterval);
        result.healthCheckTimeout =
            readSeconds("health_check_timeout_s", result.healthCheckTimeout);
        if (config.contains("health_window_size")) {
            result.healthWindowSize = py::cast<size_t>(config["health_window_size"]);
        }
        if (config.contains("failure_threshold")) {
            result.failureThreshold = py::cast<size_t>(config["failure_threshold"]);
        }
        ThrowIfFailed(result.Validate());
        return result;
    }
    static Detail::TaskDesc MakeTaskDesc(const pybind11::buffer& ids,
                                         const pybind11::buffer& indexes,
                                         const pybind11::buffer& addrs)
    {
        BufferArrayView<Detail::BlockId> idArr{ids};
        BufferArrayView<size_t> idxArr{indexes};
        Buffer2DArrayView<void*> addrArr{addrs};
        if (idArr.num != idxArr.num || idArr.num != addrArr.rows) {
            ThrowIfFailed(
                Status::InvalidParam("invalid dim: {},{},{}", idArr.num, idxArr.num, addrArr.rows));
        }
        Detail::TaskDesc desc;
        desc.reserve(idArr.num);
        for (size_t i = 0; i < idArr.num; i++) {
            Detail::Shard shard;
            shard.owner = *idArr[i];
            shard.index = *idxArr[i];
            shard.addrs.assign(addrArr[i], addrArr[i] + addrArr.cols);
            desc.push_back(std::move(shard));
        }
        return desc;
    }

public:
    explicit PipelineStore(const py::dict& healthConfig = {})
        : healthConfig_{ParseHealthConfig(healthConfig)}
    {
    }
    ~PipelineStore()
    {
        for (auto& breaker : breakers_) { breaker->Stop(); }
        breakers_.clear();
        while (!stores_.empty()) { stores_.pop_back(); }
    }
    void Stack(const std::string& name, const std::string& path, const py::dict& dict)
    {
        py::dict storeDict(dict);
        Detail::Dictionary config;
        ThrowIfFailed(ConfigParser::Parse(config, storeDict));
        config.Set<StoreV1*>("store_backend", StoreBack());
        StoreLoader loader{path, "Make" + name + "Store"};
        ThrowIfFailed(loader.LoadLibrary());
        auto store = loader.CreateObject();
        if (!store) { throw std::runtime_error{"failed to create store(" + name + ")"}; }
        ThrowIfFailed(store->Setup(config));
        loaders_.push_back(std::move(loader));
        stores_.push_back(std::move(store));
        entry_ = stores_.back().get();
        if (healthConfig_.enabled) {
            const auto storeId = "pipeline/" + std::to_string(stores_.size() - 1) + ":" +
                                 stores_.back()->Readme();
            auto breaker =
                std::make_shared<BreakerStore>(stores_.back().get(), storeId, healthConfig_);
            ThrowIfFailed(breaker->Start());
            breakers_.push_back(std::move(breaker));
            entry_ = breakers_.back().get();
        }
    }
    uintptr_t Self() const { return (uintptr_t)(void*)StoreBack(); }
    pybind11::bytes Lookup(const pybind11::buffer& ids)
    {
        BufferArrayView<Detail::BlockId> idArr{ids};
        auto res = StoreBack()->Lookup(idArr.data, idArr.num);
        if (res) {
            auto& v = res.Value();
            return pybind11::bytes(reinterpret_cast<const char*>(v.data()), v.size());
        }
        ThrowError(res.Error());
    }
    ssize_t LookupOnPrefix(const pybind11::buffer& ids)
    {
        BufferArrayView<Detail::BlockId> idArr{ids};
        auto res = StoreBack()->LookupOnPrefix(idArr.data, idArr.num);
        if (res) { return res.Value(); }
        ThrowError(res.Error());
    }
    void Prefetch(const pybind11::buffer& ids)
    {
        BufferArrayView<Detail::BlockId> idArr{ids};
        StoreBack()->Prefetch(idArr.data, idArr.num);
    }
    Detail::TaskHandle Load(const pybind11::buffer& ids, const pybind11::buffer& indexes,
                            const pybind11::buffer& addrs)
    {
        auto desc = MakeTaskDesc(ids, indexes, addrs);
        desc.brief = "Load";
        auto res = StoreBack()->Load(std::move(desc));
        if (res) { return res.Value(); }
        ThrowError(res.Error());
    }
    Detail::TaskHandle Dump(const pybind11::buffer& ids, const pybind11::buffer& indexes,
                            const pybind11::buffer& addrs, uintptr_t prerequisite_handle = 0)
    {
        auto desc = MakeTaskDesc(ids, indexes, addrs);
        desc.brief = "Dump";
        desc.prerequisiteHandle = prerequisite_handle;
        auto res = StoreBack()->Dump(desc);
        if (res) { return res.Value(); }
        ThrowError(res.Error());
    }
    bool Check(Detail::TaskHandle taskId)
    {
        auto res = StoreBack()->Check(taskId);
        if (res) { return res.Value(); }
        ThrowError(res.Error());
    }
    void Wait(Detail::TaskHandle taskId)
    {
        auto status = Status::OK();
        {
            pybind11::gil_scoped_release release;
            status = StoreBack()->Wait(taskId);
        }
        ThrowIfFailed(status);
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
    auto errors = py::module_::import("ucm.store.pipeline.errors");
    py::register_exception<StoreNotFoundError>(m, "StoreNotFoundError",
                                               errors.attr("StoreNotFoundError").ptr());
    py::register_exception<StoreUnhealthyError>(m, "StoreUnhealthyError",
                                                errors.attr("StoreUnhealthyError").ptr());
    auto s = py::class_<PipelineStore, std::unique_ptr<PipelineStore>>(m, "PipelineStore");
    s.def(py::init<const py::dict&>(), py::arg("store_health") = py::dict());
    s.def("Stack", &PipelineStore::Stack);
    s.def("Self", &PipelineStore::Self);
    s.def("Lookup", &PipelineStore::Lookup, py::arg("ids").noconvert());
    s.def("LookupOnPrefix", &PipelineStore::LookupOnPrefix, py::arg("ids").noconvert());
    s.def("Prefetch", &PipelineStore::Prefetch, py::arg("ids").noconvert());
    s.def("Load", &PipelineStore::Load, py::arg("ids").noconvert(), py::arg("indexes").noconvert(),
          py::arg("addrs").noconvert());
    s.def("Dump", &PipelineStore::Dump, py::arg("ids").noconvert(), py::arg("indexes").noconvert(),
          py::arg("addrs").noconvert(), py::arg("prerequisite_handle") = 0);
    s.def("Check", &PipelineStore::Check);
    s.def("Wait", &PipelineStore::Wait);
}
