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
#include "localstore.h"
#include <pybind11/pybind11.h>
#include "template/singleton.h"

namespace py = pybind11;
using StoreImpl = UC::LocalStore;

namespace UC {

inline void* CCStoreImpl() { return Singleton<StoreImpl>::Instance(); }
inline int32_t Setup(const StoreImpl::Config& config)
{
    return ((StoreImpl*)CCStoreImpl())->Setup(config);
}
inline int32_t Alloc(const std::string& block) { return ((StoreImpl*)CCStoreImpl())->Alloc(block); }
inline bool Lookup(const std::string& block) { return ((StoreImpl*)CCStoreImpl())->Lookup(block); }
inline void Commit(const std::string& block, const bool success)
{
    return ((StoreImpl*)CCStoreImpl())->Commit(block, success);
}
inline py::list AllocBatch(const py::list& blocks)
{
    py::list results;
    for (auto& block : blocks) { results.append(Alloc(block.cast<std::string>())); }
    return results;
}
inline py::list LookupBatch(const py::list& blocks)
{
    py::list founds;
    for (auto& block : blocks) { founds.append(Lookup(block.cast<std::string>())); }
    return founds;
}
inline void CommitBatch(const py::list& blocks, const bool success)
{
    for (auto& block : blocks) { Commit(block.cast<std::string>(), success); }
}
inline int32_t Wait(const size_t task) { return ((StoreImpl*)CCStoreImpl())->Wait(task); }
inline py::tuple Check(const size_t task)
{
    auto finish = false;
    auto ret = ((StoreImpl*)CCStoreImpl())->Check(task, finish);
    return py::make_tuple(ret, finish);
}
size_t Submit(const py::list& blockIds, const py::list& offsets, const py::list& addresses,
              const py::list& lengths, const CCStore::Task::Type type,
              const CCStore::Task::Location location, const std::string& brief)
{
    CCStore::Task task{type, location, brief};
    auto blockId = blockIds.begin();
    auto offset = offsets.begin();
    auto address = addresses.begin();
    auto length = lengths.begin();
    while ((blockId != blockIds.end()) && (offset != offsets.end()) &&
           (address != addresses.end()) && (length != lengths.end())) {
        auto ret = task.Append(blockId->cast<std::string>(), offset->cast<size_t>(),
                               address->cast<uintptr_t>(), length->cast<size_t>());
        if (ret != 0) { return CCStore::invalidTaskId; }
        blockId++;
        offset++;
        address++;
        length++;
    }
    return ((StoreImpl*)CCStoreImpl())->Submit(std::move(task));
}
inline size_t Load(const py::list& blockIds, const py::list& offsets, const py::list& addresses,
                   const py::list& lengths)
{
    return Submit(blockIds, offsets, addresses, lengths, CCStore::Task::Type::LOAD,
                  CCStore::Task::Location::DEVICE, "Local::S2D");
}
inline size_t Dump(const py::list& blockIds, const py::list& offsets, const py::list& addresses,
                   const py::list& lengths)
{
    return Submit(blockIds, offsets, addresses, lengths, CCStore::Task::Type::DUMP,
                  CCStore::Task::Location::DEVICE, "Local::D2S");
}

} // namespace UC

PYBIND11_MODULE(ucmdramstore, module)
{
    module.attr("project") = UC_VAR_PROJECT_NAME;
    module.attr("version") = UC_VAR_PROJECT_VERSION;
    module.attr("commit_id") = UC_VAR_GIT_COMMIT_ID;
    module.attr("build_type") = UC_VAR_BUILD_TYPE;
    auto store = module.def_submodule("LocalStore");
    auto config = py::class_<StoreImpl::Config>(store, "Config");
    config.def(py::init<const size_t, const size_t>(), py::arg("ioSize"), py::arg("capacity"));
    config.def_readwrite("ioSize", &StoreImpl::Config::ioSize);
    config.def_readwrite("capacity", &StoreImpl::Config::capacity);
    config.def_readwrite("backend", &StoreImpl::Config::backend);
    config.def_readwrite("deviceId", &StoreImpl::Config::deviceId);
    module.def("CCStoreImpl", &UC::CCStoreImpl);
    module.def("Setup", &UC::Setup);
    module.def("Alloc", &UC::Alloc);
    module.def("AllocBatch", &UC::AllocBatch);
    module.def("Lookup", &UC::Lookup);
    module.def("LookupBatch", &UC::LookupBatch);
    module.def("Load", &UC::Load);
    module.def("Dump", &UC::Dump);
    module.def("Wait", &UC::Wait);
    module.def("Check", &UC::Check);
    module.def("Commit", &UC::Commit);
    module.def("CommitBatch", &UC::CommitBatch);
}
