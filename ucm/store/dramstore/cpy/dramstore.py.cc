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
#include "api/dramstore.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace UC {

class DRAMStorePy : public DRAMStore {
public:
    void* CCStoreImpl() { return this; }
    py::list AllocBatch(const py::list& blocks)
    {
        py::list results;
        for (auto& block : blocks) { results.append(this->Alloc(block.cast<std::string>())); }
        return results;
    }
    py::list LookupBatch(const py::list& blocks)
    {
        py::list founds;
        for (auto& block : blocks) { founds.append(this->Lookup(block.cast<std::string>())); }
        return founds;
    }
    void CommitBatch(const py::list& blocks, const bool success)
    {
        for (auto& block : blocks) { this->Commit(block.cast<std::string>(), success); }
    }
    py::tuple CheckPy(const size_t task)
    {
        auto finish = false;
        auto ret = this->Check(task, finish);
        return py::make_tuple(ret, finish);
    }
    size_t Load(const py::list& blockIds, const py::list& offsets, const py::list& addresses,
                const py::list& lengths)
    {
        return this->SubmitPy(blockIds, offsets, addresses, lengths, CCStore::Task::Type::LOAD,
                              CCStore::Task::Location::DEVICE, "DRAM::H2D");
    }
    size_t Dump(const py::list& blockIds, const py::list& offsets, const py::list& addresses,
                const py::list& lengths)
    {
        return this->SubmitPy(blockIds, offsets, addresses, lengths, CCStore::Task::Type::DUMP,
                              CCStore::Task::Location::DEVICE, "DRAM::D2H");
    }

private:
    size_t SubmitPy(const py::list& blockIds, const py::list& offsets, const py::list& addresses,
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
        return this->Submit(std::move(task));
    }
};

} // namespace UC

PYBIND11_MODULE(ucmdramstore, module)
{
    module.attr("project") = UCM_PROJECT_NAME;
    module.attr("version") = UCM_PROJECT_VERSION;
    module.attr("commit_id") = UCM_COMMIT_ID;
    module.attr("build_type") = UCM_BUILD_TYPE;
    auto store = py::class_<UC::DRAMStorePy>(module, "DRAMStore");
    auto config = py::class_<UC::DRAMStorePy::Config>(store, "Config");
    config.def(py::init<const size_t, const size_t>(), py::arg("ioSize"), py::arg("capacity"));
    config.def_readwrite("ioSize", &UC::DRAMStorePy::Config::ioSize);
    config.def_readwrite("capacity", &UC::DRAMStorePy::Config::capacity);
    config.def_readwrite("deviceId", &UC::DRAMStorePy::Config::deviceId);
    store.def(py::init<>());
    store.def("CCStoreImpl", &UC::DRAMStorePy::CCStoreImpl);
    store.def("Setup", &UC::DRAMStorePy::Setup);
    store.def("Alloc", py::overload_cast<const std::string&>(&UC::DRAMStorePy::Alloc));
    store.def("AllocBatch", &UC::DRAMStorePy::AllocBatch);
    store.def("Lookup", py::overload_cast<const std::string&>(&UC::DRAMStorePy::Lookup));
    store.def("LookupBatch", &UC::DRAMStorePy::LookupBatch);
    store.def("Load", &UC::DRAMStorePy::Load);
    store.def("Dump", &UC::DRAMStorePy::Dump);
    store.def("Wait", &UC::DRAMStorePy::Wait);
    store.def("Check", &UC::DRAMStorePy::Check);
    store.def("Commit",
              py::overload_cast<const std::string&, const bool>(&UC::DRAMStorePy::Commit));
    store.def("CommitBatch", &UC::DRAMStorePy::CommitBatch);
}
