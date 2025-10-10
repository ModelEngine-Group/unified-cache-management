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
#include "fsstore.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace UC {

class FSStorePy : public FSStore {
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
    size_t LoadToDevice(const py::list& blockIds, const py::list& offsets,
                        const py::list& addresses, const py::list& lengths)
    {
        return this->SubmitPy(blockIds, offsets, addresses, lengths, CCStore::Task::Type::LOAD,
                              CCStore::Task::Location::DEVICE, "FS::S2D");
    }
    size_t LoadToHost(const py::list& blockIds, const py::list& offsets, const py::list& addresses,
                      const py::list& lengths)
    {
        return this->SubmitPy(blockIds, offsets, addresses, lengths, CCStore::Task::Type::LOAD,
                              CCStore::Task::Location::HOST, "FS::S2H");
    }
    size_t DumpFromDevice(const py::list& blockIds, const py::list& offsets,
                          const py::list& addresses, const py::list& lengths)
    {
        return this->SubmitPy(blockIds, offsets, addresses, lengths, CCStore::Task::Type::DUMP,
                              CCStore::Task::Location::DEVICE, "FS::D2S");
    }
    size_t DumpFromHost(const py::list& blockIds, const py::list& offsets,
                        const py::list& addresses, const py::list& lengths)
    {
        return this->SubmitPy(blockIds, offsets, addresses, lengths, CCStore::Task::Type::DUMP,
                              CCStore::Task::Location::HOST, "FS::H2S");
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

PYBIND11_MODULE(ucmfsstore, module)
{
    module.attr("project") = UCM_PROJECT_NAME;
    module.attr("version") = UCM_PROJECT_VERSION;
    module.attr("commit_id") = UCM_COMMIT_ID;
    module.attr("build_type") = UCM_BUILD_TYPE;
    auto store = py::class_<UC::FSStorePy>(module, "FSStore");
    auto config = py::class_<UC::FSStorePy::Config>(store, "Config");
    config.def(py::init<const std::vector<std::string>&, const size_t, const bool>(),
               py::arg("storageBackends"), py::arg("kvcacheBlockSize"), py::arg("transferEnable"));
    config.def_readwrite("storageBackends", &UC::FSStorePy::Config::storageBackends);
    config.def_readwrite("kvcacheBlockSize", &UC::FSStorePy::Config::kvcacheBlockSize);
    config.def_readwrite("transferEnable", &UC::FSStorePy::Config::transferEnable);
    config.def_readwrite("transferDeviceId", &UC::FSStorePy::Config::transferDeviceId);
    config.def_readwrite("transferStreamNumber", &UC::FSStorePy::Config::transferStreamNumber);
    config.def_readwrite("transferIoSize", &UC::FSStorePy::Config::transferIoSize);
    config.def_readwrite("transferBufferNumber", &UC::FSStorePy::Config::transferBufferNumber);
    store.def(py::init<>());
    store.def("CCStoreImpl", &UC::FSStorePy::CCStoreImpl);
    store.def("Setup", &UC::FSStorePy::Setup);
    store.def("Alloc", py::overload_cast<const std::string&>(&UC::FSStorePy::Alloc));
    store.def("AllocBatch", &UC::FSStorePy::AllocBatch);
    store.def("Lookup", py::overload_cast<const std::string&>(&UC::FSStorePy::Lookup));
    store.def("LookupBatch", &UC::FSStorePy::LookupBatch);
    store.def("LoadToDevice", &UC::FSStorePy::LoadToDevice);
    store.def("LoadToHost", &UC::FSStorePy::LoadToHost);
    store.def("DumpFromDevice", &UC::FSStorePy::DumpFromDevice);
    store.def("DumpFromHost", &UC::FSStorePy::DumpFromHost);
    store.def("Wait", &UC::FSStorePy::Wait);
    store.def("Check", &UC::FSStorePy::CheckPy);
    store.def("Commit", py::overload_cast<const std::string&, const bool>(&UC::FSStorePy::Commit));
    store.def("CommitBatch", &UC::FSStorePy::CommitBatch);
}
