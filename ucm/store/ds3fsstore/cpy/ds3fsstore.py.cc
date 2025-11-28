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
#include "ds3fsstore.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace UC {

class Ds3FsStorePy : public Ds3FsStore {
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
    size_t LoadToDevice(const py::list& blockIds, const py::list& addresses)
    {
        return this->SubmitPy(blockIds, addresses, TransTask::Type::LOAD, "DS3FS::S2D");
    }
    size_t DumpFromDevice(const py::list& blockIds, const py::list& addresses)
    {
        return this->SubmitPy(blockIds, addresses, TransTask::Type::DUMP, "DS3FS::D2S");
    }

private:
    size_t SubmitPy(const py::list& blockIds, const py::list& addresses, TransTask::Type&& type,
                    std::string&& brief)
    {
        TransTask task{std::move(type), std::move(brief)};
        auto blockId = blockIds.begin();
        auto address = addresses.begin();
        while ((blockId != blockIds.end()) && (address != addresses.end())) {
            task.Append(blockId->cast<std::string>(), address->cast<uintptr_t>());
            blockId++;
            address++;
        }
        return this->Submit(std::move(task));
    }
};

} // namespace UC

PYBIND11_MODULE(ucmds3fsstore, module)
{
    module.attr("project") = UCM_PROJECT_NAME;
    module.attr("version") = UCM_PROJECT_VERSION;
    module.attr("commit_id") = UCM_COMMIT_ID;
    module.attr("build_type") = UCM_BUILD_TYPE;
    auto store = py::class_<UC::Ds3FsStorePy>(module, "Ds3FsStore");
    auto config = py::class_<UC::Ds3FsStorePy::Config>(store, "Config");
    config.def(py::init<const std::vector<std::string>&, const size_t, const bool>(),
               py::arg("storageBackends"), py::arg("kvcacheBlockSize"), py::arg("transferEnable"));
    config.def_readwrite("storageBackends", &UC::Ds3FsStorePy::Config::storageBackends);
    config.def_readwrite("kvcacheBlockSize", &UC::Ds3FsStorePy::Config::kvcacheBlockSize);
    config.def_readwrite("transferEnable", &UC::Ds3FsStorePy::Config::transferEnable);
    config.def_readwrite("mountPoint", &UC::Ds3FsStorePy::Config::mountPoint);
    config.def_readwrite("transferIoDirect", &UC::Ds3FsStorePy::Config::transferIoDirect);
    config.def_readwrite("transferLocalRankSize", &UC::Ds3FsStorePy::Config::transferLocalRankSize);
    config.def_readwrite("transferDeviceId", &UC::Ds3FsStorePy::Config::transferDeviceId);
    config.def_readwrite("transferStreamNumber", &UC::Ds3FsStorePy::Config::transferStreamNumber);
    config.def_readwrite("transferIoSize", &UC::Ds3FsStorePy::Config::transferIoSize);
    config.def_readwrite("transferBufferNumber", &UC::Ds3FsStorePy::Config::transferBufferNumber);
    config.def_readwrite("transferTimeoutMs", &UC::Ds3FsStorePy::Config::transferTimeoutMs);
    store.def(py::init<>());
    store.def("CCStoreImpl", &UC::Ds3FsStorePy::CCStoreImpl);
    store.def("Setup", &UC::Ds3FsStorePy::Setup);
    store.def("Alloc", py::overload_cast<const std::string&>(&UC::Ds3FsStorePy::Alloc));
    store.def("AllocBatch", &UC::Ds3FsStorePy::AllocBatch);
    store.def("Lookup", py::overload_cast<const std::string&>(&UC::Ds3FsStorePy::Lookup));
    store.def("LookupBatch", &UC::Ds3FsStorePy::LookupBatch);
    store.def("LoadToDevice", &UC::Ds3FsStorePy::LoadToDevice);
    store.def("DumpFromDevice", &UC::Ds3FsStorePy::DumpFromDevice);
    store.def("Wait", &UC::Ds3FsStorePy::Wait);
    store.def("Check", &UC::Ds3FsStorePy::CheckPy);
    store.def("Commit",
              py::overload_cast<const std::string&, const bool>(&UC::Ds3FsStorePy::Commit));
    store.def("CommitBatch", &UC::Ds3FsStorePy::CommitBatch);
}
