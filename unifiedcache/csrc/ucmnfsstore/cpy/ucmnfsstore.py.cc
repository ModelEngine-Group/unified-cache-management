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
#include "ucmnfsstore/ucmnfsstore.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace UC {

inline int32_t AllocBatch(const py::list& blockIds)
{
    int32_t ret = 0;
    for (auto id : blockIds) {
        if ((ret = Alloc(id.cast<std::string>())) != 0) { break; }
    }
    return ret;
}

inline py::list LookupBatch(const py::list& blockIds)
{
    py::list founds;
    for (auto id : blockIds) { founds.append(Lookup(id.cast<std::string>())); }
    return founds;
}

inline size_t SubmitTsfTasks(const py::list& blockIdList, const py::list& offsetList, const py::list& addressList,
                             const py::list& lengthList, const TsfTask::Type type, const TsfTask::Location location,
                             const std::string& brief)
{
    std::list<TsfTask> tasks;
    size_t size = 0;
    size_t number = 0;
    auto blockId = blockIdList.begin();
    auto offset = offsetList.begin();
    auto address = addressList.begin();
    auto length = lengthList.begin();
    while ((blockId != blockIdList.end()) && (offset != offsetList.end()) && (address != addressList.end()) &&
           (length != lengthList.end())) {
        tasks.emplace_back(type, location, blockId->cast<std::string>(), offset->cast<size_t>(),
                           address->cast<uintptr_t>(), length->cast<size_t>());
        size += length->cast<size_t>();
        number++;
        blockId++;
        offset++;
        address++;
        length++;
    }
    return Submit(tasks, size, number, brief);
}

inline size_t LoadToDevice(const py::list& blockIdList, const py::list& offsetList, const py::list& addressList,
                           const py::list& lengthList)
{
    return SubmitTsfTasks(blockIdList, offsetList, addressList, lengthList, TsfTask::Type::LOAD,
                          TsfTask::Location::DEVICE, "S2D");
}

inline size_t LoadToHost(const py::list& blockIdList, const py::list& offsetList, const py::list& addressList,
                         const py::list& lengthList)
{
    return SubmitTsfTasks(blockIdList, offsetList, addressList, lengthList, TsfTask::Type::LOAD,
                          TsfTask::Location::HOST, "S2H");
}

inline size_t DumpFromDevice(const py::list& blockIdList, const py::list& offsetList, const py::list& addressList,
                             const py::list& lengthList)
{
    return SubmitTsfTasks(blockIdList, offsetList, addressList, lengthList, TsfTask::Type::DUMP,
                          TsfTask::Location::DEVICE, "D2S");
}

inline size_t DumpFromHost(const py::list& blockIdList, const py::list& offsetList, const py::list& addressList,
                           const py::list& lengthList)
{
    return SubmitTsfTasks(blockIdList, offsetList, addressList, lengthList, TsfTask::Type::DUMP,
                          TsfTask::Location::HOST, "H2S");
}

inline void CommitBatch(const py::list& blockIds, const bool success)
{
    for (auto id : blockIds) { Commit(id.cast<std::string>(), success); }
}

} // namespace UC

PYBIND11_MODULE(ucmnfsstore, module)
{
    py::class_<UC::SetupParam>(module, "SetupParam")
        .def(py::init<const std::vector<std::string>&, const size_t, const bool>(), py::arg("storageBackends"),
             py::arg("kvcacheBlockSize"), py::arg("transferEnable"))
        .def_readwrite("storageBackends", &UC::SetupParam::storageBackends)
        .def_readwrite("kvcacheBlockSize", &UC::SetupParam::kvcacheBlockSize)
        .def_readwrite("transferEnable", &UC::SetupParam::transferEnable)
        .def_readwrite("transferDeviceId", &UC::SetupParam::transferDeviceId)
        .def_readwrite("transferStreamNumber", &UC::SetupParam::transferStreamNumber);
    module.def("Setup", &UC::Setup);
    module.def("Alloc", &UC::AllocBatch);
    module.def("Lookup", &UC::LookupBatch);
    module.def("LoadToDevice", &UC::LoadToDevice);
    module.def("LoadToHost", &UC::LoadToHost);
    module.def("DumpFromDevice", &UC::DumpFromDevice);
    module.def("DumpFromHost", &UC::DumpFromHost);
    module.def("Wait", &UC::Wait);
    module.def("Commit", &UC::CommitBatch);
}
