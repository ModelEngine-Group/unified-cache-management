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
#include "ucmlocalstore/ucmlocalstore.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "status/status.h"

namespace py = pybind11;

namespace UCM {

py::list Lookup(const std::list<std::string>& blockIdList)
{
    py::list pyResult;
    std::list<bool> result = LookupBatch(blockIdList);
    for (auto& s : result) {
        pyResult.append(s);
    }
    return pyResult;
}

size_t WriteToDram(const std::list<std::string>& blockIdList, const std::list<uintptr_t>& addressList)
{
    return SubmitWrite(blockIdList, addressList);
}

size_t ReadFromDram(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList,
                    const std::list<uintptr_t>& addressList, const std::list<size_t>& lengthList)
{
    return SubmitRead(blockIdList, addressList, lengthList, offsetList);
}

} // namespace UCM

PYBIND11_MODULE(ucmlocalstore, module)
{
    py::class_<UCM::SetupParam>(module, "SetupParam")
        .def(py::init<const size_t, const size_t>(), py::arg("capacity"), py::arg("cacheSize"))
        .def_readwrite("deviceId", &UCM::SetupParam::deviceId)
        .def_readwrite("ioSize", &UCM::SetupParam::ioSize);
    module.def("Setup", &UCM::Setup);
    module.def("Lookup", &UCM::Lookup);
    module.def("WriteToDram", &UCM::WriteToDram);
    module.def("ReadFromDram", &UCM::ReadFromDram);
    module.def("Wait", &UCM::Wait);
}