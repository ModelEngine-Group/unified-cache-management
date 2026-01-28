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
#include <pybind11/pybind11.h>
#include "logger.h"

namespace py = pybind11;
namespace UC::Logger {

std::string format(std::string fmt, py::args args)
{
    std::string msg = fmt;
    if (args.size() > 1) {
        py::object formatted = py::str(fmt).format(*args);
        msg = formatted.cast<std::string>();
    }
    return msg;
}

void LogWrapper(Level lv, std::string file, std::string func, int line, std::string msg)
{
    Log(std::move(lv), std::move(file), std::move(func), line, std::move(msg));
}

PYBIND11_MODULE(ucmlogger, m)
{
    m.def("format", &format);
    m.def("setup", &Setup);
    m.def("flush", &Flush);
    m.def("log", &LogWrapper);
    py::enum_<Level>(m, "Level")
        .value("DEBUG", Level::DEBUG)
        .value("INFO", Level::INFO)
        .value("WARNING", Level::WARN)
        .value("ERROR", Level::ERROR);
}

}  // namespace UC::Logger