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
#include "logger/spdlog/spdlog_logger.h"
#define UC_SOURCE_LOCATION {__FILE__, __FUNCTION__, __LINE__}
namespace py = pybind11;

using SourceLocation = UC::Logger::SourceLocation;
using Level = UC::Logger::Level;

struct SourceLocationWrapper {
    std::string file_str;
    std::string func_str;
    int32_t line;
    
    SourceLocationWrapper(const std::string &file, const std::string &func, int line_val)
        : file_str(file), func_str(func), line(line_val) {}
    
    SourceLocation toSourceLocation() const {
        return SourceLocation{file_str.c_str(), func_str.c_str(), line};
    }
};

class PyLogger {
public:
    PyLogger() {}
    ~PyLogger() {}

    void info(const std::string &fmt, const SourceLocationWrapper &loc_wrapper, py::args args) {
        log(Level::INFO, fmt, loc_wrapper.toSourceLocation(), args);
    }

    void warning(const std::string &fmt, const SourceLocationWrapper &loc_wrapper, py::args args) {
        log(Level::WARN, fmt, loc_wrapper.toSourceLocation(), args);
    }

    void error(const std::string &fmt, const SourceLocationWrapper &loc_wrapper, py::args args) {
        log(Level::ERROR, fmt, loc_wrapper.toSourceLocation(), args);
    }

    void debug(const std::string &fmt, const SourceLocationWrapper &loc_wrapper, py::args args) {
        log(Level::DEBUG, fmt, loc_wrapper.toSourceLocation(), args);
    }

private:
    void log(Level level, const std::string &fmt, SourceLocation loc, py::args args) {
        auto *logger = UC::Logger::Make();
        py::object formatted = py::str(fmt).format(*args);
        std::string msg = formatted.cast<std::string>();
        logger->Log(std::move(level), std::move(loc), std::move(msg));
    }
};

PYBIND11_MODULE(spdlog_logger, m) {
    py::class_<PyLogger>(m, "Logger")
        .def(py::init<>())
        .def("info", &PyLogger::info, py::arg("fmt"), py::arg("loc"))
        .def("warning", &PyLogger::warning, py::arg("fmt"), py::arg("loc"))
        .def("error", &PyLogger::error, py::arg("fmt"), py::arg("loc"))
        .def("debug", &PyLogger::debug, py::arg("fmt"), py::arg("loc"));
}
