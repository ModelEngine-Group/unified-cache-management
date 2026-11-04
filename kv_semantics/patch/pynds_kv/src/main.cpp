#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <set>
#include <stdexcept>

#include "nds_interface.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace NDS;

static size_t get_buffer_size(const py::buffer_info &info)
{
    return static_cast<size_t>(info.size * info.itemsize);
}

static void check_range(const py::buffer_info &info, size_t offset, size_t length)
{
    size_t buf_size = get_buffer_size(info);

    if (offset > buf_size || length > buf_size - offset) {
        throw std::runtime_error("Buffer is too small for offset + length");
    }
}

// 80-bit block_key 拆分：Python flat int（[0, 2^80)）→ key_low(64) + key_high(16)。
// 所有接口接受 flat int；< 2^64 的旧 64-bit key 天然兼容（key_high 拆出 0）。
static bool split_key80(PyObject *obj, uint64_t &low, uint16_t &high)
{
    if (!PyLong_Check(obj)) {
        return false;
    }
    // 拒绝负数
    PyObject *zero = PyLong_FromLong(0);
    if (zero == nullptr) {
        PyErr_Clear();
        return false;
    }
    int ge = PyObject_RichCompareBool(obj, zero, Py_GE);
    Py_DECREF(zero);
    if (ge != 1) {
        PyErr_Clear();
        return false;
    }
    // 低 64 位（Mask 版本直接截断，不抛溢出）
    low = PyLong_AsUnsignedLongLongMask(obj);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        return false;
    }
    // 高段 = key >> 64，必须在 [0, 0xFFFF]
    PyObject *shift = PyLong_FromLong(64);
    if (shift == nullptr) {
        PyErr_Clear();
        return false;
    }
    PyObject *hi = PyNumber_Rshift(obj, shift);
    Py_DECREF(shift);
    if (hi == nullptr) {
        PyErr_Clear();
        return false;
    }
    unsigned long long h = PyLong_AsUnsignedLongLong(hi);
    Py_DECREF(hi);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        return false;
    }
    if (h > 0xFFFF) {
        return false;
    }
    high = static_cast<uint16_t>(h);
    return true;
}

static void parse_key80(PyObject *obj, uint64_t &low, uint16_t &high)
{
    if (!split_key80(obj, low, high)) {
        throw std::invalid_argument(
            "block_id must be a non-negative int < 2^80 (80-bit block_key: "
            "key_low 64 bits + key_high 16 bits)");
    }
}

static void parse_key80_sequence(const py::sequence &seq,
                                 std::vector<uint64_t> &lows,
                                 std::vector<uint16_t> &highs)
{
    size_t n = seq.size();
    lows.resize(n);
    highs.resize(n);
    for (size_t i = 0; i < n; ++i) {
        if (!split_key80(seq[i].ptr(), lows[i], highs[i])) {
            throw std::invalid_argument(
                "block_ids must contain non-negative ints < 2^80 (80-bit block_key: "
                "key_low 64 bits + key_high 16 bits)");
        }
    }
}

PYBIND11_MODULE(pynds_kv, m) {
    m.doc() = R"pbdoc(
        Pybind11 NDS (Network Data Service) Plugin
    )pbdoc";

    m.def("init", [](py::buffer b, size_t length, py::object path_obj) {
        py::buffer_info info = b.request();
        size_t buf_size = get_buffer_size(info);
        if (length > buf_size) {
            throw std::runtime_error("init length is larger than buffer size");
        }

        std::string path_str;
        const char* path = nullptr;
        if (!path_obj.is_none()) {
            path_str = path_obj.cast<std::string>();
            path = path_str.c_str();
        }

        return init(info.ptr, length, path);
    }, py::arg("buffer"), py::arg("length"), py::arg("nds_config_path") = py::none(),
       "Initialize NDS with a memory buffer and optional nds_config file path.");

    m.def("init_multi", [](const std::vector<py::buffer>& buffers,
                           const std::vector<size_t>& lengths,
                           py::object path_obj) {
        if (buffers.empty()) {
            throw std::invalid_argument("buffers must not be empty");
        }
        if (buffers.size() != lengths.size()) {
            throw std::invalid_argument("buffers and lengths must have same length");
        }

        std::vector<void*> buffers_ptr;
        std::vector<uint64_t> lengths_u64;
        buffers_ptr.reserve(buffers.size());
        lengths_u64.reserve(lengths.size());
        for (size_t i = 0; i < buffers.size(); ++i) {
            py::buffer_info info = buffers[i].request();
            if (lengths[i] > get_buffer_size(info)) {
                throw std::runtime_error("init_multi length is larger than buffer size");
            }
            buffers_ptr.push_back(info.ptr);
            lengths_u64.push_back(static_cast<uint64_t>(lengths[i]));
        }

        std::string path_str;
        const char* path = nullptr;
        if (!path_obj.is_none()) {
            path_str = path_obj.cast<std::string>();
            path = path_str.c_str();
        }

        return initMulti(buffers_ptr, lengths_u64, path);
    }, py::arg("buffers"), py::arg("lengths"), py::arg("nds_config_path") = py::none(),
       "Initialize NDS with multiple memory buffers (each registered as an MR, mr_id in order) and optional nds_config file path.");

    m.def("is_exists", [](py::sequence block_ids, uint32_t nsid) {
        std::vector<uint64_t> lows;
        std::vector<uint16_t> highs;
        parse_key80_sequence(block_ids, lows, highs);
        py::gil_scoped_release release;
        return isExists(lows, highs, nsid);
    }, py::arg("blockIds"), py::arg("nsid") = 0,
       "Check existence of blocks (prefix hit count). blockIds are flat ints < 2^80.");

    m.def("get", [](py::int_ block_id, py::buffer b, size_t offset, size_t length, uint32_t nsid) {
        uint64_t low;
        uint16_t high;
        parse_key80(block_id.ptr(), low, high);
        py::buffer_info info = b.request();
        check_range(info, offset, length);
        py::gil_scoped_release release;
        return get(low, high, static_cast<uint8_t *>(info.ptr), offset, length, nsid);
    }, py::arg("block_id"), py::arg("buffer"), py::arg("offset"), py::arg("length"), py::arg("nsid"));

    m.def("batchGet", [](py::sequence block_ids,
                        const std::vector<py::buffer>& buffers,
                        const std::vector<size_t> offsets,
                        const std::vector<size_t> lengths,
                        const std::vector<uint32_t> &nsids) {
        std::vector<uint64_t> lows;
        std::vector<uint16_t> highs;
        parse_key80_sequence(block_ids, lows, highs);
        if (lows.size() != buffers.size() ||
            lows.size() != offsets.size() ||
            lows.size() != nsids.size() ||
            lows.size() != lengths.size()) {
            throw std::invalid_argument("block_ids, buffers, offsets, nsids, lengths must have same length");
        }

        std::vector<uint8_t *> buffers_ptr;
        buffers_ptr.reserve(lows.size());
        for (size_t i = 0; i < lows.size(); ++i) {
            py::buffer_info info = buffers[i].request();
            check_range(info, offsets[i], lengths[i]);
            buffers_ptr.push_back(static_cast<uint8_t *>(info.ptr));
        }

        py::gil_scoped_release release;
        return batchGet(lows, highs, buffers_ptr, offsets, lengths, nsids);
    }, py::arg("block_ids"), py::arg("buffers"), py::arg("offsets"), py::arg("lengths"), py::arg("nsids"),
       "Batch get blocks. block_ids are flat ints < 2^80.");

    m.def("put", [](py::int_ block_id, py::buffer b, size_t offset, size_t length, uint32_t nsid) {
        uint64_t low;
        uint16_t high;
        parse_key80(block_id.ptr(), low, high);
        py::buffer_info info = b.request();
        check_range(info, offset, length);
        py::gil_scoped_release release;
        return put(low, high, static_cast<uint8_t *>(info.ptr), offset, length, nsid);
    }, py::arg("block_id"), py::arg("buffer"), py::arg("offset"), py::arg("length"), py::arg("nsid"));

    m.def("batchPut", [](py::sequence block_ids,
                        const std::vector<py::buffer>& buffers,
                        const std::vector<size_t> offsets,
                        const std::vector<size_t> lengths,
                        const std::vector<uint32_t> &nsids) {
        std::vector<uint64_t> lows;
        std::vector<uint16_t> highs;
        parse_key80_sequence(block_ids, lows, highs);
        if (lows.size() != buffers.size() ||
            lows.size() != offsets.size() ||
            lows.size() != nsids.size() ||
            lows.size() != lengths.size()) {
            throw std::invalid_argument("block_ids, buffers, offsets, nsids, lengths must have same length");
        }

        std::vector<uint8_t *> buffers_ptr;
        buffers_ptr.reserve(lows.size());
        for (size_t i = 0; i < lows.size(); ++i) {
            py::buffer_info info = buffers[i].request();
            check_range(info, offsets[i], lengths[i]);
            buffers_ptr.push_back(static_cast<uint8_t *>(info.ptr));
        }

        py::gil_scoped_release release;
        return batchPut(lows, highs, buffers_ptr, offsets, lengths, nsids);
    }, py::arg("block_ids"), py::arg("buffers"), py::arg("offsets"), py::arg("lengths"), py::arg("nsids"),
       "Batch put blocks. block_ids are flat ints < 2^80.");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
