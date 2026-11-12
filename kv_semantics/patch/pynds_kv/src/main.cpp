// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#include <cstdint>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "nds_interface.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

static size_t get_buffer_size(const py::buffer_info &info)
{
    return static_cast<size_t>(info.size * info.itemsize);
}

static uint8_t *get_buffer_ptr(const py::buffer &buffer, size_t offset, size_t length)
{
    py::buffer_info info = buffer.request();
    const size_t size = get_buffer_size(info);
    if (offset > size || length > size - offset) {
        throw std::runtime_error("Buffer is too small for offset + length");
    }
    return static_cast<uint8_t *>(info.ptr);
}

static std::optional<std::string> parse_config_path(const py::object &path)
{
    if (path.is_none()) { return std::nullopt; }
    return path.cast<std::string>();
}

// Python flat int（[0, 2^80)）→ key_low(64) + key_high(16)；旧 64-bit key 自然兼容。
static bool split_key80(PyObject *obj, uint64_t &low, uint16_t &high)
{
    if (!PyLong_Check(obj)) { return false; }
    PyObject *zero = PyLong_FromLong(0);
    if (zero == nullptr) {
        PyErr_Clear();
        return false;
    }
    const int ge = PyObject_RichCompareBool(obj, zero, Py_GE);
    Py_DECREF(zero);
    if (ge != 1) {
        PyErr_Clear();
        return false;
    }
    low = PyLong_AsUnsignedLongLongMask(obj);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        return false;
    }
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
    const unsigned long long h = PyLong_AsUnsignedLongLong(hi);
    Py_DECREF(hi);
    if (PyErr_Occurred() || h > 0xFFFF) {
        PyErr_Clear();
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

static void parse_key80_sequence(const py::sequence &seq, std::vector<uint64_t> &lows,
                                 std::vector<uint16_t> &highs)
{
    const size_t count = seq.size();
    lows.resize(count);
    highs.resize(count);
    for (size_t i = 0; i < count; ++i) {
        if (!split_key80(seq[i].ptr(), lows[i], highs[i])) {
            throw std::invalid_argument(
                "block_ids must contain non-negative ints < 2^80 (80-bit block_key: "
                "key_low 64 bits + key_high 16 bits)");
        }
    }
}

template <typename Func>
static int32_t invoke_batch(Func func, py::sequence block_ids,
                            const std::vector<py::buffer> &buffers, std::vector<size_t> offsets,
                            std::vector<size_t> lengths, std::vector<uint32_t> nsids)
{
    std::vector<uint64_t> lows;
    std::vector<uint16_t> highs;
    parse_key80_sequence(block_ids, lows, highs);
    const size_t count = lows.size();
    if (count != buffers.size() || count != offsets.size() || count != lengths.size() ||
        count != nsids.size()) {
        throw std::invalid_argument(
            "block_ids, buffers, offsets, lengths, nsids must have same length");
    }

    std::vector<uint8_t *> ptrs;
    ptrs.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        ptrs.push_back(get_buffer_ptr(buffers[i], offsets[i], lengths[i]));
    }

    py::gil_scoped_release release;
    return func(std::move(lows), std::move(highs), std::move(ptrs), std::move(offsets),
                std::move(lengths), std::move(nsids));
}

PYBIND11_MODULE(pynds_kv, m)
{
    m.doc() = "Pybind11 NDS (Network Data Service) Plugin";

    m.def(
        "init",
        [](py::buffer buffer, size_t length, py::object path_obj) {
            py::buffer_info info = buffer.request();
            if (length > get_buffer_size(info)) {
                throw std::runtime_error("init length is larger than buffer size");
            }
            auto path = parse_config_path(path_obj);
            return NDS::init(info.ptr, length, path ? path->c_str() : nullptr);
        },
        py::arg("buffer"), py::arg("length"), py::arg("nds_config_path") = py::none(),
        "Initialize NDS with a memory buffer and optional nds_config file path.");

    m.def(
        "init_multi",
        [](const std::vector<py::buffer> &buffers, const std::vector<size_t> &lengths,
           py::object path_obj) {
            if (buffers.empty()) { throw std::invalid_argument("buffers must not be empty"); }
            if (buffers.size() != lengths.size()) {
                throw std::invalid_argument("buffers and lengths must have same length");
            }
            std::vector<void *> ptrs;
            std::vector<uint64_t> lengths_u64;
            ptrs.reserve(buffers.size());
            lengths_u64.reserve(lengths.size());
            for (size_t i = 0; i < buffers.size(); ++i) {
                py::buffer_info info = buffers[i].request();
                if (lengths[i] > get_buffer_size(info)) {
                    throw std::runtime_error("init_multi length is larger than buffer size");
                }
                ptrs.push_back(info.ptr);
                lengths_u64.push_back(static_cast<uint64_t>(lengths[i]));
            }
            auto path = parse_config_path(path_obj);
            return NDS::initMulti(ptrs, lengths_u64, path ? path->c_str() : nullptr);
        },
        py::arg("buffers"), py::arg("lengths"), py::arg("nds_config_path") = py::none(),
        "Initialize NDS with multiple memory buffers (each registered as an MR, mr_id in order) "
        "and optional nds_config file path.");

    m.def(
        "is_exists",
        [](py::sequence block_ids, uint32_t nsid) {
            std::vector<uint64_t> lows;
            std::vector<uint16_t> highs;
            parse_key80_sequence(block_ids, lows, highs);
            py::gil_scoped_release release;
            return NDS::isExists(lows, highs, nsid);
        },
        py::arg("blockIds"), py::arg("nsid") = 0,
        "Check existence of blocks (prefix hit count). blockIds are flat ints < 2^80.");

    m.def(
        "get",
        [](py::int_ block_id, py::buffer buffer, size_t offset, size_t length, uint32_t nsid) {
            uint64_t low;
            uint16_t high;
            parse_key80(block_id.ptr(), low, high);
            uint8_t *ptr = get_buffer_ptr(buffer, offset, length);
            py::gil_scoped_release release;
            return NDS::get(low, high, ptr, offset, length, nsid);
        },
        py::arg("block_id"), py::arg("buffer"), py::arg("offset"), py::arg("length"),
        py::arg("nsid"), "Read one block. block_id is a flat int < 2^80.");

    m.def(
        "put",
        [](py::int_ block_id, py::buffer buffer, size_t offset, size_t length, uint32_t nsid) {
            uint64_t low;
            uint16_t high;
            parse_key80(block_id.ptr(), low, high);
            uint8_t *ptr = get_buffer_ptr(buffer, offset, length);
            py::gil_scoped_release release;
            return NDS::put(low, high, ptr, offset, length, nsid);
        },
        py::arg("block_id"), py::arg("buffer"), py::arg("offset"), py::arg("length"),
        py::arg("nsid"), "Write one block. block_id is a flat int < 2^80.");

    m.def(
        "batchGet",
        [](py::sequence block_ids, const std::vector<py::buffer> &buffers,
           std::vector<size_t> offsets, std::vector<size_t> lengths, std::vector<uint32_t> nsids) {
            return invoke_batch(NDS::batchGet, block_ids, buffers, std::move(offsets),
                                std::move(lengths), std::move(nsids));
        },
        py::arg("block_ids"), py::arg("buffers"), py::arg("offsets"), py::arg("lengths"),
        py::arg("nsids"), "Batch get blocks. block_ids are flat ints < 2^80.");

    m.def(
        "batchPut",
        [](py::sequence block_ids, const std::vector<py::buffer> &buffers,
           std::vector<size_t> offsets, std::vector<size_t> lengths, std::vector<uint32_t> nsids) {
            return invoke_batch(NDS::batchPut, block_ids, buffers, std::move(offsets),
                                std::move(lengths), std::move(nsids));
        },
        py::arg("block_ids"), py::arg("buffers"), py::arg("offsets"), py::arg("lengths"),
        py::arg("nsids"), "Batch put blocks. block_ids are flat ints < 2^80.");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
