
#include <cstdint>
#include <functional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

static constexpr int HASH_SHIFT_LEFT = 6;
static constexpr int HASH_SHIFT_RIGHT = 2;
static constexpr uint64_t INVALID_HASH_VALUE = 0ULL;
static constexpr uint64_t EXTRA_HASH = 0ULL;
static constexpr uint64_t CONST = 0x9e3779b97f4a7c15ULL;

template <class T>
static inline uint64_t HashCombine(uint64_t seed, const T& v)
{
    uint64_t result = seed;
    std::hash<T> hasher;
    uint64_t hv = static_cast<uint64_t>(hasher(v));
    constexpr uint64_t kMul = CONST;
    result ^= hv + kMul + (result << HASH_SHIFT_LEFT) + (result >> HASH_SHIFT_RIGHT);
    if (result == INVALID_HASH_VALUE) { result = 1; }
    return result;
}

template <typename T>
static inline uint64_t ToU64Mask(T v)
{
    using U = std::make_unsigned_t<T>;
    return static_cast<uint64_t>(static_cast<U>(v));
}

static inline uint64_t PyintToU64Mask(const py::handle& obj)
{
    unsigned long long v = PyLong_AsUnsignedLongLongMask(obj.ptr());
    if (PyErr_Occurred()) { throw py::error_already_set(); }
    return static_cast<uint64_t>(v);
}

uint64_t HashBlockU64(py::handle prefix, py::array_t<uint64_t, py::array::c_style> arr)
{
    uint64_t prefix_u = PyintToU64Mask(prefix);

    auto buf = arr.request();
    const uint64_t* p = static_cast<const uint64_t*>(buf.ptr);
    size_t n = static_cast<size_t>(buf.size);

    uint64_t seed = 0;
    if (prefix_u != 0) { seed = HashCombine(seed, prefix_u); }
    for (size_t i = 0; i < n; ++i) { seed = HashCombine(seed, p[i]); }
    seed = HashCombine(seed, 0);
    return seed;
}

template <typename T>
static inline uint64_t HashBlock(uint64_t prefix_hash, const T* p, size_t n)
{
    uint64_t seed = 0;
    if (prefix_hash != 0) { seed = HashCombine(seed, prefix_hash); }
    for (size_t i = 0; i < n; ++i) { seed = HashCombine(seed, ToU64Mask(p[i])); }
    seed = HashCombine(seed, EXTRA_HASH);
    return seed;
}

template <typename T>
py::array_t<uint64_t> HashPrefix(py::handle prefix0, py::array_t<T, py::array::c_style> tokens,
                                 size_t block_size, size_t start_block, size_t end_block)
{
    uint64_t prefix_hash_value = PyintToU64Mask(prefix0);

    auto buf = tokens.request();
    const T* p = static_cast<const T*>(buf.ptr);
    const size_t total = static_cast<size_t>(buf.size);

    if (block_size == 0) { throw std::runtime_error("block_size must be > 0"); }
    if (end_block < start_block) { throw std::runtime_error("end_block must be >= start_block"); }
    if (end_block * block_size > total) { throw std::runtime_error("tokens too short"); }

    const size_t out_n = end_block - start_block;
    py::array_t<uint64_t> out(out_n);
    auto outb = out.request();
    uint64_t* o = static_cast<uint64_t*>(outb.ptr);

    for (size_t b = start_block; b < end_block; ++b) {
        const T* block_ptr = p + b * block_size;
        prefix_hash_value = HashBlock(prefix_hash_value, block_ptr, block_size);
        o[b - start_block] = prefix_hash_value;
    }
    return out;
}

PYBIND11_MODULE(uc_hash_ext, m)
{
    m.doc() = "Fast hash_block compatible with mindie hash_combine",
    m.def("hash_prefix", &HashPrefix<uint64_t>, py::arg("prefix0"), py::arg("tokens"),
          py::arg("block_size"), py::arg("start_block"), py::arg("end_block"));
    m.def("hash_prefix", &HashPrefix<int32_t>, py::arg("prefix0"), py::arg("tokens"),
          py::arg("block_size"), py::arg("start_block"), py::arg("end_block"));
    m.def("hash_prefix", &HashPrefix<int64_t>, py::arg("prefix0"), py::arg("tokens"),
          py::arg("block_size"), py::arg("start_block"), py::arg("end_block"));
}
