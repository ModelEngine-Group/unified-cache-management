/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "status/status.h"
#include "store_assembler.h"
#include "store_scheduler.h"
#include "store_worker.h"
#include "type/block.h"
#include "type/dictionary.h"
#include "type/request.h"
#include "type/token.h"

namespace py = pybind11;

using namespace UC;
using namespace UC::Proxy;

namespace {

class UcmError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

template <typename T>
T Unwrap(Expected<T> result)
{
    if (!result) { throw UcmError(result.Error().ToString()); }
    return std::move(result).Value();
}

void CheckStatus(const Status& status)
{
    if (status.Failure()) { throw UcmError(status.ToString()); }
}

std::vector<std::int64_t> ToInt64Vector(const py::sequence& sequence, const std::string& key)
{
    std::vector<std::int64_t> values;
    values.reserve(sequence.size());
    for (auto element : sequence) {
        if (!py::isinstance<py::int_>(element)) {
            throw py::type_error("expected int elements for key '" + key + "'");
        }
        values.push_back(element.cast<std::int64_t>());
    }
    return values;
}

std::vector<std::string> ToStringVector(const py::sequence& sequence, const std::string& key)
{
    std::vector<std::string> values;
    values.reserve(sequence.size());
    for (auto element : sequence) {
        if (!py::isinstance<py::str>(element)) {
            throw py::type_error("expected str elements for key '" + key + "'");
        }
        values.push_back(element.cast<std::string>());
    }
    return values;
}

Dictionary ToDictionary(const py::dict& source);

std::vector<Dictionary> ToDictionaryVector(const py::sequence& sequence, const std::string& key)
{
    std::vector<Dictionary> values;
    values.reserve(sequence.size());
    for (auto element : sequence) {
        if (!py::isinstance<py::dict>(element)) {
            throw py::type_error("expected dict elements for key '" + key + "'");
        }
        values.push_back(ToDictionary(py::reinterpret_borrow<py::dict>(element)));
    }
    return values;
}

void SetSequence(Dictionary& target, const std::string& key, const py::sequence& sequence)
{
    if (sequence.empty()) {
        target.SetNumbers(key, std::vector<std::int64_t>{});
    } else if (py::isinstance<py::dict>(sequence[0])) {
        target.Set(key, ToDictionaryVector(sequence, key));
    } else if (py::isinstance<py::int_>(sequence[0])) {
        target.SetNumbers(key, ToInt64Vector(sequence, key));
    } else if (py::isinstance<py::str>(sequence[0])) {
        target.Set(key, ToStringVector(sequence, key));
    } else {
        throw py::type_error("unsupported element type for key '" + key + "'");
    }
}

Dictionary ToDictionary(const py::dict& source)
{
    Dictionary target;
    for (auto item : source) {
        if (!py::isinstance<py::str>(item.first)) {
            throw py::type_error("configuration keys must be str");
        }
        const std::string key = item.first.cast<std::string>();
        const py::handle value = item.second;
        if (py::isinstance<py::dict>(value)) {
            target.Set(key, ToDictionary(py::reinterpret_borrow<py::dict>(value)));
        } else if (py::isinstance<py::str>(value)) {
            target.Set(key, value.cast<std::string>());
        } else if (py::isinstance<py::int_>(value)) {
            target.SetInt64(key, value.cast<std::int64_t>());
        } else if (py::isinstance<py::float_>(value)) {
            target.SetDouble(key, value.cast<double>());
        } else if (py::isinstance<py::bytes>(value)) {
            throw py::type_error("unsupported value type for key '" + key + "'");
        } else if (py::isinstance<py::sequence>(value)) {
            SetSequence(target, key, py::reinterpret_borrow<py::sequence>(value));
        } else {
            throw py::type_error("unsupported value type for key '" + key + "'");
        }
    }
    return target;
}

std::string_view AsStringView(const py::bytes& bytes) { return bytes; }

std::vector<BlockId> ToBlockIdVector(const py::bytes& keys)
{
    const std::string_view data = AsStringView(keys);
    if (data.size() % sizeof(BlockId) != 0) {
        throw py::value_error("keys byte length must be a multiple of " +
                              std::to_string(sizeof(BlockId)));
    }
    std::vector<BlockId> result(data.size() / sizeof(BlockId));
    std::memcpy(result.data(), data.data(), data.size());
    return result;
}

}  // namespace

PYBIND11_MODULE(ucm_proxy, m)
{
    py::register_exception<UcmError>(m, "UcmError", PyExc_RuntimeError);

    py::class_<BlockId>(m, "BlockId")
        .def(py::init([](const py::bytes& bytes) {
                 const std::string_view data = AsStringView(bytes);
                 BlockId id;
                 if (data.size() != id.bytes.size()) {
                     throw py::value_error("BlockId requires exactly 16 bytes");
                 }
                 std::memcpy(id.bytes.data(), data.data(), data.size());
                 return id;
             }),
             py::arg("bytes"))
        .def("to_bytes",
             [](const BlockId& id) {
                 return py::bytes(reinterpret_cast<const char*>(id.bytes.data()), id.bytes.size());
             })
        .def(py::self == py::self)
        .def("__hash__", [](const BlockId& id) { return std::hash<BlockId>{}(id); });

    py::class_<RequestContext>(m, "RequestContext")
        .def(py::init<>())
        .def_readwrite("request_id", &RequestContext::requestId)
        .def_property(
            "ttl_ms",
            [](const RequestContext& context) -> std::optional<std::int64_t> {
                if (!context.ttl.has_value()) { return std::nullopt; }
                return context.ttl->count();
            },
            [](RequestContext& context, const std::optional<std::int64_t>& value) {
                if (value.has_value()) {
                    context.ttl = std::chrono::milliseconds(*value);
                } else {
                    context.ttl = std::nullopt;
                }
            })
        .def_readwrite("total_block_count", &RequestContext::totalBlockCount);

    py::class_<AddressedToken>(m, "AddressedToken")
        .def(py::init<>())
        .def_readwrite("keys", &AddressedToken::keys)
        .def_property(
            "payload",
            [](const AddressedToken& token) {
                return py::bytes(reinterpret_cast<const char*>(token.payload.data()),
                                 token.payload.size());
            },
            [](AddressedToken& token, const py::bytes& bytes) {
                const std::string_view data = AsStringView(bytes);
                const auto* first = reinterpret_cast<const std::byte*>(data.data());
                token.payload.assign(first, first + data.size());
            });

    py::class_<BlockPayload>(m, "BlockPayload")
        .def(py::init<>())
        .def_property(
            "pointers",
            [](const BlockPayload& payload) {
                std::vector<std::uintptr_t> pointers;
                pointers.reserve(payload.pointers.size());
                for (void* pointer : payload.pointers) {
                    pointers.push_back(reinterpret_cast<std::uintptr_t>(pointer));
                }
                return pointers;
            },
            [](BlockPayload& payload, const std::vector<std::uintptr_t>& pointers) {
                payload.pointers.clear();
                payload.pointers.reserve(pointers.size());
                for (std::uintptr_t pointer : pointers) {
                    payload.pointers.push_back(reinterpret_cast<void*>(pointer));
                }
            })
        .def_readwrite("sizes", &BlockPayload::sizes)
        .def_readwrite("memory_region_indices", &BlockPayload::memoryRegionIndices)
        .def_readwrite("group_sizes", &BlockPayload::groupSizes);

    py::class_<IoContext>(m, "IoContext")
        .def(py::init<>())
        .def_readwrite("prerequisite_handle", &IoContext::prerequisiteHandle);

    py::class_<IoResult>(m, "IoResult")
        .def(py::init<>())
        .def_readwrite("job_id", &IoResult::jobId)
        .def_readwrite("success", &IoResult::success);

    py::class_<StoreScheduler>(m, "StoreScheduler")
        .def(
            "on_new_request",
            [](StoreScheduler& scheduler, const RequestContext& request) {
                CheckStatus(scheduler.OnNewRequest(request));
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "on_request_finished",
            [](StoreScheduler& scheduler, const RequestId& requestId) {
                scheduler.OnRequestFinished(requestId);
            },
            py::call_guard<py::gil_scoped_release>())
        .def("lookup",
             [](StoreScheduler& scheduler, const RequestId& requestId,
                const py::bytes& keys) -> py::object {
                 const std::vector<BlockId> blockIds = ToBlockIdVector(keys);
                 auto result = [&] {
                     py::gil_scoped_release release;
                     return scheduler.Lookup(requestId, blockIds);
                 }();
                 if (!result) { throw UcmError(result.Error().ToString()); }
                 auto hits = std::move(result).Value();
                 if (!hits.has_value()) { return py::none(); }
                 return py::bytes(reinterpret_cast<const char*>(hits->data()), hits->size());
             })
        .def("lookup_on_prefix",
             [](StoreScheduler& scheduler, const RequestId& requestId, const py::bytes& keys) {
                 const std::vector<BlockId> blockIds = ToBlockIdVector(keys);
                 auto result = [&] {
                     py::gil_scoped_release release;
                     return scheduler.LookupOnPrefix(requestId, blockIds);
                 }();
                 return Unwrap(std::move(result));
             })
        .def("lookup_on_reverse",
             [](StoreScheduler& scheduler, const RequestId& requestId, const py::bytes& keys) {
                 const std::vector<BlockId> blockIds = ToBlockIdVector(keys);
                 auto result = [&] {
                     py::gil_scoped_release release;
                     return scheduler.LookupOnReverse(requestId, blockIds);
                 }();
                 return Unwrap(std::move(result));
             })
        .def("touch",
             [](StoreScheduler& scheduler, const RequestId& requestId, const py::bytes& keys) {
                 const std::vector<BlockId> blockIds = ToBlockIdVector(keys);
                 py::gil_scoped_release release;
                 scheduler.Touch(requestId, blockIds);
             })
        .def(
            "prepare_prefetch",
            [](StoreScheduler& scheduler, const RequestId& requestId) {
                return Unwrap(scheduler.PreparePrefetch(requestId));
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "complete_prefetch",
            [](StoreScheduler& scheduler, const RequestId& requestId, bool success) {
                scheduler.CompletePrefetch(requestId, success);
            },
            py::call_guard<py::gil_scoped_release>())
        .def("prepare_get",
             [](StoreScheduler& scheduler, const RequestId& requestId, const py::bytes& keys) {
                 const std::vector<BlockId> blockIds = ToBlockIdVector(keys);
                 auto result = [&] {
                     py::gil_scoped_release release;
                     return scheduler.PrepareGet(requestId, blockIds);
                 }();
                 return Unwrap(std::move(result));
             })
        .def(
            "complete_get",
            [](StoreScheduler& scheduler, const RequestId& requestId) {
                scheduler.CompleteGet(requestId);
            },
            py::call_guard<py::gil_scoped_release>())
        .def("prepare_put",
             [](StoreScheduler& scheduler, const RequestId& requestId, const py::bytes& keys) {
                 const std::vector<BlockId> blockIds = ToBlockIdVector(keys);
                 auto result = [&] {
                     py::gil_scoped_release release;
                     return scheduler.PreparePut(requestId, blockIds);
                 }();
                 return Unwrap(std::move(result));
             })
        .def(
            "complete_put",
            [](StoreScheduler& scheduler, const RequestId& requestId, bool success) {
                scheduler.CompletePut(requestId, success);
            },
            py::call_guard<py::gil_scoped_release>());

    py::class_<StoreWorker>(m, "StoreWorker")
        .def(
            "submit_prefetch",
            [](StoreWorker& worker, std::size_t jobId, AddressedToken token) {
                CheckStatus(worker.SubmitPrefetch(jobId, std::move(token)));
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "submit_get",
            [](StoreWorker& worker, std::size_t jobId, std::vector<BlockPayload> payloads,
               AddressedToken token, const IoContext& context) {
                CheckStatus(
                    worker.SubmitGet(jobId, std::move(payloads), std::move(token), context));
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "submit_put",
            [](StoreWorker& worker, std::size_t jobId, std::vector<BlockPayload> payloads,
               AddressedToken token, const IoContext& context) {
                CheckStatus(
                    worker.SubmitPut(jobId, std::move(payloads), std::move(token), context));
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "get_finished", [](StoreWorker& worker) { return worker.GetFinished(); },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "wait",
            [](StoreWorker& worker, const std::vector<std::size_t>& jobIds) {
                return worker.Wait(jobIds);
            },
            py::call_guard<py::gil_scoped_release>());

    py::class_<StoreAssembler>(m, "StoreAssembler")
        .def(py::init([](const py::dict& config) {
                 const Dictionary dictionary = ToDictionary(config);
                 py::gil_scoped_release release;
                 auto result = StoreAssembler::Create(dictionary);
                 if (!result) { throw UcmError(result.Error().ToString()); }
                 return std::move(result.Value());
             }),
             py::arg("config"))
        .def("get_scheduler", &StoreAssembler::GetScheduler,
             py::return_value_policy::reference_internal)
        .def("get_worker", &StoreAssembler::GetWorker, py::return_value_policy::reference_internal);
}
