#ifndef UCM_PIPELINE_EARLY_BINDINGS_H
#define UCM_PIPELINE_EARLY_BINDINGS_H

#include <pybind11/pybind11.h>
#include <stdexcept>
#include "ucmstore_v1.h"

namespace UC::PipelineStore::Early {

// Reuse the exception types registered by the native pipeline module.
template <class NotFoundError, class UnhealthyError, class T>
T ValueOrThrow(Expected<T> result)
{
    if (result) { return result.Value(); }
    const auto& status = result.Error();
    if (status == Status::NotFound()) { throw NotFoundError{status.ToString()}; }
    if (status == Status::StoreUnhealthy()) { throw UnhealthyError{status.ToString()}; }
    throw std::runtime_error{status.ToString()};
}

template <class Pipeline, class NotFoundError, class UnhealthyError>
void BindOperations(pybind11::module_& module)
{
    namespace py = pybind11;
    // The synchronous binding keeps the Python Pipeline argument alive. Self()
    // returns its existing entry, including the native health wrapper.
    module.def("EarlyReservePrefix", [](Pipeline& pipeline, const py::buffer& ids) {
        const auto info = ids.request(false);
        const auto* blocks = static_cast<const Detail::BlockId*>(info.ptr);
        const auto scale = sizeof(Detail::BlockId) / info.itemsize;
        const auto count = static_cast<size_t>(info.shape[0]) / scale;
        auto* store = reinterpret_cast<StoreV1*>(pipeline.Self());
        PrefixReservation result;
        {
            py::gil_scoped_release release;
            result = ValueOrThrow<NotFoundError, UnhealthyError>(
                store->ReservePrefix(blocks, count));
        }
        py::dict out;
        out["handle"] = result.handle;
        out["total_hit_blocks"] = result.totalHitBlocks;
        out["remaining_l2_work"] = result.remainingL2Work;
        return out;
    }, py::arg("pipeline"), py::arg("ids").noconvert());

    module.def("EarlyPrefetchReserved", [](Pipeline& pipeline, uint64_t handle) {
        auto* store = reinterpret_cast<StoreV1*>(pipeline.Self());
        PrefetchResult result;
        {
            py::gil_scoped_release release;
            result = ValueOrThrow<NotFoundError, UnhealthyError>(
                store->PrefetchReserved(handle));
        }
        py::dict out;
        out["l2_read_shards"] = result.l2ReadShards;
        out["l2_read_bytes"] = result.l2ReadBytes;
        out["coalesced_shards"] = result.coalescedShards;
        return out;
    }, py::arg("pipeline"), py::arg("handle"));

    module.def("EarlyReleaseReservation", [](Pipeline& pipeline, uint64_t handle) {
        auto* store = reinterpret_cast<StoreV1*>(pipeline.Self());
        py::gil_scoped_release release;
        store->ReleaseReservation(handle);
    }, py::arg("pipeline"), py::arg("handle"));
}

}  // namespace UC::PipelineStore::Early
#endif
