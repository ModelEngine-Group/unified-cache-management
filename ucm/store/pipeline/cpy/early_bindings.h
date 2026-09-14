#ifndef UCM_PIPELINE_EARLY_BINDINGS_H
#define UCM_PIPELINE_EARLY_BINDINGS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include "ucmstore_v1.h"
#include "early_batch.h"

namespace UC::PipelineStore::Early {

struct BoundBatch {
    pybind11::object pipeline;  // keep the exact native store and health wrapper alive
    std::shared_ptr<UC::Early::Batch> batch;
    ~BoundBatch()
    {
        pybind11::gil_scoped_release release;
        batch.reset();  // abandoned accepted I/O drains before pipeline destruction
    }
};

// Narrow access to the native argument conversion; it stays private and is
// shared with the native Load entry instead of duplicating buffer parsing.
struct LoadBinding {
    template <class Pipeline>
    static Detail::TaskDesc Describe(const pybind11::buffer& ids,
                                     const pybind11::buffer& indexes,
                                     const pybind11::buffer& addrs)
    {
        return Pipeline::MakeTaskDesc(ids, indexes, addrs);
    }
};

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
    py::class_<BoundBatch>(module, "EarlyBatch")
        .def_property_readonly("window_bytes", [](const BoundBatch& b) { return b.batch->windowBytes; })
        .def_property_readonly("shard_bytes", [](const BoundBatch& b) { return b.batch->shardBytes; })
        .def_property_readonly("protection_limit_bytes", [](const BoundBatch& b) { return b.batch->protectionLimitBytes; })
        .def_property_readonly("protected_bytes", [](const BoundBatch& b) { return b.batch->ProtectedBytes(); })
        .def("step", [](BoundBatch& b, size_t availableBytes) {
            UC::Early::Progress progress;
            {
                py::gil_scoped_release release;
                progress = ValueOrThrow<NotFoundError, UnhealthyError>(b.batch->Step(availableBytes));
            }
            py::dict out;
            out["done"] = progress.done;
            out["advanced"] = progress.advanced;
            out["can_submit"] = progress.canSubmit;
            out["inflight_bytes"] = progress.inflightBytes;
            out["remaining_l2_work"] = progress.remainingL2Work;
            out["l2_read_shards"] = progress.result.l2ReadShards;
            out["l2_read_bytes"] = progress.result.l2ReadBytes;
            out["coalesced_shards"] = progress.result.coalescedShards;
            return out;
        });
    module.def("EarlyLoad", [](Pipeline& pipeline, const py::buffer& ids,
                                const py::buffer& indexes, const py::buffer& addrs,
                                std::vector<uint64_t> reservations) {
        auto task = LoadBinding::Describe<Pipeline>(ids, indexes, addrs);
        task.loadReservations = std::move(reservations);
        auto* store = reinterpret_cast<StoreV1*>(pipeline.Self());
        py::gil_scoped_release release;
        return ValueOrThrow<NotFoundError, UnhealthyError>(store->Load(std::move(task)));
    }, py::arg("pipeline"), py::arg("ids").noconvert(),
       py::arg("indexes").noconvert(), py::arg("addrs").noconvert(),
       py::arg("reservations"));

    // The synchronous binding keeps the Python Pipeline argument alive. Self()
    // returns its existing entry, including the native health wrapper.
    module.def("EarlyReservePrefix", [](py::object keeper, const py::buffer& ids) {
        auto& pipeline = keeper.cast<Pipeline&>();
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
        try {
            py::dict out;
            out["handle"] = result.handle;
            out["total_hit_blocks"] = result.totalHitBlocks;
            out["remaining_l2_work"] = result.remainingL2Work;
            if (result.batch) {
                auto owned = std::make_unique<BoundBatch>();
                owned->pipeline = keeper;
                owned->batch = std::move(result.batch);
                out["batch"] = std::move(owned);
            }
            return out;
        } catch (...) {
            // No Python caller received ownership. Keep the exact pipeline
            // alive while returning its registered reservation on conversion
            // or allocation failure.
            py::gil_scoped_release release;
            store->ReleaseReservation(result.handle);
            throw;
        }
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
