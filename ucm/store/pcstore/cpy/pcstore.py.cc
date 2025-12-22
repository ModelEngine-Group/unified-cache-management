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
#include "pcstore.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "status/status.h"
#include "thread/latch.h"
#include "thread/thread_pool.h"

namespace py = pybind11;

namespace UC {

class PcStorePy : public PcStore {
    struct LookupCtx {
        std::string block;
        uint8_t* found;
        Latch* waiter;
        std::atomic<int32_t>* status;
    };
    ThreadPool<LookupCtx> lookupService_;

public:
    void* CCStoreImpl() { return this; }
    int32_t SetupPy(const Config& config)
    {
        auto ret = Setup(config);
        if (config.transferEnable || ret != Status::OK().Underlying()) { return ret; }
        auto success = lookupService_.SetNWorker(4)
                           .SetWorkerFn([this](auto& ctx, auto) { OnLookup(ctx); })
                           .SetWorkerTimeoutFn([this](auto& ctx, auto) { OnLookupTimeouted(ctx); },
                                               config.transferTimeoutMs)
                           .Run();
        if (!success) {
            UC_ERROR("Failed to start lookup service.");
            return Status::Error().Underlying();
        }
        return Status::OK().Underlying();
    }
    py::list AllocBatch(const py::list& blocks)
    {
        py::list results;
        for (auto& block : blocks) { results.append(this->Alloc(block.cast<std::string>())); }
        return results;
    }
    std::vector<uint8_t> LookupBatch(const py::list& blocks)
    {
        const auto number = blocks.size();
        const auto ok = Status::OK().Underlying();
        std::atomic<int32_t> status{ok};
        Latch waiter{};
        waiter.Set(number);
        std::vector<uint8_t> founds(number);
        size_t idx = 0;
        for (auto& block : blocks) {
            lookupService_.Push({block.cast<std::string>(), founds.data() + idx, &waiter, &status});
            idx++;
        }
        waiter.Wait();
        const auto ret = status.load();
        if (ret == ok) { return founds; }
        throw std::runtime_error{fmt::format("error({}) when performing LookupBatch", ret)};
    }
    void CommitBatch(const py::list& blocks, const bool success)
    {
        for (auto& block : blocks) { this->Commit(block.cast<std::string>(), success); }
    }
    py::tuple CheckPy(const size_t task)
    {
        auto finish = false;
        auto ret = this->Check(task, finish);
        return py::make_tuple(ret, finish);
    }
    size_t LoadToDevice(const py::list& blockIds, const py::list& addresses)
    {
        return this->SubmitPy(blockIds, addresses, TransTask::Type::LOAD, "PC::S2D");
    }
    size_t DumpFromDevice(const py::list& blockIds, const py::list& addresses)
    {
        return this->SubmitPy(blockIds, addresses, TransTask::Type::DUMP, "PC::D2S");
    }

private:
    size_t SubmitPy(const py::list& blockIds, const py::list& addresses, TransTask::Type&& type,
                    std::string&& brief)
    {
        TransTask task{std::move(type), std::move(brief)};
        auto blockId = blockIds.begin();
        auto address = addresses.begin();
        while ((blockId != blockIds.end()) && (address != addresses.end())) {
            task.Append(blockId->cast<std::string>(), address->cast<uintptr_t>());
            blockId++;
            address++;
        }
        return this->Submit(std::move(task));
    }
    void OnLookup(LookupCtx& ctx)
    {
        const auto ok = Status::OK().Underlying();
        if (ctx.status->load() == ok) { *ctx.found = Lookup(ctx.block); }
        ctx.waiter->Done();
    }
    void OnLookupTimeouted(LookupCtx& ctx)
    {
        auto ok = Status::OK().Underlying();
        auto timeout = Status::Timeout().Underlying();
        ctx.status->compare_exchange_weak(ok, timeout, std::memory_order_acq_rel);
        ctx.waiter->Done();
    }
};

}  // namespace UC

PYBIND11_MODULE(ucmpcstore, module)
{
    module.attr("project") = UCM_PROJECT_NAME;
    module.attr("version") = UCM_PROJECT_VERSION;
    module.attr("commit_id") = UCM_COMMIT_ID;
    module.attr("build_type") = UCM_BUILD_TYPE;
    auto store = py::class_<UC::PcStorePy>(module, "PcStore");
    auto config = py::class_<UC::PcStorePy::Config>(store, "Config");
    config.def(py::init<const std::vector<std::string>&, const size_t, const bool>(),
               py::arg("storageBackends"), py::arg("kvcacheBlockSize"), py::arg("transferEnable"));
    config.def_readwrite("storageBackends", &UC::PcStorePy::Config::storageBackends);
    config.def_readwrite("kvcacheBlockSize", &UC::PcStorePy::Config::kvcacheBlockSize);
    config.def_readwrite("transferEnable", &UC::PcStorePy::Config::transferEnable);
    config.def_readwrite("uniqueId", &UC::PcStorePy::Config::uniqueId);
    config.def_readwrite("transferIoDirect", &UC::PcStorePy::Config::transferIoDirect);
    config.def_readwrite("transferLocalRankSize", &UC::PcStorePy::Config::transferLocalRankSize);
    config.def_readwrite("transferDeviceId", &UC::PcStorePy::Config::transferDeviceId);
    config.def_readwrite("transferStreamNumber", &UC::PcStorePy::Config::transferStreamNumber);
    config.def_readwrite("transferIoSize", &UC::PcStorePy::Config::transferIoSize);
    config.def_readwrite("transferBufferNumber", &UC::PcStorePy::Config::transferBufferNumber);
    config.def_readwrite("transferTimeoutMs", &UC::PcStorePy::Config::transferTimeoutMs);
    config.def_readwrite("transferScatterGatherEnable",
                         &UC::PcStorePy::Config::transferScatterGatherEnable);
    store.def(py::init<>());
    store.def("CCStoreImpl", &UC::PcStorePy::CCStoreImpl);
    store.def("Setup", &UC::PcStorePy::SetupPy);
    store.def("Alloc", py::overload_cast<const std::string&>(&UC::PcStorePy::Alloc));
    store.def("AllocBatch", &UC::PcStorePy::AllocBatch);
    store.def("Lookup", py::overload_cast<const std::string&>(&UC::PcStorePy::Lookup));
    store.def("LookupBatch", &UC::PcStorePy::LookupBatch);
    store.def("LoadToDevice", &UC::PcStorePy::LoadToDevice);
    store.def("DumpFromDevice", &UC::PcStorePy::DumpFromDevice);
    store.def("Wait", &UC::PcStorePy::Wait);
    store.def("Check", &UC::PcStorePy::CheckPy);
    store.def("Commit", py::overload_cast<const std::string&, const bool>(&UC::PcStorePy::Commit));
    store.def("CommitBatch", &UC::PcStorePy::CommitBatch);
}
