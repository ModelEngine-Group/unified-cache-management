#include <torch/extension.h>
#include <vector>
#include <c10/util/Optional.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "kvstar_retrieve/kvstar_retrieve.h"
#include "retrieve_task/retrieve_task.h"

namespace py = pybind11;

namespace KVStar {

inline size_t AsyncRetrieveByCPU(
        const torch::Tensor& queryGroup,
        const torch::Tensor& blkRepre,
        const py::object& dPrunedIndex,
        int topK,
        int reqId,
        DeviceType deviceType
) {
    PlainTensor plainQuery, plainBlkRepre;
    std::optional<PlainTensor> plainPrunedIndex;

    plainQuery.data = queryGroup.data_ptr();
    plainQuery.shape.assign(queryGroup.sizes().begin(), queryGroup.sizes().end());
    plainQuery.strides.assign(queryGroup.strides().begin(), queryGroup.strides().end());

    plainBlkRepre.data = blkRepre.data_ptr();
    plainBlkRepre.shape.assign(blkRepre.sizes().begin(), blkRepre.sizes().end());
    plainBlkRepre.strides.assign(blkRepre.strides().begin(), blkRepre.strides().end());

    if (!dPrunedIndex.is_none()) {
        auto pruned_tensor = dPrunedIndex.cast<torch::Tensor>();
        PlainTensor p_index;
        p_index.data = pruned_tensor.data_ptr();
        p_index.shape.assign(pruned_tensor.sizes().begin(), pruned_tensor.sizes().end());
        p_index.strides.assign(pruned_tensor.strides().begin(), pruned_tensor.strides().end());
        plainPrunedIndex = p_index;
    }

    RetrieveTask task(std::move(plainQuery), std::move(plainBlkRepre), std::move(plainPrunedIndex), topK, reqId, deviceType);

    size_t taskId = 0;

    auto status = Singleton<RetrieveTaskManager>::Instance()->SubmitSingleTask(std::move(task), taskId);

    if (status.Failure()) {
        KVSTAR_ERROR("Failed to submit task {}.", taskId);
    }

    return taskId;
}

py::object GetTaskResult(size_t taskId) {
    std::shared_ptr<TaskResult> result;
    auto status = Singleton<RetrieveTaskManager>::Instance()->GetResult(taskId, result);

    if (status.Failure()) {
        return py::none();
    }

    py::dict resultDict;

    TaskStatus taskStatus = result->status.load(std::memory_order_relaxed);

    switch(taskStatus) {
        case TaskStatus::PENDING:
            resultDict["status"] = "PENDING";
            break;

        case TaskStatus::RUNNING:
            resultDict["status"] = "RUNNING";
            break;

        case TaskStatus::SUCCESS:
            resultDict["status"] = "SUCCESS";
            {
                std::lock_guard<std::mutex> lock(result->mtx);
                resultDict["data"] = result->topkIndices;
            }
            break;

        case TaskStatus::FAILURE:
            resultDict["status"] = "FAILURE";
            {
                std::lock_guard<std::mutex> lock(result->mtx);
                resultDict["error"] = result->errorMessage;
            }
            break;
    }

    return resultDict;
}



} // namespace KVStar

PYBIND11_MODULE(kvstar_retrieve, module)
{
    py::enum_<KVStar::DeviceType>(module, "DeviceType")
    .value("CPU", KVStar::DeviceType::CPU)
    .value("GPU", KVStar::DeviceType::GPU)
    .export_values();

    py::class_<KVStar::SetupParam>(module, "SetupParam")
        .def(py::init<const std::vector<int>&,
                      const int,
                      const float,
                      const size_t,
                      const KVStar::DeviceType,
                      const int,
                      const int>(),
             py::arg("cpuNumaIds"),
             py::arg("physicalCorePerNuma"),
             py::arg("allocRatio"),
             py::arg("blkRepreSize"),
             py::arg("deviceType"),
             py::arg("totalTpSize"),
             py::arg("localRankId"))
        .def_readwrite("cpuNumaIds", &KVStar::SetupParam::cpuNumaIds)
        .def_readwrite("physicalCorePerNuma", &KVStar::SetupParam::physicalCorePerNuma)
        .def_readwrite("allocRatio", &KVStar::SetupParam::allocRatio)
        .def_readwrite("blkRepreSize", &KVStar::SetupParam::blkRepreSize)
        .def_readwrite("deviceType", &KVStar::SetupParam::deviceType)
        .def_readwrite("totalTpSize", &KVStar::SetupParam::totalTpSize)
        .def_readwrite("localRankId", &KVStar::SetupParam::localRankId);

    module.def("Setup", &KVStar::Setup);
    module.def("AsyncRetrieveByCPU", &KVStar::AsyncRetrieveByCPU);
    module.def("Wait", &KVStar::Wait);
    module.def("GetTaskResult", &KVStar::GetTaskResult);
}