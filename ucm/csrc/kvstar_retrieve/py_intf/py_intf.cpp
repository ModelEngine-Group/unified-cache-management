/**
    绕开GIL锁
 *  哪些是“Python 感知”的类型？
        torch::Tensor
        py::object, py::handle, py::str, py::list, py::dict
        任何由 PYBIND11_DECLARE_HOLDER_TYPE 定义的自定义 C++ 对象

    当C++函数签名里只包含C++原生类型或可转换为原生类型的标准库类型时
        int, float, double, bool, size_t
        void*, uintptr_t (这正是传下来data_ptr的情况)
        const char*
        std::string
        std::vector<int> (其中 int 是原生类型)
 *
 * */
#include <torch/extension.h>
#include <vector>
#include <c10/util/Optional.h> // 用于可选参数

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "kvstar_retrieve/kvstar_retrieve.h"
#include "retrieve_task/retrieve_task.h"

namespace py = pybind11;

namespace KVStar {

// NOTE: C++侧Tensor对象本身是个类似智能指针的对象，不用担心按值传递引起数据拷贝;
// 异步提交任务的接口，参数列表包含 blk_repre
inline size_t AsyncRetrieveByCPU(
        const torch::Tensor& queryGroup,
        const torch::Tensor& blkRepre, // Python 每次都把它传下来
        const py::object& dPrunedIndex,
        int topK,
        int reqId,
        DeviceType deviceType
) {
    // 持有GIL, 剥离torch::Tensor对象

    // 1. 创建torch无关的PlainTensor
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

    // 2. 使用torch无关的元数据构造RetrieveTask
    RetrieveTask task(std::move(plainQuery), std::move(plainBlkRepre), std::move(plainPrunedIndex), topK, reqId, deviceType);

    // 3. 提交到后台
    size_t taskId = 0;

    auto status = Singleton<RetrieveTaskManager>::Instance()->SubmitSingleTask(std::move(task), taskId);

    if (status.Failure()) {
        KVSTAR_ERROR("Failed to submit task {}.", taskId);
    }

    return taskId;
}

py::object GetTaskResult(size_t taskId) {
    // 1. 从 TaskManager 获取指向结果容器的共享指针
    std::shared_ptr<TaskResult> result;
    auto status = Singleton<RetrieveTaskManager>::Instance()->GetResult(taskId, result);

    // 2. 如果任务ID不存在，TaskManager 会返回 Failure，我们将其转换为 Python 的 None
    if (status.Failure()) {
        return py::none();
    }

    // 3. 创建一个 Python 字典来存放结果
    py::dict resultDict;

    // 4. 原子地加载当前任务状态
    TaskStatus taskStatus = result->status.load(std::memory_order_relaxed);

    // 5. 根据任务状态，填充字典内容
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
                // 读取非原子成员前，必须加锁，防止数据竞争
                std::lock_guard<std::mutex> lock(result->mtx);
                // NOTE: pybind11 会自动将 std::vector<int64_t> 转换为 Python list
                resultDict["data"] = result->topkIndices;
            }
            break;

        case TaskStatus::FAILURE:
            resultDict["status"] = "FAILURE";
            {
                // 读取非原子成员前，必须加锁
                std::lock_guard<std::mutex> lock(result->mtx);
                resultDict["error"] = result->errorMessage;
            }
            break;
    }

    // 6. 返回填充好的 Python 字典
    return resultDict;
}



} // namespace KVStar

PYBIND11_MODULE(kvstar_retrieve, module)
{
    // 创建一个名为 "DeviceType" 的新 Python 类型
    py::enum_<KVStar::DeviceType>(module, "DeviceType")
    .value("CPU", KVStar::DeviceType::CPU)       // 将 C++ 的值绑定到 Python 中的名字
    .value("GPU", KVStar::DeviceType::GPU)
    .export_values();

    py::class_<KVStar::SetupParam>(module, "SetupParam")
        .def(py::init<const std::vector<int>&,
                      const int,
                      const float,
                      const size_t,
                      const KVStar::DeviceType, // pybind11 现在知道这个参数对应上面绑定的 DeviceType
                      const int,
                      const int>(),
             py::arg("cpuNumaIds"),
             py::arg("physicalCorePerNuma"),
             py::arg("allocRatio"),
             py::arg("blkRepreSize"),
             py::arg("deviceType"),
             py::arg("totalTpSize"),
             py::arg("localRankId"))
        // (可选但推荐) 把成员变量也暴露出来，方便在 Python 中访问
        .def_readwrite("cpuNumaIds", &KVStar::SetupParam::cpuNumaIds)
        .def_readwrite("physicalCorePerNuma", &KVStar::SetupParam::physicalCorePerNuma)
        .def_readwrite("allocRatio", &KVStar::SetupParam::allocRatio)
        .def_readwrite("blkRepreSize", &KVStar::SetupParam::blkRepreSize) // TODO: 后续增加块表征卸载接口, C++侧维护更加CPU计算友好的块表征数据
        .def_readwrite("deviceType", &KVStar::SetupParam::deviceType)
        .def_readwrite("totalTpSize", &KVStar::SetupParam::totalTpSize)
        .def_readwrite("localRankId", &KVStar::SetupParam::localRankId);

    module.def("Setup", &KVStar::Setup);
    module.def("AsyncRetrieveByCPU", &KVStar::AsyncRetrieveByCPU);
    module.def("Wait", &KVStar::Wait);
    module.def("GetTaskResult", &KVStar::GetTaskResult);
}