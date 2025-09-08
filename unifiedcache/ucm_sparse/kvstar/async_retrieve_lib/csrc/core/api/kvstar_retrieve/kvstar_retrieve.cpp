#include <spdlog/fmt/ranges.h>

#include "kvstar_retrieve.h"
#include "status/status.h"
#include "logger/logger.h"
#include "template/singleton.h"
#include "retrieve_task/retrieve_task_manager.h"

namespace KVStar {
SetupParam::SetupParam(const std::vector<int>& cpuNumaIds, const int physicalCorePerNuma, const float allocRatio, const size_t blkRepreSize,
           const DeviceType deviceType, const int totalTpSize, const int localRankId)
        : cpuNumaIds{cpuNumaIds}, physicalCorePerNuma{physicalCorePerNuma}, allocRatio{allocRatio}, blkRepreSize{blkRepreSize}, deviceType{deviceType},
          totalTpSize{totalTpSize}, localRankId{localRankId}
{
    // 根据一些设置, 确定线程池的线程数量和对应绑定的核心id
    int coreNumPerNumaAlloc = static_cast<int>(this->physicalCorePerNuma * this->allocRatio);

    // 清理并准备perNumaCoreIds容器
    this->perNumaCoreIds.clear();
    this->perNumaCoreIds.reserve(this->cpuNumaIds.size());

    // 遍历所有分配给该rank的NUMA节点ID
    for (const int numaId : this->cpuNumaIds) {
        // 计算该NUMA节点的起始核心ID
        int startCoreId = numaId * this->physicalCorePerNuma;

        // 创建一个代表该NUMA节点将要使用的核心ID列表
        std::vector<int> curNumaCoreIdAlloc(coreNumPerNumaAlloc);

        // 生成核心ID序列, 例如 startCoreId=32, coreNumPerNumaAlloc=24 -> 生成 [32, 33, ..., 55]
        std::iota(curNumaCoreIdAlloc.begin(), curNumaCoreIdAlloc.end(), startCoreId);

        // 将这个列表添加到最终结果中
        this->perNumaCoreIds.push_back(curNumaCoreIdAlloc);

        KVSTAR_DEBUG("Alloc core ids {} in numa {}.", curNumaCoreIdAlloc, numaId);
    }

    this->threadNum = static_cast<int>(coreNumPerNumaAlloc * this->cpuNumaIds.size());
    KVSTAR_DEBUG("Successfully configured. Total threads = {}.", this->threadNum);
}


int32_t Setup(const SetupParam& param)
{

    auto status = Singleton<RetrieveTaskManager>::Instance()->Setup(param.threadNum, param.cpuNumaIds, param.perNumaCoreIds);
    if (status.Failure()) {
        KVSTAR_ERROR("Failed({}) to setup RetrieveTaskManager.", status);
        return status.Underlying();
    }
    KVSTAR_DEBUG("Setup RetrieveTaskManager success.");

    return Status::OK().Underlying();
}

int32_t Wait(const size_t taskId) {
    return Singleton<RetrieveTaskManager>::Instance()->Wait(taskId).Underlying();
}


}