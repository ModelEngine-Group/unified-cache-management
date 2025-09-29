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

    int coreNumPerNumaAlloc = static_cast<int>(this->physicalCorePerNuma * this->allocRatio);

    this->perNumaCoreIds.clear();
    this->perNumaCoreIds.reserve(this->cpuNumaIds.size());

    for (const int numaId : this->cpuNumaIds) {
        int startCoreId = numaId * this->physicalCorePerNuma;

        std::vector<int> curNumaCoreIdAlloc(coreNumPerNumaAlloc);

        std::iota(curNumaCoreIdAlloc.begin(), curNumaCoreIdAlloc.end(), startCoreId);

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