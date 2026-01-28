#include "kvstar_retrieve.h"
#include "logger/logger.h"
#include "retrieve_task/retrieve_task_manager.h"
#include "status/status.h"
#include "template/singleton.h"

namespace KVStar {
SetupParam::SetupParam(const std::vector<int>& cpuNumaIds,
                       const std::vector<std::pair<int, int>>& bindInfo,
                       const DeviceType deviceType, const int totalTpSize, const int localRankId)
    : cpuNumaIds{cpuNumaIds},
      bindInfo{bindInfo},
      deviceType{deviceType},
      totalTpSize{totalTpSize},
      localRankId{localRankId}
{
    this->threadNum = this->bindInfo.size();
    UC_DEBUG("Successfully configured. Total threads = {}.", this->threadNum);
}

int32_t Setup(const SetupParam& param)
{
    auto status =
        Singleton<RetrieveTaskManager>::Instance()->Setup(param.threadNum, param.bindInfo);
    if (status.Failure()) {
        UC_ERROR("Failed({}) to setup RetrieveTaskManager.", status.Underlying());
        return status.Underlying();
    }
    UC_DEBUG("Setup RetrieveTaskManager success.");

    return Status::OK().Underlying();
}

int32_t Wait(const size_t taskId)
{
    return Singleton<RetrieveTaskManager>::Instance()->Wait(taskId).Underlying();
}

}  // namespace KVStar
