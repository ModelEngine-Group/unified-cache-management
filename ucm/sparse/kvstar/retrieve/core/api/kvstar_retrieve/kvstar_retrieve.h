#ifndef KVSTAR_RETRIEVE_CLIB_KVSTAR_RETRIEVE_H
#define KVSTAR_RETRIEVE_CLIB_KVSTAR_RETRIEVE_H

#include <list>
#include <string>
#include <vector>
#include <numeric> // for std::iota
#include "retrieve_task/retrieve_task.h"
#include "retrieve_task/retrieve_task_manager.h"
#include "template/singleton.h"

namespace KVStar {

struct SetupParam {
    std::vector<int> cpuNumaIds;
    std::vector<std::pair<int, int>> bindInfo; // coreId, numaId
    DeviceType deviceType;
    int totalTpSize;
    int localRankId;
    int threadNum;

    SetupParam(const std::vector<int>& cpuNumaIds, const std::vector<std::pair<int, int>>& bindInfo,
               const DeviceType deviceType, const int totalTpSize, const int localRankId);

};

int32_t Setup(const SetupParam& param);

int32_t Wait(const size_t taskId);


} // namespace KVStar



#endif //KVSTAR_RETRIEVE_CLIB_KVSTAR_RETRIEVE_H
