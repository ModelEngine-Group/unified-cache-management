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
    int physicalCorePerNuma;
    float allocRatio;
    size_t blkRepreSize;
    DeviceType deviceType;
    int totalTpSize;
    int localRankId;
    std::vector<std::vector<int>> perNumaCoreIds;
    int threadNum;

    SetupParam(const std::vector<int>& cpuNumaIds, const int physicalCorePerNuma, const float allocRatio, const size_t blkRepreSize,
               const DeviceType deviceType, const int totalTpSize, const int localRankId);

};

int32_t Setup(const SetupParam& param);

int32_t Wait(const size_t taskId);


} // namespace KVStar



#endif //KVSTAR_RETRIEVE_CLIB_KVSTAR_RETRIEVE_H