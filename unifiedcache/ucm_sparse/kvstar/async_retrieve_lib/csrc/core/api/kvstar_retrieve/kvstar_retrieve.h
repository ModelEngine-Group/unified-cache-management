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

// vLLM每个TP域(Worker进程), 各自有一个检索CLIB实例
struct SetupParam {
    std::vector<int> cpuNumaIds; // 该tp rank能接管的numa id, 例如NUMA_NUM = 8, TP = 2, 那一个rank能分到4个NUMA节点
    int physicalCorePerNuma;
    float allocRatio;
    size_t blkRepreSize;
    DeviceType deviceType;
    int totalTpSize;
    int localRankId;
    std::vector<std::vector<int>> perNumaCoreIds;
    int threadNum;
    // TODO: 按需设置检索引擎的配置项

    SetupParam(const std::vector<int>& cpuNumaIds, const int physicalCorePerNuma, const float allocRatio, const size_t blkRepreSize,
               const DeviceType deviceType, const int totalTpSize, const int localRankId);

};

int32_t Setup(const SetupParam& param);

int32_t Wait(const size_t taskId);


} // namespace KVStar



#endif //KVSTAR_RETRIEVE_CLIB_KVSTAR_RETRIEVE_H