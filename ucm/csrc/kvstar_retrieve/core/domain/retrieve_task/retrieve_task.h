#ifndef UCM_SPARSE_KVSTAR_RETRIEVE_RETRIEVE_TASK_H
#define UCM_SPARSE_KVSTAR_RETRIEVE_RETRIEVE_TASK_H

#include <cstdint>
#include <string>
#include <vector>
#include <optional>
#include "retrieve_task_waiter.h"
#include "computation_task.h"

namespace KVStar {

enum DeviceType {
    CPU = 0,
    NPU,
    GPU,
    TYPE_END
};

struct RetrieveTask {
    PlainTensor queryGroup;
    PlainTensor blkRepre;
    std::optional<PlainTensor> dPrunedIndex;

    int topK;
    int reqId;
    DeviceType deviceType;
    size_t allocTaskId;
    std::shared_ptr<RetrieveTaskWaiter> waiter;

    RetrieveTask(
            PlainTensor qGroup, PlainTensor bRepre, std::optional<PlainTensor> pIndex,
            int tK, int rId, DeviceType devType
    ) : queryGroup(std::move(qGroup)),
        blkRepre(std::move(bRepre)),
        dPrunedIndex(std::move(pIndex)),
        topK(tK),
        reqId(rId),
        deviceType(devType),
        allocTaskId(0) {}

    RetrieveTask() = default;
    RetrieveTask(RetrieveTask&& other) noexcept = default;
    RetrieveTask& operator=(RetrieveTask&& other) noexcept = default;
};

}

#endif //UCM_SPARSE_KVSTAR_RETRIEVE_RETRIEVE_TASK_H