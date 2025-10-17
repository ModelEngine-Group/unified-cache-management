#pragma once
#include <list>
#include "status/status.h"
#include "tsf_task.h"

namespace UC {

class ITsfTaskQueue {
public:
    virtual ~ITsfTaskQueue() = default;

    virtual Status Setup(int32_t deviceId, size_t bufferSize, size_t bufferNumber,
                         class TsfTaskSet* failureSet, const class SpaceLayout* layout, bool transferUseDirect) = 0;
    virtual void Push(std::list<TsfTask>& tasks) = 0; 
};

} // namespace UC