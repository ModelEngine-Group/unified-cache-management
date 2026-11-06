/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include "status/status.h"
#include "type/types.h"

namespace UC::Trans {
class Event;
}

namespace UC::Detail {

// Every event has been recorded after this task's source reads, on every copy
// stream involved. The consumer must retain the fence until its stream waits
// have executed. This is independent of the Store's Wait/Check completion.
struct DumpSourceFence {
    std::vector<std::shared_ptr<Trans::Event>> events;
    std::vector<uintptr_t> handles;

    std::vector<uintptr_t> NativeHandles() const { return handles; }
};

// Optional interface: do not change StoreV1's ABI or require vendor backends to
// implement fences. nullptr means the caller must use the ordinary Wait path.
class DumpSourceFenceProvider {
public:
    virtual ~DumpSourceFenceProvider() = default;
    virtual Expected<std::shared_ptr<DumpSourceFence>> GetDumpSourceFence(TaskHandle taskId) = 0;
};

}  // namespace UC::Detail
