/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef UNIFIEDCACHE_TRANS_ASCEND_SHARD_IO_AGGREGATOR_H
#define UNIFIEDCACHE_TRANS_ASCEND_SHARD_IO_AGGREGATOR_H

#include <acl/acl.h>
#include <cstdint>
#include <memory>
#include <vector>
#include "../ffts/ffts_sdma_dispatcher.h"
#include "status/status.h"
#include "trans/event.h"

namespace UC::Trans {

struct AscendShardIOAggregatorConfig {
    size_t streamNumber{1};
    size_t pipelineDepth{2};
    uint16_t maxReadyLanes{8};
    size_t objectBytes{0};
    size_t maxFragments{0};
};

class AscendShardIOAggregator {
    struct InFlightObject {
        std::vector<AscendFftsCopySpec> specs;
        FftsSdmaDispatcher dispatcher;
    };
    struct Lane {
        size_t nextSlotIndex{0};
        aclrtStream copyStream{nullptr};
        aclrtStream fftsStream{nullptr};
        std::vector<void*> stagingBuffers{};
        std::vector<aclrtEvent> slotReady{};
        std::vector<aclrtEvent> slotFree{};
        std::vector<std::unique_ptr<InFlightObject>> inFlight{};
    };

public:
    AscendShardIOAggregator() = default;
    ~AscendShardIOAggregator();
    AscendShardIOAggregator(const AscendShardIOAggregator&) = delete;
    AscendShardIOAggregator& operator=(const AscendShardIOAggregator&) = delete;

    Status Setup(const AscendShardIOAggregatorConfig& config);
    Status WaitEvent(const Event& event);
    Status SubmitLoadObject(void* host, void** devices, const std::vector<size_t>& sizes);
    Status SubmitDumpObject(void** devices, void* host, const std::vector<size_t>& sizes);
    Status Synchronize();

private:
    void Cleanup() noexcept;
    Status BuildScatterSpecs(InFlightObject& object, void* staging, void** devices,
                             const std::vector<size_t>& sizes) const;
    Status BuildGatherSpecs(InFlightObject& object, void* staging, void** devices,
                            const std::vector<size_t>& sizes) const;
    Status LaunchFfts(InFlightObject& object, aclrtStream stream) const;

    bool setup_{false};
    size_t streamNumber_{0};
    size_t pipelineDepth_{0};
    uint16_t maxReadyLanes_{0};
    size_t objectBytes_{0};
    size_t maxFragments_{0};
    size_t nextObjectIndex_{0};
    std::vector<Lane> lanes_{};
};

}  // namespace UC::Trans

#endif
