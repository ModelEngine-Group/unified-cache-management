/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef UNIFIEDCACHE_TRANS_ASCEND_SDMA_DIRECT_COPIER_H
#define UNIFIEDCACHE_TRANS_ASCEND_SDMA_DIRECT_COPIER_H

#include <acl/acl.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "../ffts/ffts_sdma_dispatcher.h"
#include "status/status.h"

namespace UC::Trans {

class AscendSdmaDirectCopier {
    struct InFlightObject {
        std::vector<AscendFftsCopySpec> specs;
        FftsSdmaDispatcher dispatcher;
    };
    struct Lane {
        aclrtStream fftsStream{nullptr};
        std::vector<std::unique_ptr<InFlightObject>> inFlight{};
    };

public:
    AscendSdmaDirectCopier() = default;
    ~AscendSdmaDirectCopier();
    AscendSdmaDirectCopier(const AscendSdmaDirectCopier&) = delete;
    AscendSdmaDirectCopier& operator=(const AscendSdmaDirectCopier&) = delete;

    Status Setup();
    Status WaitEvent(void* event);
    Status SubmitLoadObject(const void* hostDevicePtr, void** devices,
                            const std::vector<size_t>& sizes);
    Status SubmitLoadTask(const std::vector<void*>& hostDevicePtrs,
                          const std::vector<void**>& devices, const std::vector<size_t>& sizes);
    Status SubmitDumpObject(void** devices, void* hostDevicePtr, const std::vector<size_t>& sizes);
    Status SubmitDumpTask(const std::vector<void**>& devices,
                          const std::vector<void*>& hostDevicePtrs,
                          const std::vector<size_t>& sizes);
    Status Synchronize();

private:
    void Cleanup() noexcept;
    Status BuildHostToDeviceSpecs(const void* hostDevicePtr, void** devices,
                                  const std::vector<size_t>& sizes,
                                  std::vector<AscendFftsCopySpec>& specs) const;
    Status BuildDeviceToHostSpecs(void** devices, void* hostDevicePtr,
                                  const std::vector<size_t>& sizes,
                                  std::vector<AscendFftsCopySpec>& specs) const;
    Status LaunchSpecs(std::vector<AscendFftsCopySpec>&& specs, Lane& lane);
    Lane& NextLane();
    static Status AclStatus(aclError ret, const char* expr);

    size_t nextLaneIndex_{0};
    std::vector<Lane> lanes_{};
    bool setup_{false};
};

}  // namespace UC::Trans

#endif
