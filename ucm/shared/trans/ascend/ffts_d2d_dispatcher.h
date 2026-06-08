/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef UNIFIEDCACHE_TRANS_ASCEND_FFTS_D2D_DISPATCHER_H
#define UNIFIEDCACHE_TRANS_ASCEND_FFTS_D2D_DISPATCHER_H

#include <acl/acl.h>
#include <cstdint>
#include <vector>
#include "status/status.h"

#if __has_include("runtime/rt_ffts_plus.h")
#include "runtime/rt_ffts_plus.h"
#elif __has_include("rt_external_ffts.h")
#include "rt_external_ffts.h"
#else
#error "FFTS Plus header was not found. Configure Ascend FFTS include directories in CMake."
#endif

namespace UC::Trans {

struct AscendFftsCopySpec {
    void* dst{nullptr};
    const void* src{nullptr};
    size_t size{0};
};

class FftsD2DDispatcher {
public:
    Status BuildCopies(const std::vector<AscendFftsCopySpec>& copies, uint16_t maxReadyLanes,
                       uint16_t& readyContextNum);
    Status Launch(aclrtStream stream, uint16_t readyContextNum);

private:
    Status AddMemcpy(void* dst, const void* src, size_t size);
    Status AddDependency(uint32_t predecessorId, uint32_t successorId);
    void Reset();
    static void BuildSdmaCtx(void* dst, const void* src, size_t size, rtFftsPlusSdmaCtx_t* ctx);

    std::vector<rtFftsPlusComCtx_t> contexts_{};
    bool completed_{false};
};

}  // namespace UC::Trans

#endif
