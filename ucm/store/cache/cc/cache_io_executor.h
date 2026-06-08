/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef UNIFIEDCACHE_CACHE_STORE_CC_CACHE_IO_EXECUTOR_H
#define UNIFIEDCACHE_CACHE_STORE_CC_CACHE_IO_EXECUTOR_H

#include <memory>
#include "global_config.h"
#include "status/status.h"

namespace UC::CacheStore {

class CacheIOExecutor {
public:
    virtual ~CacheIOExecutor() = default;
    virtual Status Setup(const Config& config) = 0;
    virtual Status WaitEvent(void* event) = 0;
    virtual Status HostToDevice(void* host, void** devices) = 0;
    virtual Status DeviceToHost(void** devices, void* host) = 0;
    virtual Status Synchronize() = 0;
};

std::unique_ptr<CacheIOExecutor> MakeCacheIOExecutor(const Config& config);

}  // namespace UC::CacheStore

#endif
