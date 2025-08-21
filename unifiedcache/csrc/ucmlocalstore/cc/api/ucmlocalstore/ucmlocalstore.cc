/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#include "ucmlocalstore.h"
#include "logger/logger.h"
#include "status/status.h"
#include "template/singleton.h"
#include "cache/cache_manager.h"

namespace UCM {

void ShowSetupParam(const SetupParam& param)
{
    UCM_INFO("Set UCM::Capacity to {}.", param.capacity);
    UCM_INFO("Set UCM::CacheSize to {}.", param.cacheSize);
    UCM_INFO("Set UCM::DeviceId to {}.", param.deviceId);
    UCM_INFO("Set UCM::IOSize to {}.", param.ioSize);
}

int32_t Setup(const SetupParam& param)
{
    auto status = Singleton<CacheManager>::Instance()->Setup(param.capacity, param.cacheSize, param.deviceId, param.ioSize);
    if (status.Failure()) {
        UCM_ERROR("Failed({}) to setup CacheManager.", status);
        return status.Underlying();
    }
    ShowSetupParam(param);
    return Status::OK().Underlying();
}

std::list<bool> LookupBatch(const std::list<std::string>& blockIdList)
{
    return Singleton<CacheManager>::Instance()->LookupBatch(blockIdList);
}

size_t SubmitRead(const std::list<std::string>& blockIdList, const std::list<uintptr_t> dstList,
        const std::list<size_t> lengthList, const std::list<size_t> offsetList)
{
    return Singleton<CacheManager>::Instance()->SubmitRead(blockIdList, dstList, lengthList, offsetList);
}

size_t SubmitWrite(const std::list<std::string>& blockIdList, const std::list<uintptr_t> srcList)
{
    return Singleton<CacheManager>::Instance()->SubmitWrite(blockIdList, srcList);
}

int32_t Wait(const size_t cacheId) { return Singleton<CacheManager>::Instance()->Wait(cacheId).Underlying(); }

} // namespace UCM
