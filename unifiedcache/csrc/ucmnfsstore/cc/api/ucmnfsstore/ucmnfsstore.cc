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
#include "ucmnfsstore.h"
#include "space/space_manager.h"
#include "template/singleton.h"
#include "logger/logger.h"
#include "status/status.h"

namespace UC {

int32_t Setup(const SetupParam& param)
{
    auto mgr = Singleton<SpaceManager>::Instance();
    auto s = mgr.Setup(param.storageBackends, param.kvcacheBlockSize);
    if (s.Failure()) {
        UC_ERROR("Failed to initialize SpaceManager, error code: {}.", s.Underlying());
    } else {
        UC_INFO("Succeed in initializing SpaceManager.");
    }
    return s.Underlying();
}

int32_t Alloc(const std::string& blockId)
{
    auto mgr = Singleton<SpaceManager>::Instance();
    auto s = mgr.NewBlock(blockId);
    if (s.Failure()) {
        UC_ERROR("Failed to allocate kv cache block space, block id: {}, error code: {}.", blockId, s.Underlying());
    } else {
        UC_INFO("Succeed in allocating kv cache block space, block id: {}.", blockId);
    }
    return s.Underlying();
}

bool Lookup(const std::string& blockId)
{
    auto mgr = Singleton<SpaceManager>::Instance();
    auto ok = mgr.LookupBlock(blockId);
    if (!ok) {
        UC_ERROR("Failed to lookup kv cache block space, block id: {}.", blockId);
    } else {
        UC_INFO("Succeed in looking up kv cache block space.");
    }
    return ok;
}

size_t Submit(std::list<TsfTask> tasks)
{
    return size_t();
}

int32_t Wait(const size_t taskId)
{
    return 0;
}

void Commit(const std::string& blockId, const bool success)
{
    auto mgr = Singleton<SpaceManager>::Instance();
    auto s = mgr.CommitBlock(blockId, success);
    if (s.Failure()) {
        UC_ERROR("Failed to commit kv cache block space, block id: {}, error code: {}.", blockId, s.Underlying());
    } else {
        UC_INFO("Succeed in committing kv cache block space, block id: {}.", blockId);
    }
}

} // namespace UC
