/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
#include <chrono>
#include <cstdlib>
#include <new>
#include <thread>
#include "KVClientAgentABI.h"

struct kv_client_agent_handle {
    int placeholder{0};
};

namespace {

int32_t ResultFromEnvironment()
{
    const char* value = std::getenv("M800_FAKE_GET_IN_CODE");
    return value == nullptr ? 0 : static_cast<int32_t>(std::atoi(value));
}

void InitializeOutputs(kv_client_agent_node_view** views, size_t* viewCount)
{
    if (views != nullptr) { *views = nullptr; }
    if (viewCount != nullptr) { *viewCount = 0; }
}

void BlockIfRequested()
{
    if (std::getenv("M800_FAKE_GET_IN_BLOCK") == nullptr) { return; }
    while (std::getenv("M800_FAKE_GET_IN_UNBLOCK") == nullptr) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

}  // namespace

extern "C" {

uint32_t kv_client_agent_get_abi_version(void) { return KV_CLIENT_AGENT_ABI_VERSION; }

kv_client_agent_handle* kv_client_agent_create(void)
{
    return new (std::nothrow) kv_client_agent_handle{};
}

int32_t kv_client_agent_init(kv_client_agent_handle*, const kv_client_agent_config*, uint32_t,
                             size_t)
{
    return KV_CLIENT_AGENT_OK;
}

int32_t kv_client_agent_start(kv_client_agent_handle*) { return KV_CLIENT_AGENT_OK; }
int32_t kv_client_agent_stop(kv_client_agent_handle*)
{
    setenv("M800_FAKE_GET_IN_UNBLOCK", "1", 1);
    return KV_CLIENT_AGENT_OK;
}

int32_t kv_client_agent_get_epoch(kv_client_agent_handle*, uint64_t* epoch)
{
    if (epoch != nullptr) { *epoch = 1; }
    return KV_CLIENT_AGENT_OK;
}

int32_t kv_client_agent_get_all_ns_name(kv_client_agent_handle*, const char*** names, size_t* count)
{
    if (names != nullptr) { *names = nullptr; }
    if (count != nullptr) { *count = 0; }
    return KV_CLIENT_AGENT_OK;
}

int32_t kv_client_agent_get_view_by_ns(kv_client_agent_handle*, const char*, int32_t, int32_t,
                                       kv_client_agent_node_view** views, size_t* viewCount)
{
    InitializeOutputs(views, viewCount);
    return KV_CLIENT_AGENT_OK;
}

int32_t kv_client_agent_get_in_by_ns(kv_client_agent_handle*, const char*,
                                     kv_client_agent_node_view** views, size_t* viewCount)
{
    InitializeOutputs(views, viewCount);
    BlockIfRequested();
    return ResultFromEnvironment();
}

void kv_client_agent_free_ns_names(kv_client_agent_handle*, const char**, size_t) {}
void kv_client_agent_free_node_views(kv_client_agent_handle*, kv_client_agent_node_view*, size_t) {}
void kv_client_agent_destroy(kv_client_agent_handle* handle) { delete handle; }

}  // extern "C"
