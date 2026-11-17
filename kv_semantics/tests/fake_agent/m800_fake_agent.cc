#include "KVClientAgentABI.h"

#include <cstdlib>
#include <new>

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
int32_t kv_client_agent_stop(kv_client_agent_handle*) { return KV_CLIENT_AGENT_OK; }

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
    return ResultFromEnvironment();
}

void kv_client_agent_free_ns_names(kv_client_agent_handle*, const char**, size_t) {}
void kv_client_agent_free_node_views(kv_client_agent_handle*, kv_client_agent_node_view*, size_t) {}
void kv_client_agent_destroy(kv_client_agent_handle* handle) { delete handle; }

}  // extern "C"
