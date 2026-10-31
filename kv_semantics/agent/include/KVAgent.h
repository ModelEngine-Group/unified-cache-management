#pragma once

#include <stddef.h>
#include <stdint.h>

#include "AgentLogger.h"

#define KV_OSD_AGENT_ABI_VERSION 1U

#define INVALID_AGENT_HANDLE (-20000)

#if defined(__GNUC__)
#define KV_AGENT_EXPORT __attribute__((visibility("default")))
#else
#define KV_AGENT_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kv_agent_handle kv_agent_handle;

typedef struct kv_osd_agent_config {
    uint32_t dpu_id;
    const char *mon_host;
    int32_t osd_id;
    // Raw CephX secret (for example, "AQ...=="), not a keyring file path.
    const char *osd_keyring;
    const char *osd_uuid;
    uint32_t osd_mon_heartbeat_interval;
    uint32_t boot_beacon_delay;
    const char *addrs;
} kv_osd_agent_config;

KV_AGENT_EXPORT uint32_t kv_agent_get_abi_version(void);
KV_AGENT_EXPORT kv_agent_handle *kv_agent_create(void);
KV_AGENT_EXPORT int32_t kv_agent_init(
    kv_agent_handle *agent,
    const kv_osd_agent_config *config,
    uint32_t abi_version,
    size_t config_size,
    const char *log_dir);
KV_AGENT_EXPORT int32_t kv_agent_is_initialized(kv_agent_handle *agent);
// Starts the asynchronous heartbeat timer and returns immediately.
KV_AGENT_EXPORT int32_t kv_agent_start_heartbeat(kv_agent_handle *agent);
// 主动停止发心跳：仅停MOSDBeacon发送并将状态置为down
// 保留通信层与句柄，可再次kv_agent_start_heartbeat恢复
KV_AGENT_EXPORT int32_t kv_agent_stop_heartbeat(kv_agent_handle *agent);
KV_AGENT_EXPORT void kv_agent_destroy(kv_agent_handle *agent);

/* 
* 异步回调API —— agent内部用线程池执行mon命令
* 避免调用方在LWT协作式调用上下文被pthread_cond_wait阻塞
* 回调原型： void cb(int32_t result, void *user_data)
*   - result 操作返回值 0表示成功 负数表示错误码
*   - user_data 调用方传入的不透明指针
* 调用方需保证：
*   - cb非空，user_data可以为null
*   - 回调在agent worker线程内被调用，禁止在其中执行长耗时操作或者反向调用agent的其他API（会与worker现场串行死锁）
*   - 回调里唤醒调用方线程/任务的同步原语由调用方提供（例如LWT信号量）
*/
typedef void (*kv_agent_async_cb)(int32_t result, void *user_data);

KV_AGENT_EXPORT int32_t kv_agent_create_crush_root_async(kv_agent_handle *agent,
    char* kvPoolName, uint32_t nameLen, int32_t* osdIdList, uint32_t listSize,
    kv_agent_async_cb cb, void* user_data);
KV_AGENT_EXPORT int32_t kv_agent_delete_crush_root_async(kv_agent_handle *agent,
    char* kvPoolName, uint32_t nameLen, int32_t* osdIdList, uint32_t listSize,
    kv_agent_async_cb cb, void* user_data);
KV_AGENT_EXPORT int32_t kv_agent_create_ceph_pool_async(kv_agent_handle *agent,
    char* kvPoolName, uint32_t kvPoolLen, char* kvnsName, uint32_t kvnsLen,
    kv_agent_async_cb cb, void* user_data);
KV_AGENT_EXPORT int32_t kv_agent_delete_ceph_pool_async(kv_agent_handle *agent,
    char* kvnsName, uint32_t nameLen,
    kv_agent_async_cb cb, void* user_data);
#ifdef __cplusplus
}
#endif
