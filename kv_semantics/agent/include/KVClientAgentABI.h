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
#ifndef KV_CLIENT_AGENT_ABI_H
#define KV_CLIENT_AGENT_ABI_H

#include <stddef.h>
#include <stdint.h>

#define KV_CLIENT_AGENT_ABI_VERSION 1U

// Stable status codes for the C boundary. Positive values leave zero available
// for success and avoid mixing this ABI with process exit codes.
#define KV_CLIENT_AGENT_OK 0
#define KV_CLIENT_AGENT_INVALID_ARGUMENT 1
#define KV_CLIENT_AGENT_ABI_VERSION_MISMATCH 2
#define KV_CLIENT_AGENT_NOT_INITIALIZED 3
#define KV_CLIENT_AGENT_EMPTY_VIEW 4
#define KV_CLIENT_AGENT_TIMEOUT 5
#define KV_CLIENT_AGENT_INTERNAL_ERROR 6

#if defined(__GNUC__)
#define KV_CLIENT_AGENT_EXPORT __attribute__((visibility("default")))
#else
#define KV_CLIENT_AGENT_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kv_client_agent_handle kv_client_agent_handle;

typedef struct kv_client_agent_config {
    const char* mon_host;
    const char* client_name;
    // Raw CephX secret. The M800 master parses keyring files before crossing
    // this ABI; paths are not accepted here.
    const char* client_key;
    const char* log_dir;
    // Values are seconds; M800ViewServerConfig stores milliseconds, so
    // callers must convert explicitly. Zero selects the agent default.
    int32_t view_refresh_interval;
    int32_t view_fetch_timeout;
} kv_client_agent_config;

typedef struct kv_client_agent_node_view {
    int32_t osd_id;
    int32_t up;
    int32_t in;
    const char** addrs;
    size_t addr_count;
} kv_client_agent_node_view;

KV_CLIENT_AGENT_EXPORT uint32_t kv_client_agent_get_abi_version(void);

KV_CLIENT_AGENT_EXPORT kv_client_agent_handle* kv_client_agent_create(void);

KV_CLIENT_AGENT_EXPORT int32_t kv_client_agent_init(kv_client_agent_handle* handle,
                                                    const kv_client_agent_config* config,
                                                    uint32_t abi_version, size_t config_size);

KV_CLIENT_AGENT_EXPORT int32_t kv_client_agent_start(kv_client_agent_handle* handle);

KV_CLIENT_AGENT_EXPORT int32_t kv_client_agent_stop(kv_client_agent_handle* handle);

KV_CLIENT_AGENT_EXPORT int32_t kv_client_agent_get_epoch(kv_client_agent_handle* handle,
                                                         uint64_t* epoch);

// The caller owns the returned array and every string until the matching free
// function is called with the same handle, pointer, and count.
KV_CLIENT_AGENT_EXPORT int32_t kv_client_agent_get_all_ns_name(kv_client_agent_handle* handle,
                                                               const char*** ns_names,
                                                               size_t* ns_count);

// up and in are requested filter values; the M800 master requests 1/1.
// The caller owns the returned node-view array, its address arrays, and every
// string until the matching free function is called.
KV_CLIENT_AGENT_EXPORT int32_t kv_client_agent_get_view_by_ns(kv_client_agent_handle* handle,
                                                              const char* ns_name, int32_t up,
                                                              int32_t in,
                                                              kv_client_agent_node_view** views,
                                                              size_t* view_count);

KV_CLIENT_AGENT_EXPORT void kv_client_agent_free_ns_names(kv_client_agent_handle* handle,
                                                          const char** ns_names, size_t ns_count);

KV_CLIENT_AGENT_EXPORT void kv_client_agent_free_node_views(kv_client_agent_handle* handle,
                                                            kv_client_agent_node_view* views,
                                                            size_t view_count);

KV_CLIENT_AGENT_EXPORT void kv_client_agent_destroy(kv_client_agent_handle* handle);

#ifdef __cplusplus
}
#endif

#endif
