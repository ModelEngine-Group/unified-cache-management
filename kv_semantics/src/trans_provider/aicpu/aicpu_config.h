#pragma once

#ifdef UCM_ASU_ENABLE_AICPU_PROVIDER
#include <cstdint>
#include <string>
#include <utility>
#include "parser_common.h"
#if __has_include(<hcomm/hcomm_res.h>)
#include <hcomm/hcomm_res.h>
#elif __has_include(<hcomm_res.h>)
#include <hcomm_res.h>
#else
#error "UCM_ASU_ENABLE_AICPU_PROVIDER requires hcomm_res.h"
#endif

#ifndef UCM_ASU_AICPU_USE_STAGED_CHANNEL_API
#define UCM_ASU_AICPU_USE_STAGED_CHANNEL_API 1
#endif

#if UCM_ASU_AICPU_USE_STAGED_CHANNEL_API != 0 && UCM_ASU_AICPU_USE_STAGED_CHANNEL_API != 1
#error "UCM_ASU_AICPU_USE_STAGED_CHANNEL_API must be 0 or 1"
#endif

namespace kv::aicpu_config {
const char* CommProtocolName(CommProtocol protocol);
bool IsUbProtocol(CommProtocol protocol);
const char* CommAddrTypeName(CommAddrType type);
std::pair<std::string, std::string> ResolveLocalEndpointAddress(const TransportConfig& config,
                                                                std::uint32_t logicalDeviceId,
                                                                const std::string& fallback,
                                                                CommProtocol protocol);
std::uint32_t ResolveRemoteDeviceId(const NodeEndpoint* endpoint, std::uint32_t fallback);
Status ResolveProtocol(const TransportConfig& config, CommProtocol& protocol);
#if UCM_ASU_AICPU_USE_STAGED_CHANNEL_API
std::string ResolveStagedOobHost(const TransportConfig& config, const NodeEndpoint* endpoint,
                                 const std::string& remoteIp);
std::uint16_t ResolveStagedOobPort(const TransportConfig& config, const NodeEndpoint* endpoint,
                                   std::uint32_t port);
std::uint32_t ResolveStagedClientId(const TransportConfig& config, const NodeEndpoint* endpoint);
#endif
EndpointLocType ResolveLocType(const TransportConfig& config, const NodeEndpoint* endpoint);
Status FillCommAddr(const std::string& text, CommAddr& out);
std::string ResolveHixlKernelJsonPath(const TransportConfig& config);

}  // namespace kv::aicpu_config
#endif
