#include "aicpu_config.h"

#ifdef UCM_ASU_ENABLE_AICPU_PROVIDER
#include <arpa/inet.h>
#include <cstdlib>
#include <cstring>
#include <limits>
#include "logger.h"

namespace kv::aicpu_config {
namespace {
constexpr const char* kDefaultAscendHome = "/usr/local/Ascend/cann";
constexpr const char* kHixlKernelJsonSuffix =
    "/opp/built-in/op_impl/aicpu/config/libcann_hixl_kernel.json";
int HexNibble(char value)
{
    if (value >= '0' && value <= '9') { return value - '0'; }
    if (value >= 'a' && value <= 'f') { return value - 'a' + 10; }
    if (value >= 'A' && value <= 'F') { return value - 'A' + 10; }
    return -1;
}

std::string NormalizeEidLiteral(const std::string& text)
{
    std::string input = text;
    if (input.size() > 4U && ToLower(input.substr(0, 4U)) == "eid:") { input = input.substr(4U); }
    std::string normalized;
    normalized.reserve(input.size());
    std::size_t start = 0U;
    if (input.size() > 2U && input[0] == '0' && (input[1] == 'x' || input[1] == 'X')) {
        start = 2U;
    }
    for (std::size_t index = start; index < input.size(); ++index) {
        if (input[index] != ':' && input[index] != '-') { normalized.push_back(input[index]); }
    }
    return normalized;
}

bool TryFillEidCommAddr(const std::string& text, CommAddr& out)
{
    const std::string normalized = NormalizeEidLiteral(text);
    if (normalized.size() != COMM_ADDR_EID_LEN * 2U) { return false; }

    std::uint8_t eid[COMM_ADDR_EID_LEN]{};
    for (std::size_t index = 0U; index < COMM_ADDR_EID_LEN; ++index) {
        const int high = HexNibble(normalized[index * 2U]);
        const int low = HexNibble(normalized[index * 2U + 1U]);
        if (high < 0 || low < 0) { return false; }
        eid[index] = static_cast<std::uint8_t>((high << 4U) | low);
    }

    out.type = COMM_ADDR_TYPE_EID;
    std::memset(out.raws, 0, sizeof(out.raws));
    std::memcpy(out.eid, eid, sizeof(eid));
    return true;
}

}  // namespace

const char* CommProtocolName(CommProtocol protocol)
{
    switch (protocol) {
        case COMM_PROTOCOL_ROCE: return "roce";
        case COMM_PROTOCOL_UBC_TP: return "ubc_tp";
        case COMM_PROTOCOL_UB_MEM: return "ub_mem";
        case COMM_PROTOCOL_UBOE: return "uboe";
        case COMM_PROTOCOL_UBG: return "ubg";
        case COMM_PROTOCOL_UBC_CTP: return "ubc_ctp";
        default: return "unknown";
    }
}

bool IsUbProtocol(CommProtocol protocol)
{
    return protocol == COMM_PROTOCOL_UBC_TP || protocol == COMM_PROTOCOL_UB_MEM ||
           protocol == COMM_PROTOCOL_UBOE || protocol == COMM_PROTOCOL_UBC_CTP ||
           protocol == COMM_PROTOCOL_UBG;
}

const char* CommAddrTypeName(CommAddrType type)
{
    switch (type) {
        case COMM_ADDR_TYPE_IP_V4: return "ipv4";
        case COMM_ADDR_TYPE_IP_V6: return "ipv6";
        case COMM_ADDR_TYPE_ID: return "id";
        case COMM_ADDR_TYPE_EID: return "eid";
        default: return "unknown";
    }
}

std::pair<std::string, std::string> ResolveLocalEndpointAddress(const TransportConfig& config,
                                                                std::uint32_t logicalDeviceId,
                                                                const std::string& fallback,
                                                                CommProtocol protocol)
{
    const bool useEid = protocol == COMM_PROTOCOL_UBG;
    const std::string addressKey = useEid ? "aicpu_local_eid" : "aicpu_local_ip";
    const auto deviceKey = addressKey + "." + std::to_string(logicalDeviceId);
    auto it = config.attrs.find(deviceKey);
    if (it != config.attrs.end() && !it->second.empty()) { return {it->second, deviceKey}; }

    auto configured = GetConfigAttr(config, {addressKey.c_str()});
    if (!configured.empty()) { return {std::move(configured), addressKey}; }
    if (!fallback.empty()) { return {fallback, "localIp_argument"}; }

    configured = GetConfigAttr(config, {"localIp", "local_ip"});
    return {std::move(configured), "localIp"};
}

std::uint32_t ResolveRemoteDeviceId(const NodeEndpoint* endpoint, std::uint32_t fallback)
{
    const auto explicitDevice =
        GetEndpointAttr(endpoint, {"device_id", "deviceId", "remote_device_id"});
    if (!explicitDevice.empty()) { return ParseConfigUint32(explicitDevice, fallback); }
    return fallback;
}

Status ResolveProtocol(const TransportConfig& config, CommProtocol& protocol)
{
    auto value = ToLower(GetConfigAttr(config, {"aicpu_hcomm_protocol"}));
    if (value == "ubg") {
        protocol = COMM_PROTOCOL_UBG;
        return Status::OK();
    }
    if (value == "ub" || value == "ubc_ctp" || value == "ub_ctp") {
        protocol = COMM_PROTOCOL_UBC_CTP;
        return Status::OK();
    }
    return Status::Error(StatusCode::INVALID_ARGUMENT,
                         "AICPUTransProvider: unsupported or missing aicpu_hcomm_protocol '" +
                             value + "'; supported values are UBG and UBC_CTP");
}

#if UCM_ASU_AICPU_USE_STAGED_CHANNEL_API
std::string ResolveStagedOobHost(const TransportConfig& config, const NodeEndpoint* endpoint,
                                 const std::string& remoteIp)
{
    auto host = GetEndpointAttr(endpoint, {"aicpu_staged_oob_host", "staged_oob_host", "oob_host"});
    if (host.empty()) {
        host = GetConfigAttr(config, {"aicpu_staged_oob_host", "staged_oob_host", "oob_host"});
    }
    return host.empty() ? remoteIp : host;
}

std::uint16_t ResolveStagedOobPort(const TransportConfig& config, const NodeEndpoint* endpoint,
                                   std::uint32_t port)
{
    auto value =
        GetEndpointAttr(endpoint, {"aicpu_staged_oob_port", "staged_oob_port", "oob_port"});
    if (value.empty()) {
        value = GetConfigAttr(config, {"aicpu_staged_oob_port", "staged_oob_port", "oob_port"});
    }
    return ParseConfigUint16(value, static_cast<std::uint16_t>(port));
}

std::uint32_t ResolveStagedClientId(const TransportConfig& config, const NodeEndpoint* endpoint)
{
    auto value =
        GetEndpointAttr(endpoint, {"aicpu_staged_client_id", "staged_client_id", "client_id"});
    if (value.empty()) {
        value = GetConfigAttr(config, {"aicpu_staged_client_id", "staged_client_id", "client_id"});
    }
    return ParseConfigUint32(value, static_cast<std::uint32_t>(config.nodeId));
}
#endif

EndpointLocType ResolveLocType(const TransportConfig& config, const NodeEndpoint* endpoint)
{
    auto value = GetEndpointAttr(endpoint, {"endpoint_loc", "placement", "loc"});
    if (value.empty()) { value = GetConfigAttr(config, {"endpoint_loc", "placement", "loc"}); }
    value = ToLower(std::move(value));
    return value == "host" ? ENDPOINT_LOC_TYPE_HOST : ENDPOINT_LOC_TYPE_DEVICE;
}

Status FillCommAddr(const std::string& text, CommAddr& out)
{
    if (TryFillEidCommAddr(text, out)) {
        KV_INFO("AICPUTransProvider: parsed HCOMM endpoint address as EID addr={}", text);
        return Status::OK();
    }
    if (inet_pton(AF_INET, text.c_str(), &out.addr) == 1) {
        out.type = COMM_ADDR_TYPE_IP_V4;
        return Status::OK();
    }
    if (inet_pton(AF_INET6, text.c_str(), &out.addr6) == 1) {
        out.type = COMM_ADDR_TYPE_IP_V6;
        return Status::OK();
    }
    const auto parsed = ParseConfigUint64(text, std::numeric_limits<std::uint64_t>::max());
    if (parsed <= std::numeric_limits<std::uint32_t>::max()) {
        out.type = COMM_ADDR_TYPE_ID;
        out.id = static_cast<std::uint32_t>(parsed);
        return Status::OK();
    }
    return Status::Error(StatusCode::INVALID_ARGUMENT, "invalid hcomm endpoint address: " + text);
}

std::string ResolveHixlKernelJsonPath(const TransportConfig& config)
{
    auto path = GetConfigAttr(config, {"hixl_kernel_json", "aicpu_hixl_kernel_json"});
    if (!path.empty()) { return path; }
    if (const char* env = std::getenv("HIXL_KERNEL_JSON"); env != nullptr && env[0] != '\0') {
        return env;
    }
    const char* ascendHome = std::getenv("ASCEND_HOME_PATH");
    path = (ascendHome == nullptr || ascendHome[0] == '\0') ? kDefaultAscendHome : ascendHome;
    path += kHixlKernelJsonSuffix;
    return path;
}

}  // namespace kv::aicpu_config
#endif
