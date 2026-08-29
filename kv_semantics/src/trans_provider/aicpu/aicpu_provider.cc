#include "aicpu_provider.h"

#ifdef UCM_ASU_ENABLE_AICPU_PROVIDER

#if __has_include(<hcomm/hcomm_res.h>)
#include <hcomm/hcomm_res.h>
#include <hcomm/hcomm_primitives.h>
#elif __has_include(<hcomm_res.h>)
#include <hcomm_res.h>
#include <hcomm_primitives.h>
#else
#error "UCM_ASU_ENABLE_AICPU_PROVIDER requires hcomm_res.h and hcomm_primitives.h"
#endif

#include <acl/acl.h>
#include <arpa/inet.h>
#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "logger.h"
#include "ascend/ascend_buffer.h"

#ifndef UCM_ASU_AICPU_USE_STAGED_CHANNEL_API
#define UCM_ASU_AICPU_USE_STAGED_CHANNEL_API 1
#endif

#if UCM_ASU_AICPU_USE_STAGED_CHANNEL_API != 0 && UCM_ASU_AICPU_USE_STAGED_CHANNEL_API != 1
#error "UCM_ASU_AICPU_USE_STAGED_CHANNEL_API must be 0 or 1"
#endif

namespace kv {
namespace {

constexpr std::uint32_t kDefaultNotifyNum = 1;
constexpr std::uint32_t kDefaultUbSqDepth = 0xFFFFFFFFU;
constexpr std::uint32_t kDefaultSendTimeoutMs = 1836U * 1000U;
constexpr std::uint32_t kAclSyncGraceMs = 5000U;
constexpr std::uint32_t kCpuKernelMode = 0U;
constexpr std::uint32_t kKernelBlockDim = 1U;
constexpr std::uintptr_t kHostRegisterAlignment = 4096U;
constexpr const char* kDefaultChannelName = "ucm_asu_aicpu";
constexpr const char* kProviderSignature =
    "UCM_ASU_AICPU_PROVIDER_UBC_CTP_UBG_HCOMM_HIXL_V10";
constexpr const char* kBatchSendKernelName = "HixlBatchSend";
constexpr const char* kDefaultAscendHome = "/usr/local/Ascend/cann";
constexpr const char* kHixlKernelJsonSuffix =
    "/opp/built-in/op_impl/aicpu/config/libcann_hixl_kernel.json";
#if UCM_ASU_AICPU_USE_STAGED_CHANNEL_API
constexpr const char* kChannelApiMode = "HcommChannelCreateStaged";
#else
constexpr const char* kChannelApiMode = "HcommChannelCreate";
#endif

// Local mirror of the HixlBatchSend AICPU launch ABI. It assumes 64-bit handles and pointers:
// UcmHixlSendIoBatch is 32 bytes with imm_data at offset 24, and UcmHixlBatchSendParam is 56 bytes.
// Keep this layout synchronized with hixl_send.h when extending either structure.
struct UcmHixlSendIoBatch {
    ::ChannelHandle channel;
    const void* local_src;
    std::uint64_t len;
    std::uint32_t imm_data;
    std::uint32_t reserved;
};

struct UcmHixlBatchSendParam {
    ::ThreadHandle thread;
    UcmHixlSendIoBatch* io_batches;
    std::uint64_t batch_size;
    std::uint32_t* status_array;
    std::uint32_t timeout_ms;
    void* stats;
    std::uint32_t complete_sender_cqe;
};

static_assert(sizeof(::ChannelHandle) == 8U, "HixlBatchSend requires 64-bit Hcomm handles");
static_assert(sizeof(::ThreadHandle) == 8U, "HixlBatchSend requires 64-bit Hcomm threads");
static_assert(sizeof(UcmHixlSendIoBatch) == 32U, "UcmHixlSendIoBatch ABI changed");
static_assert(offsetof(UcmHixlSendIoBatch, imm_data) == 24U,
              "UcmHixlSendIoBatch IMM offset changed");
static_assert(sizeof(UcmHixlBatchSendParam) == 56U, "UcmHixlBatchSendParam ABI changed");
static_assert(offsetof(UcmHixlBatchSendParam, timeout_ms) == 32U,
              "UcmHixlBatchSendParam timeout offset changed");
static_assert(offsetof(UcmHixlBatchSendParam, stats) == 40U,
              "UcmHixlBatchSendParam stats offset changed");
static_assert(offsetof(UcmHixlBatchSendParam, complete_sender_cqe) == 48U,
              "UcmHixlBatchSendParam completion offset changed");

template <typename T>
auto SetHcommChannelNameIfSupported(T& desc, const char* channelName, int)
    -> decltype((void)(desc.channelName = channelName), void())
{
    desc.channelName = channelName;
}

template <typename T>
void SetHcommChannelNameIfSupported(T&, const char*, ...)
{
}

struct StagedPublishTarget {
    std::string oobHost;
    std::uint16_t oobPort{0};
    std::uint32_t clientId{0};
    std::uint32_t laneToken{0};
};

#if UCM_ASU_AICPU_USE_STAGED_CHANNEL_API
bool operator<(const StagedPublishTarget& lhs, const StagedPublishTarget& rhs)
{
    if (lhs.oobHost != rhs.oobHost) { return lhs.oobHost < rhs.oobHost; }
    if (lhs.oobPort != rhs.oobPort) { return lhs.oobPort < rhs.oobPort; }
    if (lhs.clientId != rhs.clientId) { return lhs.clientId < rhs.clientId; }
    return lhs.laneToken < rhs.laneToken;
}

bool operator==(const StagedPublishTarget& lhs, const StagedPublishTarget& rhs)
{
    return lhs.oobHost == rhs.oobHost && lhs.oobPort == rhs.oobPort &&
           lhs.clientId == rhs.clientId && lhs.laneToken == rhs.laneToken;
}

std::string StagedPublishTargetKey(const StagedPublishTarget& target)
{
    return target.oobHost + ":" + std::to_string(target.oobPort) + ":" +
           std::to_string(target.clientId) + ":" + std::to_string(target.laneToken);
}
#endif

struct ConnectionRecord {
    ::ChannelHandle channel{0};
    ::ThreadHandle thread{0};
    bool hasImmOverride{false};
    std::uint32_t immOverride{0};
#if UCM_ASU_AICPU_USE_STAGED_CHANNEL_API
    HcommStagedChannelInfo stagedInfo{};
#endif
    ServerKvCapabilities serverCapabilities{};
    bool hasServerCapabilities{false};
    std::string stagedOobHost;
    std::uint16_t stagedOobPort{0};
};

struct MemoryRecord {
    MRHandle handle{kInvalidMRHandle};
    HcommMemHandle mem{nullptr};
    EndpointHandle endpoint{nullptr};
    std::string tag;
    std::uintptr_t originalAddr{0};
    std::uintptr_t localAddr{0};
    std::uintptr_t transportAddr{0};
    std::size_t size{0};
    TransProvider::MemType memoryType{TransProvider::MemType::MEM_HOST};
    bool ownsHostMapping{false};
    std::uintptr_t hostMappingBase{0};
    std::uint32_t tokenId{0};
    bool hasToken{false};
    std::uint32_t stagedMrId{0};
    std::unordered_set<std::string> stagedPublicationKeys;
};

Status HcommError(const std::string& op, HcommResult ret, StatusCode code = StatusCode::INTERNAL_ERROR)
{
    auto message = op + " failed ret=" + std::to_string(ret);
    KV_ERROR("AICPUProvider: {}", message);
    return Status::Error(code, std::move(message));
}

Status HcommConnectionError(const std::string& op, HcommResult ret)
{
    return HcommError(op, ret, StatusCode::CONNECTION_ERROR);
}

Status AclError(const std::string& op, aclError ret, StatusCode code = StatusCode::INTERNAL_ERROR)
{
    std::string message = op + " failed ret=" + std::to_string(static_cast<int>(ret));
    if (const char* recent = aclGetRecentErrMsg(); recent != nullptr && recent[0] != '\0') {
        message += " msg=";
        message += recent;
    }
    KV_ERROR("AICPUProvider: {}", message);
    return Status::Error(code, std::move(message));
}

// aclrtSetDevice changes the current ACL context of the calling thread. Preserve an existing
// framework context while the provider temporarily binds the ASU device for HCOMM/HIXL calls.
class AclContextScope final {
public:
    explicit AclContextScope(const char* stage) : stage_(stage)
    {
        const auto ret = aclrtGetCurrentContext(&savedContext_);
        if (ret != ACL_SUCCESS || savedContext_ == nullptr) {
            KV_DEBUG("AICPUProvider: no ACL context to preserve stage={} "
                     "aclrtGetCurrentContext_ret={}",
                     stage_, static_cast<int>(ret));
            savedContext_ = nullptr;
            return;
        }

        const auto getDeviceRet = aclrtGetDevice(&savedDevice_);
        KV_DEBUG("AICPUProvider: preserved caller ACL context stage={} context={} "
                 "device_id={} aclrtGetDevice_ret={}",
                 stage_, static_cast<const void*>(savedContext_), savedDevice_,
                 static_cast<int>(getDeviceRet));
    }

    ~AclContextScope()
    {
        if (savedContext_ == nullptr) { return; }

        aclrtContext currentContext = nullptr;
        const auto getRet = aclrtGetCurrentContext(&currentContext);
        if (getRet == ACL_SUCCESS && currentContext == savedContext_) { return; }

        const auto setRet = aclrtSetCurrentContext(savedContext_);
        if (setRet != ACL_SUCCESS) {
            const char* recent = aclGetRecentErrMsg();
            KV_ERROR("AICPUProvider: failed to restore caller ACL context stage={} "
                     "context={} device_id={} ret={} msg={}",
                     stage_, static_cast<const void*>(savedContext_), savedDevice_,
                     static_cast<int>(setRet), recent == nullptr ? "" : recent);
            return;
        }

        KV_DEBUG("AICPUProvider: restored caller ACL context stage={} context={} "
                 "device_id={} previous_context={} aclrtGetCurrentContext_ret={}",
                 stage_, static_cast<const void*>(savedContext_), savedDevice_,
                 static_cast<const void*>(currentContext), static_cast<int>(getRet));
    }

    AclContextScope(const AclContextScope&) = delete;
    AclContextScope& operator=(const AclContextScope&) = delete;

private:
    const char* stage_{nullptr};
    aclrtContext savedContext_{nullptr};
    std::int32_t savedDevice_{-1};
};

std::string Normalize(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::string GetAttr(const std::unordered_map<std::string, std::string>& attrs,
                    std::initializer_list<const char*> names)
{
    for (const auto* name : names) {
        auto it = attrs.find(name);
        if (it != attrs.end()) { return it->second; }
    }
    return {};
}

std::string GetConfigAttr(const TransportConfig& config, std::initializer_list<const char*> names)
{
    return GetAttr(config.attrs, names);
}

std::string GetEndpointAttr(const NodeEndpoint* endpoint, std::initializer_list<const char*> names)
{
    return endpoint == nullptr ? std::string{} : GetAttr(endpoint->attrs, names);
}

std::uint32_t ParseUint32(std::string value, std::uint32_t fallback)
{
    if (value.empty()) { return fallback; }
    char* end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(value.c_str(), &end, 0);
    if (errno != 0 || end == value.c_str() || *end != '\0' ||
        parsed > std::numeric_limits<std::uint32_t>::max()) {
        return fallback;
    }
    return static_cast<std::uint32_t>(parsed);
}

std::uint64_t ParseUint64(std::string value, std::uint64_t fallback)
{
    if (value.empty()) { return fallback; }
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(value.c_str(), &end, 0);
    if (errno != 0 || end == value.c_str() || *end != '\0') { return fallback; }
    return static_cast<std::uint64_t>(parsed);
}

std::uint16_t ParseUint16(std::string value, std::uint16_t fallback)
{
    const auto parsed = ParseUint32(std::move(value), fallback);
    if (parsed > std::numeric_limits<std::uint16_t>::max()) { return fallback; }
    return static_cast<std::uint16_t>(parsed);
}

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
    if (input.size() > 4U && Normalize(input.substr(0, 4U)) == "eid:") {
        input = input.substr(4U);
    }
    std::string normalized;
    normalized.reserve(input.size());
    std::size_t start = 0U;
    if (input.size() > 2U && input[0] == '0' && (input[1] == 'x' || input[1] == 'X')) {
        start = 2U;
    }
    for (std::size_t index = start; index < input.size(); ++index) {
        if (input[index] != ':' && input[index] != '-') {
            normalized.push_back(input[index]);
        }
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

struct LocalDeviceSelection {
    std::uint32_t deviceId{0};
    aclrtContext context{nullptr};
    std::string source{"default"};
};

LocalDeviceSelection ResolveConfiguredDevice(const TransportConfig& config)
{
    if (config.deviceId >= 0) {
        return {static_cast<std::uint32_t>(config.deviceId), nullptr, "config.deviceId"};
    }
    const auto explicitDevice =
        GetConfigAttr(config, {"device_id", "deviceId", "logical_device_id"});
    if (!explicitDevice.empty()) {
        return {ParseUint32(explicitDevice, 0), nullptr, "config"};
    }

    // endpoint.deviceId belongs to the remote ASU and must not select the local ACL device.
    return {0, nullptr, "default_device_0"};
}

LocalDeviceSelection ResolveLocalDevice(const TransportConfig& config)
{
    auto fallback = ResolveConfiguredDevice(config);

    std::int32_t currentDevice = -1;
    const auto deviceRet = aclrtGetDevice(&currentDevice);
    if (deviceRet != ACL_SUCCESS || currentDevice < 0) {
        KV_WARN("AICPUProvider: current ACL logical device unavailable ret={} device_id={}; "
                "using fallback logical_device_id={} source={}",
                static_cast<int>(deviceRet), currentDevice, fallback.deviceId, fallback.source);
        fallback.source = "current_acl_fallback_" + fallback.source;
        return fallback;
    }

    aclrtContext currentContext = nullptr;
    const auto contextRet = aclrtGetCurrentContext(&currentContext);
    if (contextRet != ACL_SUCCESS) {
        KV_WARN("AICPUProvider: current ACL context unavailable for logical_device_id={} "
                "ret={}; device binding will use aclrtSetDevice",
                currentDevice, static_cast<int>(contextRet));
        currentContext = nullptr;
    }
    return {static_cast<std::uint32_t>(currentDevice), currentContext, "current_acl"};
}

std::pair<std::string, std::string> ResolveLocalEndpointAddress(
    const TransportConfig& config, std::uint32_t logicalDeviceId, const std::string& fallback,
    CommProtocol protocol)
{
    const bool useEid = protocol == COMM_PROTOCOL_UBG;
    const std::string addressKey = useEid ? "aicpu_local_eid" : "aicpu_local_ip";
    const auto deviceKey = addressKey + "." + std::to_string(logicalDeviceId);
    auto it = config.attrs.find(deviceKey);
    if (it != config.attrs.end() && !it->second.empty()) {
        return {it->second, deviceKey};
    }

    auto configured = GetConfigAttr(config, {addressKey.c_str()});
    if (!configured.empty()) { return {std::move(configured), addressKey}; }
    if (!fallback.empty()) { return {fallback, "localIp_argument"}; }

    configured = GetConfigAttr(config, {"localIp", "local_ip"});
    return {std::move(configured), "localIp"};
}

std::uint32_t ResolveRemoteDeviceId(const NodeEndpoint* endpoint, std::uint32_t fallback)
{
    const auto explicitDevice = GetEndpointAttr(endpoint, {"device_id", "deviceId", "remote_device_id"});
    if (!explicitDevice.empty()) { return ParseUint32(explicitDevice, fallback); }
    return fallback;
}

Status ResolveProtocol(const TransportConfig& config, const NodeEndpoint* endpoint,
                       CommProtocol& protocol)
{
    auto value = GetConfigAttr(config, {"aicpu_hcomm_protocol", "hcomm_protocol"});
    if (value.empty()) {
        value = GetEndpointAttr(endpoint, {"aicpu_hcomm_protocol", "hcomm_protocol", "protocol"});
    }
    value = Normalize(std::move(value));
    if (value == "ubg") {
        protocol = COMM_PROTOCOL_UBG;
        return Status::OK();
    }
    if (value == "ub" || value == "ubc_ctp" || value == "ub_ctp") {
        protocol = COMM_PROTOCOL_UBC_CTP;
        return Status::OK();
    }
    return Status::Error(
        StatusCode::INVALID_ARGUMENT,
        "AICPUProvider: unsupported or missing HCOMM protocol '" + value +
            "'; supported values are UBG and UBC_CTP");
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
    auto value = GetEndpointAttr(endpoint, {"aicpu_staged_oob_port", "staged_oob_port", "oob_port"});
    if (value.empty()) {
        value = GetConfigAttr(config, {"aicpu_staged_oob_port", "staged_oob_port", "oob_port"});
    }
    return ParseUint16(value, static_cast<std::uint16_t>(port));
}

std::uint32_t ResolveStagedClientId(const TransportConfig& config, const NodeEndpoint* endpoint)
{
    auto value = GetEndpointAttr(endpoint, {"aicpu_staged_client_id", "staged_client_id", "client_id"});
    if (value.empty()) {
        value = GetConfigAttr(config, {"aicpu_staged_client_id", "staged_client_id", "client_id"});
    }
    return ParseUint32(value, static_cast<std::uint32_t>(config.nodeId));
}
#endif

EndpointLocType ResolveLocType(const TransportConfig& config, const NodeEndpoint* endpoint)
{
    auto value = GetEndpointAttr(endpoint, {"endpoint_loc", "placement", "loc"});
    if (value.empty()) { value = GetConfigAttr(config, {"endpoint_loc", "placement", "loc"}); }
    value = Normalize(std::move(value));
    return value == "host" ? ENDPOINT_LOC_TYPE_HOST : ENDPOINT_LOC_TYPE_DEVICE;
}

Status FillCommAddr(const std::string& text, CommAddr& out)
{
    if (TryFillEidCommAddr(text, out)) {
        KV_INFO("AICPUProvider: parsed HCOMM endpoint address as EID addr={}", text);
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
    char* end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(text.c_str(), &end, 0);
    if (errno == 0 && end != text.c_str() && *end == '\0' &&
        parsed <= std::numeric_limits<std::uint32_t>::max()) {
        out.type = COMM_ADDR_TYPE_ID;
        out.id = static_cast<std::uint32_t>(parsed);
        return Status::OK();
    }
    return Status::Error(StatusCode::INVALID_ARGUMENT, "invalid hcomm endpoint address: " + text);
}

Status BuildEndpointDesc(const TransportConfig& config, const NodeEndpoint* endpoint,
                         const std::string& addr, std::uint32_t deviceId, CommProtocol protocol,
                         const char* role, EndpointDesc& out)
{
    const auto init = EndpointDescInit(&out, 1U);
    if (init != 0) { return HcommError("EndpointDescInit", init); }

    out.protocol = protocol;
    auto status = FillCommAddr(addr, out.commAddr);
    if (!status.ok()) { return status; }
    if (protocol == COMM_PROTOCOL_UBG && out.commAddr.type != COMM_ADDR_TYPE_EID) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             std::string("AICPUProvider: UBG ") + role +
                                 " endpoint must use an EID: " + addr);
    }
    KV_DEBUG("AICPUProvider: EndpointDesc resolved role={} addr={} addr_type={} "
             "device_id={} protocol={}",
             role, addr, CommAddrTypeName(out.commAddr.type), deviceId,
             CommProtocolName(protocol));

    out.loc.locType = ResolveLocType(config, endpoint);
    if (out.loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {
        out.loc.device.devPhyId = deviceId;
        out.loc.device.superDevId =
            ParseUint32(GetEndpointAttr(endpoint, {"super_device_id", "superDevId"}), 0);
        out.loc.device.serverIdx =
            ParseUint32(GetEndpointAttr(endpoint, {"server_idx", "serverIdx"}), 0);
        out.loc.device.superPodIdx =
            ParseUint32(GetEndpointAttr(endpoint, {"super_pod_idx", "superPodIdx"}), 0);
    } else {
        out.loc.host.id = ParseUint32(GetEndpointAttr(endpoint, {"host_id", "hostId"}), 0);
    }
    return Status::OK();
}

CommMemType ToHcommMemType(TransProvider::MemType type)
{
    return type == TransProvider::MemType::MEM_HOST ? COMM_MEM_TYPE_HOST : COMM_MEM_TYPE_DEVICE;
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

int32_t MakeAclSyncTimeoutMs(std::uint32_t hcommTimeoutMs)
{
    const std::uint32_t effective =
        hcommTimeoutMs == 0U ? kDefaultSendTimeoutMs : hcommTimeoutMs;
    const std::uint64_t timeout = static_cast<std::uint64_t>(effective) + kAclSyncGraceMs;
    const auto max = static_cast<std::uint64_t>(std::numeric_limits<int32_t>::max());
    return static_cast<int32_t>(timeout > max ? max : timeout);
}

std::uint32_t MakeKernelTimeoutSeconds(std::uint32_t hcommTimeoutMs)
{
    return static_cast<std::uint32_t>(MakeAclSyncTimeoutMs(hcommTimeoutMs) / 1000) + 1U;
}

struct DeviceAllocation {
    void* ptr{nullptr};

    DeviceAllocation() = default;
    explicit DeviceAllocation(void* value) : ptr(value) {}
    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;
    ~DeviceAllocation()
    {
        if (ptr != nullptr) { (void)aclrtFree(ptr); }
    }
};

ConnectionRecord* ToConnectionRecord(TransProvider::ConnectionHandle handle)
{
    return static_cast<ConnectionRecord*>(handle);
}

}  // namespace

struct AICPUProvider::Impl {
    struct HostMapping {
        std::size_t size{0};
        std::uintptr_t deviceAddr{0};
        std::size_t refCount{0};
    };

    explicit Impl(const TransportConfig& configIn)
        : config(configIn),
          notifyNum(ParseUint32(GetConfigAttr(configIn, {"aicpu_notify_num", "notify_num"}),
                                kDefaultNotifyNum)),
          ubSqDepth(ParseUint32(GetConfigAttr(configIn, {"aicpu_ub_sq_depth", "ub_sq_depth"}),
                                kDefaultUbSqDepth)),
          qos(ParseUint32(GetConfigAttr(configIn, {"aicpu_qos", "qos"}), 0)),
          sendTimeoutMs(ParseUint32(GetConfigAttr(configIn, {"aicpu_send_timeout_ms", "send_timeout_ms",
                                                            "timeout"}),
                                    kDefaultSendTimeoutMs)),
          channelName(GetConfigAttr(configIn, {"aicpu_channel_name", "channel_name"})),
          hixlKernelJsonPath(ResolveHixlKernelJsonPath(configIn)),
          stagedKato(ParseUint32(GetConfigAttr(configIn, {"aicpu_staged_kato", "staged_kato"}), 0)),
          stagedRmUasid(ParseUint32(GetConfigAttr(configIn, {"aicpu_staged_rm_uasid",
                                                             "staged_rm_uasid"}),
                                    0)),
          stagedMamiTag(ParseUint64(GetConfigAttr(configIn, {"aicpu_staged_mami_tag",
                                                             "staged_mami_tag"}),
                                    0))
    {
        auto selection = ResolveLocalDevice(configIn);
        localDeviceId = selection.deviceId;
        providerContext = selection.context;
        deviceSelectionSource = std::move(selection.source);
        if (channelName.empty()) { channelName = kDefaultChannelName; }
    }

    ~Impl()
    {
        AclContextScope contextScope("AICPUProvider cleanup");
        if (hixlBin != nullptr || stream != nullptr || endpoint != nullptr) {
            const auto deviceStatus = EnsureAclDeviceBound("AICPUProvider cleanup");
            if (!deviceStatus.ok()) {
                KV_WARN("AICPUProvider: cleanup continuing after device bind failure: {}",
                        deviceStatus.message);
            }
        }
        if (hixlBin != nullptr) {
            (void)aclrtBinaryUnLoad(hixlBin);
            hixlBin = nullptr;
        }
        if (stream != nullptr) {
            (void)aclrtDestroyStream(stream);
            stream = nullptr;
        }
        if (endpoint != nullptr) {
            (void)HcommEndpointDestroy(endpoint);
            endpoint = nullptr;
        }
    }

    const NodeEndpoint* FindEndpoint(const std::string& remoteIp, std::uint32_t port) const
    {
        auto byAddr = std::find_if(config.endpoints.begin(), config.endpoints.end(),
                                   [&](const NodeEndpoint& endpoint) {
                                       return endpoint.ip == remoteIp && endpoint.port == port;
                                   });
        if (byAddr != config.endpoints.end()) { return &*byAddr; }

        byAddr = std::find_if(config.endpoints.begin(), config.endpoints.end(),
                              [&](const NodeEndpoint& endpoint) {
                                  return endpoint.ip == remoteIp;
                              });
        return byAddr == config.endpoints.end() ? nullptr : &*byAddr;
    }

    Status RefreshLocalDeviceFromCaller()
    {
        std::int32_t currentDevice = -1;
        const auto deviceRet = aclrtGetDevice(&currentDevice);
        if (deviceRet != ACL_SUCCESS || currentDevice < 0) {
            std::lock_guard<std::mutex> lock(mu);
            KV_WARN("AICPUProvider: caller ACL device unavailable before "
                    "CreateConnection ret={} device_id={}; keeping logical_device_id={} "
                    "source={} and binding it explicitly",
                    static_cast<int>(deviceRet), currentDevice, localDeviceId,
                    deviceSelectionSource);
            if (endpoint == nullptr && connections.empty()) {
                providerContext = nullptr;
                deviceSelectionSource = "current_acl_create_connection_fallback";
            }
            return Status::OK();
        }

        aclrtContext currentContext = nullptr;
        const auto contextRet = aclrtGetCurrentContext(&currentContext);
        if (contextRet != ACL_SUCCESS) {
            KV_WARN("AICPUProvider: cannot capture caller ACL context before "
                    "CreateConnection logical_device_id={} ret={}; device binding will use "
                    "aclrtSetDevice",
                    currentDevice, static_cast<int>(contextRet));
            currentContext = nullptr;
        }

        std::lock_guard<std::mutex> lock(mu);
        if (endpoint != nullptr || !connections.empty()) {
            if (localDeviceId != static_cast<std::uint32_t>(currentDevice)) {
                return Status::Error(
                    StatusCode::INVALID_ARGUMENT,
                    "AICPUProvider: caller logical device changed after endpoint creation");
            }
            return Status::OK();
        }

        KV_INFO("AICPUProvider: refreshed Provider ACL binding at CreateConnection "
                "old_logical_device_id={} new_logical_device_id={} old_context={} "
                "new_context={}",
                localDeviceId, currentDevice, static_cast<const void*>(providerContext),
                static_cast<const void*>(currentContext));
        localDeviceId = static_cast<std::uint32_t>(currentDevice);
        providerContext = currentContext;
        deviceSelectionSource = "current_acl_create_connection";
        return Status::OK();
    }

    Status EnsureEndpointLocked(const std::string& localIp, CommProtocol protocol)
    {
        if (endpoint != nullptr) {
            if (endpointProtocol != protocol) {
                KV_ERROR("AICPUProvider: mixed hcomm protocols are not supported "
                         "existing_protocol={} requested_protocol={} endpoint_ip={}",
                         CommProtocolName(endpointProtocol), CommProtocolName(protocol), endpointIp);
                return Status::Error(StatusCode::INVALID_ARGUMENT,
                                     "AICPUProvider: mixed hcomm protocols are not supported");
            }
            KV_DEBUG("AICPUProvider: reusing HCOMM endpoint local_addr={} protocol={}",
                     endpointIp, CommProtocolName(protocol));
            return Status::OK();
        }

        auto [resolvedLocalIp, addressSource] =
            ResolveLocalEndpointAddress(config, localDeviceId, localIp, protocol);
        if (resolvedLocalIp.empty()) {
            const char* addressKind = protocol == COMM_PROTOCOL_UBG ? "UBG EID" : "UBC_CTP IP";
            KV_ERROR("AICPUProvider: local {} is required for hcomm endpoint "
                     "logical_device_id={}", addressKind, localDeviceId);
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 std::string("AICPUProvider: local ") + addressKind +
                                     " is required for hcomm endpoint");
        }

        KV_INFO("AICPUProvider: creating HCOMM endpoint local_addr={} address_source={} "
                "logical_device_id={} device_source={} context={} protocol={}",
                resolvedLocalIp, addressSource, localDeviceId, deviceSelectionSource,
                static_cast<const void*>(providerContext), CommProtocolName(protocol));
        EndpointDesc localDesc{};
        auto status = BuildEndpointDesc(config, nullptr, resolvedLocalIp, localDeviceId, protocol,
                                        "local", localDesc);
        if (!status.ok()) {
            KV_ERROR("AICPUProvider: local EndpointDesc build failed local_addr={} "
                     "local_device_id={} protocol={} message={}",
                     resolvedLocalIp, localDeviceId, CommProtocolName(protocol), status.message);
            return status;
        }

        EndpointHandle created = nullptr;
        const auto ret = HcommEndpointCreate(&localDesc, &created);
        if (ret != 0) { return HcommConnectionError("HcommEndpointCreate", ret); }

        endpoint = created;
        endpointIp = resolvedLocalIp;
        endpointProtocol = protocol;
        KV_INFO("AICPUProvider: HCOMM endpoint created local_addr={} local_device_id={} "
                "protocol={}",
                endpointIp, localDeviceId, CommProtocolName(endpointProtocol));
        return Status::OK();
    }

    // HCOMM resolves UB runtime resources through the current thread's ACL device.
    Status EnsureAclDeviceBound(const char* stage)
    {
        if (localDeviceId > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "AICPUProvider: invalid local device id " +
                                     std::to_string(localDeviceId));
        }

        const auto deviceId = static_cast<std::int32_t>(localDeviceId);
        if (providerContext != nullptr) {
            aclrtContext currentContext = nullptr;
            const auto getContextRet = aclrtGetCurrentContext(&currentContext);
            if (getContextRet == ACL_SUCCESS && currentContext == providerContext) {
                return Status::OK();
            }

            const auto setContextRet = aclrtSetCurrentContext(providerContext);
            if (setContextRet != ACL_SUCCESS) {
                return AclError(std::string("aclrtSetCurrentContext before ") + stage +
                                    " logical_device_id=" + std::to_string(deviceId),
                                setContextRet);
            }

            std::int32_t currentDevice = -1;
            const auto getDeviceRet = aclrtGetDevice(&currentDevice);
            if (getDeviceRet != ACL_SUCCESS || currentDevice != deviceId) {
                return Status::Error(
                    StatusCode::INTERNAL_ERROR,
                    "AICPUProvider: captured ACL context resolved unexpected logical device " +
                        std::to_string(currentDevice) + " expected " + std::to_string(deviceId));
            }
            KV_INFO("AICPUProvider: bound captured ACL context stage={} context={} "
                    "logical_device_id={} previous_context={} aclrtGetCurrentContext_ret={}",
                    stage, static_cast<const void*>(providerContext), deviceId,
                    static_cast<const void*>(currentContext), static_cast<int>(getContextRet));
            return Status::OK();
        }

        thread_local std::int32_t readyDeviceId = -1;

        std::int32_t currentDevice = -1;
        if (readyDeviceId == deviceId) {
            const auto getRet = aclrtGetDevice(&currentDevice);
            if (getRet == ACL_SUCCESS && currentDevice == deviceId) { return Status::OK(); }
        }

        const auto initRet = aclInit(nullptr);
        if (initRet != ACL_SUCCESS && initRet != ACL_ERROR_REPEAT_INITIALIZE) {
            return AclError(std::string("aclInit before ") + stage, initRet);
        }

        currentDevice = -1;
        const auto getRet = aclrtGetDevice(&currentDevice);
        if (getRet == ACL_SUCCESS && currentDevice == deviceId) {
            readyDeviceId = deviceId;
            return Status::OK();
        }

        const auto setRet = aclrtSetDevice(deviceId);
        if (setRet != ACL_SUCCESS) {
            return AclError(std::string("aclrtSetDevice before ") + stage +
                                " device_id=" + std::to_string(deviceId),
                            setRet);
        }

        KV_INFO("AICPUProvider: aclrtSetDevice bound logical_device_id={} stage={} "
                "previous_device={} aclrtGetDevice_ret={} aclInit_ret={}",
                deviceId, stage, currentDevice, static_cast<int>(getRet),
                static_cast<int>(initRet));
        readyDeviceId = deviceId;
        return Status::OK();
    }

    Status UpdateChannelMemory(ConnectionRecord& connection, const MemoryRecord& record)
    {
        if (connection.channel == 0U || record.mem == nullptr) { return Status::OK(); }

        HcommMemHandle memHandle = record.mem;
#if UCM_ASU_AICPU_USE_STAGED_CHANNEL_API
        KV_INFO("AICPUProvider: HcommChannelUpdateStagedLocalMemInfo begin "
                "tag={} staged_mr_id={} channel={}",
                record.tag, record.stagedMrId, connection.channel);
        const auto ret =
            HcommChannelUpdateStagedLocalMemInfo(&memHandle, 1U, connection.channel);
        constexpr const char* operation = "HcommChannelUpdateStagedLocalMemInfo";
#else
        KV_INFO("AICPUProvider: HcommChannelUpdateMemInfo begin tag={} channel={}",
                record.tag, connection.channel);
        const auto ret = HcommChannelUpdateMemInfo(&memHandle, 1U, connection.channel);
        constexpr const char* operation = "HcommChannelUpdateMemInfo";
#endif
        return ret == 0 ? Status::OK() : HcommConnectionError(operation, ret);
    }

    Status PublishMemoryStaged(
        MemoryRecord& record,
        const std::vector<StagedPublishTarget>& additionalTargets = {})
    {
#if !UCM_ASU_AICPU_USE_STAGED_CHANNEL_API
        // Standard Channel API mode assumes HCOMM owns peer MR exchange together with
        // connection negotiation. Do not publish a duplicate Staged MR_REGISTER frame.
        (void)record;
        (void)additionalTargets;
        return Status::OK();
#else
        std::vector<StagedPublishTarget> targets;
        {
            std::lock_guard<std::mutex> lock(mu);
            for (auto* conn : connections) {
                if (conn != nullptr) {
                    targets.push_back(
                        {conn->stagedOobHost, conn->stagedOobPort, conn->stagedInfo.clientId,
                         conn->stagedInfo.controllerId});
                }
            }
        }
        targets.insert(targets.end(), additionalTargets.begin(), additionalTargets.end());
        std::sort(targets.begin(), targets.end());
        targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
        targets.erase(
            std::remove_if(targets.begin(), targets.end(),
                           [&](const StagedPublishTarget& target) {
                               return target.oobHost.empty() || target.oobPort == 0U ||
                                      target.clientId == 0U || target.laneToken == 0U;
                           }),
            targets.end());
        if (targets.empty()) {
            KV_DEBUG("AICPUProvider: defer staged MR publication tag={} until a "
                     "connection target is available",
                     record.tag);
            return Status::OK();
        }
        targets.erase(
            std::remove_if(targets.begin(), targets.end(),
                           [&](const StagedPublishTarget& target) {
                               return record.stagedPublicationKeys.count(
                                          StagedPublishTargetKey(target)) != 0U;
                           }),
            targets.end());
        if (targets.empty()) { return Status::OK(); }

        HcommStagedMrDesc mr{};
        mr.memHandle = record.mem;
        mr.mrId = record.stagedMrId;

        for (const auto& target : targets) {
            KV_INFO("AICPUProvider: publishing staged MR tag={} mr_id={} original_addr={} "
                    "transport_addr={} size={} oob={}:{} client_id={} lane_token={}",
                    record.tag, record.stagedMrId, record.originalAddr, record.transportAddr,
                    record.size, target.oobHost, target.oobPort, target.clientId,
                    target.laneToken);

            HcommStagedMrPublishDesc publish{};
            const auto initRet = HcommStagedMrPublishDescInit(&publish, 1U);
            if (initRet != 0) { return HcommError("HcommStagedMrPublishDescInit", initRet); }
            publish.oobHost = target.oobHost.c_str();
            publish.oobPort = target.oobPort;
            publish.timeoutMs = sendTimeoutMs;
            publish.clientId = target.clientId;
            publish.requestId = target.laneToken;
            publish.mrNum = 1U;
            publish.mrs = &mr;

            const auto ret = HcommMemPublishStaged(record.endpoint, &publish);
            if (ret != 0) { return HcommConnectionError("HcommMemPublishStaged", ret); }
            record.stagedPublicationKeys.insert(StagedPublishTargetKey(target));
        }
        KV_INFO("AICPUProvider: staged MR published tag={} mr_id={} original_addr={} "
                "transport_addr={} size={} token_id={} has_token={} client_count={}",
                record.tag, record.stagedMrId, record.originalAddr, record.transportAddr,
                record.size, record.tokenId, record.hasToken ? 1 : 0, targets.size());
        return Status::OK();
#endif
    }

    Status AttachExistingMemoriesToConnection(ConnectionRecord& connection)
    {
        std::vector<MemoryRecord*> records;
        {
            std::lock_guard<std::mutex> lock(mu);
            records.reserve(memories.size());
            for (const auto& item : memories) { records.push_back(item.second.get()); }
        }

        for (auto* record : records) {
            if (record == nullptr) { continue; }
            auto status = UpdateChannelMemory(connection, *record);
            if (!status.ok()) { return status; }
        }

#if UCM_ASU_AICPU_USE_STAGED_CHANNEL_API
        const std::vector<StagedPublishTarget> targets{
            {connection.stagedOobHost, connection.stagedOobPort,
             connection.stagedInfo.clientId, connection.stagedInfo.controllerId}};
        for (auto* record : records) {
            if (record == nullptr) { continue; }
            auto status = PublishMemoryStaged(*record, targets);
            if (!status.ok()) { return status; }
        }
#endif
        return Status::OK();
    }

    Status AcquireHostMapping(std::uintptr_t hostAddr, std::size_t size,
                              std::uintptr_t& mappingBase, std::uintptr_t& deviceAddr)
    {
        mappingBase = hostAddr & ~(kHostRegisterAlignment - 1U);
        const auto offset = hostAddr - mappingBase;
        if (size > std::numeric_limits<std::size_t>::max() - offset) {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "AICPUProvider: aligned host mapping size overflows");
        }
        const auto mappingSize = size + static_cast<std::size_t>(offset);

        std::lock_guard<std::mutex> lock(hostMappingMu);
        auto iter = hostMappings.find(mappingBase);
        if (iter != hostMappings.end()) {
            if (iter->second.size < mappingSize) {
                return Status::Error(
                    StatusCode::BUFFER_NOT_SUPPORTED,
                    "AICPUProvider: host mapping overlaps a smaller active mapping");
            }
            ++iter->second.refCount;
            deviceAddr = iter->second.deviceAddr + offset;
            KV_INFO("AICPUProvider: reused host mapping host_addr={} mapping_base={} "
                    "device_addr={} mapping_size={} request_size={} ref_count={}",
                    hostAddr, mappingBase, deviceAddr, iter->second.size, size,
                    iter->second.refCount);
            return Status::OK();
        }

        void* mapped = nullptr;
        const auto mapStatus = runtime::RegisterHostBuffer(
            reinterpret_cast<void*>(mappingBase), mappingSize, &mapped);
        if (!mapStatus.ok() || mapped == nullptr) {
            return Status::Error(StatusCode::BUFFER_NOT_SUPPORTED,
                                 "AICPUProvider: host mapping failed status=" +
                                     mapStatus.message);
        }

        const auto mappedBase = reinterpret_cast<std::uintptr_t>(mapped);
        deviceAddr = mappedBase + offset;
        hostMappings.emplace(mappingBase, HostMapping{mappingSize, mappedBase, 1});
        KV_INFO("AICPUProvider: created host mapping host_addr={} mapping_base={} "
                "mapped_base={} device_addr={} mapping_size={} request_size={} device_id={}",
                hostAddr, mappingBase, mappedBase, deviceAddr, mappingSize, size, localDeviceId);
        return Status::OK();
    }

    void ReleaseHostMapping(std::uintptr_t mappingBase)
    {
        std::lock_guard<std::mutex> lock(hostMappingMu);
        auto iter = hostMappings.find(mappingBase);
        if (iter == hostMappings.end()) {
            KV_WARN("AICPUProvider: host mapping release missed mapping_base={}",
                    mappingBase);
            return;
        }
        if (--iter->second.refCount != 0) {
            KV_INFO("AICPUProvider: retained host mapping mapping_base={} mapped_base={} "
                    "ref_count={}",
                    mappingBase, iter->second.deviceAddr, iter->second.refCount);
            return;
        }
        runtime::UnregisterHostBuffer(reinterpret_cast<void*>(mappingBase));
        KV_INFO("AICPUProvider: released host mapping mapping_base={} mapped_base={} "
                "mapping_size={}",
                mappingBase, iter->second.deviceAddr, iter->second.size);
        hostMappings.erase(iter);
    }

    Status ReleaseMemoryRecord(MemoryRecord& record)
    {
        if (record.mem != nullptr) {
            const auto ret = HcommMemUnreg(record.endpoint, record.mem);
            if (ret != 0) {
                return HcommError("HcommMemUnreg", ret, StatusCode::BUFFER_NOT_REGISTERED);
            }
            record.mem = nullptr;
        }
        if (record.ownsHostMapping) {
            ReleaseHostMapping(record.hostMappingBase);
            record.ownsHostMapping = false;
        }
        return Status::OK();
    }

    MRHandle InsertMemoryRecord(std::unique_ptr<MemoryRecord> record)
    {
        std::lock_guard<std::mutex> lock(mu);
        MRHandle handle = kInvalidMRHandle;
        do {
            handle = nextMemoryHandle++;
        } while (handle == kInvalidMRHandle || memories.find(handle) != memories.end());
        record->handle = handle;
        memories.emplace(handle, std::move(record));
        return handle;
    }

    Status EnsureAclStreamLocked()
    {
        if (stream != nullptr) { return Status::OK(); }
        const auto ret = aclrtCreateStream(&stream);
        if (ret != ACL_SUCCESS) { return AclError("aclrtCreateStream", ret); }
        return Status::OK();
    }

    Status LoadHixlBatchSendLocked(aclrtFuncHandle& func)
    {
        if (hixlBin == nullptr) {
            aclrtBinaryLoadOption option{};
            option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
            option.value.cpuKernelMode = kCpuKernelMode;
            aclrtBinaryLoadOptions options{};
            options.options = &option;
            options.numOpt = 1U;
            const auto ret =
                aclrtBinaryLoadFromFile(hixlKernelJsonPath.c_str(), &options, &hixlBin);
            if (ret != ACL_SUCCESS) {
                return AclError("aclrtBinaryLoadFromFile " + hixlKernelJsonPath, ret);
            }
        }

        const auto getRet = aclrtBinaryGetFunction(hixlBin, kBatchSendKernelName, &func);
        if (getRet != ACL_SUCCESS) { return AclError("aclrtBinaryGetFunction HixlBatchSend", getRet); }
        return Status::OK();
    }

    Status LaunchBatchSendLocked(const std::vector<TransProvider::SendIoBatch>& ioBatches,
                                 std::vector<std::uint32_t>& hixlStatuses)
    {
        hixlStatuses.assign(ioBatches.size(), 1U);
        auto status = EnsureAclStreamLocked();
        if (!status.ok()) { return status; }

        std::vector<UcmHixlSendIoBatch> batches;
        batches.reserve(ioBatches.size());
        ::ThreadHandle thread = 0;
        for (const auto& io : ioBatches) {
            auto* conn = ToConnectionRecord(io.connectionHandle);
            if (conn == nullptr || conn->channel == 0U || conn->thread == 0U) {
                return Status::Error(StatusCode::CONNECTION_ERROR,
                                     "AICPUProvider::Send: Hcomm channel is not ready");
            }
            if (!conn->hasImmOverride) {
                return Status::Error(
                    StatusCode::CONNECTION_ERROR,
                    "AICPUProvider::Send: negotiated SendWithImm value is unavailable");
            }
            if (thread == 0) {
                thread = conn->thread;
            } else if (thread != conn->thread) {
                return Status::Error(StatusCode::UNSUPPORTED,
                                     "AICPUProvider::Send: mixed AICPU threads in one batch "
                                     "are not supported by HixlBatchSend");
            }
            batches.push_back(UcmHixlSendIoBatch{
                conn->channel, io.sendBuffer, io.len, conn->immOverride, 0U});
            KV_DEBUG("AICPUProvider: batch send entry channel={} thread={} addr={} len={} imm=0x{:x}",
                     conn->channel, conn->thread, io.sendBuffer, io.len, conn->immOverride);
        }
        if (thread == 0) {
            return Status::Error(StatusCode::CONNECTION_ERROR,
                                 "AICPUProvider::Send: no AICPU thread available");
        }

        void* deviceBatchesRaw = nullptr;
        auto ret = aclrtMalloc(&deviceBatchesRaw, batches.size() * sizeof(UcmHixlSendIoBatch),
                               ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_SUCCESS) { return AclError("aclrtMalloc HIXL batch entries", ret); }
        DeviceAllocation deviceBatches(deviceBatchesRaw);

        void* deviceStatusRaw = nullptr;
        ret = aclrtMalloc(&deviceStatusRaw, hixlStatuses.size() * sizeof(std::uint32_t),
                          ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_SUCCESS) { return AclError("aclrtMalloc HIXL status array", ret); }
        DeviceAllocation deviceStatus(deviceStatusRaw);

        ret = aclrtMemcpy(deviceBatches.ptr, batches.size() * sizeof(UcmHixlSendIoBatch),
                          batches.data(), batches.size() * sizeof(UcmHixlSendIoBatch),
                          ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) { return AclError("aclrtMemcpy HIXL batch entries", ret); }

        aclrtFuncHandle func = nullptr;
        status = LoadHixlBatchSendLocked(func);
        if (!status.ok()) { return status; }

        UcmHixlBatchSendParam param{};
        param.thread = thread;
        param.io_batches = static_cast<UcmHixlSendIoBatch*>(deviceBatches.ptr);
        param.batch_size = static_cast<std::uint64_t>(batches.size());
        param.status_array = static_cast<std::uint32_t*>(deviceStatus.ptr);
        param.timeout_ms = sendTimeoutMs;
        param.stats = nullptr;
        // Staged Hcomm channels use USER_CTL sender CQs. The paired HixlBatchSend
        // consumes one sender CQE and advances CQ/SQ CI before HcommThreadJoin.
        param.complete_sender_cqe = 1U;

        aclrtArgsHandle args = nullptr;
        aclrtParamHandle paramHandle = nullptr;
        ret = aclrtKernelArgsInit(func, &args);
        if (ret != ACL_SUCCESS) { return AclError("aclrtKernelArgsInit HixlBatchSend", ret); }
        ret = aclrtKernelArgsAppend(args, &param, sizeof(param), &paramHandle);
        if (ret != ACL_SUCCESS) { return AclError("aclrtKernelArgsAppend HixlBatchSend", ret); }
        ret = aclrtKernelArgsFinalize(args);
        if (ret != ACL_SUCCESS) { return AclError("aclrtKernelArgsFinalize HixlBatchSend", ret); }

        aclrtLaunchKernelAttr attr{};
        attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
        attr.value.timeout = MakeKernelTimeoutSeconds(sendTimeoutMs);
        aclrtLaunchKernelCfg cfg{};
        cfg.numAttrs = 1U;
        cfg.attrs = &attr;
        ret = aclrtLaunchKernelWithConfig(func, kKernelBlockDim, stream, &cfg, args, nullptr);
        if (ret != ACL_SUCCESS) { return AclError("aclrtLaunchKernelWithConfig HixlBatchSend", ret); }
        ret = aclrtSynchronizeStreamWithTimeout(stream, MakeAclSyncTimeoutMs(sendTimeoutMs));
        if (ret != ACL_SUCCESS) {
            return AclError("aclrtSynchronizeStreamWithTimeout HixlBatchSend", ret);
        }

        ret = aclrtMemcpy(hixlStatuses.data(), hixlStatuses.size() * sizeof(std::uint32_t),
                          deviceStatus.ptr, hixlStatuses.size() * sizeof(std::uint32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) { return AclError("aclrtMemcpy HIXL status array", ret); }
        return Status::OK();
    }

    TransportConfig config;
    std::uint32_t localDeviceId{0};
    aclrtContext providerContext{nullptr};
    std::string deviceSelectionSource{"default"};
    std::uint32_t notifyNum{kDefaultNotifyNum};
    std::uint32_t ubSqDepth{kDefaultUbSqDepth};
    std::uint32_t qos{0};
    std::uint32_t sendTimeoutMs{kDefaultSendTimeoutMs};
    std::string channelName;
    std::string hixlKernelJsonPath;
    std::uint32_t stagedKato{0};
    std::uint32_t stagedRmUasid{0};
    std::uint64_t stagedMamiTag{0};
    std::uint32_t nextStagedMrId{0};

    std::mutex mu;
    // Channel creation and MR mutation must observe one another atomically. Rollback paths call
    // the public release helpers while retaining this lock, hence the recursive mutex.
    std::recursive_mutex resourceMu;
    EndpointHandle endpoint{nullptr};
    std::string endpointIp;
    CommProtocol endpointProtocol{COMM_PROTOCOL_RESERVED};
    std::unordered_set<ConnectionRecord*> connections;
    std::unordered_map<MRHandle, std::unique_ptr<MemoryRecord>> memories;
    MRHandle nextMemoryHandle{1};
    std::uint64_t nextMemTag{1};

    std::mutex hostMappingMu;
    std::unordered_map<std::uintptr_t, HostMapping> hostMappings;

    aclrtStream stream{nullptr};
    aclrtBinHandle hixlBin{nullptr};
};

AICPUProvider::AICPUProvider(const TransportConfig& config)
    : impl_(std::make_unique<Impl>(config))
{
    KV_INFO("AICPU_TRANSPORT_PROVIDER_SIGNATURE={} pid={} asu_id={} logical_device_id={} "
            "device_source={} provider_context={} protocol=ubg channel_api={} "
            "send_with_imm=1 complete_sender_cqe=1 publish_mrs=1 channel_name={}",
            kProviderSignature, static_cast<long>(::getpid()), impl_->config.nodeId,
            impl_->localDeviceId, impl_->deviceSelectionSource,
            static_cast<const void*>(impl_->providerContext), kChannelApiMode, impl_->channelName);
}

AICPUProvider::~AICPUProvider()
{
    if (!impl_) { return; }

    std::vector<MRHandle> memoryHandles;
    std::vector<ConnectionHandle> connHandles;
    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        memoryHandles.reserve(impl_->memories.size());
        for (const auto& item : impl_->memories) { memoryHandles.push_back(item.first); }
        connHandles.reserve(impl_->connections.size());
        for (auto* conn : impl_->connections) { connHandles.push_back(conn); }
    }

    // HCOMM can represent repeated/subrange registrations as aliases of an earlier handle.
    // Release newer local handles first so aliases normally disappear before their parent.
    std::sort(memoryHandles.begin(), memoryHandles.end());
    if (!memoryHandles.empty()) { (void)ReleaseMemory(memoryHandles, "CleanupMemory"); }
    if (!connHandles.empty()) { (void)DeleteConnections(connHandles); }
}

Status AICPUProvider::CreateConnection(const std::string& localIp, const std::string& remoteIp,
                                       uint32_t port, uint32_t qpNum, uint32_t timeout,
                                       std::vector<ConnectionHandle>& connectionHandles)
{
    connectionHandles.clear();
    if (qpNum == 0) { return Status::OK(); }

    AclContextScope contextScope("CreateConnection");
    std::lock_guard<std::recursive_mutex> resourceLock(impl_->resourceMu);
    auto status = impl_->RefreshLocalDeviceFromCaller();
    if (!status.ok()) { return status; }
    status = impl_->EnsureAclDeviceBound("CreateConnection");
    if (!status.ok()) { return status; }

    const auto* endpoint = impl_->FindEndpoint(remoteIp, port);
    CommProtocol protocol = COMM_PROTOCOL_RESERVED;
    status = ResolveProtocol(impl_->config, endpoint, protocol);
    if (!status.ok()) { return status; }
    const auto remoteDeviceId = ResolveRemoteDeviceId(endpoint, impl_->localDeviceId);
    KV_INFO("AICPUProvider: CreateConnection start signature={} pid={} local_addr={} "
            "remote_addr={} remote_port={} qp_num={} timeout_ms={} logical_device_id={} "
            "device_source={} provider_context={} remote_device_id={} protocol={} "
            "endpoint_matched={}",
            kProviderSignature, static_cast<long>(::getpid()), localIp, remoteIp, port, qpNum,
            timeout, impl_->localDeviceId, impl_->deviceSelectionSource,
            static_cast<const void*>(impl_->providerContext), remoteDeviceId,
            CommProtocolName(protocol), endpoint == nullptr ? 0 : 1);

    EndpointDesc remoteDesc{};
    status = BuildEndpointDesc(impl_->config, endpoint, remoteIp, remoteDeviceId, protocol,
                               "remote", remoteDesc);
    if (!status.ok()) {
        KV_ERROR("AICPUProvider: remote EndpointDesc build failed remote_addr={} "
                 "remote_port={} remote_device_id={} protocol={} message={}",
                 remoteIp, port, remoteDeviceId, CommProtocolName(protocol), status.message);
        return status;
    }

    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        status = impl_->EnsureEndpointLocked(localIp, protocol);
        if (!status.ok()) {
            KV_ERROR("AICPUProvider: EnsureEndpointLocked failed local_addr={} "
                     "local_device_id={} protocol={} message={}",
                     localIp, impl_->localDeviceId, CommProtocolName(protocol), status.message);
            return status;
        }
    }

    std::vector<ConnectionHandle> createdHandles;
    createdHandles.reserve(qpNum);
#if UCM_ASU_AICPU_USE_STAGED_CHANNEL_API
    const std::string stagedOobHost = ResolveStagedOobHost(impl_->config, endpoint, remoteIp);
    const std::uint16_t stagedOobPort = ResolveStagedOobPort(impl_->config, endpoint, port);
    const std::uint32_t stagedClientId = ResolveStagedClientId(impl_->config, endpoint);
    KV_INFO("AICPUProvider: CreateConnection resolved mode=staged_{} channel_api={} "
            "staged_oob={}:{} staged_client_id={} notify_num={} ub_sq_depth={} qos={} "
            "send_with_imm={} complete_sender_cqe={}",
            CommProtocolName(protocol), kChannelApiMode, stagedOobHost, stagedOobPort,
            stagedClientId, impl_->notifyNum, impl_->ubSqDepth, impl_->qos, 1, 0);
#else
    KV_INFO("AICPUProvider: CreateConnection resolved mode=staged_{} channel_api={} "
            "notify_num={} ub_sq_depth={} qos={} send_with_imm={} complete_sender_cqe={}",
            CommProtocolName(protocol), kChannelApiMode, impl_->notifyNum, impl_->ubSqDepth,
            impl_->qos, 1, 0);
#endif

    for (std::uint32_t remaining = qpNum; remaining > 0; --remaining) {
        const std::uint32_t qpIndex = static_cast<std::uint32_t>(createdHandles.size());
        KV_DEBUG("AICPUProvider: creating connection handle qp_index={} qp_num={} "
                 "channel_api={}",
                 qpIndex, qpNum, kChannelApiMode);
        auto* record = new ConnectionRecord{};
        const std::uint32_t notify = impl_->notifyNum;
        const auto threadRet = HcommThreadAlloc(COMM_ENGINE_AICPU, 1U, &notify, &record->thread);
        if (threadRet != 0) {
            delete record;
            if (!createdHandles.empty()) { (void)DeleteConnections(createdHandles); }
            return HcommConnectionError("HcommThreadAlloc", threadRet);
        }
        const auto cleanupRecord = [&]() {
            {
                std::lock_guard<std::mutex> lock(impl_->mu);
                impl_->connections.insert(record);
            }
            const auto cleanupStatuses = DeleteConnections({record});
            if (cleanupStatuses.empty() || !cleanupStatuses.front().ok()) {
                KV_WARN("AICPUProvider: retained partially initialized connection "
                        "handle={} channel={} thread={} for later cleanup",
                        static_cast<void*>(record), record->channel, record->thread);
            }
        };

        HcommChannelDesc desc{};
        const auto descRet = HcommChannelDescInit(&desc, 1U);
        if (descRet != 0) {
            cleanupRecord();
            if (!createdHandles.empty()) { (void)DeleteConnections(createdHandles); }
            return HcommError("HcommChannelDescInit", descRet);
        }
        desc.remoteEndpoint = remoteDesc;
        desc.notifyNum = notify;
        desc.exchangeAllMems = true;
        desc.role = HCOMM_SOCKET_ROLE_CLIENT;
        desc.port = ParseUint16(GetEndpointAttr(endpoint, {"aicpu_port", "hcomm_port"}),
                                static_cast<std::uint16_t>(port));
        desc.ubAttr.sqDepth = impl_->ubSqDepth;
        desc.qos = impl_->qos;
        SetHcommChannelNameIfSupported(desc, impl_->channelName.c_str(), 0);

        EndpointHandle hcommEndpoint = nullptr;
        {
            std::lock_guard<std::mutex> lock(impl_->mu);
            hcommEndpoint = impl_->endpoint;
        }
#if UCM_ASU_AICPU_USE_STAGED_CHANNEL_API
        HcommStagedChannelDesc stagedDesc{};
        const auto stagedInitRet = HcommStagedChannelDescInit(&stagedDesc, 1U);
        if (stagedInitRet != 0) {
            cleanupRecord();
            if (!createdHandles.empty()) { (void)DeleteConnections(createdHandles); }
            return HcommError("HcommStagedChannelDescInit", stagedInitRet);
        }
        stagedDesc.channelDesc = desc;
        stagedDesc.oobHost = stagedOobHost.c_str();
        stagedDesc.oobPort = stagedOobPort;
        stagedDesc.timeoutMs = impl_->sendTimeoutMs;
        stagedDesc.clientId = stagedClientId;
        stagedDesc.qpIndex = qpIndex;
        stagedDesc.kato = impl_->stagedKato;
        stagedDesc.connMode = HCOMM_STAGED_CONN_MODE_RM;
        stagedDesc.rmUasid = impl_->stagedRmUasid;
        stagedDesc.mamiTag = impl_->stagedMamiTag;
        KV_INFO("AICPUProvider: HcommChannelCreateStaged begin qp_index={} "
                "oob={}:{} client_id={} timeout_ms={} rm_uasid={} "
                "mami_tag={} channel_port={}",
                qpIndex, stagedOobHost, stagedOobPort, stagedClientId, stagedDesc.timeoutMs,
                stagedDesc.rmUasid, stagedDesc.mamiTag, desc.port);
        const auto chanRet =
            HcommChannelCreateStaged(hcommEndpoint, COMM_ENGINE_AICPU, &stagedDesc, 1U,
                                     &record->channel);
        if (chanRet != 0) {
            cleanupRecord();
            if (!createdHandles.empty()) { (void)DeleteConnections(createdHandles); }
            return HcommConnectionError("HcommChannelCreateStaged", chanRet);
        }
        KV_INFO("AICPUProvider: staged channel created qp_index={} channel={} thread={}",
                qpIndex, record->channel, record->thread);

        HcommStagedChannelInfo stagedInfo{};
        const auto infoInitRet = HcommStagedChannelInfoInit(&stagedInfo, 1U);
        if (infoInitRet != 0) {
            cleanupRecord();
            if (!createdHandles.empty()) { (void)DeleteConnections(createdHandles); }
            return HcommError("HcommStagedChannelInfoInit", infoInitRet);
        }
        const auto infoRet = HcommChannelGetStagedInfo(record->channel, &stagedInfo);
        if (infoRet != 0) {
            cleanupRecord();
            if (!createdHandles.empty()) { (void)DeleteConnections(createdHandles); }
            return HcommConnectionError("HcommChannelGetStagedInfo", infoRet);
        }
        record->stagedInfo = stagedInfo;
        if (record->stagedInfo.clientId == 0U) {
            record->stagedInfo.clientId = stagedClientId;
        }

        HcommStagedServerCapabilities hcommCapabilities{};
        const auto capabilitiesInitRet =
            HcommStagedServerCapabilitiesInit(&hcommCapabilities, 1U);
        if (capabilitiesInitRet != 0) {
            cleanupRecord();
            if (!createdHandles.empty()) { (void)DeleteConnections(createdHandles); }
            return HcommError("HcommStagedServerCapabilitiesInit", capabilitiesInitRet);
        }
        const auto capabilitiesRet =
            HcommChannelGetStagedServerCapabilities(record->channel, &hcommCapabilities);
        if (capabilitiesRet != 0) {
            cleanupRecord();
            if (!createdHandles.empty()) { (void)DeleteConnections(createdHandles); }
            return HcommConnectionError("HcommChannelGetStagedServerCapabilities", capabilitiesRet);
        }
        record->serverCapabilities.queueNum = hcommCapabilities.queueNum;
        record->serverCapabilities.ioQueueDepth = hcommCapabilities.ioQueueDepth;
        record->serverCapabilities.ioQueueKeyConcurrency =
            hcommCapabilities.ioQueueKeyConcurrency;
        record->serverCapabilities.connectionKeyConcurrency =
            hcommCapabilities.connectionKeyConcurrency;
        record->serverCapabilities.singleValueMaxBytes = hcommCapabilities.singleValueMaxBytes;
        record->serverCapabilities.batchValueMaxBytes = hcommCapabilities.batchValueMaxBytes;
        record->serverCapabilities.batchStoreKeys = hcommCapabilities.batchStoreKeys;
        record->serverCapabilities.batchLoadKeys = hcommCapabilities.batchLoadKeys;
        record->serverCapabilities.deleteKeys = hcommCapabilities.deleteKeys;
        record->serverCapabilities.queryKeys = hcommCapabilities.queryKeys;
        record->serverCapabilities.keyLength = hcommCapabilities.keyLength;
        record->serverCapabilities.kvCapabilities = hcommCapabilities.kvCapabilities;
        record->hasServerCapabilities = true;
        record->hasImmOverride = true;
        record->immOverride = stagedInfo.sendImm;
        record->stagedOobHost = stagedOobHost;
        record->stagedOobPort = stagedOobPort;
        KV_INFO("AICPUProvider: staged channel info qp_index={} controller_id={} "
                "send_imm={} client_id={} configured_client_id={} remote_jetty_id={} "
                "remote_token_value={}",
                stagedInfo.qpIndex, stagedInfo.controllerId, stagedInfo.sendImm,
                record->stagedInfo.clientId, stagedClientId, stagedInfo.remoteJettyId,
                stagedInfo.remoteTokenValue);
        KV_INFO("AICPUProvider: staged server capabilities qp_index={} queue_num={} "
                "ioq_depth={} ioq_key_concurrency={} connection_key_concurrency={} "
                "single_value_max_bytes={} batch_value_max_bytes={} batch_store_keys={} "
                "batch_load_keys={} delete_keys={} query_keys={} key_length={} kv_capabilities=0x{:x}",
                qpIndex, record->serverCapabilities.queueNum,
                record->serverCapabilities.ioQueueDepth,
                record->serverCapabilities.ioQueueKeyConcurrency,
                record->serverCapabilities.connectionKeyConcurrency,
                record->serverCapabilities.singleValueMaxBytes,
                record->serverCapabilities.batchValueMaxBytes,
                record->serverCapabilities.batchStoreKeys,
                record->serverCapabilities.batchLoadKeys,
                record->serverCapabilities.deleteKeys, record->serverCapabilities.queryKeys,
                record->serverCapabilities.keyLength,
                record->serverCapabilities.kvCapabilities);
#else
        // This mode is for the future standard HCOMM API after it owns the complete Staged URMA
        // negotiation contract. Until then it is not compatible with the current Staged Server.
        KV_INFO("AICPUProvider: HcommChannelCreate begin qp_index={} channel_port={}",
                qpIndex, desc.port);
        const auto chanRet =
            HcommChannelCreate(hcommEndpoint, COMM_ENGINE_AICPU, &desc, 1U, &record->channel);
        if (chanRet != 0) {
            cleanupRecord();
            if (!createdHandles.empty()) { (void)DeleteConnections(createdHandles); }
            return HcommConnectionError("HcommChannelCreate", chanRet);
        }
        KV_INFO("AICPUProvider: standard channel created qp_index={} channel={} thread={}",
                qpIndex, record->channel, record->thread);
#endif

        status = impl_->AttachExistingMemoriesToConnection(*record);
        if (!status.ok()) {
            KV_ERROR("AICPUProvider: failed to attach existing memory to new channel "
                     "qp_index={} channel={} code={} message={}",
                     qpIndex, record->channel, static_cast<int>(status.code), status.message);
            cleanupRecord();
            if (!createdHandles.empty()) { (void)DeleteConnections(createdHandles); }
            return status;
        }

        {
            std::lock_guard<std::mutex> lock(impl_->mu);
            impl_->connections.insert(record);
        }
        createdHandles.push_back(record);
        connectionHandles.push_back(record);
        KV_INFO("AICPUProvider: connection handle ready qp_index={} created_handles={}/{}",
                qpIndex, connectionHandles.size(), qpNum);
    }

    KV_INFO("AICPUProvider: CreateConnection complete remote_addr={} remote_port={} "
            "handles={}",
            remoteIp, port, connectionHandles.size());
    return Status::OK();
}

std::vector<Status> AICPUProvider::DeleteConnections(
    const std::vector<ConnectionHandle>& connectionHandles)
{
    std::vector<Status> results(connectionHandles.size(), Status::OK());
    if (connectionHandles.empty()) { return results; }

    AclContextScope contextScope("DeleteConnections");
    std::lock_guard<std::recursive_mutex> resourceLock(impl_->resourceMu);
    const auto deviceStatus = impl_->EnsureAclDeviceBound("DeleteConnections");
    if (!deviceStatus.ok()) {
        return std::vector<Status>(connectionHandles.size(), deviceStatus);
    }
    for (std::size_t index = 0; index < connectionHandles.size(); ++index) {
        auto* record = ToConnectionRecord(connectionHandles[index]);
        {
            std::lock_guard<std::mutex> lock(impl_->mu);
            if (record == nullptr || impl_->connections.find(record) == impl_->connections.end()) {
                results[index] = Status::Error(StatusCode::INVALID_ARGUMENT,
                                               "AICPUProvider: invalid connection handle");
                continue;
            }
            impl_->connections.erase(record);
        }

        Status status = Status::OK();
        if (record->channel != 0) {
            auto channel = record->channel;
            const auto ret = HcommChannelDestroy(&channel, 1U);
            if (ret != 0) {
                status = HcommConnectionError("HcommChannelDestroy", ret);
            } else {
                record->channel = 0;
            }
        }
        if (status.ok() && record->thread != 0) {
            auto thread = record->thread;
            const auto ret = HcommThreadFree(&thread, 1U);
            if (ret != 0) {
                status = HcommConnectionError("HcommThreadFree", ret);
            } else {
                record->thread = 0;
            }
        }
        results[index] = status;
        if (status.ok()) {
            delete record;
        } else {
            std::lock_guard<std::mutex> lock(impl_->mu);
            impl_->connections.insert(record);
            KV_WARN("AICPUProvider: retained connection handle={} channel={} thread={} "
                    "after delete failure for later cleanup",
                    static_cast<void*>(record), record->channel, record->thread);
        }
    }
    return results;
}

Status AICPUProvider::GetServerCapabilities(ConnectionHandle connectionHandle,
                                                 ServerKvCapabilities& capabilities)
{
    capabilities = {};
    auto* record = ToConnectionRecord(connectionHandle);
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (record == nullptr || impl_->connections.find(record) == impl_->connections.end()) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "AICPUProvider: invalid connection handle");
    }
    if (!record->hasServerCapabilities) {
        return Status::Error(StatusCode::UNSUPPORTED,
                             "AICPUProvider: server capability query is not available");
    }
    capabilities = record->serverCapabilities;
    return Status::OK();
}

std::vector<Status> AICPUProvider::Send(const std::vector<SendIoBatch>& ioBatches,
                                        uint32_t kernelCount, uint32_t quietCount)
{
    (void)kernelCount;
    (void)quietCount;
    if (ioBatches.empty()) { return {}; }

    // HixlBatchSend is synchronous. Retain all channels until the kernel and its ThreadJoin
    // finish so a shared provider cannot destroy a validated handle between lookup and launch.
    std::lock_guard<std::recursive_mutex> resourceLock(impl_->resourceMu);

    struct ThreadBatchGroup {
        ::ThreadHandle thread{0};
        std::vector<std::size_t> originalIndexes;
        std::vector<SendIoBatch> batches;
    };

    std::vector<Status> results(ioBatches.size(), Status::OK());
    std::vector<ThreadBatchGroup> groups;
    std::unordered_map<::ThreadHandle, std::size_t> groupIndexByThread;
    bool valid = true;
    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        for (std::size_t index = 0; index < ioBatches.size(); ++index) {
            const auto& item = ioBatches[index];
            auto* conn = ToConnectionRecord(item.connectionHandle);
            if (conn == nullptr || impl_->connections.find(conn) == impl_->connections.end()) {
                results[index] = Status::Error(StatusCode::INVALID_ARGUMENT,
                                               "AICPUProvider::Send: invalid connection handle");
                valid = false;
                continue;
            }
            if (item.sendBuffer == nullptr || item.len == 0) {
                results[index] = Status::Error(StatusCode::INVALID_ARGUMENT,
                                               "AICPUProvider::Send: empty send buffer");
                valid = false;
                continue;
            }
            if (conn->thread == 0U) {
                results[index] = Status::Error(StatusCode::CONNECTION_ERROR,
                                               "AICPUProvider::Send: Hcomm thread is not ready");
                valid = false;
                continue;
            }

            auto [groupIt, inserted] = groupIndexByThread.emplace(conn->thread, groups.size());
            if (inserted) { groups.push_back(ThreadBatchGroup{conn->thread, {}, {}}); }
            auto& group = groups[groupIt->second];
            group.originalIndexes.push_back(index);
            group.batches.push_back(item);
        }
    }
    if (!valid) { return results; }

    AclContextScope contextScope("Send");
    const auto deviceStatus = impl_->EnsureAclDeviceBound("Send");
    if (!deviceStatus.ok()) { return std::vector<Status>(ioBatches.size(), deviceStatus); }

    for (const auto& group : groups) {
        KV_DEBUG("AICPUProvider: launching HixlBatchSend thread={} entries={}",
                 group.thread, group.batches.size());
        std::vector<std::uint32_t> hixlStatuses;
        const auto launchStatus = impl_->LaunchBatchSendLocked(group.batches, hixlStatuses);
        if (!launchStatus.ok()) {
            for (const auto originalIndex : group.originalIndexes) {
                results[originalIndex] = launchStatus;
            }
            continue;
        }
        if (hixlStatuses.size() != group.originalIndexes.size()) {
            const auto status = Status::Error(
                StatusCode::INTERNAL_ERROR,
                "HixlBatchSend returned an unexpected status count");
            for (const auto originalIndex : group.originalIndexes) {
                results[originalIndex] = status;
            }
            continue;
        }
        for (std::size_t groupIndex = 0; groupIndex < hixlStatuses.size(); ++groupIndex) {
            if (hixlStatuses[groupIndex] == 0U) { continue; }
            const auto originalIndex = group.originalIndexes[groupIndex];
            results[originalIndex] = Status::Error(
                StatusCode::INTERNAL_ERROR,
                "HixlBatchSend failed for batch index " + std::to_string(originalIndex) +
                    " status=" + std::to_string(hixlStatuses[groupIndex]));
        }
    }
    return results;
}

Status AICPUProvider::RegisterMemory(const std::vector<RegisterMemoryDesc>& memoryDescs,
                                     std::vector<MRHandle>& mrHandles)
{
    return RegisterMemoryImpl(memoryDescs, RegistrationMode::REGISTER, "RegisterMemory",
                              mrHandles);
}

Status AICPUProvider::BindMemory(const std::vector<BindMemoryDesc>& regions,
                                 std::vector<MRHandle>& mrHandles)
{
    mrHandles.clear();
    std::vector<RegisterMemoryDesc> memoryDescs;
    std::vector<std::uint32_t> expectedTokenIds;
    memoryDescs.reserve(regions.size());
    expectedTokenIds.reserve(regions.size());
    for (const auto& region : regions) {
        if (region.memoryType != MemType::MEM_DEVICE) {
            KV_ERROR("AICPUProvider: BindMemory rejected non-device memory addr={} size={} "
                     "memory_type={}",
                     region.addr, region.size, static_cast<int>(region.memoryType));
            return Status::Error(
                StatusCode::BUFFER_NOT_SUPPORTED,
                "AICPUProvider::BindMemory only supports ASCEND_DEVICE business memory");
        }
        memoryDescs.push_back({region.memoryType, region.addr, region.size, region.addr});
        expectedTokenIds.push_back(region.tokenId);
    }
    // Each independent provider owns its HCOMM registration. Bind must not acquire a second ACL
    // host mapping for canonical business memory.
    return RegisterMemoryImpl(memoryDescs, RegistrationMode::BIND, "BindMemory", mrHandles,
                              &expectedTokenIds);
}

Status AICPUProvider::RegisterMemoryImpl(
    const std::vector<RegisterMemoryDesc>& memoryDescs, RegistrationMode mode,
    const char* operation, std::vector<MRHandle>& mrHandles,
    const std::vector<std::uint32_t>* expectedTokenIds)
{
    mrHandles.clear();
    if (memoryDescs.empty()) { return Status::OK(); }
    if (expectedTokenIds != nullptr && expectedTokenIds->size() != memoryDescs.size()) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             std::string("AICPUProvider::") + operation +
                                 ": token count does not match memory count");
    }

    AclContextScope contextScope(operation);
    std::lock_guard<std::recursive_mutex> resourceLock(impl_->resourceMu);
    auto bindStatus = impl_->EnsureAclDeviceBound(operation);
    if (!bindStatus.ok()) { return bindStatus; }

    EndpointHandle endpoint = nullptr;
    CommProtocol endpointProtocol = COMM_PROTOCOL_RESERVED;
    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        if (impl_->endpoint == nullptr) {
            const auto* configuredEndpoint =
                impl_->config.endpoints.empty() ? nullptr : &impl_->config.endpoints.front();
            auto protocolStatus =
                ResolveProtocol(impl_->config, configuredEndpoint, endpointProtocol);
            if (!protocolStatus.ok()) { return protocolStatus; }
            auto endpointStatus = impl_->EnsureEndpointLocked({}, endpointProtocol);
            if (!endpointStatus.ok()) { return endpointStatus; }
        }
        endpoint = impl_->endpoint;
        endpointProtocol = impl_->endpointProtocol;
    }
    if (endpoint == nullptr) {
        return Status::Error(StatusCode::CONNECTION_ERROR,
                             std::string("AICPUProvider::") + operation +
                                 " failed to establish a local HCOMM endpoint");
    }

    const char* registrationMode =
        mode == RegistrationMode::REGISTER ? "register" : "bind";
    KV_INFO("AICPUProvider: {} begin registration_mode={} desc_count={}", operation,
            registrationMode, memoryDescs.size());
    std::vector<MRHandle> createdHandles;
    createdHandles.reserve(memoryDescs.size());
    auto cleanupCreated = [&]() {
        const auto statuses = ReleaseMemory(createdHandles, "RollbackMemory");
        std::vector<MRHandle> failedHandles;
        failedHandles.reserve(createdHandles.size());
        for (std::size_t index = 0; index < createdHandles.size(); ++index) {
            if (index >= statuses.size() || !statuses[index].ok()) {
                failedHandles.push_back(createdHandles[index]);
            }
        }
        createdHandles = failedHandles;
        mrHandles = std::move(failedHandles);
    };
    for (const auto& desc : memoryDescs) {
        if (desc.addr == 0 || desc.size == 0) {
            KV_ERROR("AICPUProvider: {} invalid desc addr={} size={} type={}", operation,
                     desc.addr, desc.size, static_cast<int>(desc.memoryType));
            cleanupCreated();
            return Status::Error(
                StatusCode::INVALID_ARGUMENT,
                std::string("AICPUProvider::") + operation + ": zero addr/size in desc");
        }

        auto record = std::make_unique<MemoryRecord>();
        record->endpoint = endpoint;
        record->originalAddr = desc.addr;
        record->localAddr = desc.localAddr == 0 ? desc.addr : desc.localAddr;
        record->transportAddr = desc.addr;
        record->size = desc.size;
        record->memoryType = desc.memoryType;
        auto releaseCurrent = [&]() {
            const auto releaseStatus = impl_->ReleaseMemoryRecord(*record);
            if (releaseStatus.ok()) { return; }
            const auto handle = impl_->InsertMemoryRecord(std::move(record));
            createdHandles.push_back(handle);
            mrHandles.push_back(handle);
            KV_ERROR("AICPUProvider: retained mr_handle={} after rollback failure: {}",
                     handle, releaseStatus.message);
        };
        {
            std::lock_guard<std::mutex> lock(impl_->mu);
            record->tag = impl_->channelName + ":mem:" + std::to_string(impl_->nextMemTag++);
            record->stagedMrId = impl_->nextStagedMrId++;
        }

        if (mode == RegistrationMode::REGISTER && desc.memoryType == MemType::MEM_HOST &&
            IsUbProtocol(endpointProtocol)) {
            auto mappingStatus = impl_->AcquireHostMapping(
                desc.addr, desc.size, record->hostMappingBase, record->transportAddr);
            if (!mappingStatus.ok()) {
                cleanupCreated();
                return mappingStatus;
            }
            record->ownsHostMapping = true;
        }

        CommMem mem{};
        mem.type = record->ownsHostMapping ? COMM_MEM_TYPE_DEVICE : ToHcommMemType(desc.memoryType);
        mem.addr = reinterpret_cast<void*>(record->transportAddr);
        mem.size = static_cast<std::uint64_t>(desc.size);
        KV_INFO(
            "AICPUProvider: HcommMemReg begin registration_mode={} tag={} original_addr={} "
            "local_addr={} transport_addr={} size={} input_mem_type={} hcomm_mem_type={} "
            "protocol={} owns_host_mapping={} staged_mr_id={}",
            registrationMode, record->tag, record->originalAddr, record->localAddr,
            record->transportAddr, record->size, static_cast<int>(desc.memoryType),
            static_cast<int>(mem.type), CommProtocolName(endpointProtocol),
            record->ownsHostMapping ? 1 : 0, record->stagedMrId);
        const auto ret = HcommMemReg(endpoint, record->tag.c_str(), &mem, &record->mem);
        if (ret != 0) {
            releaseCurrent();
            cleanupCreated();
            return HcommError("HcommMemReg", ret, StatusCode::BUFFER_NOT_SUPPORTED);
        }

        HcommMemTokenInfo tokenInfo{};
        const auto tokenRet = HcommMemGetTokenInfo(record->mem, &tokenInfo);
        if (tokenRet == 0) {
            if (tokenInfo.addr != record->transportAddr || tokenInfo.size != record->size) {
                KV_ERROR("AICPUProvider: HCOMM token range mismatch tag={} "
                         "transport_addr={} transport_size={} token_addr={} token_size={}",
                         record->tag, record->transportAddr, record->size, tokenInfo.addr,
                         tokenInfo.size);
                releaseCurrent();
                cleanupCreated();
                return Status::Error(StatusCode::BUFFER_NOT_SUPPORTED,
                                     "AICPUProvider: HCOMM token range mismatch");
            }
            if (tokenInfo.type == HCOMM_MEM_TOKEN_TYPE_UB && tokenInfo.tokenValue != 0U) {
                record->tokenId = tokenInfo.tokenValue;
                record->hasToken = true;
            } else if (tokenInfo.type == HCOMM_MEM_TOKEN_TYPE_RDMA && tokenInfo.rkey != 0U) {
                record->tokenId = tokenInfo.rkey;
                record->hasToken = true;
            }
            KV_INFO(
                "AICPUProvider: HcommMemGetTokenInfo registration_mode={} tag={} "
                "hcomm_mem_handle={} token_type={} token_id={} token_value={} rkey={} "
                "selected_mr_key={} token_addr={} token_size={} has_token={}",
                registrationMode, record->tag, record->mem, static_cast<int>(tokenInfo.type),
                tokenInfo.tokenId, tokenInfo.tokenValue, tokenInfo.rkey, record->tokenId,
                tokenInfo.addr, tokenInfo.size, record->hasToken ? 1 : 0);
        } else {
            KV_WARN("AICPUProvider: HcommMemGetTokenInfo failed registration_mode={} tag={} "
                    "hcomm_mem_handle={} ret={}",
                    registrationMode, record->tag, record->mem, tokenRet);
        }

        if (expectedTokenIds != nullptr) {
            const auto expectedToken = (*expectedTokenIds)[createdHandles.size()];
            if (!record->hasToken || record->tokenId != expectedToken) {
                KV_ERROR("AICPUProvider: {} token mismatch registration_mode={} tag={} "
                         "expected={} actual={} has_token={}",
                         operation, registrationMode, record->tag, expectedToken,
                         record->tokenId, record->hasToken ? 1 : 0);
                releaseCurrent();
                cleanupCreated();
                return Status::Error(StatusCode::BUFFER_NOT_SUPPORTED,
                                     std::string("AICPUProvider::") + operation +
                                         ": HCOMM token does not match canonical token");
            }
        }

        std::vector<ConnectionRecord*> connections;
        {
            std::lock_guard<std::mutex> lock(impl_->mu);
            connections.assign(impl_->connections.begin(), impl_->connections.end());
        }
        for (auto* conn : connections) {
            if (conn == nullptr) { continue; }
            const auto updateStatus = impl_->UpdateChannelMemory(*conn, *record);
            if (!updateStatus.ok()) {
                releaseCurrent();
                cleanupCreated();
                return updateStatus;
            }
        }

        auto publishStatus = impl_->PublishMemoryStaged(*record);
        if (!publishStatus.ok()) {
            releaseCurrent();
            cleanupCreated();
            return publishStatus;
        }

        const auto hcommMemHandle = record->mem;
        const auto handle = impl_->InsertMemoryRecord(std::move(record));
        createdHandles.push_back(handle);
        mrHandles.push_back(handle);
        KV_INFO("AICPUProvider: {} record ready registration_mode={} mr_handle={} "
                "hcomm_mem_handle={} handles={}/{}",
                operation, registrationMode, handle, hcommMemHandle, mrHandles.size(),
                memoryDescs.size());
    }

    KV_INFO("AICPUProvider: {} complete registration_mode={} handles={}", operation,
            registrationMode, mrHandles.size());
    return Status::OK();
}

std::vector<Status> AICPUProvider::UnregisterMemory(
    const std::vector<UnregisterMemoryDesc>& memoryDescs)
{
    std::vector<MRHandle> handles;
    handles.reserve(memoryDescs.size());
    for (const auto& desc : memoryDescs) { handles.push_back(desc.mrHandle); }
    return ReleaseMemory(handles, "UnregisterMemory");
}

std::vector<Status> AICPUProvider::ReleaseMemory(const std::vector<MRHandle>& mrHandles,
                                                      const char* operation)
{
    std::vector<Status> results(mrHandles.size(), Status::OK());
    if (mrHandles.empty()) { return results; }

    AclContextScope contextScope(operation);
    std::lock_guard<std::recursive_mutex> resourceLock(impl_->resourceMu);
    const auto deviceStatus = impl_->EnsureAclDeviceBound(operation);
    if (!deviceStatus.ok()) { return std::vector<Status>(mrHandles.size(), deviceStatus); }

    // The caller normally preserves registration order. Reverse release keeps HCOMM aliases
    // ahead of the original local registration while result slots still match the input.
    for (std::size_t reverseIndex = mrHandles.size(); reverseIndex > 0; --reverseIndex) {
        const auto index = reverseIndex - 1;
        const auto handle = mrHandles[index];
        if (handle == kInvalidMRHandle) { continue; }

        std::lock_guard<std::mutex> lock(impl_->mu);
        auto iter = impl_->memories.find(handle);
        if (iter == impl_->memories.end()) {
            KV_DEBUG("AICPUProvider: {} ignored released handle={}", operation, handle);
            continue;
        }
        results[index] = impl_->ReleaseMemoryRecord(*iter->second);
        if (results[index].ok()) { impl_->memories.erase(iter); }
    }
    return results;
}

Status AICPUProvider::GetMemTokenId(MRHandle mrHandle, uint32_t& tokenId)
{
    tokenId = 0;
    std::lock_guard<std::mutex> lock(impl_->mu);
    auto iter = impl_->memories.find(mrHandle);
    if (iter == impl_->memories.end()) {
        return Status::Error(StatusCode::BUFFER_NOT_REGISTERED,
                             "AICPUProvider::GetMemTokenId: memory handle not found");
    }

    const auto& record = *iter->second;
    if (!record.hasToken) {
        return Status::Error(StatusCode::UNSUPPORTED,
                             "AICPUProvider::GetMemTokenId: registered hcomm memory did "
                             "not expose a UB token value or RDMA rkey");
    }
    tokenId = record.tokenId;
    return Status::OK();
}

}  // namespace kv

#endif  // UCM_ASU_ENABLE_AICPU_PROVIDER
