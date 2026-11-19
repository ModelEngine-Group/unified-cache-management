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
#include "cpu_kv_over_roce_trans_provider.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace kv {
namespace roce {
namespace {

constexpr const char* kProviderName = "CpuKvOverRoceTransProvider";

// The low bits of a handle carry the queue pair index so that one link can hand
// out several handles while still being a single out-of-band exchange.
constexpr std::uint32_t kHandleIndexBits = 8;

constexpr std::uint32_t kDefaultOutOfBandTimeoutMs = 3000;
constexpr std::uint32_t kDefaultHandshakeTimeoutMs = 5000;

constexpr const char* kAttrRole = "roce_oob_role";
constexpr const char* kAttrBindIp = "roce_oob_bind_ip";
constexpr const char* kAttrConnectTimeout = "roce_oob_connect_timeout_ms";
constexpr const char* kAttrHandshakeTimeout = "roce_handshake_timeout_ms";
constexpr const char* kAttrQpBackend = "roce_qp_backend";

constexpr const char* kBackendVerbs = "verbs";
constexpr const char* kBackendSimulated = "simulated";

// Every region is reachable from the peer, so local and remote writes are both
// allowed on it.
constexpr int kRegistrationAccess =
    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
constexpr std::uint32_t kSimulatedRkeyBase = 0x1000;

Status NotImplemented(const std::string& operation)
{
    return Status::Error(StatusCode::UNSUPPORTED,
                         std::string(kProviderName) + ": " + operation + " is not implemented yet");
}

std::vector<Status> NotImplementedForEach(const std::string& operation, std::size_t count)
{
    return std::vector<Status>(count, NotImplemented(operation));
}

Status InvalidArgument(const std::string& message)
{
    return Status::Error(StatusCode::INVALID_ARGUMENT, std::string(kProviderName) + ": " + message);
}

Status Failure(StatusCode code, const std::string& message)
{
    return Status::Error(code, std::string(kProviderName) + ": " + message);
}

// Parses a decimal unsigned number at full width, so the caller can compare it
// against the maximum of its field *before* narrowing it: casting first would
// let 4294967296 wrap to zero and pass as a perfectly good setting.  Zero is a
// value of its own here - a traffic class, an inline threshold, a GID index and
// a retransmission timeout may all legitimately be zero - so only an empty or
// malformed text fails.
bool ParseUnsigned(const std::string& text, std::uint64_t& value)
{
    if (text.empty() || text[0] < '0' || text[0] > '9') { return false; }
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
    if (end == nullptr || *end != '\0' || errno == ERANGE) { return false; }
    value = static_cast<std::uint64_t>(parsed);
    return true;
}

// A timeout is carried as milliseconds in a 32 bit field, so anything above
// that is pinned to the largest representable value instead of wrapping.
std::uint32_t NarrowTimeout(std::uint64_t value)
{
    constexpr std::uint64_t kMaxTimeoutMs = 0xFFFFFFFFULL;
    return static_cast<std::uint32_t>(value > kMaxTimeoutMs ? kMaxTimeoutMs : value);
}

std::uint32_t AtLeastOne(std::uint32_t value) { return value == 0U ? 1U : value; }

// A zero from the store side means "not advertised", so the local wish survives.
template <typename T>
void PreferAdvertised(T& adopted, T local)
{
    if (adopted == 0) { adopted = local; }
}

// Reads an unsigned knob from the transport attributes and leaves the field
// untouched when the key is absent.  A value that is not a number or that does
// not fit the field is reported instead of being cast into something else: the
// verbs fields are narrow (timeout and retry counts are three to five bits) and
// a silent truncation would configure a link nobody can debug.
template <typename T>
bool ApplyUnsigned(const std::unordered_map<std::string, std::string>& attrs, const char* key,
                   T& field, std::uint32_t maxValue, std::string& error)
{
    const auto found = attrs.find(key);
    if (found == attrs.end()) { return true; }
    std::uint64_t value = 0;
    if (!ParseUnsigned(found->second, value)) {
        error = std::string(key) + " must be an unsigned number, got '" + found->second + "'";
        return false;
    }
    if (value > maxValue) {
        error = std::string(key) + " is " + std::to_string(value) +
                ", which is above the maximum of " + std::to_string(maxValue);
        return false;
    }
    field = static_cast<T>(value);
    return true;
}

}  // namespace

CpuKvOverRoceTransProvider::CpuKvOverRoceTransProvider(const TransportConfig& config)
    : config_(config)
{
}

CpuKvOverRoceTransProvider::~CpuKvOverRoceTransProvider()
{
    std::lock_guard<std::mutex> guard(mutex_);
    for (const auto& link : links_) {
        if (link == nullptr) { continue; }
        // Announce the teardown exactly like DeleteConnections does, but ignore a
        // failure: a destructor cannot report one, and the peer still notices the
        // closed socket.
        std::vector<std::uint8_t> frame;
        std::string ignored;
        if (EncodeDisconnect(frame).ok()) { static_cast<void>(SendFrame(*link, frame, ignored)); }
        TeardownLink(*link);
    }
    links_.clear();
    handleToLink_.clear();
    // The device and the registration table belong to the process: only the
    // entries this provider created are released here, so what another provider
    // registered stays usable for it.
    ProcessState& state = ProcessState::Instance();
    std::vector<Registration> owned;
    {
        std::lock_guard<std::mutex> memoryGuard(state.memoryMutex);
        std::vector<MRHandle> released;
        for (auto entry = state.registrations.begin(); entry != state.registrations.end();) {
            if (entry->second.owner != this) {
                ++entry;
                continue;
            }
            released.push_back(entry->first);
            owned.push_back(entry->second);
            entry = state.registrations.erase(entry);
        }
        // A binding another provider made describes one of these regions, so it
        // cannot stay resolvable once the region is gone.
        for (auto binding = state.registrations.begin(); binding != state.registrations.end();) {
            const auto source = std::find(released.begin(), released.end(), binding->second.source);
            if (source != released.end()) {
                binding = state.registrations.erase(binding);
            } else {
                ++binding;
            }
        }
    }
    // Regions the client never unregistered still have to be released, but the
    // protection domain itself now outlives this provider.
    for (Registration& registration : owned) {
        if (registration.mr != nullptr) { ibv_dereg_mr(registration.mr); }
    }
}

VerbDevice& CpuKvOverRoceTransProvider::Device()
{
    // Only the verbs path reaches the device, and every such path opens it
    // first.  The static fallback keeps a misuse from becoming a null
    // dereference instead of a visible "no device" error.
    static VerbDevice unused;
    VerbDevice* device = device_.load(std::memory_order_acquire);
    return device != nullptr ? *device : unused;
}

std::string CpuKvOverRoceTransProvider::Attr(const std::string& key) const
{
    const auto found = config_.attrs.find(key);
    return found == config_.attrs.end() ? std::string() : found->second;
}

std::uint32_t CpuKvOverRoceTransProvider::ResolveTimeout(std::uint32_t timeout) const
{
    if (timeout != 0) { return timeout; }
    std::uint64_t configured = 0;
    // Zero keeps meaning "not configured" for a timeout: it would make the very
    // first poll return instead of waiting for the peer.
    if (ParseUnsigned(Attr(kAttrConnectTimeout), configured) && configured != 0) {
        return NarrowTimeout(configured);
    }
    return kDefaultOutOfBandTimeoutMs;
}

std::uint32_t CpuKvOverRoceTransProvider::HandshakeTimeout() const
{
    std::uint64_t configured = 0;
    if (ParseUnsigned(Attr(kAttrHandshakeTimeout), configured) && configured != 0) {
        return NarrowTimeout(configured);
    }
    return kDefaultHandshakeTimeoutMs;
}

bool CpuKvOverRoceTransProvider::UseVerbsBackend() const
{
    const std::string backend = Attr(kAttrQpBackend);
    return backend.empty() || backend == kBackendVerbs;
}

bool CpuKvOverRoceTransProvider::IsInitiator() const
{
    const std::string role = Attr(kAttrRole);
    return role.empty() || role == "client";
}

CpuKvOverRoceTransProvider::ConnectionHandle CpuKvOverRoceTransProvider::MakeHandle(
    std::uint64_t linkId, std::uint32_t qpIndex)
{
    const std::uint64_t value = (linkId << kHandleIndexBits) | (qpIndex + 1U);
    return reinterpret_cast<ConnectionHandle>(static_cast<std::uintptr_t>(value));
}

Status CpuKvOverRoceTransProvider::OpenLink(const std::string& localIp, const std::string& remoteIp,
                                            std::uint32_t port, std::uint32_t timeout,
                                            TcpChannel& channel) const
{
    const std::string role = Attr(kAttrRole);
    const std::uint32_t effectiveTimeout = ResolveTimeout(timeout);
    std::string error;

    if (role.empty() || role == "client") {
        Status status =
            TcpChannel::Connect(localIp, remoteIp, port, effectiveTimeout, channel, error);
        if (!status.ok()) {
            return Failure(status.code, "out-of-band TCP connect to " + remoteIp + ":" +
                                            std::to_string(port) + " failed: " + error);
        }
        return Status::OK();
    }

    if (role != "server") {
        return InvalidArgument("unsupported " + std::string(kAttrRole) + " '" + role +
                               "', expected 'client' or 'server'");
    }

    std::string bindIp = Attr(kAttrBindIp);
    if (bindIp.empty()) { bindIp = localIp; }
    TcpChannel listener;
    Status listened =
        TcpChannel::Listen(bindIp, static_cast<std::uint16_t>(port), 1, listener, error);
    if (!listened.ok()) {
        // A listening socket cannot be shared, so a second provider on the same
        // address fails here; say so instead of leaving a bare bind error.
        const std::string hint = error.find("Address already in use") == std::string::npos
                                     ? std::string()
                                     : std::string(
                                           " (another provider is already listening on"
                                           " that address; the server side of a link"
                                           " cannot share the port)");
        return Failure(listened.code, "out-of-band TCP listen on " +
                                          (bindIp.empty() ? std::string("*") : bindIp) + ":" +
                                          std::to_string(port) + " failed: " + error + hint);
    }
    Status accepted = TcpChannel::Accept(listener, effectiveTimeout, channel, error);
    if (!accepted.ok()) {
        return Failure(accepted.code, "out-of-band TCP accept on port " + std::to_string(port) +
                                          " failed: " + error);
    }
    return Status::OK();
}

Status CpuKvOverRoceTransProvider::BuildVerbConfig(const std::string& localIp, VerbQpConfig& config,
                                                   std::string& error) const
{
    error.clear();
    config.localIp = localIp;
    config.hcaName = Attr("roce_hca_name");

    // The port number has to stay inside the byte the verbs layer takes. A value
    // that does not parse is an error like every other knob, keeping the default
    // instead would hide a typo.
    if (const std::string value = Attr("roce_hca_port"); !value.empty()) {
        std::uint64_t hcaPort = 0;
        if (!ParseUnsigned(value, hcaPort)) {
            error = "roce_hca_port '" + value + "' is not a number";
            return Failure(StatusCode::INVALID_ARGUMENT, error);
        }
        if (hcaPort == 0U || hcaPort > 0xFFU) {
            error = "roce_hca_port is " + std::to_string(hcaPort) + ", which is not a port number";
            return Failure(StatusCode::INVALID_ARGUMENT, error);
        }
        config.port = static_cast<std::uint8_t>(hcaPort);
    }

    // The limits are the widths the verbs attributes and the protocol table give
    // these fields: 8 bits for sl/traffic class, 5 for the timeout and the RNR
    // timer, 3 for the retry counters, 8 for the queue pair atomicity.
    if (config_.attrs.find("roce_gid_index") == config_.attrs.end()) {
        error = "roce_gid_index is required (the provider does not choose a GID automatically)";
        return Failure(StatusCode::INVALID_ARGUMENT, error);
    }
    if (!ApplyUnsigned(config_.attrs, "roce_gid_index", config.gidIndex, 0xFFU, error) ||
        !ApplyUnsigned(config_.attrs, "roce_sq_depth", config.sendDepth, 0x7FFFU, error) ||
        !ApplyUnsigned(config_.attrs, "roce_rq_depth", config.recvDepth, 0x7FFFU, error) ||
        !ApplyUnsigned(config_.attrs, "roce_cq_depth", config.cqDepth, 0xFFFFU, error) ||
        !ApplyUnsigned(config_.attrs, "roce_inline_threshold", config.inlineThreshold, 0x1000U,
                       error) ||
        !ApplyUnsigned(config_.attrs, "roce_sl", config.sl, 0xFFU, error) ||
        !ApplyUnsigned(config_.attrs, "roce_traffic_class", config.trafficClass, 0xFFU, error) ||
        !ApplyUnsigned(config_.attrs, "roce_qp_timeout", config.timeout, 31U, error) ||
        !ApplyUnsigned(config_.attrs, "roce_retry_cnt", config.retryCnt, 7U, error) ||
        !ApplyUnsigned(config_.attrs, "roce_rnr_retry", config.rnrRetryCnt, 7U, error) ||
        !ApplyUnsigned(config_.attrs, "roce_min_rnr_timer", config.rnrTimer, 31U, error) ||
        !ApplyUnsigned(config_.attrs, "roce_qp_rd_atom", config.qpRdAtom, 0xFFU, error)) {
        return Failure(StatusCode::INVALID_ARGUMENT, error);
    }
    return Status::OK();
}

Status CpuKvOverRoceTransProvider::OpenDevice(std::string& error)
{
    if (device_.load(std::memory_order_acquire) != nullptr) {
        // The device was fixed by the first thing that needed it, either this
        // provider's own configuration or a registration it bound.
        return Status::OK();
    }
    VerbQpConfig verbConfig;
    Status built = BuildVerbConfig(std::string(), verbConfig, error);
    if (!built.ok()) { return built; }

    // The GID is only known once the port has been queried, and it is part of the
    // identity: two providers on the same adapter and port but with different
    // GIDs (two IP addresses on one NIC, say) must not share a protection domain,
    // because the GID index is what every queue pair puts into the global route
    // header.  So open first and let the resolved identity decide the bucket; a
    // candidate that turns out to be a duplicate is simply closed again.
    auto candidate = std::make_unique<DeviceEntry>();
    Status opened = VerbDevice::Open(verbConfig, candidate->device, error);
    if (!opened.ok()) { return opened; }
    candidate->name = candidate->device.Name();
    candidate->port = verbConfig.port;
    const DeviceKey key{candidate->name, candidate->port, candidate->device.GidIndex()};

    ProcessState& state = ProcessState::Instance();
    std::lock_guard<std::mutex> guard(state.deviceMutex);
    auto existing = state.devices.find(key);
    if (existing != state.devices.end()) {
        device_.store(&existing->second->device, std::memory_order_release);
        return Status::OK();
    }
    device_.store(&candidate->device, std::memory_order_release);
    state.devices.emplace(key, std::move(candidate));
    return Status::OK();
}

Status CpuKvOverRoceTransProvider::SendFrame(const Link& link,
                                             const std::vector<std::uint8_t>& frame,
                                             std::string& error) const
{
    Status status = link.channel.SendAll(frame.data(), frame.size(), HandshakeTimeout(), error);
    if (!status.ok()) {
        error = "send of command " + std::to_string(frame[kWireCmdOffset]) + " failed: " + error;
        return Failure(status.code, error);
    }
    return Status::OK();
}

Status CpuKvOverRoceTransProvider::ReceiveFrame(const Link& link, Message& message,
                                                std::string& error) const
{
    std::uint8_t header[kWireHeaderBytes] = {0};
    Status status = link.channel.RecvAll(header, sizeof(header), HandshakeTimeout(), error);
    if (!status.ok()) {
        error = "receive of the message header failed: " + error;
        return Failure(status.code, error);
    }
    std::uint32_t bodyLength = 0;
    if (!PeekBodyLength(header, sizeof(header), bodyLength)) {
        error = "peer sent a truncated message header";
        return Failure(StatusCode::INVALID_ARGUMENT, error);
    }
    if (bodyLength > kWireBodyBytes) {
        error = "peer announced a body of " + std::to_string(bodyLength) + " bytes";
        return Failure(StatusCode::INVALID_ARGUMENT, error);
    }

    std::vector<std::uint8_t> frame(kWireHeaderBytes + bodyLength, 0U);
    std::memcpy(frame.data(), header, sizeof(header));
    if (bodyLength > 0U) {
        status = link.channel.RecvAll(frame.data() + kWireHeaderBytes, bodyLength,
                                      HandshakeTimeout(), error);
        if (!status.ok()) {
            error = "receive of the message body failed: " + error;
            return Failure(status.code, error);
        }
    }

    std::string decodeError;
    status = DecodeMessage(frame.data(), frame.size(), message, decodeError);
    if (!status.ok()) {
        error = decodeError;
        return Failure(StatusCode::INVALID_ARGUMENT, error);
    }
    return Status::OK();
}

Status CpuKvOverRoceTransProvider::ReceiveMessage(const Link& link,
                                                  const std::vector<std::uint8_t>& acceptedCommands,
                                                  Message& message, std::string& error) const
{
    Status status = ReceiveFrame(link, message, error);
    if (!status.ok()) { return status; }
    const bool accepted = std::any_of(acceptedCommands.begin(), acceptedCommands.end(),
                                      [&message](std::uint8_t cmd) { return message.cmd == cmd; });
    if (!accepted) {
        error = "unexpected message command " + std::to_string(message.cmd);
        return Failure(StatusCode::INVALID_ARGUMENT, error);
    }
    return Status::OK();
}

Status CpuKvOverRoceTransProvider::ExchangeCapabilities(Link& link, std::string& error)
{
    const bool initiator = IsInitiator();

    std::vector<std::uint8_t> frame;
    Message message;

    if (initiator) {
        CapabilityRequest request;
        Status encoded = EncodeCapabilityRequest(request, frame);
        if (!encoded.ok()) {
            error = "failed to encode the capability request";
            return encoded;
        }
        Status sent = SendFrame(link, frame, error);
        if (!sent.ok()) { return sent; }
        Status received = ReceiveMessage(link, {kWireCmdCapability}, message, error);
        if (!received.ok()) { return received; }
        if (!message.capabilityReply) {
            error = "peer answered the capability request with a request";
            return Failure(StatusCode::INVALID_ARGUMENT, error);
        }
        link.peerCapabilities = message.capabilities;
        if (message.capabilities.majorVersion != link.localCapabilities.majorVersion) {
            error = "peer speaks capability version " +
                    std::to_string(message.capabilities.majorVersion) + "." +
                    std::to_string(message.capabilities.minorVersion) +
                    " while this provider speaks " +
                    std::to_string(link.localCapabilities.majorVersion) + "." +
                    std::to_string(link.localCapabilities.minorVersion);
            return Failure(StatusCode::INVALID_ARGUMENT, error);
        }
        link.negotiatedCapabilities =
            AdoptCapabilities(link.localCapabilities, link.peerCapabilities);
        return Status::OK();
    }

    Status received = ReceiveMessage(link, {kWireCmdCapability}, message, error);
    if (!received.ok()) { return received; }
    if (message.capabilityReply) {
        error = "peer answered before sending its capability request";
        return Failure(StatusCode::INVALID_ARGUMENT, error);
    }
    Status encoded = EncodeCapabilityReply(link.localCapabilities, frame);
    if (!encoded.ok()) {
        error = "failed to encode the capability reply";
        return encoded;
    }
    return SendFrame(link, frame, error);
}

Status CpuKvOverRoceTransProvider::ValidateCapabilityLimits(std::string& error) const
{
    // These limits travel in 16 bit capability fields. Truncating them would
    // advertise a value the provider cannot keep and would corrupt the handshake
    // silently, so a limit that does not fit is rejected up front.
    constexpr std::uint32_t kMaxCapabilityValue = 0xFFFFU;
    struct Limit {
        const char* name;
        std::size_t value;
    };
    const Limit limits[] = {
        {"max_inflight_tasks",     config_.maxInflightTasks},
        {"asu_batch_store_io_num", config_.batchStoreIoNum },
        {"asu_batch_load_io_num",  config_.batchLoadIoNum  },
        {"asu_query_io_num",       config_.queryIoNum      },
        {"asu_delete_io_num",      config_.deleteIoNum     },
    };
    for (const Limit& limit : limits) {
        if (limit.value > kMaxCapabilityValue) {
            error = std::string(limit.name) + " is " + std::to_string(limit.value) +
                    ", which exceeds the " + std::to_string(kMaxCapabilityValue) +
                    " limit of a capability field";
            return Failure(StatusCode::INVALID_ARGUMENT, error);
        }
    }
    return Status::OK();
}

Capabilities CpuKvOverRoceTransProvider::LocalCapabilities(std::uint32_t qpNum) const
{
    // Every value below fits: the queue pair count is capped by kMaxQpPerLink and
    // CreateConnection rejects limits that do not fit a capability field.
    Capabilities local;
    local.majorVersion = kWireCapabilityMajorVersion;
    local.minorVersion = kWireCapabilityMinorVersion;
    local.queueNum = static_cast<std::uint16_t>(qpNum);
    local.ioqKeyBatch = static_cast<std::uint16_t>(AtLeastOne(config_.maxInflightTasks));
    local.totalKeyBatch = static_cast<std::uint16_t>(AtLeastOne(config_.maxInflightTasks));
    local.storeKeyBatch = static_cast<std::uint16_t>(AtLeastOne(config_.batchStoreIoNum));
    local.retrieveKeyBatch = static_cast<std::uint16_t>(AtLeastOne(config_.batchLoadIoNum));
    local.deleteKeyBatch = static_cast<std::uint16_t>(AtLeastOne(config_.deleteIoNum));
    local.existKeyBatch = static_cast<std::uint16_t>(AtLeastOne(config_.queryIoNum));
    local.keyLength = 8;
    return local;
}

Status CpuKvOverRoceTransProvider::CreateQueuePairs(Link& link, std::uint32_t qpNum,
                                                    const std::string& localIp, std::string& error)
{
    link.localQpInfo = QpInfo{};
    // qpNum never exceeds kMaxQpPerLink here, so this byte cannot wrap.
    link.localQpInfo.totalQpNum = static_cast<std::uint8_t>(qpNum);
    link.localQpInfo.mtu = 5;

    if (!UseVerbsBackend()) {
        // Test backend: the protocol runs end to end while the queue pairs are
        // simulated, so the handshake can be exercised without RDMA hardware.
        link.verbs = false;
        link.linkPsn = NextPsn();
        link.localQpInfo.startPsn = link.linkPsn;
        for (std::uint32_t index = 0; index < qpNum; ++index) {
            link.localQpInfo.qpn[index] = 0x1000U + index;
        }
        link.linkQpCount = qpNum;
        return Status::OK();
    }

    Status opened = OpenDevice(error);
    if (!opened.ok()) { return opened; }
    VerbQpConfig verbConfig;
    Status built = BuildVerbConfig(localIp, verbConfig, error);
    if (!built.ok()) { return built; }
    link.verbs = true;
    link.linkPsn = NextPsn();
    link.localQpInfo.startPsn = link.linkPsn;
    for (std::uint32_t index = 0; index < qpNum; ++index) {
        auto qp = std::make_unique<VerbQp>();
        Status created = VerbQp::Create(Device(), verbConfig, *qp, error);
        if (!created.ok()) { return created; }
        link.localQpInfo.qpn[index] = qp->Qpn();
        link.queuePairs.push_back(std::move(qp));
    }
    link.linkQpCount = static_cast<std::uint32_t>(link.queuePairs.size());
    std::memcpy(link.localQpInfo.gid, Device().Gid(), sizeof(link.localQpInfo.gid));
    link.localQpInfo.lid = Device().Lid();
    link.localQpInfo.mtu = Device().MtuCode();
    return Status::OK();
}

Status CpuKvOverRoceTransProvider::ExchangeConnectRequests(Link& link, std::string& error)
{
    const bool initiator = IsInitiator();
    const std::uint8_t cmd = initiator ? kWireCmdConnectRequest : kWireCmdConnectResponse;

    std::vector<std::uint8_t> frame;
    Status encoded = EncodeConnectRequest(cmd, link.localQpInfo, frame);
    if (!encoded.ok()) {
        error = "failed to encode the connect request";
        return encoded;
    }
    Status sent = SendFrame(link, frame, error);
    if (!sent.ok()) { return sent; }

    Message message;
    Status received =
        ReceiveMessage(link, {kWireCmdConnectRequest, kWireCmdConnectResponse}, message, error);
    if (!received.ok()) { return received; }
    if (message.qp.totalQpNum == 0U) {
        error = "peer reported zero queue pairs";
        return Failure(StatusCode::INVALID_ARGUMENT, error);
    }
    // Both sides keep the smaller of the two counts, so a store that accepts
    // fewer queue pairs than requested still links successfully.
    const std::uint32_t local = link.localQpInfo.totalQpNum;
    const std::uint32_t remote = message.qp.totalQpNum;
    const std::uint32_t effective = local < remote ? local : remote;
    link.peerQpInfo = message.qp;
    link.linkQpCount = effective;
    if (effective < local) {
        TrimQueuePairs(link, effective);
        link.localQpInfo.totalQpNum = static_cast<std::uint8_t>(effective);
    }
    return Status::OK();
}

Status CpuKvOverRoceTransProvider::ConnectQueuePairs(Link& link, std::string& error)
{
    if (!link.verbs) { return Status::OK(); }
    VerbQpConfig verbConfig;
    Status built = BuildVerbConfig(std::string(), verbConfig, error);
    if (!built.ok()) { return built; }
    for (std::size_t index = 0; index < link.queuePairs.size(); ++index) {
        Status connected = link.queuePairs[index]->Connect(
            Device(), verbConfig, link.peerQpInfo, link.peerQpInfo.qpn[index], link.linkPsn, error);
        if (!connected.ok()) {
            error = "queue pair " + std::to_string(index) + ": " + error;
            return connected;
        }
    }
    return Status::OK();
}

Status CpuKvOverRoceTransProvider::ExchangeConnectDone(Link& link, std::string& error)
{
    std::vector<std::uint8_t> frame;
    Status encoded = EncodeConnectDone(frame);
    if (!encoded.ok()) {
        error = "failed to encode the connect done message";
        return encoded;
    }
    // The connect done message is a one-way notification: the peer does not
    // answer it, so waiting for a frame here would only stall until the
    // handshake timeout expires.
    return SendFrame(link, frame, error);
}

auto CpuKvOverRoceTransProvider::FindLinkPosition(std::uint64_t linkId)
    -> std::vector<std::unique_ptr<Link>>::iterator
{
    return std::find_if(links_.begin(), links_.end(), [linkId](const std::unique_ptr<Link>& link) {
        return link != nullptr && link->id == linkId;
    });
}

void CpuKvOverRoceTransProvider::TeardownLink(Link& link) noexcept { link.queuePairs.clear(); }

Capabilities CpuKvOverRoceTransProvider::AdoptCapabilities(const Capabilities& local,
                                                           const Capabilities& peer)
{
    // The values the store side reported become the connection's operating point.
    Capabilities adopted = peer;
    PreferAdvertised(adopted.queueNum, local.queueNum);
    // The contract hands back at most the requested number of handles, so a
    // store that offers more queue pairs than we asked for is pulled down to
    // the request instead of widening the connection.
    if (adopted.queueNum > local.queueNum) { adopted.queueNum = local.queueNum; }
    PreferAdvertised(adopted.adminqDepth, local.adminqDepth);
    PreferAdvertised(adopted.ioqDepth, local.ioqDepth);
    PreferAdvertised(adopted.ioqKeyBatch, local.ioqKeyBatch);
    PreferAdvertised(adopted.totalKeyBatch, local.totalKeyBatch);
    PreferAdvertised(adopted.srMdts, local.srMdts);
    PreferAdvertised(adopted.bsrMdts, local.bsrMdts);
    PreferAdvertised(adopted.storeKeyBatch, local.storeKeyBatch);
    PreferAdvertised(adopted.retrieveKeyBatch, local.retrieveKeyBatch);
    PreferAdvertised(adopted.existKeyBatch, local.existKeyBatch);
    PreferAdvertised(adopted.deleteKeyBatch, local.deleteKeyBatch);
    PreferAdvertised(adopted.keyLength, local.keyLength);

    // A single link carries at most this many queue pairs on the wire.
    if (adopted.queueNum > kMaxQpPerLink) {
        adopted.queueNum = static_cast<std::uint16_t>(kMaxQpPerLink);
    }
    return adopted;
}

std::uint32_t CpuKvOverRoceTransProvider::NegotiatedQpNum(const Link& link) const
{
    // AdoptCapabilities() substituted the local wish when the store advertised
    // nothing, so only the protocol ceiling is left to apply.
    const std::uint32_t agreed = link.negotiatedCapabilities.queueNum;
    return agreed > kMaxQpPerLink ? kMaxQpPerLink : agreed;
}

void CpuKvOverRoceTransProvider::TrimQueuePairs(Link& link, std::uint32_t count)
{
    if (link.queuePairs.size() > count) { link.queuePairs.resize(count); }
    for (std::uint32_t index = count; index < kMaxQpPerLink; ++index) {
        link.localQpInfo.qpn[index] = 0U;
    }
    link.linkQpCount = count;
}

Status CpuKvOverRoceTransProvider::CreateConnection(
    const std::string& localIp, const std::string& remoteIp, std::uint32_t port,
    std::uint32_t qpNum, std::uint32_t timeout, std::vector<ConnectionHandle>& connectionHandles)
{
    // The contract requires the handle list to stay empty when the call fails.
    connectionHandles.clear();

    if (qpNum == 0) { return InvalidArgument("qpNum must be greater than zero"); }
    if (qpNum > kMaxQpPerLink) {
        return InvalidArgument("qpNum " + std::to_string(qpNum) +
                               " exceeds the protocol maximum of " + std::to_string(kMaxQpPerLink) +
                               " queue pairs per link");
    }
    if (remoteIp.empty()) { return InvalidArgument("remote IP must not be empty"); }
    if (port == 0 || port > 0xFFFFU) {
        return InvalidArgument("port " + std::to_string(port) + " is out of range");
    }
    const std::string backend = Attr(kAttrQpBackend);
    if (!backend.empty() && backend != kBackendVerbs && backend != kBackendSimulated) {
        return InvalidArgument("unsupported " + std::string(kAttrQpBackend) + " '" + backend +
                               "', expected 'verbs' or 'simulated'");
    }
    std::string limitError;
    Status limits = ValidateCapabilityLimits(limitError);
    if (!limits.ok()) { return Failure(limits.code, limitError); }

    // Validate the queue pair knobs before a TCP connection is opened: a bad
    // value would otherwise only surface on the first real problem, and on the
    // simulated backend never at all.
    {
        VerbQpConfig checked;
        std::string configError;
        Status built = BuildVerbConfig(localIp, checked, configError);
        if (!built.ok()) { return Failure(built.code, configError); }
    }

    TcpChannel channel;
    Status opened = OpenLink(localIp, remoteIp, port, timeout, channel);
    if (!opened.ok()) { return opened; }

    auto link = std::make_unique<Link>();
    link->remoteIp = remoteIp;
    link->remotePort = static_cast<std::uint16_t>(port);
    link->channel = std::move(channel);
    link->localCapabilities = LocalCapabilities(qpNum);
    link->negotiatedCapabilities = link->localCapabilities;

    std::string error;
    Status handshake = ExchangeCapabilities(*link, error);
    if (handshake.ok()) {
        // The store side decided how many queue pairs it accepts, so create that
        // many instead of insisting on the requested count.
        handshake = CreateQueuePairs(*link, NegotiatedQpNum(*link), localIp, error);
    }
    if (handshake.ok()) { handshake = ExchangeConnectRequests(*link, error); }
    if (handshake.ok()) { handshake = ConnectQueuePairs(*link, error); }
    if (handshake.ok()) { handshake = ExchangeConnectDone(*link, error); }
    if (!handshake.ok()) {
        TeardownLink(*link);
        return Failure(handshake.code, "linking with " + remoteIp + ":" + std::to_string(port) +
                                           " failed: " + error);
    }

    std::lock_guard<std::mutex> guard(mutex_);
    link->id = nextLinkId_++;
    const std::uint64_t linkId = link->id;

    connectionHandles.reserve(link->linkQpCount);
    for (std::uint32_t index = 0; index < link->linkQpCount; ++index) {
        const ConnectionHandle handle = MakeHandle(linkId, index);
        handleToLink_.emplace(handle, linkId);
        connectionHandles.push_back(handle);
    }
    links_.push_back(std::move(link));
    return Status::OK();
}

std::vector<Status> CpuKvOverRoceTransProvider::DeleteConnections(
    const std::vector<ConnectionHandle>& connectionHandles)
{
    std::vector<Status> results;
    results.reserve(connectionHandles.size());

    for (const ConnectionHandle handle : connectionHandles) {
        std::uint64_t linkId = 0;
        bool lastHandle = false;
        {
            std::lock_guard<std::mutex> guard(mutex_);
            const auto found = handleToLink_.find(handle);
            if (found == handleToLink_.end()) {
                results.push_back(InvalidArgument("unknown connection handle"));
                continue;
            }
            linkId = found->second;
            handleToLink_.erase(found);
            lastHandle =
                std::none_of(handleToLink_.begin(), handleToLink_.end(),
                             [linkId](const auto& entry) { return entry.second == linkId; });
        }

        if (lastHandle) {
            std::unique_ptr<Link> closing;
            {
                std::lock_guard<std::mutex> guard(mutex_);
                const auto position = FindLinkPosition(linkId);
                if (position != links_.end()) {
                    closing = std::move(*position);
                    links_.erase(position);
                }
            }
            if (closing != nullptr) {
                std::string error;
                std::vector<std::uint8_t> frame;
                if (EncodeDisconnect(frame).ok()) {
                    // The disconnect request is one-way as well: announce the
                    // teardown, then drop the link without waiting for an answer.
                    static_cast<void>(SendFrame(*closing, frame, error));
                }
                TeardownLink(*closing);
            }
        }
        results.push_back(Status::OK());
    }
    return results;
}

Status CpuKvOverRoceTransProvider::GetServerCapabilities(ConnectionHandle connectionHandle,
                                                         ServerKvCapabilities& capabilities)
{
    capabilities = ServerKvCapabilities{};
    std::lock_guard<std::mutex> guard(mutex_);
    const auto found = handleToLink_.find(connectionHandle);
    if (found == handleToLink_.end()) { return InvalidArgument("unknown connection handle"); }
    const auto position = FindLinkPosition(found->second);
    if (position == links_.end()) {
        return InvalidArgument("connection handle refers to a closed link");
    }
    const Link& link = **position;
    const Capabilities& peer = link.peerCapabilities;
    // The negotiated count, not what the store advertised: it is what the link
    // actually carries, and the caller sizes its channels from the handles.
    capabilities.queueNum = static_cast<std::uint16_t>(link.linkQpCount);
    capabilities.ioQueueDepth = peer.ioqDepth;
    capabilities.ioQueueKeyConcurrency = peer.ioqKeyBatch;
    capabilities.connectionKeyConcurrency = peer.totalKeyBatch;
    capabilities.singleValueMaxBytes = peer.srMdts;
    capabilities.batchValueMaxBytes = peer.bsrMdts;
    capabilities.batchStoreKeys = peer.storeKeyBatch;
    capabilities.batchLoadKeys = peer.retrieveKeyBatch;
    capabilities.deleteKeys = peer.deleteKeyBatch;
    capabilities.queryKeys = peer.existKeyBatch;
    capabilities.keyLength = peer.keyLength;
    capabilities.kvCapabilities = 0;
    return Status::OK();
}

std::vector<Status> CpuKvOverRoceTransProvider::Send(const std::vector<SendIoBatch>& ioBatches,
                                                     std::uint32_t kernelCount,
                                                     std::uint32_t quietCount)
{
    (void)kernelCount;
    (void)quietCount;
    return NotImplementedForEach("Send", ioBatches.size());
}

Status CpuKvOverRoceTransProvider::RegisterMemory(
    const std::vector<RegisterMemoryDesc>& memoryDescs, std::vector<MRHandle>& mrHandles)
{
    mrHandles.clear();
    if (memoryDescs.empty()) { return InvalidArgument("no memory descriptor was given"); }
    for (const RegisterMemoryDesc& desc : memoryDescs) {
        if (desc.addr == 0) { return InvalidArgument("memory address must not be zero"); }
        if (desc.size == 0) { return InvalidArgument("memory size must not be zero"); }
    }

    ProcessState& state = ProcessState::Instance();
    // The simulated backend has no protection domain to register against; the
    // bookkeeping still runs so the rest of the client path stays identical.
    const bool verbs = UseVerbsBackend();
    if (verbs) {
        std::string openError;
        Status opened = OpenDevice(openError);
        if (!opened.ok()) { return opened; }
    }

    std::vector<MRHandle> handles;
    std::vector<Registration> staged;
    handles.reserve(memoryDescs.size());
    staged.reserve(memoryDescs.size());

    for (const RegisterMemoryDesc& desc : memoryDescs) {
        Registration registration;
        registration.addr = desc.addr;
        registration.size = desc.size;
        registration.owner = this;
        MRHandle handle = 0;
        if (verbs) {
            registration.device = device_.load(std::memory_order_acquire);
            registration.mr = ibv_reg_mr(Device().Pd(), reinterpret_cast<void*>(desc.addr),
                                         desc.size, kRegistrationAccess);
            if (registration.mr == nullptr) {
                std::string hint;
                if (desc.memoryType == MemType::MEM_DEVICE) {
                    hint =
                        "; device memory needs peer-memory support on the HCA, check "
                        "nvidia-peermem or the dmabuf support of the driver";
                }
                for (const Registration& done : staged) {
                    if (done.mr != nullptr) { ibv_dereg_mr(done.mr); }
                }
                return Failure(StatusCode::IO_ERROR, "ibv_reg_mr of " + std::to_string(desc.size) +
                                                         " bytes failed: " + std::strerror(errno) +
                                                         hint);
            }
            registration.tokenId = registration.mr->rkey;
            handle = static_cast<MRHandle>(reinterpret_cast<std::uintptr_t>(registration.mr));
        } else {
            std::lock_guard<std::mutex> guard(state.memoryMutex);
            handle = state.nextBookkeepingId++;
            registration.tokenId = kSimulatedRkeyBase + static_cast<std::uint32_t>(handle);
        }

        staged.push_back(registration);
        handles.push_back(handle);
    }

    {
        std::lock_guard<std::mutex> guard(state.memoryMutex);
        for (std::size_t index = 0; index < handles.size(); ++index) {
            state.registrations[handles[index]] = staged[index];
        }
    }
    mrHandles = std::move(handles);
    return Status::OK();
}

Status CpuKvOverRoceTransProvider::BindMemory(const std::vector<BindMemoryDesc>& memoryDescs,
                                              std::vector<MRHandle>& mrHandles)
{
    mrHandles.clear();
    if (memoryDescs.empty()) { return InvalidArgument("no memory descriptor was given"); }
    for (const BindMemoryDesc& desc : memoryDescs) {
        if (desc.addr == 0) { return InvalidArgument("memory address must not be zero"); }
        if (desc.size == 0) { return InvalidArgument("memory size must not be zero"); }
    }

    ProcessState& state = ProcessState::Instance();
    std::lock_guard<std::mutex> guard(state.memoryMutex);
    // The process is the memory provider for every region its providers touch:
    // an address is only bindable when some provider in the process registered
    // it and the requested range fits inside that registration.  A token that
    // describes memory registered nowhere cannot be honoured, so it is refused
    // outright instead of being recorded against a stale key.
    std::vector<MRHandle> sources(memoryDescs.size(), 0);
    // A provider that has not opened its device yet may adopt the device of the
    // regions it binds, but only once the whole batch passed: pinning the device
    // on the way to a failure would leave the provider unable to serve the
    // retry, which is the batch the caller actually wants.
    VerbDevice* seeded = device_.load(std::memory_order_acquire);
    for (std::size_t index = 0; index < memoryDescs.size(); ++index) {
        const BindMemoryDesc& desc = memoryDescs[index];
        bool backendMismatch = false;
        bool deviceMismatch = false;
        const MRHandle source =
            FindCoveringRegistration(desc.addr, desc.size, seeded, backendMismatch, deviceMismatch);
        if (source == 0) {
            // A region registered by the simulated backend has no access record,
            // so a verbs provider cannot address it, and the other way round the
            // token of a real region would be replaced by a synthetic one.
            if (backendMismatch) {
                return Failure(StatusCode::UNSUPPORTED,
                               "the covering registration was made with a different queue pair "
                               "backend");
            }
            // The token of a real registration is its rkey, and an rkey is only
            // valid on the device that produced it.
            if (deviceMismatch) {
                return Failure(StatusCode::UNSUPPORTED,
                               "the covering registration belongs to another (adapter, port) "
                               "pair");
            }
            return Failure(StatusCode::UNSUPPORTED,
                           "binding memory that this provider did not register is not supported");
        }
        const Registration& covering = state.registrations.at(source);
        // Every descriptor of the batch is checked against the same device, so a
        // batch that spans two adapters fails instead of pinning one of them.
        if (seeded == nullptr && covering.device != nullptr) {
            seeded = const_cast<VerbDevice*>(covering.device);
        }
        sources[index] = source;
    }

    // Commit the adopted device only now that every descriptor passed.
    if (seeded != nullptr && device_.load(std::memory_order_acquire) == nullptr) {
        VerbDevice* expected = nullptr;
        if (!device_.compare_exchange_strong(expected, seeded, std::memory_order_acq_rel)) {
            if (expected != seeded) {
                return Failure(StatusCode::UNSUPPORTED,
                               "the covering registrations belong to another (adapter, port) pair");
            }
        }
    }

    // The token travels through untouched, so GetMemTokenId() on the returned
    // handle reports exactly what the caller passed in, which is what the
    // client verifies once the call returns.
    mrHandles.reserve(memoryDescs.size());
    for (std::size_t index = 0; index < memoryDescs.size(); ++index) {
        const BindMemoryDesc& desc = memoryDescs[index];
        Registration registration;
        registration.addr = desc.addr;
        registration.size = desc.size;
        registration.tokenId = desc.tokenId;
        registration.source = sources[index];
        registration.owner = this;
        const MRHandle handle = state.nextBookkeepingId++;
        state.registrations[handle] = registration;
        mrHandles.push_back(handle);
    }
    return Status::OK();
}

MRHandle CpuKvOverRoceTransProvider::FindCoveringRegistration(std::uintptr_t addr, std::size_t size,
                                                              const VerbDevice* device,
                                                              bool& backendMismatch,
                                                              bool& deviceMismatch) const
{
    backendMismatch = false;
    deviceMismatch = false;
    const bool verbs = UseVerbsBackend();
    MRHandle adoptable = 0;
    for (const auto& entry : ProcessState::Instance().registrations) {
        const Registration& registration = entry.second;
        // A binding is not a registration: only entries created by
        // RegisterMemory() can cover another range.
        if (registration.source != 0) { continue; }
        // Containment without the overflow `addr + size` would risk.
        if (addr < registration.addr || size > registration.size ||
            addr - registration.addr > registration.size - size) {
            continue;
        }
        // A simulated region has no access record and a real one only has one on
        // its own device, so a registration of the other backend cannot be used
        // whatever the addresses say.
        if ((registration.mr != nullptr) != verbs) {
            backendMismatch = true;
            continue;
        }
        if (registration.device != device) {
            // Usable only when this provider has not fixed its own device yet;
            // otherwise it belongs to another (adapter, port) pair.
            if (device == nullptr) {
                if (adoptable == 0) { adoptable = entry.first; }
            } else {
                deviceMismatch = true;
            }
            continue;
        }
        return entry.first;
    }
    return adoptable;
}

std::vector<Status> CpuKvOverRoceTransProvider::UnregisterMemory(
    const std::vector<UnregisterMemoryDesc>& memoryDescs)
{
    std::vector<Status> results;
    results.reserve(memoryDescs.size());
    ProcessState& state = ProcessState::Instance();
    for (const UnregisterMemoryDesc& desc : memoryDescs) {
        Registration registration;
        {
            std::lock_guard<std::mutex> guard(state.memoryMutex);
            const auto entry = state.registrations.find(desc.mrHandle);
            if (entry == state.registrations.end()) {
                results.push_back(InvalidArgument("unknown memory registration handle"));
                continue;
            }
            // Another provider may be using this region right now, so only the
            // provider that registered it may take it away.
            if (entry->second.owner != this) {
                results.push_back(
                    InvalidArgument("memory registration belongs to another provider"));
                continue;
            }
            registration = entry->second;
            // Claim the entry before dropping the lock: a second thread that
            // unregisters the same handle then sees an unknown handle instead
            // of deregistering the same memory twice.
            state.registrations.erase(entry);
        }
        if (registration.mr != nullptr && ibv_dereg_mr(registration.mr) != 0) {
            const int savedErrno = errno;
            std::lock_guard<std::mutex> guard(state.memoryMutex);
            // Roll the claim back so the caller can retry the release.
            state.registrations[desc.mrHandle] = registration;
            results.push_back(Failure(StatusCode::IO_ERROR, std::string("ibv_dereg_mr failed: ") +
                                                                std::strerror(savedErrno)));
            continue;
        }
        {
            std::lock_guard<std::mutex> guard(state.memoryMutex);
            // A binding is only meaningful while the registration it points at
            // exists, so it retires together with it.
            for (auto binding = state.registrations.begin();
                 binding != state.registrations.end();) {
                if (binding->second.source == desc.mrHandle) {
                    binding = state.registrations.erase(binding);
                } else {
                    ++binding;
                }
            }
        }
        results.push_back(Status::OK());
    }
    return results;
}

Status CpuKvOverRoceTransProvider::AllocThread(std::uint32_t threadNum,
                                               const std::vector<std::uint32_t>& notifyNumPerThread,
                                               std::vector<ThreadHandle>& threads)
{
    (void)threadNum;
    (void)notifyNumPerThread;
    // IO submitted through this provider runs on the caller thread, so there is
    // no SDK thread to hand out.  Reporting success keeps the client code path
    // identical to the other providers.
    threads.clear();
    return Status::OK();
}

std::vector<Status> CpuKvOverRoceTransProvider::FreeThread(const std::vector<ThreadHandle>& threads)
{
    (void)threads;
    return std::vector<Status>(threads.size(), Status::OK());
}

Status CpuKvOverRoceTransProvider::GetMemTokenId(MRHandle mrHandle, std::uint32_t& tokenId)
{
    tokenId = 0;
    ProcessState& state = ProcessState::Instance();
    std::lock_guard<std::mutex> guard(state.memoryMutex);
    const auto entry = state.registrations.find(mrHandle);
    if (entry == state.registrations.end()) {
        return InvalidArgument("unknown memory registration handle");
    }
    // The remote key is what the peer needs; the local lkey is not exported.
    tokenId = entry->second.tokenId;
    return Status::OK();
}

std::size_t CpuKvOverRoceTransProvider::ConnectionCount() const
{
    std::lock_guard<std::mutex> guard(mutex_);
    return links_.size();
}

std::size_t CpuKvOverRoceTransProvider::HandleCount() const
{
    std::lock_guard<std::mutex> guard(mutex_);
    return handleToLink_.size();
}

const char* CpuKvOverRoceTransProvider::Name() const noexcept { return kProviderName; }

const TransportConfig& CpuKvOverRoceTransProvider::Config() const noexcept { return config_; }

std::shared_ptr<TransProvider> CreateCpuKvOverRoceTransProvider(const TransportConfig& config)
{
    return std::make_shared<CpuKvOverRoceTransProvider>(config);
}

}  // namespace roce
}  // namespace kv
