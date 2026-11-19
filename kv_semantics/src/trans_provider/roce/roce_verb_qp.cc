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
#include "roce_verb_qp.h"
#include <atomic>
#include <cerrno>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace kv {
namespace roce {
namespace {

Status Fail(StatusCode code, const std::string& message) { return Status::Error(code, message); }

std::string VerbsError(const std::string& operation)
{
    return operation + " failed: " + std::strerror(errno);
}

bool IsZeroGid(const ibv_gid& gid)
{
    for (std::uint8_t byte : gid.raw) {
        if (byte != 0U) { return false; }
    }
    return true;
}

ibv_mtu ToVerbsMtu(std::uint8_t code)
{
    switch (code) {
        case 1: return IBV_MTU_256;
        case 2: return IBV_MTU_512;
        case 3: return IBV_MTU_1024;
        case 4: return IBV_MTU_2048;
        default: return IBV_MTU_4096;
    }
}

}  // namespace

std::uint32_t NextPsn() noexcept
{
    static std::atomic<std::uint32_t> counter{0};
    const std::uint32_t step = 0x1000U;
    return (counter.fetch_add(step) + step) & 0xFFFFFFU;
}

VerbDevice::VerbDevice(VerbDevice&& other) noexcept
    : context_(other.context_),
      pd_(other.pd_),
      lid_(other.lid_),
      mtuCode_(other.mtuCode_),
      maxQpRdAtom_(other.maxQpRdAtom_),
      gidIndex_(other.gidIndex_),
      name_(std::move(other.name_))
{
    std::memcpy(gid_, other.gid_, sizeof(gid_));
    other.context_ = nullptr;
    other.pd_ = nullptr;
}

VerbDevice& VerbDevice::operator=(VerbDevice&& other) noexcept
{
    if (this != &other) {
        if (pd_ != nullptr) { ibv_dealloc_pd(pd_); }
        if (context_ != nullptr) { ibv_close_device(context_); }
        context_ = other.context_;
        pd_ = other.pd_;
        lid_ = other.lid_;
        mtuCode_ = other.mtuCode_;
        maxQpRdAtom_ = other.maxQpRdAtom_;
        gidIndex_ = other.gidIndex_;
        name_ = std::move(other.name_);
        std::memcpy(gid_, other.gid_, sizeof(gid_));
        other.context_ = nullptr;
        other.pd_ = nullptr;
    }
    return *this;
}

VerbDevice::~VerbDevice()
{
    if (pd_ != nullptr) {
        ibv_dealloc_pd(pd_);
        pd_ = nullptr;
    }
    if (context_ != nullptr) {
        ibv_close_device(context_);
        context_ = nullptr;
    }
}

Status VerbDevice::Open(const VerbQpConfig& config, VerbDevice& device, std::string& error)
{
    error.clear();

    ibv_device** devices = ibv_get_device_list(nullptr);
    if (devices == nullptr) {
        error = VerbsError("ibv_get_device_list");
        return Fail(StatusCode::IO_ERROR, error);
    }

    ibv_device* selected = nullptr;
    std::string selectedName;
    std::size_t available = 0;
    for (std::size_t index = 0; devices[index] != nullptr; ++index) {
        ++available;
        const char* name = ibv_get_device_name(devices[index]);
        if (config.hcaName.empty() || (name != nullptr && config.hcaName == name)) {
            selected = devices[index];
            selectedName = name != nullptr ? name : std::string();
            break;
        }
    }
    if (selected == nullptr) {
        ibv_free_device_list(devices);
        error = config.hcaName.empty()
                    ? "no RDMA device found (is the RNIC driver loaded?)"
                    : "RDMA device '" + config.hcaName + "' was not found among " +
                          std::to_string(available) + " device(s)";
        return Fail(StatusCode::IO_ERROR, error);
    }

    VerbDevice opened;
    opened.context_ = ibv_open_device(selected);
    if (opened.context_ == nullptr) {
        error = VerbsError("ibv_open_device");
        ibv_free_device_list(devices);
        return Fail(StatusCode::IO_ERROR, error);
    }
    ibv_free_device_list(devices);

    ibv_port_attr portAttributes{};
    if (ibv_query_port(opened.context_, config.port, &portAttributes) != 0) {
        error = VerbsError("ibv_query_port");
        return Fail(StatusCode::IO_ERROR, error);
    }
    if (portAttributes.state != IBV_PORT_ACTIVE) {
        error = "RDMA port " + std::to_string(config.port) + " is not active (state " +
                std::to_string(static_cast<int>(portAttributes.state)) + ")";
        return Fail(StatusCode::CONNECTION_ERROR, error);
    }
    opened.lid_ = portAttributes.lid;
    opened.mtuCode_ = static_cast<std::uint8_t>(portAttributes.active_mtu);

    // The index is configured, never guessed: which entry of the GID table
    // carries a usable address depends on the fabric configuration of the host.
    if (config.gidIndex < 0) {
        error = "roce_gid_index is required (the provider does not choose a GID automatically)";
        return Fail(StatusCode::INVALID_ARGUMENT, error);
    }
    if (config.gidIndex >= portAttributes.gid_tbl_len) {
        error = "gid_index " + std::to_string(config.gidIndex) + " is outside the " +
                std::to_string(portAttributes.gid_tbl_len) + " entries of port " +
                std::to_string(config.port);
        return Fail(StatusCode::INVALID_ARGUMENT, error);
    }
    ibv_gid gid{};
    if (ibv_query_gid(opened.context_, config.port, config.gidIndex, &gid) != 0) {
        error = "configured gid_index " + std::to_string(config.gidIndex) + " is not usable";
        return Fail(StatusCode::INVALID_ARGUMENT, error);
    }
    if (IsZeroGid(gid)) {
        error = "gid_index " + std::to_string(config.gidIndex) + " of port " +
                std::to_string(config.port) + " is the all zero GID; check the IP configuration";
        return Fail(StatusCode::INVALID_ARGUMENT, error);
    }
    opened.gidIndex_ = static_cast<std::uint32_t>(config.gidIndex);
    std::memcpy(opened.gid_, gid.raw, sizeof(opened.gid_));
    opened.name_ = std::move(selectedName);

    opened.pd_ = ibv_alloc_pd(opened.context_);
    if (opened.pd_ == nullptr) {
        error = VerbsError("ibv_alloc_pd");
        return Fail(StatusCode::IO_ERROR, error);
    }

    // max_dest_rd_atomic has to stay inside what the adapter can honour, so ask
    // the device instead of trusting the peer's advertisement.
    ibv_device_attr deviceAttributes{};
    if (ibv_query_device(opened.context_, &deviceAttributes) == 0 &&
        deviceAttributes.max_qp_rd_atom > 0) {
        const int capacity =
            deviceAttributes.max_qp_rd_atom > 0xFF ? 0xFF : deviceAttributes.max_qp_rd_atom;
        opened.maxQpRdAtom_ = static_cast<std::uint8_t>(capacity);
    }

    device = std::move(opened);
    return Status::OK();
}

const std::string& VerbDevice::Name() const noexcept { return name_; }

bool VerbDevice::Opened() const noexcept { return pd_ != nullptr; }

ibv_context* VerbDevice::Context() const noexcept { return context_; }

ibv_pd* VerbDevice::Pd() const noexcept { return pd_; }

std::uint16_t VerbDevice::Lid() const noexcept { return lid_; }

std::uint8_t VerbDevice::MtuCode() const noexcept { return mtuCode_; }

std::uint8_t VerbDevice::MaxQpRdAtom() const noexcept { return maxQpRdAtom_; }

std::uint32_t VerbDevice::GidIndex() const noexcept { return gidIndex_; }

const std::uint8_t* VerbDevice::Gid() const noexcept { return gid_; }

VerbQp::VerbQp(VerbQp&& other) noexcept : cq_(other.cq_), qp_(other.qp_)
{
    other.cq_ = nullptr;
    other.qp_ = nullptr;
}

VerbQp& VerbQp::operator=(VerbQp&& other) noexcept
{
    if (this != &other) {
        Reset();
        cq_ = other.cq_;
        qp_ = other.qp_;
        other.cq_ = nullptr;
        other.qp_ = nullptr;
    }
    return *this;
}

VerbQp::~VerbQp() { Reset(); }

void VerbQp::Reset() noexcept
{
    if (qp_ != nullptr) {
        ibv_qp_attr attributes{};
        attributes.qp_state = IBV_QPS_ERR;
        static_cast<void>(ibv_modify_qp(qp_, &attributes, IBV_QP_STATE));
        ibv_destroy_qp(qp_);
        qp_ = nullptr;
    }
    if (cq_ != nullptr) {
        ibv_destroy_cq(cq_);
        cq_ = nullptr;
    }
}

std::uint32_t VerbQp::Qpn() const noexcept { return qp_ == nullptr ? 0U : qp_->qp_num; }

Status VerbQp::Create(VerbDevice& device, const VerbQpConfig& config, VerbQp& qp,
                      std::string& error)
{
    error.clear();
    if (!device.Opened()) {
        error = "queue pair creation needs an opened device";
        return Fail(StatusCode::CONNECTION_ERROR, error);
    }

    VerbQp created;
    const std::uint32_t cqDepth = config.cqDepth == 0U ? 512U : config.cqDepth;
    created.cq_ = ibv_create_cq(device.Context(), static_cast<int>(cqDepth), nullptr, nullptr, 0);
    if (created.cq_ == nullptr) {
        error = VerbsError("ibv_create_cq");
        return Fail(StatusCode::IO_ERROR, error);
    }

    ibv_qp_init_attr attributes{};
    attributes.qp_type = IBV_QPT_RC;
    attributes.sq_sig_all = 0;
    attributes.send_cq = created.cq_;
    attributes.recv_cq = created.cq_;
    attributes.cap.max_send_wr = config.sendDepth == 0U ? 256U : config.sendDepth;
    attributes.cap.max_recv_wr = config.recvDepth == 0U ? 64U : config.recvDepth;
    attributes.cap.max_send_sge = 1;
    attributes.cap.max_recv_sge = 1;
    attributes.cap.max_inline_data = config.inlineThreshold;

    created.qp_ = ibv_create_qp(device.Pd(), &attributes);
    if (created.qp_ == nullptr) {
        // Only an inline reservation the adapter refused is worth retrying
        // without it; the other failures must keep the error that explains them.
        const int inlineFailure = errno;
        if (config.inlineThreshold > 0U && inlineFailure == EINVAL) {
            attributes.cap.max_inline_data = 0;
            created.qp_ = ibv_create_qp(device.Pd(), &attributes);
        }
    }
    if (created.qp_ == nullptr) {
        error = VerbsError("ibv_create_qp");
        return Fail(StatusCode::IO_ERROR, error);
    }

    qp = std::move(created);
    return Status::OK();
}

namespace {
std::uint8_t LocalReadAtomicDepth(const VerbQpConfig& config, const VerbDevice& device)
{
    const std::uint8_t requested = config.qpRdAtom == 0U ? 1U : config.qpRdAtom;
    const std::uint8_t capacity = device.MaxQpRdAtom() == 0U ? 1U : device.MaxQpRdAtom();
    return requested < capacity ? requested : capacity;
}
}  // namespace

Status VerbQp::Connect(VerbDevice& device, const VerbQpConfig& config, const QpInfo& peer,
                       std::uint32_t peerQpn, std::uint32_t linkPsn, std::string& error)
{
    error.clear();
    if (qp_ == nullptr) {
        error = "queue pair is not created";
        return Fail(StatusCode::INVALID_ARGUMENT, error);
    }
    if (peerQpn == 0U) {
        error = "peer queue pair number is zero";
        return Fail(StatusCode::INVALID_ARGUMENT, error);
    }

    ibv_qp_attr init{};
    init.qp_state = IBV_QPS_INIT;
    init.pkey_index = 0;
    init.port_num = config.port;
    init.qp_access_flags =
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    int mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    if (ibv_modify_qp(qp_, &init, mask) != 0) {
        error = VerbsError("ibv_modify_qp(INIT)");
        return Fail(StatusCode::CONNECTION_ERROR, error);
    }

    ibv_qp_attr rtr{};
    rtr.qp_state = IBV_QPS_RTR;
    // The peer's path MTU is an upper bound, not a promise this port can carry:
    // modifying the queue pair with a value above the local active MTU fails.
    const std::uint8_t peerMtu = peer.mtu == 0U ? device.MtuCode() : peer.mtu;
    rtr.path_mtu = ToVerbsMtu(peerMtu < device.MtuCode() ? peerMtu : device.MtuCode());
    rtr.dest_qp_num = peerQpn;
    // The receive sequence number is the one the peer announced for its sends.
    rtr.rq_psn = peer.startPsn;
    // A responder has to accept as many outstanding read and atomic operations
    // as the peer will post, otherwise the difference turns into a steady stream
    // of RNR NAKs; the adapter limit keeps the modification legal.
    const std::uint8_t peerDepth = peer.qpRdAtom == 0U ? 1U : peer.qpRdAtom;
    const std::uint8_t responderCapacity = device.MaxQpRdAtom() == 0U ? 1U : device.MaxQpRdAtom();
    rtr.max_dest_rd_atomic = peerDepth < responderCapacity ? peerDepth : responderCapacity;
    rtr.min_rnr_timer = peer.rnrTimer;
    rtr.ah_attr.is_global = 1;
    rtr.ah_attr.dlid = peer.lid;
    rtr.ah_attr.sl = peer.sl;
    rtr.ah_attr.src_path_bits = 0;
    rtr.ah_attr.port_num = config.port;
    // ibv_gid is a union of 64 bit members, so casting the wire bytes to it would
    // assume an alignment and an aliasing rule the peer description does not
    // promise; copying the 16 bytes is what the struct is for.
    std::memcpy(rtr.ah_attr.grh.dgid.raw, peer.gid, sizeof(peer.gid));
    rtr.ah_attr.grh.sgid_index = device.GidIndex();
    rtr.ah_attr.grh.hop_limit = 64;
    rtr.ah_attr.grh.traffic_class = peer.trafficClass;
    mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
           IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    if (ibv_modify_qp(qp_, &rtr, mask) != 0) {
        error = VerbsError("ibv_modify_qp(RTR)");
        return Fail(StatusCode::CONNECTION_ERROR, error);
    }

    ibv_qp_attr rts{};
    rts.qp_state = IBV_QPS_RTS;
    // Retransmission is local policy: how often this side retries its own
    // sends is ours to decide, while the peer's numbers describe how it sends
    // to us and therefore only drive the RTR state.
    rts.timeout = config.timeout;
    rts.retry_cnt = config.retryCnt;
    rts.rnr_retry = config.rnrRetryCnt;
    // One PSN covers the whole link, so every queue pair sends from it.
    rts.sq_psn = linkPsn;
    rts.max_rd_atomic = LocalReadAtomicDepth(config, device);
    mask = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
           IBV_QP_MAX_QP_RD_ATOMIC;
    if (ibv_modify_qp(qp_, &rts, mask) != 0) {
        error = VerbsError("ibv_modify_qp(RTS)");
        return Fail(StatusCode::CONNECTION_ERROR, error);
    }

    return Status::OK();
}

}  // namespace roce
}  // namespace kv
