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
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include "kv_types.h"
#include "roce_wire.h"

// The verbs types are only used through pointers here, so this header stays
// free of <infiniband/verbs.h>: the provider, its tests and the store build do
// not need rdma-core, and only roce_verb_qp.cc may include the real header.
struct ibv_context;
struct ibv_pd;
struct ibv_cq;
struct ibv_qp;
struct ibv_mr;

namespace kv {
namespace roce {

// Access flags accepted by RegisterMemory.  The values match the verbs ones,
// so the implementation can pass them straight to ibv_reg_mr.
enum MemoryAccess : int {
    kMemoryAccessLocalWrite = 1,
    kMemoryAccessRemoteWrite = 2,
    kMemoryAccessRemoteRead = 4,
};

// Settings that come from the ASU configuration (asu_roce_*) and describe how
// the local HCA and its queue pairs are brought up.
struct VerbQpConfig {
    std::string hcaName;
    std::uint8_t port{1};
    int gidIndex{-1};
    std::string localIp;
    std::uint32_t sendDepth{256};
    std::uint32_t recvDepth{64};
    std::uint32_t cqDepth{512};
    std::uint32_t inlineThreshold{0};
    std::uint8_t sl{3};
    std::uint8_t trafficClass{104};
    std::uint8_t rnrTimer{12};
    std::uint8_t rnrRetryCnt{7};
    std::uint8_t timeout{17};
    std::uint8_t retryCnt{7};
    std::uint8_t qpRdAtom{16};
};

// Owns the verbs context and protection domain of one HCA.
class VerbDevice {
public:
    VerbDevice() = default;

    ~VerbDevice();

    VerbDevice(const VerbDevice&) = delete;
    VerbDevice& operator=(const VerbDevice&) = delete;

    VerbDevice(VerbDevice&& other) noexcept;

    VerbDevice& operator=(VerbDevice&& other) noexcept;

    static Status Open(const VerbQpConfig& config, VerbDevice& device, std::string& error);

    // Name of the adapter this device was opened on; empty until Open succeeds.
    const std::string& Name() const noexcept;

    bool Opened() const noexcept;

    ibv_context* Context() const noexcept;

    ibv_pd* Pd() const noexcept;

    std::uint16_t Lid() const noexcept;

    std::uint8_t MtuCode() const noexcept;

    // How many outstanding RDMA read and atomic operations this adapter
    // supports per queue pair; the local policy is capped by it.
    std::uint8_t MaxQpRdAtom() const noexcept;

    std::uint32_t GidIndex() const noexcept;
    const std::uint8_t* Gid() const noexcept;
    // Registers a memory region for the queue pairs of this device.  The
    // returned region is opaque to callers and rkey is what a peer needs to
    // reach it; both stay valid until UnregisterMemory.
    Status RegisterMemory(void* addr, std::size_t size, int access, void*& region,
                          std::uint32_t& rkey, std::string& error);
    Status UnregisterMemory(void* region, std::string& error) const;

private:
    ibv_context* context_{nullptr};
    ibv_pd* pd_{nullptr};
    std::uint16_t lid_{0};
    std::uint8_t mtuCode_{5};
    std::uint8_t maxQpRdAtom_{1};
    std::uint32_t gidIndex_{0};
    std::uint8_t gid_[16]{};
    std::string name_;
};

// One RC queue pair with its own completion queue.
class VerbQp {
public:
    VerbQp() = default;

    ~VerbQp();

    VerbQp(const VerbQp&) = delete;
    VerbQp& operator=(const VerbQp&) = delete;

    VerbQp(VerbQp&& other) noexcept;

    VerbQp& operator=(VerbQp&& other) noexcept;

    static Status Create(VerbDevice& device, const VerbQpConfig& config, VerbQp& qp,
                         std::string& error);

    // INIT -> RTR -> RTS using the queue pair description received from the peer.
    Status Connect(VerbDevice& device, const VerbQpConfig& config, const QpInfo& peer,
                   std::uint32_t peerQpn, std::uint32_t linkPsn, std::string& error);

    void Reset() noexcept;

    std::uint32_t Qpn() const noexcept;

private:
    ibv_cq* cq_{nullptr};
    ibv_qp* qp_{nullptr};
};

// Next pseudo random PSN in the range used by the handshake.
std::uint32_t NextPsn() noexcept;

}  // namespace roce
}  // namespace kv
