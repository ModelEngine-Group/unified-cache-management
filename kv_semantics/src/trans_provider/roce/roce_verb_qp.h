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

#include <cstdint>
#include <infiniband/verbs.h>
#include <string>
#include "kv_types.h"
#include "roce_wire.h"

namespace kv {
namespace roce {

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

    std::uint32_t GidIndex() const noexcept;

    const std::uint8_t* Gid() const noexcept;

private:
    ibv_context* context_{nullptr};
    ibv_pd* pd_{nullptr};
    std::uint16_t lid_{0};
    std::uint8_t mtuCode_{5};
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
