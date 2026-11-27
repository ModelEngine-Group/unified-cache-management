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
// Builds picked when BUILD_KV_CLIENT_PROVIDER_ROCE is off.  Nothing here talks to
// a NIC: the point is that the provider, its handshake and its tests keep
// building and running without rdma-core, and that asking for a device fails
// with an explanation instead of a missing symbol.
#include "roce_verb_qp.h"
namespace kv {
namespace roce {
namespace {
constexpr const char* kNotBuilt =
    "the CPU-over-RoCE provider was built without BUILD_KV_CLIENT_PROVIDER_ROCE";
}  // namespace
VerbDevice::~VerbDevice() = default;
VerbDevice::VerbDevice(VerbDevice&& other) noexcept = default;
VerbDevice& VerbDevice::operator=(VerbDevice&& other) noexcept = default;
Status VerbDevice::Open(const VerbQpConfig& config, VerbDevice& device, std::string& error)
{
    static_cast<void>(config);
    static_cast<void>(device);
    error = kNotBuilt;
    return Status::Error(StatusCode::UNSUPPORTED, error);
}
const std::string& VerbDevice::Name() const noexcept
{
    static const std::string empty;
    return empty;
}
bool VerbDevice::Opened() const noexcept { return false; }
ibv_context* VerbDevice::Context() const noexcept { return nullptr; }
ibv_pd* VerbDevice::Pd() const noexcept { return nullptr; }
std::uint16_t VerbDevice::Lid() const noexcept { return 0; }
std::uint8_t VerbDevice::MtuCode() const noexcept { return 5; }
std::uint8_t VerbDevice::MaxQpRdAtom() const noexcept { return 1; }
std::uint32_t VerbDevice::GidIndex() const noexcept { return 0; }
const std::uint8_t* VerbDevice::Gid() const noexcept
{
    static const std::uint8_t zero[16] = {};
    return zero;
}
Status VerbDevice::RegisterMemory(void* addr, std::size_t size, int access, void*& region,
                                  std::uint32_t& rkey, std::string& error)
{
    static_cast<void>(addr);
    static_cast<void>(size);
    static_cast<void>(access);
    region = nullptr;
    rkey = 0;
    error = kNotBuilt;
    return Status::Error(StatusCode::UNSUPPORTED, error);
}
Status VerbDevice::UnregisterMemory(void* region, std::string& error) const
{
    static_cast<void>(region);
    error = kNotBuilt;
    return Status::Error(StatusCode::UNSUPPORTED, error);
}
VerbQp::~VerbQp() = default;
VerbQp::VerbQp(VerbQp&& other) noexcept = default;
VerbQp& VerbQp::operator=(VerbQp&& other) noexcept = default;
Status VerbQp::Create(VerbDevice& device, const VerbQpConfig& config, VerbQp& qp,
                      std::string& error)
{
    static_cast<void>(device);
    static_cast<void>(config);
    static_cast<void>(qp);
    error = kNotBuilt;
    return Status::Error(StatusCode::UNSUPPORTED, error);
}
Status VerbQp::Connect(VerbDevice& device, const VerbQpConfig& config, const QpInfo& peer,
                       std::uint32_t peerQpn, std::uint32_t linkPsn, std::string& error)
{
    static_cast<void>(device);
    static_cast<void>(config);
    static_cast<void>(peer);
    static_cast<void>(peerQpn);
    static_cast<void>(linkPsn);
    error = kNotBuilt;
    return Status::Error(StatusCode::UNSUPPORTED, error);
}
void VerbQp::Reset() noexcept {}
std::uint32_t VerbQp::Qpn() const noexcept { return 0; }
std::uint32_t NextPsn() noexcept { return 0; }
}  // namespace roce
}  // namespace kv
