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

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "kv_types.h"
#include "roce_process_state.h"
#include "roce_tcp_channel.h"
#include "roce_verb_qp.h"
#include "roce_wire.h"
#include "trans_provider.h"

namespace kv {
namespace roce {

// TransProvider implementation that moves KV blocks over RoCEv2 from the CPU
// side: the caller hands over host or device addresses, the provider registers
// them with the verbs API and exchanges the resulting remote keys over a TCP
// out-of-band channel.
//
// The link establishment follows the protocol tables shipped with the product
// specification:
//
//   1. TCP      connect (client) or listen/accept (server) on the control port
//   2. caps     table 1 request from the initiator, table 2 reply from the peer
//   3. queue    completion queue + RC queue pair per connection, INIT/RTR/RTS
//   4. connect  table 3 exchange carrying gid/lid/mtu/qpn/psn, then table 4
//   5. teardown table 5 disconnect followed by queue pair destruction
//
// Only Send still reports UNSUPPORTED: link establishment, teardown and the
// memory registration path are implemented.
class CpuKvOverRoceTransProvider final : public TransProvider {
public:
    explicit CpuKvOverRoceTransProvider(const TransportConfig& config);

    ~CpuKvOverRoceTransProvider() override;

    CpuKvOverRoceTransProvider(const CpuKvOverRoceTransProvider&) = delete;
    CpuKvOverRoceTransProvider& operator=(const CpuKvOverRoceTransProvider&) = delete;

    // Creates one handle per queue pair. The store may accept fewer queue pairs than
    // requested, in which case the list is shorter than qpNum but never empty; the
    // agreed count is visible through GetServerCapabilities. On failure the list stays
    // empty.
    Status CreateConnection(const std::string& localIp, const std::string& remoteIp,
                            std::uint32_t port, std::uint32_t qpNum, std::uint32_t timeout,
                            std::vector<ConnectionHandle>& connectionHandles) override;

    std::vector<Status> DeleteConnections(
        const std::vector<ConnectionHandle>& connectionHandles) override;

    Status GetServerCapabilities(ConnectionHandle connectionHandle,
                                 ServerKvCapabilities& capabilities) override;

    std::vector<Status> Send(const std::vector<SendIoBatch>& ioBatches, std::uint32_t kernelCount,
                             std::uint32_t quietCount) override;

    Status RegisterMemory(const std::vector<RegisterMemoryDesc>& memoryDescs,
                          std::vector<MRHandle>& mrHandles) override;

    Status BindMemory(const std::vector<BindMemoryDesc>& memoryDescs,
                      std::vector<MRHandle>& mrHandles) override;

    std::vector<Status> UnregisterMemory(
        const std::vector<UnregisterMemoryDesc>& memoryDescs) override;

    Status AllocThread(std::uint32_t threadNum,
                       const std::vector<std::uint32_t>& notifyNumPerThread,
                       std::vector<ThreadHandle>& threads) override;

    std::vector<Status> FreeThread(const std::vector<ThreadHandle>& threads) override;

    Status GetMemTokenId(MRHandle mrHandle, std::uint32_t& tokenId) override;

    // Diagnostics used by the smoke test and by config dumps.
    const char* Name() const noexcept;

    const TransportConfig& Config() const noexcept;

    // Number of live out-of-band links, and number of handles handed out.
    std::size_t ConnectionCount() const;

    std::size_t HandleCount() const;

private:
    // One out-of-band link: a TCP channel, the negotiated capabilities and the
    // queue pairs whose handles are addressed from the low bits of each handle.
    struct Link {
        std::uint64_t id{0};
        std::string remoteIp;
        std::uint16_t remotePort{0};
        bool verbs{false};
        TcpChannel channel;
        Capabilities localCapabilities;
        Capabilities peerCapabilities;
        QpInfo localQpInfo;
        QpInfo peerQpInfo;
        std::vector<std::unique_ptr<VerbQp>> queuePairs;
        // One PSN covers the whole link, so every queue pair sends from it.
        std::uint32_t linkPsn{0};
        // Queue pairs both sides agreed on; it may be smaller than the request.
        std::uint32_t linkQpCount{0};
        // Capabilities the store side accepted at setup time.
        Capabilities negotiatedCapabilities;
    };

    static ConnectionHandle MakeHandle(std::uint64_t linkId, std::uint32_t qpIndex);

    std::string Attr(const std::string& key) const;

    std::uint32_t ResolveTimeout(std::uint32_t timeout) const;

    std::uint32_t HandshakeTimeout() const;

    bool UseVerbsBackend() const;

    // True when this side opens the TCP channel; the other side listens.
    bool IsInitiator() const;

    Status OpenLink(const std::string& localIp, const std::string& remoteIp, std::uint32_t port,
                    std::uint32_t timeout, TcpChannel& channel) const;

    Status OpenDevice(std::string& error);

    // Capability block this provider advertises for a link that asked for qpNum
    // queue pairs.
    Capabilities LocalCapabilities(std::uint32_t qpNum) const;

    /// Rejects transport limits that would not fit the capability message fields.
    Status ValidateCapabilityLimits(std::string& error) const;

    // Reads the asu_roce_* knobs into a queue pair configuration.  A value that
    // does not fit the field it configures is rejected instead of being silently
    // truncated into something the caller never asked for.
    Status BuildVerbConfig(const std::string& localIp, VerbQpConfig& config,
                           std::string& error) const;

    Status ExchangeCapabilities(Link& link, std::string& error);

    Status CreateQueuePairs(Link& link, std::uint32_t qpNum, const std::string& localIp,
                            std::string& error);

    static Capabilities AdoptCapabilities(const Capabilities& local, const Capabilities& peer);

    std::uint32_t NegotiatedQpNum(const Link& link) const;

    void TrimQueuePairs(Link& link, std::uint32_t count);

    Status ExchangeConnectRequests(Link& link, std::string& error);

    Status ConnectQueuePairs(Link& link, std::string& error);

    Status ExchangeConnectDone(Link& link, std::string& error);

    Status SendFrame(const Link& link, const std::vector<std::uint8_t>& frame,
                     std::string& error) const;

    Status SendFrame(const Link& link, const std::vector<std::uint8_t>& frame, std::string& error,
                     std::uint32_t timeoutMs) const;

    Status ReceiveFrame(const Link& link, Message& message, std::string& error) const;

    Status ReceiveMessage(const Link& link, const std::vector<std::uint8_t>& acceptedCommands,
                          Message& message, std::string& error) const;

    void TeardownLink(Link& link) noexcept;

    // Callers hold mutex_.
    auto FindLinkPosition(std::uint64_t linkId) -> std::vector<std::unique_ptr<Link>>::iterator;

    TransportConfig config_;

    mutable std::mutex mutex_;

    std::vector<std::unique_ptr<Link>> links_;

    std::unordered_map<ConnectionHandle, std::uint64_t> handleToLink_;

    std::uint64_t nextLinkId_{1};

    // Handle of the registration that covers the range and that this provider
    // can actually use, or zero when there is none.  A registration made on the
    // very (adapter, port) pair this provider talks to wins over one made on
    // another pair: the same buffer is routinely registered on every adapter of
    // the host, and only the matching access record is addressable, so picking
    // whichever entry the map happens to yield first would fail at random.  The
    // caller holds the process state memory lock.
    MRHandle FindCoveringRegistration(std::uintptr_t addr, std::size_t size,
                                      const VerbDevice* device, bool& backendMismatch,
                                      bool& deviceMismatch) const;

    // Device of the (adapter, port) pair this provider talks to, shared with
    // every provider that resolved to the same pair; null until the verbs path
    // opens it.  The entry lives in the process state, so the pointer stays
    // valid for as long as the provider does.
    VerbDevice& Device();

    // Device of the (adapter, port) pair this provider talks to.  Public methods
    // may run concurrently (the interface requires thread safety) and one of
    // them can adopt a device while another registers memory, so the pointer is
    // published atomically instead of under one of two different locks.
    std::atomic<VerbDevice*> device_{nullptr};
};

// Factory used by CreateTransProvider() once the ROCE provider type is wired in.
std::shared_ptr<TransProvider> CreateCpuKvOverRoceTransProvider(const TransportConfig& config);

}  // namespace roce
}  // namespace kv
