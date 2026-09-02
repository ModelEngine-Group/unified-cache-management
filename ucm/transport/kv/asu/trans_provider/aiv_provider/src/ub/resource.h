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
#include <memory>
#include <string>
#include <vector>
#include "src/protocol/ub_internal_types.h"
#include "src/protocol/ub_policy.h"
#include "src/ub/raii.h"
#include "src/ub/status.h"

namespace umc::comm {

struct LocalSegSpec {
    void* baseVa{nullptr};
    uint64_t size{0};
    uint32_t access{0xE};
    uint32_t tokenValue{4242};
    bool cacheable{true};
    bool nonPin{false};
};

struct LocalJettyHandle {
    uint32_t jettyId{0};

    TokenIdHandleRAII token;
    LmemHandleRAII lmem;
    ChanHandleRAII chan;
    CqHandleRAII rcq;
    CqHandleRAII scq;
    QpHandleRAII qp;

    uint8_t qpKey[64]{};
    uint8_t qpKeySize{0};
    uint8_t memKey[128]{};
    uint8_t memKeySize{0};
    uint32_t tokenId{0};
    uint32_t tokenValue{0};

    uint32_t uasid{0};
    uint32_t txPsn{0};
    uint64_t tag{0};

    void* segBaseVa{nullptr};
    uint64_t segSize{0};

    uint64_t sqRingVa{0};
    uint32_t sqRingDepth{0};
    uint64_t dbAddr{0};
    uint64_t wqebbSize{0};

    uint64_t cqRingVa{0};
    uint32_t cqeSize{0};
    uint32_t cqDepth{0};
    uint64_t cqDbAddr{0};
};

class UbV2ResourceManager {
public:
    struct InitConfig {
        uint32_t deviceId{0};
        UdmaEid localEid{};
        uint32_t qpDepth{4096};
        uint32_t cqDepth{16384};
        TransportProfile profile{TransportProfile::Ubc};

        JettyConnMode connMode{JettyConnMode::Rc};

        UbLocalNetAddr localNetAddr{};

        bool probeUboeCapability{true};

        bool skipTokenIdAlloc{false};
    };

    UbV2ResourceManager();
    ~UbV2ResourceManager();

    UbV2ResourceManager(const UbV2ResourceManager&) = delete;
    UbV2ResourceManager& operator=(const UbV2ResourceManager&) = delete;

    UbStatus Init(const InitConfig& cfg);

    UbStatus CreateLocalJetty(uint32_t jettyId, const LocalSegSpec& seg, LocalJettyHandle* out);
    UbStatus DestroyLocalJetty(LocalJettyHandle* local);

    struct ExportedSeg {
        LmemHandleRAII lmem;
        uint8_t memKey[128]{};
        uint8_t memKeySize{0};
        uint32_t tokenId{0};
        uint32_t tokenValue{0};
    };
    UbStatus RegisterExportableSeg(const LocalSegSpec& seg, ExportedSeg* out);

    struct RemoteJettyDescriptor {
        uint32_t jettyId{0};

        uint32_t uasid{0};

        uint8_t qpKeyRaw[64]{};
        uint8_t qpKeySize{0};
        uint8_t memKeyRaw[128]{};
        uint8_t memKeySize{0};
        uint32_t tokenId{0};
        uint32_t tokenValue{0};
        uint64_t remoteAddr{0};
        uint64_t remoteSize{0};
        int32_t transportMode{0};

        UdmaEid remoteEid{};

        UbTpType tpType{UbTpType::Ctp};

        UbLocalNetAddr remoteNetAddr{};

        uint64_t peerTpHandle{0};
        uint64_t tag{0};
        uint32_t peerTxPsn{0};
    };

    struct RemoteJettyHandle {
        uint32_t jettyId{0};
        std::shared_ptr<void> tpLease;
        RmemHandleRAII rmem;
        RemQpHandleRAII remQp;
        uint32_t tpn{0};
        uint64_t remoteAddr{0};
        uint64_t remoteSize{0};
        uint32_t remoteTokenValue{0};
        uint32_t remoteTokenId{0};
        UdmaEid remoteEid{};
        bool bound{false};
    };

    UbStatus ImportRemoteJetty(const LocalJettyHandle& local, const RemoteJettyDescriptor& remote,
                               RemoteJettyHandle* out);
    UbStatus DestroyRemoteJetty(const LocalJettyHandle& local, RemoteJettyHandle* remote);

    UbStatus GetLocalTpHandle(const UdmaEid& localEid, const UdmaEid& peerEid, UbTpType tpType,
                              int32_t transportMode, TransportProfile profile,
                              uint64_t* outTpHandle);

    struct AivInfoBuildSpec {
        uint32_t qpNum{1};
        uint32_t peerCount{1};
        uint32_t stripeCount{256};
        std::vector<const LocalJettyHandle*> locals;
        std::vector<const RemoteJettyHandle*> remotes;
    };

    struct DeviceAivInfo {
        uint64_t aivInfoDevVa{0};
        uint64_t signalSlotDevVa{0};
        uint64_t flagSlotDevVa{0};
        std::size_t totalBytes{0};
        uint64_t handleId{0};
    };

    UbStatus BuildAndDownloadAivInfo(const AivInfoBuildSpec& spec, DeviceAivInfo* out);

    UbStatus ReleaseDeviceAivInfo(const DeviceAivInfo& info);

    UbStatus Teardown();

    bool IsInitialized() const;
    void* CtxHandle() const;
    void* TokenHandle() const;
    const InitConfig& Config() const;

    UbStatus BindThreadContext() const;

    uint32_t PhyId() const;

private:
    UbStatus AcquireLocalTpHandle(const UdmaEid& localEid, const UdmaEid& peerEid, UbTpType tpType,
                                  int32_t transportMode, TransportProfile profile,
                                  uint64_t* outTpHandle, std::shared_ptr<void>* outLease);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

constexpr uint64_t kUrmaSegPageBytes = 4096ULL;

inline uint64_t AlignUrmaSegSize(uint64_t size)
{
    return (size + kUrmaSegPageBytes - 1U) / kUrmaSegPageBytes * kUrmaSegPageBytes;
}

inline bool IsUrmaSegPageAligned(uint64_t value)
{
    return (value & (kUrmaSegPageBytes - 1U)) == 0U;
}

inline void UrmaSegPageSpan(uint64_t addr, uint64_t len, uint64_t* outAddr, uint64_t* outLen)
{
    const uint64_t base = addr & ~(kUrmaSegPageBytes - 1U);
    const uint64_t end = addr + len;
    const uint64_t alignedEnd = AlignUrmaSegSize(end);
    *outAddr = base;
    *outLen = alignedEnd - base;
}

UbStatus AllocDeviceMemory(void** devPtr, uint64_t size);
UbStatus FreeDeviceMemory(void* devPtr);

}  // namespace umc::comm
