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

#include "src/ub/resource.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include "src/protocol/ub_internal_types.h"
#include "src/protocol/ub_signal_value.h"
#include "src/runtime/acl_rt_loader.h"
#include "src/runtime/hccp_v2_loader.h"
#include "src/ub/log.h"
#include "src/ub/queue_alias.h"

namespace umc::comm {

namespace {

constexpr int kHccpRaInitRepeated = 328002;

UbStatus AdoptHandle(HccpV2Handle& target, HccpV2Handle&& source, const char* operation)
{
    const auto code = target.Adopt(std::move(source));
    return code == UbErrorCode::Ok
               ? UbStatus::Ok()
               : UbStatus(code, std::string(operation) + ": target is not empty");
}

inline uint32_t IntLog2(uint64_t v)
{
    uint32_t r = 0;
    while ((uint64_t{1} << r) < v && r < 63) ++r;
    return r;
}

inline UdmaEid ToImportedEid(const UdmaEid& src)
{
    UdmaEid swapped{};
    uint64_t eidL = 0, eidH = 0;
    std::memcpy(&eidL, src.raw, sizeof(uint64_t));
    std::memcpy(&eidH, src.raw + sizeof(uint64_t), sizeof(uint64_t));
    eidL = __builtin_bswap64(eidL);
    eidH = __builtin_bswap64(eidH);
    std::memcpy(swapped.raw, &eidH, sizeof(uint64_t));
    std::memcpy(swapped.raw + sizeof(uint64_t), &eidL, sizeof(uint64_t));
    return swapped;
}

inline v2::HccpNetworkMode ToHccpNicPosition(TransportProfile profile)
{
    return profile == TransportProfile::Uboe ? v2::NETWORK_PEER_ONLINE : v2::NETWORK_OFFLINE;
}

inline const char* TransportProfileStr(TransportProfile p)
{
    return p == TransportProfile::Uboe ? "UBoE/NETWORK_PEER_ONLINE" : "UBC/NETWORK_OFFLINE";
}

inline const char* UbTpTypeStr(UbTpType t)
{
    switch (t) {
        case UbTpType::Rtp: return "URMA_RTP(0)";
        case UbTpType::Ctp: return "URMA_CTP(1)";
        case UbTpType::Utp: return "URMA_UTP(2)";
    }
    return "?";
}

inline bool EffectiveJettyRm(const UbV2ResourceManager::InitConfig& cfg)
{
    return cfg.profile == TransportProfile::Ubc || cfg.connMode == JettyConnMode::Rm;
}

constexpr uint32_t kUdmaRqDepthDefault = 256;

inline UbTpType NormalizeTpTypeForProfile(TransportProfile profile, UbTpType requested)
{
    if (profile == TransportProfile::Uboe && requested == UbTpType::Ctp) { return UbTpType::Rtp; }
    return requested;
}

#pragma pack(push, 8)
union RmUrmaEid {
    uint8_t raw[16];
    uint64_t align8_[2];
};
struct RmUrmaJettyId {
    RmUrmaEid eid;
    uint32_t uasid;
    uint32_t id;
};
struct RmJettyKeyInfo {
    RmUrmaJettyId jettyId;
    uint32_t transMode;  // urma_transport_mode_t（enum → int）
};
#pragma pack(pop)
static_assert(sizeof(RmUrmaEid) == 16, "urma_eid_t mirror must be 16B");
static_assert(sizeof(RmUrmaJettyId) == 24, "urma_jetty_id_t mirror must be 24B");
static_assert(sizeof(RmJettyKeyInfo) == 32, "RsJettyKeyInfo mirror must be 32B (28 used + 4 pad)");
static_assert(offsetof(RmUrmaJettyId, uasid) == 16, "uasid @16");
static_assert(offsetof(RmUrmaJettyId, id) == 20, "id @20");
static_assert(offsetof(RmJettyKeyInfo, transMode) == 24, "transMode @24");

constexpr uint32_t kUrmaTmRm = 0x1;
constexpr uint32_t kUrmaTmRc = 0x2;

inline uint8_t BuildRmQpKey(const UdmaEid& remoteEid, uint32_t uasid, uint32_t jettyId,
                            int32_t descTransportMode, uint8_t (&outKey)[64])
{
    RmJettyKeyInfo info{};
    std::memcpy(info.jettyId.eid.raw, remoteEid.raw, sizeof(info.jettyId.eid.raw));
    info.jettyId.uasid = uasid;
    info.jettyId.id = jettyId;
    info.transMode = (descTransportMode == 1) ? kUrmaTmRc : kUrmaTmRm;
    std::memset(outKey, 0, sizeof(outKey));
    std::memcpy(outKey, &info, sizeof(info));
    return static_cast<uint8_t>(sizeof(info));
}

}  // namespace

struct UbV2ResourceManager::Impl {
    struct TpCacheKey {
        std::array<uint8_t, 16> localEid{};
        std::array<uint8_t, 16> peerEid{};
        uint32_t tpType{0};
        int32_t transportMode{0};
        uint32_t profile{0};

        bool operator==(const TpCacheKey& other) const
        {
            return localEid == other.localEid && peerEid == other.peerEid &&
                   tpType == other.tpType && transportMode == other.transportMode &&
                   profile == other.profile;
        }
    };

    struct TpHandleLease {
        uint64_t handle{0};
    };

    struct TpCacheEntry {
        TpCacheKey key{};
        std::weak_ptr<TpHandleLease> lease;
    };

    InitConfig cfg{};
    bool initialized{false};

    TsdProcRAII tsdProc;      // TsdProcessOpen ↔ TsdProcessClose
    CtxHandleRAII ctx;        // RaCtxInit ↔ RaCtxDeinit
    TokenIdHandleRAII token;  // RaCtxTokenIdAlloc ↔ RaCtxTokenIdFree

    v2::RaInitConfig raInitCfg{};
    bool raInitCalled{false};
    bool externalRaSession{false};
    bool deviceSet{false};

    void* aclContext{nullptr};

    std::unordered_map<uint64_t, std::vector<void*>> deviceInfoBufs;
    uint64_t nextDeviceHandleId{1};

    std::mutex tpCacheMutex;
    std::vector<TpCacheEntry> tpCache;

    UbStatus FreeDeviceInfo(uint64_t handleId)
    {
        auto it = deviceInfoBufs.find(handleId);
        if (it == deviceInfoBufs.end()) return UbStatus::Ok();
        UbStatus firstError = UbStatus::Ok();
        std::vector<void*> retry;
        for (void* p : it->second) {
            auto st = acl::DlAclRt::Free(p);
            if (st.IsError()) {
                if (firstError.IsOk()) firstError = st;
                retry.push_back(p);
            }
        }
        if (retry.empty()) {
            deviceInfoBufs.erase(it);
        } else {
            it->second = std::move(retry);
        }
        return firstError;
    }
    UbStatus FreeAllDeviceInfo()
    {
        UbStatus firstError = UbStatus::Ok();
        std::vector<uint64_t> handles;
        handles.reserve(deviceInfoBufs.size());
        for (const auto& [id, bufs] : deviceInfoBufs) {
            (void)bufs;
            handles.push_back(id);
        }
        for (uint64_t id : handles) {
            auto status = FreeDeviceInfo(id);
            if (firstError.IsOk() && status.IsError()) firstError = status;
        }
        return firstError;
    }
};

UbV2ResourceManager::UbV2ResourceManager() : impl_(std::make_unique<Impl>()) {}

UbV2ResourceManager::~UbV2ResourceManager()
{
    if (impl_) {
        const auto status = Teardown();
        if (status.IsError()) {
            UB_LOG_ERROR(
                "UbV2ResourceManager: preserving resources after teardown "
                "failure: {}",
                status.Message().c_str());
            (void)impl_.release();
        }
    }
}

bool UbV2ResourceManager::IsInitialized() const { return impl_ && impl_->initialized; }

void* UbV2ResourceManager::CtxHandle() const { return impl_ ? impl_->ctx.Raw() : nullptr; }

void* UbV2ResourceManager::TokenHandle() const { return impl_ ? impl_->token.Raw() : nullptr; }

uint32_t UbV2ResourceManager::PhyId() const
{
    if (!impl_) return 0;
    return impl_->initialized ? impl_->raInitCfg.phyId : impl_->cfg.deviceId;
}

const UbV2ResourceManager::InitConfig& UbV2ResourceManager::Config() const
{
    static const InitConfig kEmpty{};
    return impl_ ? impl_->cfg : kEmpty;
}

UbStatus UbV2ResourceManager::BindThreadContext() const
{
    if (!impl_) return UbStatus(UbErrorCode::InvalidArgument, "BindThreadContext: null impl");
    if (!impl_->deviceSet) { return UbStatus::Ok(); }
    if (impl_->aclContext != nullptr) {
        auto st = acl::DlAclRt::SetCurrentContext(impl_->aclContext);
        if (st.IsError()) { return st.WithContext("BindThreadContext: aclrtSetCurrentContext"); }
        return UbStatus::Ok();
    }
    auto st = acl::DlAclRt::SetDevice(static_cast<int32_t>(impl_->cfg.deviceId));
    if (st.IsError()) { return st.WithContext("BindThreadContext: fallback aclrtSetDevice"); }
    return UbStatus::Ok();
}

UbStatus UbV2ResourceManager::Init(const InitConfig& cfg)
{
    if (impl_->initialized) { return UbStatus::Ok(); }
    if (impl_->tsdProc || impl_->ctx || impl_->token || impl_->raInitCalled) {
        return UbStatus(
            UbErrorCode::InvalidArgument,
            "UbV2ResourceManager::Init requires Teardown after a partial initialization");
    }
    impl_->cfg = cfg;

    UbErrorCode loadRc = v2::DlHccpV2Api::LoadLibrary();
    if (loadRc != UbErrorCode::Ok) {
        UB_LOG_ERROR("DlHccpV2Api::LoadLibrary failed: {}", UbErrorCodeToString(loadRc));
        return UbStatus(loadRc, "DlHccpV2Api::LoadLibrary failed");
    }

    {
        auto sd = acl::DlAclRt::SetDevice(static_cast<int32_t>(cfg.deviceId));
        if (sd.IsError()) {
            UB_LOG_ERROR("aclrtSetDevice({}) failed: {}", cfg.deviceId, sd.Message().c_str());
            return sd.WithContext("UbV2ResourceManager::Init aclrtSetDevice");
        }
        impl_->deviceSet = true;
        UB_LOG_DEBUG("aclrtSetDevice({}) ok", cfg.deviceId);

        void* ctx = nullptr;
        auto gc = acl::DlAclRt::GetCurrentContext(&ctx);
        if (gc.IsError()) {
            impl_->aclContext = nullptr;
            UB_LOG_WARN(
                "aclrtGetCurrentContext failed ({}); device operations will use "
                "aclrtSetDevice({}) per thread",
                gc.Message().c_str(), cfg.deviceId);
        } else {
            impl_->aclContext = ctx;
            UB_LOG_DEBUG("captured ACL context={} (device={}) for cross-thread bind", ctx,
                         cfg.deviceId);
        }
    }

    uint32_t phyId = cfg.deviceId;
    {
        int32_t phy = static_cast<int32_t>(cfg.deviceId);
        auto pst = acl::DlAclRt::GetPhyDevIdByLogicDevId(static_cast<int32_t>(cfg.deviceId), &phy);
        if (pst.IsError()) {
            UB_LOG_ERROR(
                "GetPhyDevIdByLogicDevId(logicId={}) failed: {}; "
                "HCCP physical device ID is unavailable; RaInit cannot proceed",
                cfg.deviceId, pst.Message().c_str());
            return pst.WithContext("UbV2ResourceManager::Init GetPhyDevIdByLogicDevId");
        }
        phyId = static_cast<uint32_t>(phy);
        if (phyId != cfg.deviceId) {
            UB_LOG_DEBUG("logical device {} maps to physical device {}", cfg.deviceId, phyId);
        } else {
            UB_LOG_DEBUG("device logicId={} == phyId={}", cfg.deviceId, phyId);
        }
    }

    v2::ProcOpenArgs openArgs{};
    openArgs.procType = v2::TSD_SUB_PROC_HCCP;
    pid_t hccpSubPid = 0;
    openArgs.subPid = &hccpSubPid;
    char hdcParam[24];
    int hdcParamLen = std::snprintf(hdcParam, sizeof(hdcParam), "--hdcType=%d",
                                    static_cast<int>(v2::HDC_SERVICE_TYPE_RDMA_V2));
    v2::ProcExtParam hdcExt{};
    hdcExt.paramInfo = hdcParam;
    hdcExt.paramLen = static_cast<uint64_t>(hdcParamLen) + 1;
    openArgs.extParamList = &hdcExt;
    openArgs.extParamCnt = 1;
    {
        uint32_t rc = v2::DlHccpV2Api::TsdProcessOpen(cfg.deviceId, &openArgs);
        if (rc != 0) {
            UB_LOG_ERROR(
                "TsdProcessOpen(device={}, RDMA_V2) failed rc={}; "
                "is hccp_service.bin running on this NPU node?",
                cfg.deviceId, rc);
            return UbStatus(UbErrorCode::HccpV2TsdOpenFailed, "TsdProcessOpen failed");
        }
    }
    UB_LOG_DEBUG("TsdProcessOpen(device={}, RDMA_V2) subPid={}", cfg.deviceId,
                 static_cast<int>(hccpSubPid));
    if (hccpSubPid <= 0) {
        UB_LOG_ERROR("TsdProcessOpen(device={}) returned invalid subPid={}", cfg.deviceId,
                     static_cast<int>(hccpSubPid));
        return UbStatus(UbErrorCode::HccpV2TsdOpenFailed, "TsdProcessOpen returned invalid subPid");
    }
    HandleAssoc tsdAssoc{};
    tsdAssoc.logicDeviceId = static_cast<int32_t>(cfg.deviceId);
    UB_RETURN_IF_ERROR(AdoptHandle(
        impl_->tsdProc,
        TsdProcRAII(HandleKind::kTsdProc,
                    reinterpret_cast<void*>(static_cast<uintptr_t>(hccpSubPid)), tsdAssoc),
        "UbV2ResourceManager::Init TsdProcessOpen"));

    const v2::HccpNetworkMode nicPos = ToHccpNicPosition(cfg.profile);
    impl_->raInitCfg.phyId = phyId;
    impl_->raInitCfg.nicPosition = nicPos;
    impl_->raInitCfg.hdcType = v2::HDC_SERVICE_TYPE_RDMA_V2;
    impl_->raInitCfg.enableHdcAsync = true;
    {
        int rc = v2::DlHccpV2Api::RaInit(&impl_->raInitCfg);
        if (rc == kHccpRaInitRepeated) {
            impl_->externalRaSession = true;
            (void)impl_->tsdProc.Release();
            UB_LOG_WARN(
                "RaInit returned repeated rc={} (profile={}, logicId={}, phyId={}); "
                "reuse existing RDMA_V2 HDC session and skip RaDeinit/TsdProcessClose/"
                "aclrtResetDevice on teardown",
                rc, TransportProfileStr(cfg.profile), cfg.deviceId, phyId);
        } else if (rc != 0) {
            UB_LOG_ERROR(
                "RaInit failed rc={} (profile={}, logicId={}, phyId={}); "
                "rc=228002 means -ENODEV, rc=328002 means repeated RaInit/-EEXIST",
                rc, TransportProfileStr(cfg.profile), cfg.deviceId, phyId);
            return UbStatus(UbErrorCode::HccpV2RaInitFailed, "RaInit failed");
        } else {
            impl_->raInitCalled = true;
            UB_LOG_DEBUG("RaInit ok (logicId={}, phyId={}, profile={}, hdc=RDMA_V2)", cfg.deviceId,
                         phyId, TransportProfileStr(cfg.profile));
        }
    }

    v2::CtxInitCfg ctxCfg{};
    ctxCfg.mode = nicPos;
    v2::CtxInitAttr ctxAttr{};
    ctxAttr.phyId = phyId;

    bool localEidIsZero = true;
    for (int i = 0; i < 16; ++i) {
        if (cfg.localEid.raw[i] != 0) {
            localEidIsZero = false;
            break;
        }
    }
    if (localEidIsZero) {
        v2::RaInfo eidQ{};
        eidQ.mode = nicPos;
        eidQ.phyId = phyId;
        unsigned int eidNum = 0;
        int qn = v2::DlHccpV2Api::RaGetDevEidInfoNum(eidQ, &eidNum);
        if (qn != 0 || eidNum == 0) {
            UB_LOG_ERROR(
                "RaGetDevEidInfoNum failed rc={} eidNum={} (phyId={}); "
                "device EID is unavailable; RaCtxInit cannot proceed",
                qn, eidNum, phyId);
            return UbStatus(UbErrorCode::HccpV2RaCtxInitFailed, "RaGetDevEidInfoNum failed");
        }
        std::vector<v2::DevEidInfo> eidList(eidNum);
        unsigned int got = eidNum;
        int ql = v2::DlHccpV2Api::RaGetDevEidInfoList(eidQ, eidList.data(), &got);
        if (ql != 0 || got == 0) {
            UB_LOG_ERROR("RaGetDevEidInfoList failed rc={} got={} (phyId={})", ql, got, phyId);
            return UbStatus(UbErrorCode::HccpV2RaCtxInitFailed, "RaGetDevEidInfoList failed");
        }
        std::memcpy(ctxAttr.ub.eid.raw, eidList[0].eid.raw, 16);
        ctxAttr.ub.eidIndex = eidList[0].eidIndex;
        std::memcpy(impl_->cfg.localEid.raw, eidList[0].eid.raw, 16);
        UB_LOG_DEBUG("RaGetDevEidInfo ok: eidNum={} use eidIndex={} eid[0..1]=0x{:02x}{:02x}",
                     eidNum, eidList[0].eidIndex, eidList[0].eid.raw[0], eidList[0].eid.raw[1]);
    } else {
        ctxAttr.ub.eidIndex = 0;
        std::memcpy(ctxAttr.ub.eid.raw, cfg.localEid.raw, 16);
    }
    void* ctxHandle = nullptr;
    {
        int rc = v2::DlHccpV2Api::RaCtxInit(&ctxCfg, &ctxAttr, &ctxHandle);
        if (rc != 0 || ctxHandle == nullptr) {
            UB_LOG_ERROR("RaCtxInit failed rc={}, ctxHandle={} (profile={})", rc, ctxHandle,
                         TransportProfileStr(cfg.profile));
            return UbStatus(UbErrorCode::HccpV2RaCtxInitFailed, "RaCtxInit failed");
        }
    }
    UB_RETURN_IF_ERROR(AdoptHandle(impl_->ctx,
                                   CtxHandleRAII(HandleKind::kCtx, ctxHandle, HandleAssoc{}),
                                   "UbV2ResourceManager::Init RaCtxInit"));
    UB_LOG_DEBUG("RaCtxInit ok, ctxHandle={} (mode={}, eidIndex={}, eid[0]=0x{:02x})", ctxHandle,
                 TransportProfileStr(cfg.profile), ctxAttr.ub.eidIndex, ctxAttr.ub.eid.raw[0]);

    if (cfg.profile == TransportProfile::Uboe && cfg.probeUboeCapability) {
        uint64_t probeTpHandle = 0;
        const int32_t probeTransMode = (cfg.connMode == JettyConnMode::Rm) ? 0 : 1;
        auto probeSt = GetLocalTpHandle(cfg.localEid, cfg.localEid, UbTpType::Rtp, probeTransMode,
                                        cfg.profile, &probeTpHandle);
        if (probeSt.IsError()) {
            UB_LOG_WARN(
                "UBoE capability probe GetLocalTpHandle failed: {} "
                "(driver may not expose uboe flag / no UBoE TP available; "
                "confirm UB NIC vlan/ip configured & link up via hccn_tool -i N -ip -g / -link -g; "
                "UBC fallback recommended)",
                probeSt.Message().c_str());
        } else {
            UB_LOG_DEBUG("UBoE capability probe ok: tpHandle=0x{:x}",
                         static_cast<unsigned long>(probeTpHandle));
        }
    }

    if (cfg.profile == TransportProfile::Ubc && cfg.connMode == JettyConnMode::Rc) {
        UB_LOG_DEBUG("profile=UBC: CTP does not support CONN_RC; using CONN_RM");
    }

    if (cfg.skipTokenIdAlloc) {
        UB_LOG_DEBUG("skipTokenIdAlloc=true: skipping RaCtxTokenIdAlloc and using token value 0");
    } else {
        v2::HccpTokenId tokenInfo{};
        tokenInfo.tokenId = 0;
        void* tokenHandle = nullptr;
        {
            int rc = v2::DlHccpV2Api::RaCtxTokenIdAlloc(ctxHandle, &tokenInfo, &tokenHandle);
            if (rc != 0 || tokenHandle == nullptr) {
                UB_LOG_ERROR("RaCtxTokenIdAlloc failed rc={}, handle={}", rc, tokenHandle);
                return UbStatus(UbErrorCode::HccpV2TokenIdAllocFailed, "RaCtxTokenIdAlloc failed");
            }
        }
        HandleAssoc tokAssoc{};
        tokAssoc.ctxHandle = ctxHandle;
        UB_RETURN_IF_ERROR(AdoptHandle(
            impl_->token, TokenIdHandleRAII(HandleKind::kTokenId, tokenHandle, tokAssoc),
            "UbV2ResourceManager::Init RaCtxTokenIdAlloc"));
        UB_LOG_DEBUG("RaCtxTokenIdAlloc ok, tokenId={}, handle={}", tokenInfo.tokenId, tokenHandle);
    }

    impl_->initialized = true;
    return UbStatus::Ok();
}

UbStatus UbV2ResourceManager::GetLocalTpHandle(const UdmaEid& localEid, const UdmaEid& peerEid,
                                               UbTpType tpType, int32_t transportMode,
                                               TransportProfile profile, uint64_t* outTpHandle)
{
    return AcquireLocalTpHandle(localEid, peerEid, tpType, transportMode, profile, outTpHandle,
                                nullptr);
}

UbStatus UbV2ResourceManager::AcquireLocalTpHandle(const UdmaEid& localEid, const UdmaEid& peerEid,
                                                   UbTpType tpType, int32_t transportMode,
                                                   TransportProfile profile, uint64_t* outTpHandle,
                                                   std::shared_ptr<void>* outLease)
{
    if (!impl_->ctx) {
        return UbStatus(UbErrorCode::InvalidArgument, "GetLocalTpHandle before Init");
    }
    if (outTpHandle == nullptr) {
        return UbStatus(UbErrorCode::InvalidArgument, "outTpHandle == nullptr");
    }
    *outTpHandle = 0;
    if (outLease != nullptr) { outLease->reset(); }
    void* ctxHandle = impl_->ctx.Raw();

    const UbTpType effTpType = NormalizeTpTypeForProfile(profile, tpType);
    if (effTpType != tpType) {
        UB_LOG_WARN("UBoE does not support {}; using {}", UbTpTypeStr(tpType),
                    UbTpTypeStr(effTpType));
    }

    v2::GetTpCfg tpCfg{};
    tpCfg.flag.value = 0;
    switch (effTpType) {
        case UbTpType::Ctp: tpCfg.flag.bs.ctp = 1; break;
        case UbTpType::Rtp: tpCfg.flag.bs.rtp = 1; break;
        case UbTpType::Utp: tpCfg.flag.bs.utp = 1; break;
    }
    if (profile == TransportProfile::Uboe) { tpCfg.flag.bs.uboe = 1; }
    tpCfg.transMode = (transportMode == 1) ? v2::CONN_RC : v2::CONN_RM;
    std::memcpy(tpCfg.localEid.raw, localEid.raw, sizeof(tpCfg.localEid.raw));
    std::memcpy(tpCfg.peerEid.raw, peerEid.raw, sizeof(tpCfg.peerEid.raw));

    void* req = nullptr;
    Impl::TpCacheKey cacheKey{};
    std::copy_n(localEid.raw, cacheKey.localEid.size(), cacheKey.localEid.begin());
    std::copy_n(peerEid.raw, cacheKey.peerEid.size(), cacheKey.peerEid.begin());
    cacheKey.tpType = static_cast<uint32_t>(effTpType);
    cacheKey.transportMode = static_cast<int32_t>(tpCfg.transMode);
    cacheKey.profile = static_cast<uint32_t>(profile);

    const bool reuseTp = tpCfg.transMode == v2::CONN_RM;
    std::unique_lock<std::mutex> cacheLock;
    if (reuseTp) {
        cacheLock = std::unique_lock<std::mutex>(impl_->tpCacheMutex);
        auto cached =
            std::find_if(impl_->tpCache.begin(), impl_->tpCache.end(),
                         [&](const Impl::TpCacheEntry& entry) { return entry.key == cacheKey; });
        if (cached != impl_->tpCache.end()) {
            if (auto lease = cached->lease.lock()) {
                *outTpHandle = lease->handle;
                if (outLease != nullptr) { *outLease = std::move(lease); }
                UB_LOG_DEBUG(
                    "GetLocalTpHandle cache hit: tpHandle=0x{:x} "
                    "(tpType={}, transMode={}, profile={})",
                    static_cast<unsigned long>(*outTpHandle), UbTpTypeStr(effTpType),
                    static_cast<int>(tpCfg.transMode), TransportProfileStr(profile));
                return UbStatus::Ok();
            }
            impl_->tpCache.erase(cached);
        }
    }

    v2::TpInfo tpInfo{};
    unsigned int num = 1;
    int rc = v2::DlHccpV2Api::RaGetTpInfoListAsync(ctxHandle, &tpCfg, &tpInfo, &num, &req);
    if (rc != 0) {
        UB_LOG_ERROR(
            "RaGetTpInfoListAsync failed rc={} (tpType={}, transMode={}, "
            "flag=0x{:x}, profile={})",
            rc, UbTpTypeStr(tpType), transportMode, tpCfg.flag.value, TransportProfileStr(profile));
        return UbStatus(UbErrorCode::HccpV2UboeNotAvailable, "RaGetTpInfoListAsync failed");
    }

    if (req != nullptr) {
        constexpr int kPollTimeoutMs = 5000;
        const auto start = std::chrono::steady_clock::now();
        while (true) {
            int result = 0;
            int qrc = v2::DlHccpV2Api::RaGetAsyncReqResult(req, &result);
            if (qrc == 0) {
                if (result == v2::HCCP_SOCK_EAGAIN) {
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                       std::chrono::steady_clock::now() - start)
                                       .count();
                    if (elapsed >= kPollTimeoutMs) {
                        UB_LOG_ERROR(
                            "RaGetAsyncReqResult SOCK_EAGAIN poll timeout ({}ms) "
                            "tpType={} transMode={}",
                            kPollTimeoutMs, UbTpTypeStr(tpType), transportMode);
                        return UbStatus(UbErrorCode::HccpV2UboeNotAvailable,
                                        "RaGetAsyncReqResult SOCK_EAGAIN poll timeout");
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
                if (result != 0) {
                    UB_LOG_ERROR(
                        "RaGetTpInfoListAsync async result error={} "
                        "(tpType={}, transMode={})",
                        result, UbTpTypeStr(tpType), transportMode);
                    return UbStatus(UbErrorCode::HccpV2UboeNotAvailable,
                                    "RaGetTpInfoListAsync async result error");
                }
                break;
            }
            if (qrc != v2::HCCP_OTHERS_EAGAIN) {
                UB_LOG_ERROR("RaGetAsyncReqResult hard error qrc={} (tpType={}, transMode={})", qrc,
                             UbTpTypeStr(tpType), transportMode);
                return UbStatus(UbErrorCode::HccpV2UboeNotAvailable,
                                "RaGetAsyncReqResult hard error");
            }
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();
            if (elapsed >= kPollTimeoutMs) {
                UB_LOG_ERROR(
                    "RaGetAsyncReqResult poll timeout (last qrc={}, {}ms) "
                    "tpType={} transMode={}",
                    qrc, kPollTimeoutMs, UbTpTypeStr(tpType), transportMode);
                return UbStatus(UbErrorCode::HccpV2UboeNotAvailable,
                                "RaGetAsyncReqResult poll timeout");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    if (num == 0) {
        UB_LOG_ERROR(
            "RaGetTpInfoListAsync returned empty TP list (no route for "
            "localEid[0]=0x{:02x} -> peerEid[0]=0x{:02x}; confirm UB NIC vlan/ip "
            "configured & link up via hccn_tool -i N -ip -g / -link -g)",
            localEid.raw[0], peerEid.raw[0]);
        return UbStatus(UbErrorCode::HccpV2UboeNotAvailable, "RaGetTpInfoListAsync empty TP list");
    }

    *outTpHandle = tpInfo.tpHandle;
    if (reuseTp && outLease != nullptr) {
        auto lease = std::make_shared<Impl::TpHandleLease>();
        lease->handle = *outTpHandle;
        impl_->tpCache.push_back(Impl::TpCacheEntry{cacheKey, lease});
        *outLease = std::move(lease);
    }
    UB_LOG_DEBUG(
        "GetLocalTpHandle queried{}: tpHandle=0x{:x} "
        "({} entry, tpType={}, transMode={}, profile={})",
        (reuseTp && outLease != nullptr) ? " and leased" : "",
        static_cast<unsigned long>(*outTpHandle), num, UbTpTypeStr(effTpType),
        static_cast<int>(tpCfg.transMode), TransportProfileStr(profile));
    return UbStatus::Ok();
}

UbStatus UbV2ResourceManager::CreateLocalJetty(uint32_t jettyId, const LocalSegSpec& seg,
                                               LocalJettyHandle* out)
{
    if (!impl_->initialized) {
        return UbStatus(UbErrorCode::InvalidArgument, "CreateLocalJetty called before Init");
    }
    if (out == nullptr) { return UbStatus(UbErrorCode::InvalidArgument, "out == nullptr"); }
    if (out->token || out->lmem || out->chan || out->rcq || out->scq || out->qp) {
        return UbStatus(UbErrorCode::InvalidArgument,
                        "CreateLocalJetty: output still owns resources");
    }
    if (auto bc = BindThreadContext(); bc.IsError()) {
        return bc.WithContext("CreateLocalJetty: BindThreadContext");
    }
    if (seg.baseVa == nullptr || seg.size == 0) {
        return UbStatus(UbErrorCode::InvalidArgument, "LocalSegSpec invalid");
    }

    void* ctxHandle = impl_->ctx.Raw();
    HandleAssoc ctxAssoc{};
    ctxAssoc.ctxHandle = ctxHandle;
    const bool isRm = EffectiveJettyRm(impl_->cfg);
    out->jettyId = jettyId;
    out->segBaseVa = seg.baseVa;
    out->segSize = seg.size;

    v2::MrRegInfoT mrInfo{};
    mrInfo.in.mem.addr = reinterpret_cast<uint64_t>(seg.baseVa);
    mrInfo.in.mem.size = seg.size;
    mrInfo.in.ub.flags.value = 0;
    const bool tokNone = impl_->cfg.skipTokenIdAlloc;
    mrInfo.in.ub.flags.bs.tokenPolicy =
        tokNone ? v2::MEM_SEG_TOKEN_NONE : v2::MEM_SEG_TOKEN_PLAIN_TEXT;
    mrInfo.in.ub.flags.bs.cacheable = seg.cacheable ? 1 : 0;
    const uint32_t segAccess =
        (seg.access == 0xF) ? static_cast<uint32_t>(v2::MEM_SEG_ACCESS_DEFAULT) : seg.access;
    mrInfo.in.ub.flags.bs.access = segAccess;
    const uint32_t initialNonPin = seg.nonPin ? 1u : 0u;
    mrInfo.in.ub.flags.bs.nonPin = initialNonPin;
    mrInfo.in.ub.flags.bs.userIova = 0;
    mrInfo.in.ub.flags.bs.tokenIdValid = tokNone ? 0 : 1;
    const uint32_t registeredTokenValue = tokNone ? 0 : seg.tokenValue;
    if (!tokNone && impl_->token.Raw() == nullptr) {
        UB_LOG_ERROR(
            "CreateLocalJetty: PLAIN_TEXT token requested but tokenIdHandle is null "
            "(skipTokenIdAlloc={})",
            impl_->cfg.skipTokenIdAlloc ? 1u : 0u);
        return UbStatus(UbErrorCode::InvalidArgument,
                        "CreateLocalJetty: tokenIdHandle null for PLAIN_TEXT token");
    }
    mrInfo.in.ub.tokenValue = registeredTokenValue;
    mrInfo.in.ub.tokenIdHandle = tokNone ? nullptr : impl_->token.Raw();
    void* lmemHandle = nullptr;
    {
        auto registerLmem = [&]() {
            lmemHandle = nullptr;
            return v2::DlHccpV2Api::RaCtxLmemRegister(ctxHandle, &mrInfo, &lmemHandle);
        };
        int rc = registerLmem();
        if ((rc != 0 || lmemHandle == nullptr) && initialNonPin == 0u) {
            UB_LOG_DEBUG(
                "RaCtxLmemRegister retry with nonPin=1 after rc={} "
                "(addr={}, size={}, access=0x{:x})",
                rc, seg.baseVa, static_cast<unsigned long>(seg.size), segAccess);
            mrInfo.in.ub.flags.bs.nonPin = 1;
            std::memset(&mrInfo.out, 0, sizeof(mrInfo.out));
            rc = registerLmem();
        }
        if (rc != 0 || lmemHandle == nullptr) {
            UB_LOG_ERROR(
                "RaCtxLmemRegister failed rc={} (addr={}, size={}, "
                "access=0x{:x} cacheable={} nonPin={} tokenValue={} tokenIdHandle={})",
                rc, seg.baseVa, static_cast<unsigned long>(seg.size), segAccess,
                seg.cacheable ? 1u : 0u, static_cast<uint32_t>(mrInfo.in.ub.flags.bs.nonPin),
                registeredTokenValue, impl_->token.Raw());
            return UbStatus(UbErrorCode::HccpV2LmemRegisterFailed, "RaCtxLmemRegister failed");
        }
    }
    UB_RETURN_IF_ERROR(AdoptHandle(out->lmem,
                                   LmemHandleRAII(HandleKind::kLmem, lmemHandle, ctxAssoc),
                                   "CreateLocalJetty RaCtxLmemRegister"));
    std::memcpy(out->memKey, mrInfo.out.key.value, sizeof(out->memKey));
    out->memKeySize = mrInfo.out.key.size;
    out->tokenId = mrInfo.out.ub.tokenId;
    out->tokenValue = registeredTokenValue;

    v2::ChanInfoT chanInfo{};
    chanInfo.in.dataPlaneFlag.bs.poolCqCstm = 1;
    void* chanHandle = nullptr;
    {
        int rc = v2::DlHccpV2Api::RaCtxChanCreate(ctxHandle, &chanInfo, &chanHandle);
        if (rc != 0 || chanHandle == nullptr) {
            UB_LOG_ERROR("RaCtxChanCreate failed rc={}", rc);
            return UbStatus(UbErrorCode::HccpV2ChanCreateFailed, "RaCtxChanCreate failed");
        }
    }
    UB_RETURN_IF_ERROR(AdoptHandle(out->chan,
                                   ChanHandleRAII(HandleKind::kChan, chanHandle, ctxAssoc),
                                   "CreateLocalJetty RaCtxChanCreate"));

    auto createCq = [&](CqHandleRAII& slot, v2::CqCreateInfo* capturedOut) -> UbStatus {
        v2::CqInfoT cqInfo{};
        cqInfo.in.chanHandle = chanHandle;
        cqInfo.in.depth = impl_->cfg.cqDepth;
        cqInfo.in.ub.mode = v2::JFC_MODE_USER_CTL_NORMAL;
        cqInfo.in.ub.userCtx = 0;
        cqInfo.in.ub.ceqn = 0;
        cqInfo.in.ub.flag.value = 0;
        void* cqHandle = nullptr;
        int rc = v2::DlHccpV2Api::RaCtxCqCreate(ctxHandle, &cqInfo, &cqHandle);
        if (rc != 0 || cqHandle == nullptr) {
            UB_LOG_ERROR("RaCtxCqCreate failed rc={}", rc);
            return UbStatus(UbErrorCode::HccpV2CqCreateFailed, "RaCtxCqCreate failed");
        }
        UB_RETURN_IF_ERROR(AdoptHandle(slot, CqHandleRAII(HandleKind::kCq, cqHandle, ctxAssoc),
                                       "CreateLocalJetty RaCtxCqCreate"));
        if (capturedOut) *capturedOut = cqInfo.out;
        return UbStatus::Ok();
    };
    v2::CqCreateInfo scqOut{};
    UB_RETURN_IF_ERROR(createCq(out->scq, &scqOut));
    UB_RETURN_IF_ERROR(createCq(out->rcq, nullptr));
    out->cqRingVa = scqOut.bufAddr;
    out->cqeSize = static_cast<uint32_t>(scqOut.cqeSize);
    out->cqDbAddr = scqOut.swdbAddr;
    out->cqDepth = impl_->cfg.cqDepth;

    v2::QpCreateAttr qpAttr{};
    qpAttr.scqHandle = out->scq.Raw();
    qpAttr.rcqHandle = out->rcq.Raw();
    qpAttr.srqHandle = out->rcq.Raw();
    qpAttr.sqDepth = impl_->cfg.qpDepth;
    qpAttr.rqDepth = kUdmaRqDepthDefault;
    qpAttr.transportMode = isRm ? v2::CONN_RM : v2::CONN_RC;
    qpAttr.ub.mode = v2::JETTY_MODE_USER_CTL_NORMAL;
    qpAttr.ub.jettyId = 0;
    qpAttr.ub.flag.value = 1;  // URMA_SHARE_JFR
    qpAttr.ub.jfsFlag.value = 2;
    qpAttr.ub.tokenIdHandle = tokNone ? nullptr : impl_->token.Raw();
    qpAttr.ub.tokenValue = registeredTokenValue;
    qpAttr.ub.priority = 0;
    qpAttr.ub.rnrRetry = 7;
    qpAttr.ub.errTimeout = 0;
    qpAttr.ub.extMode.piType = 0;
    qpAttr.ub.extMode.cstmFlag.value = 0;
    qpAttr.ub.extMode.sqebbNum = impl_->cfg.qpDepth;
    qpAttr.ub.extMode.sq.buffSize = 0;
    qpAttr.ub.extMode.sq.buffVa = 0;
    v2::QpCreateInfo qpOut{};
    void* qpHandle = nullptr;
    {
        int rc = v2::DlHccpV2Api::RaCtxQpCreate(ctxHandle, &qpAttr, &qpOut, &qpHandle);
        if (rc != 0 || qpHandle == nullptr) {
            UB_LOG_ERROR("RaCtxQpCreate failed rc={}, jettyId={} connMode={} transportMode={}", rc,
                         jettyId, isRm ? "Rm" : "Rc", isRm ? "CONN_RM" : "CONN_RC");
            return UbStatus(UbErrorCode::HccpV2QpCreateFailed, "RaCtxQpCreate failed");
        }
    }
    UB_RETURN_IF_ERROR(AdoptHandle(out->qp, QpHandleRAII(HandleKind::kQp, qpHandle, HandleAssoc{}),
                                   "CreateLocalJetty RaCtxQpCreate"));
    std::memcpy(out->qpKey, qpOut.key.value, sizeof(out->qpKey));
    out->qpKeySize = qpOut.key.size;
    out->sqRingVa = qpOut.ub.sqBuffVa;
    out->dbAddr = qpOut.ub.dbAddr;
    out->wqebbSize = qpOut.ub.wqebbSize;
    constexpr uint32_t kMaxSqeBbNum = 4;  // shmem::UDMA_MAX_SQE_BB_NUM
    out->sqRingDepth = impl_->cfg.qpDepth * kMaxSqeBbNum;

    out->uasid = qpOut.ub.uasid;
    if (isRm) { out->jettyId = qpOut.ub.id; }
    out->txPsn = static_cast<uint32_t>(::random());
    out->tag = 0;

    UB_LOG_DEBUG(
        "CreateLocalJetty ok: connMode={} jettyId={}(uasid={} id={}) "
        "qpKeySize={} memKeySize={} tokenId={}",
        isRm ? "Rm" : "Rc", out->jettyId, out->uasid, qpOut.ub.id, out->qpKeySize, out->memKeySize,
        out->tokenId);
    return UbStatus::Ok();
}

UbStatus UbV2ResourceManager::RegisterExportableSeg(const LocalSegSpec& seg, ExportedSeg* out)
{
    if (!impl_->initialized) {
        return UbStatus(UbErrorCode::InvalidArgument, "RegisterExportableSeg called before Init");
    }
    if (out == nullptr) { return UbStatus(UbErrorCode::InvalidArgument, "out == nullptr"); }
    if (out->lmem) {
        return UbStatus(UbErrorCode::InvalidArgument,
                        "RegisterExportableSeg: output still owns an MR");
    }
    if (auto bc = BindThreadContext(); bc.IsError()) {
        return bc.WithContext("RegisterExportableSeg: BindThreadContext");
    }
    if (seg.baseVa == nullptr || seg.size == 0) {
        return UbStatus(UbErrorCode::InvalidArgument, "LocalSegSpec invalid");
    }
    const uint64_t segAddr = reinterpret_cast<uint64_t>(seg.baseVa);
    if (!IsUrmaSegPageAligned(segAddr) || !IsUrmaSegPageAligned(seg.size)) {
        return UbStatus(UbErrorCode::InvalidArgument,
                        "RegisterExportableSeg: addr/size must be 4KB page aligned");
    }

    void* ctxHandle = impl_->ctx.Raw();
    HandleAssoc ctxAssoc{};
    ctxAssoc.ctxHandle = ctxHandle;
    v2::MrRegInfoT mrInfo{};
    mrInfo.in.mem.addr = reinterpret_cast<uint64_t>(seg.baseVa);
    mrInfo.in.mem.size = seg.size;
    mrInfo.in.ub.flags.value = 0;
    const bool tokNone = impl_->cfg.skipTokenIdAlloc;
    mrInfo.in.ub.flags.bs.tokenPolicy =
        tokNone ? v2::MEM_SEG_TOKEN_NONE : v2::MEM_SEG_TOKEN_PLAIN_TEXT;
    mrInfo.in.ub.flags.bs.cacheable = seg.cacheable ? 1 : 0;
    const uint32_t segAccess =
        (seg.access == 0xF) ? static_cast<uint32_t>(v2::MEM_SEG_ACCESS_DEFAULT) : seg.access;
    mrInfo.in.ub.flags.bs.access = segAccess;
    const uint32_t initialNonPin = seg.nonPin ? 1u : 0u;
    mrInfo.in.ub.flags.bs.nonPin = initialNonPin;
    mrInfo.in.ub.flags.bs.userIova = 0;
    mrInfo.in.ub.flags.bs.tokenIdValid = tokNone ? 0 : 1;
    const uint32_t registeredTokenValue = tokNone ? 0 : seg.tokenValue;
    if (!tokNone && impl_->token.Raw() == nullptr) {
        UB_LOG_ERROR(
            "RegisterExportableSeg: PLAIN_TEXT token requested but tokenIdHandle is null "
            "(skipTokenIdAlloc={})",
            impl_->cfg.skipTokenIdAlloc ? 1u : 0u);
        return UbStatus(UbErrorCode::InvalidArgument,
                        "RegisterExportableSeg: tokenIdHandle null for PLAIN_TEXT token");
    }
    mrInfo.in.ub.tokenValue = registeredTokenValue;
    mrInfo.in.ub.tokenIdHandle = tokNone ? nullptr : impl_->token.Raw();
    void* lmemHandle = nullptr;
    auto registerLmem = [&]() {
        lmemHandle = nullptr;
        return v2::DlHccpV2Api::RaCtxLmemRegister(ctxHandle, &mrInfo, &lmemHandle);
    };
    int rc = registerLmem();
    if ((rc != 0 || lmemHandle == nullptr) && initialNonPin == 0u) {
        UB_LOG_DEBUG(
            "RegisterExportableSeg retry with nonPin=1 after rc={} "
            "(addr={}, size={}, access=0x{:x})",
            rc, seg.baseVa, static_cast<unsigned long>(seg.size), segAccess);
        mrInfo.in.ub.flags.bs.nonPin = 1;
        std::memset(&mrInfo.out, 0, sizeof(mrInfo.out));
        rc = registerLmem();
    }
    if (rc != 0 || lmemHandle == nullptr) {
        UB_LOG_ERROR(
            "RegisterExportableSeg: RaCtxLmemRegister failed rc={} "
            "(addr={} size={} access=0x{:x} cacheable={} nonPin={} "
            "tokenPolicy={} tokenValue={} tokenIdHandle={})",
            rc, seg.baseVa, static_cast<unsigned long>(seg.size), segAccess,
            seg.cacheable ? 1u : 0u, static_cast<uint32_t>(mrInfo.in.ub.flags.bs.nonPin),
            static_cast<uint32_t>(mrInfo.in.ub.flags.bs.tokenPolicy), registeredTokenValue,
            impl_->token.Raw());
        return UbStatus(UbErrorCode::HccpV2LmemRegisterFailed,
                        "RegisterExportableSeg: RaCtxLmemRegister failed");
    }
    UB_RETURN_IF_ERROR(AdoptHandle(out->lmem,
                                   LmemHandleRAII(HandleKind::kLmem, lmemHandle, ctxAssoc),
                                   "RegisterExportableSeg RaCtxLmemRegister"));
    std::memcpy(out->memKey, mrInfo.out.key.value, sizeof(out->memKey));
    out->memKeySize = mrInfo.out.key.size;
    out->tokenId = mrInfo.out.ub.tokenId;
    out->tokenValue = seg.tokenValue;
    UB_LOG_DEBUG(
        "RegisterExportableSeg ok: addr={} size={} memKeySize={} tokenId={} "
        "registeredTokenValue={} exportedTokenValue={}",
        seg.baseVa, static_cast<unsigned long>(seg.size), out->memKeySize, out->tokenId,
        registeredTokenValue, out->tokenValue);
    return UbStatus::Ok();
}

UbStatus UbV2ResourceManager::DestroyLocalJetty(LocalJettyHandle* local)
{
    if (local == nullptr) { return UbStatus::Ok(); }
    if (auto bc = BindThreadContext(); bc.IsError()) {
        return bc.WithContext("DestroyLocalJetty: BindThreadContext");
    }
    auto reset = [](HccpV2Handle& handle, const char* operation) -> UbStatus {
        const auto code = handle.Reset();
        return code == UbErrorCode::Ok
                   ? UbStatus::Ok()
                   : UbStatus(code, std::string("DestroyLocalJetty: ") + operation);
    };
    UB_RETURN_IF_ERROR(reset(local->qp, "RaCtxQpDestroy"));
    UB_RETURN_IF_ERROR(reset(local->scq, "RaCtxCqDestroy(scq)"));
    UB_RETURN_IF_ERROR(reset(local->rcq, "RaCtxCqDestroy(rcq)"));
    UB_RETURN_IF_ERROR(reset(local->chan, "RaCtxChanDestroy"));
    UB_RETURN_IF_ERROR(reset(local->lmem, "RaCtxLmemUnregister"));
    UB_RETURN_IF_ERROR(reset(local->token, "RaCtxTokenIdFree"));
    return UbStatus::Ok();
}

UbStatus UbV2ResourceManager::DestroyRemoteJetty(const LocalJettyHandle& local,
                                                 RemoteJettyHandle* remote)
{
    if (remote == nullptr) { return UbStatus::Ok(); }
    if (auto bc = BindThreadContext(); bc.IsError()) {
        return bc.WithContext("DestroyRemoteJetty: BindThreadContext");
    }
    auto reset = [](HccpV2Handle& handle, const char* operation) -> UbStatus {
        const auto code = handle.Reset();
        return code == UbErrorCode::Ok
                   ? UbStatus::Ok()
                   : UbStatus(code, std::string("DestroyRemoteJetty: ") + operation);
    };
    if (remote->bound) {
        if (!local.qp) {
            return UbStatus(UbErrorCode::HccpV2HandleInvalid,
                            "DestroyRemoteJetty: bound QP has no local handle");
        }
        const int rc = v2::DlHccpV2Api::RaCtxQpUnbind(local.qp.Raw());
        if (rc != 0) {
            return UbStatus(UbErrorCode::HccpV2HandleInvalid,
                            "DestroyRemoteJetty: RaCtxQpUnbind failed rc=" + std::to_string(rc));
        }
        remote->bound = false;
    }
    UB_RETURN_IF_ERROR(reset(remote->remQp, "RaCtxQpUnimport"));
    UB_RETURN_IF_ERROR(reset(remote->rmem, "RaCtxRmemUnimport"));
    return UbStatus::Ok();
}

UbStatus UbV2ResourceManager::ImportRemoteJetty(const LocalJettyHandle& local,
                                                const RemoteJettyDescriptor& remote,
                                                RemoteJettyHandle* out)
{
    if (!impl_->initialized) {
        return UbStatus(UbErrorCode::InvalidArgument, "ImportRemoteJetty before Init");
    }
    if (out == nullptr) { return UbStatus(UbErrorCode::InvalidArgument, "out == nullptr"); }
    if (out->rmem || out->remQp) {
        return UbStatus(UbErrorCode::InvalidArgument,
                        "ImportRemoteJetty: output still owns resources");
    }
    if (auto bc = BindThreadContext(); bc.IsError()) {
        return bc.WithContext("ImportRemoteJetty: BindThreadContext");
    }
    if (!local.qp.Raw()) {
        return UbStatus(UbErrorCode::InvalidArgument, "local jetty handle invalid (no QP)");
    }
    const bool isRm = EffectiveJettyRm(impl_->cfg);
    if (!isRm && (remote.qpKeySize == 0 || remote.memKeySize == 0)) {
        return UbStatus(UbErrorCode::InvalidArgument, "remote 4-tuple sizes are zero");
    }

    void* ctxHandle = impl_->ctx.Raw();
    HandleAssoc ctxAssoc{};
    ctxAssoc.ctxHandle = ctxHandle;

    out->jettyId = remote.jettyId;
    out->remoteAddr = remote.remoteAddr;
    out->remoteSize = remote.remoteSize;
    out->remoteTokenValue = remote.tokenValue;
    out->remoteTokenId = remote.tokenId;
    out->remoteEid = remote.remoteEid;

    uint64_t localTpHandle = 0;
    std::shared_ptr<void> tpLease;
    {
        auto tpSt = AcquireLocalTpHandle(impl_->cfg.localEid, remote.remoteEid, remote.tpType,
                                         remote.transportMode, impl_->cfg.profile, &localTpHandle,
                                         &tpLease);
        if (tpSt.IsError()) {
            UB_LOG_ERROR(
                "ImportRemoteJetty: GetLocalTpHandle failed: {} "
                "(remote jetty={}, tpType={})",
                tpSt.Message().c_str(), remote.jettyId, UbTpTypeStr(remote.tpType));
            return tpSt.WithContext("ImportRemoteJetty: GetLocalTpHandle");
        }
    }
    out->tpLease = std::move(tpLease);

    const uint32_t localTxPsn = local.txPsn;
    const uint32_t rxPsn = remote.peerTxPsn;
    const uint64_t tag = remote.tag;

    v2::QpImportInfoT qpImp{};
    if (isRm) {
        qpImp.in.key.size = BuildRmQpKey(remote.remoteEid, remote.uasid, remote.jettyId,
                                         remote.transportMode, qpImp.in.key.value);
    } else {
        std::memcpy(qpImp.in.key.value, remote.qpKeyRaw, sizeof(qpImp.in.key.value));
        qpImp.in.key.size = remote.qpKeySize;
    }
    const bool impTokNone = impl_->cfg.skipTokenIdAlloc;
    qpImp.in.ub.mode = v2::JETTY_IMPORT_MODE_EXP;
    qpImp.in.ub.tokenValue = impTokNone ? 0 : remote.tokenValue;
    qpImp.in.ub.policy = v2::JETTY_GRP_POLICY_RR;
    qpImp.in.ub.type = v2::TARGET_TYPE_JETTY;
    qpImp.in.ub.flag.bs.tokenPolicy =
        impTokNone ? v2::TOKEN_POLICY_NONE : v2::TOKEN_POLICY_PLAIN_TEXT;
    qpImp.in.ub.flag.bs.orderType = 0;
    qpImp.in.ub.flag.bs.shareTp = 0;
    const UbTpType effImportTpType = NormalizeTpTypeForProfile(impl_->cfg.profile, remote.tpType);
    qpImp.in.ub.tpType = static_cast<uint32_t>(effImportTpType);
    qpImp.in.ub.expImportCfg.tpHandle = localTpHandle;
    qpImp.in.ub.expImportCfg.peerTpHandle = remote.peerTpHandle;
    qpImp.in.ub.expImportCfg.tag = tag;
    qpImp.in.ub.expImportCfg.txPsn = localTxPsn;
    qpImp.in.ub.expImportCfg.rxPsn = rxPsn;

    UB_LOG_DEBUG(
        "ImportRemoteJetty({} tokNone={}): tokenValue={} tpType={} key.size={} "
        "localTpHandle=0x{:x} peerTpHandle=0x{:x} tag=0x{:x} txPsn={} rxPsn={} "
        "(remote jetty={} uasid={})",
        "EXP", impTokNone ? 1 : 0, qpImp.in.ub.tokenValue, UbTpTypeStr(effImportTpType),
        qpImp.in.key.size, static_cast<unsigned long>(localTpHandle),
        static_cast<unsigned long>(remote.peerTpHandle), static_cast<unsigned long>(tag),
        localTxPsn, rxPsn, remote.jettyId, remote.uasid);

    void* remQpHandle = nullptr;
    {
        int rc = v2::DlHccpV2Api::RaCtxQpImport(ctxHandle, &qpImp, &remQpHandle);
        if (rc != 0 || remQpHandle == nullptr) {
            UB_LOG_ERROR("RaCtxQpImport(EXP) failed rc={} (remote jetty={}, tpType={})", rc,
                         remote.jettyId, UbTpTypeStr(effImportTpType));
            return UbStatus(UbErrorCode::HccpV2QpImportFailed, "RaCtxQpImport(EXP) failed");
        }
    }
    UB_RETURN_IF_ERROR(AdoptHandle(out->remQp,
                                   RemQpHandleRAII(HandleKind::kRemQp, remQpHandle, ctxAssoc),
                                   "ImportRemoteJetty RaCtxQpImport"));
    out->tpn = qpImp.out.ub.tpn;

    {
        v2::MrImportInfoT mrImp{};
        std::memcpy(mrImp.in.key.value, remote.memKeyRaw, sizeof(mrImp.in.key.value));
        mrImp.in.key.size = remote.memKeySize;
        mrImp.in.ub.tokenValue = remote.tokenValue;
        mrImp.in.ub.flags.bs.cacheable = 0;
        mrImp.in.ub.flags.bs.access = 0x0F;
        mrImp.in.ub.flags.bs.mapping = 0;

        void* rmemHandle = nullptr;
        int rc = v2::DlHccpV2Api::RaCtxRmemImport(ctxHandle, &mrImp, &rmemHandle);
        if (rc != 0 || rmemHandle == nullptr) {
            UB_LOG_ERROR("RaCtxRmemImport failed rc={} (remote jetty={} tokenValue={})", rc,
                         remote.jettyId, remote.tokenValue);
            return UbStatus(UbErrorCode::HccpV2RmemImportFailed, "RaCtxRmemImport failed");
        }
        UB_RETURN_IF_ERROR(AdoptHandle(out->rmem,
                                       RmemHandleRAII(HandleKind::kRmem, rmemHandle, ctxAssoc),
                                       "ImportRemoteJetty RaCtxRmemImport"));
        UB_LOG_DEBUG("RaCtxRmemImport ok: remote jetty={} tokenValue={} memKeySize={}",
                     remote.jettyId, remote.tokenValue, remote.memKeySize);
    }

    if (remote.transportMode == 1) {  // CONN_RC
        int rc = v2::DlHccpV2Api::RaCtxQpBind(local.qp.Raw(), remQpHandle);
        if (rc != 0) {
            UB_LOG_ERROR("RaCtxQpBind failed rc={} (local jetty={}, remote={})", rc, local.jettyId,
                         remote.jettyId);
            return UbStatus(UbErrorCode::HccpV2QpBindFailed, "RaCtxQpBind failed");
        }
        out->bound = true;
    }

    UB_LOG_DEBUG(
        "ImportRemoteJetty ok: local={} remote={} tpType={} tpn={} "
        "bound={} (profile={})",
        local.jettyId, remote.jettyId, UbTpTypeStr(effImportTpType), out->tpn, out->bound ? 1 : 0,
        TransportProfileStr(impl_->cfg.profile));
    return UbStatus::Ok();
}

UbStatus UbV2ResourceManager::BuildAndDownloadAivInfo(const AivInfoBuildSpec& spec,
                                                      DeviceAivInfo* out)
{
    if (out == nullptr) return UbStatus(UbErrorCode::InvalidArgument, "out == nullptr");
    if (!impl_->initialized) {
        return UbStatus(UbErrorCode::InvalidArgument, "BuildAndDownloadAivInfo before Init");
    }
    if (auto bc = BindThreadContext(); bc.IsError()) {
        return bc.WithContext("BuildAndDownloadAivInfo: BindThreadContext");
    }
    const uint32_t qpNum = spec.qpNum > 0 ? spec.qpNum : 1;
    const uint32_t peerCount = spec.peerCount;
    const uint32_t stripe = spec.stripeCount > 0 ? spec.stripeCount : 1;
    if (peerCount == 0) { return UbStatus(UbErrorCode::InvalidArgument, "peerCount == 0"); }
    const std::size_t N = std::size_t(peerCount) * qpNum;
    if (spec.locals.size() != N || spec.remotes.size() != N) {
        return UbStatus(UbErrorCode::InvalidArgument,
                        "locals/remotes size must == peerCount*qpNum");
    }
    for (std::size_t i = 0; i < N; ++i) {
        if (spec.remotes[i] == nullptr) {
            return UbStatus(UbErrorCode::InvalidArgument,
                            "null remote at logical slot " + std::to_string(i));
        }
    }
    detail::AivQueueAliasPlan aliasPlan;
    if (auto aliasSt = detail::BuildAivQueueAliasPlan(spec.locals, &aliasPlan); aliasSt.IsError()) {
        return aliasSt.WithContext("BuildAndDownloadAivInfo");
    }
    const std::size_t physicalQueueCount = aliasPlan.physicalLocals.size();

    auto loadSt = acl::DlAclRt::LoadLibrary();
    if (loadSt.IsError()) {
        return loadSt.WithContext("BuildAndDownloadAivInfo: acl runtime unavailable");
    }

    std::size_t off = sizeof(UdmaAivInfo);
    const std::size_t sqOff = off;
    off += N * sizeof(UdmaWqCtx);
    const std::size_t rqOff = off;
    off += N * sizeof(UdmaWqCtx);
    const std::size_t scqOff = off;
    off += N * sizeof(UdmaCqCtx);
    const std::size_t rcqOff = off;
    off += N * sizeof(UdmaCqCtx);
    const std::size_t memOff = off;
    off += N * sizeof(UdmaSegInfo);
    const std::size_t slotN = std::size_t(peerCount) * stripe;
    const std::size_t sigOff = off;
    off += slotN * sizeof(UbSignalSlot);
    const std::size_t flagOff = off;
    off += std::size_t(stripe) * sizeof(uint64_t);
    const std::size_t total = off;

    std::vector<void*> bufs;
    auto cleanupOnFail = [&]() {
        for (void* p : bufs) {
            const auto freeStatus = acl::DlAclRt::Free(p);
            if (freeStatus.IsError()) {
                UB_LOG_ERROR("BuildAndDownloadAivInfo rollback failed for {}: {}", p,
                             freeStatus.Message().c_str());
            }
        }
    };
    auto allocZero = [&](std::size_t sz, void** outp) -> UbStatus {
        auto st = acl::DlAclRt::Malloc(outp, sz);
        if (st.IsError()) return st;
        bufs.push_back(*outp);
        return acl::DlAclRt::Memset(*outp, sz, 0, sz);
    };

    std::vector<uint64_t> sqPi(physicalQueueCount), sqCi(physicalQueueCount);
    std::vector<uint64_t> wqeCnt(physicalQueueCount), amo(physicalQueueCount);
    std::vector<uint64_t> cqPi(physicalQueueCount), cqCi(physicalQueueCount);
    for (std::size_t i = 0; i < physicalQueueCount; ++i) {
        void *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr, *e = nullptr, *f = nullptr;
        UbStatus st;
        if ((st = allocZero(sizeof(uint32_t), &a)).IsError() ||
            (st = allocZero(sizeof(uint32_t), &b)).IsError() ||
            (st = allocZero(sizeof(uint32_t), &c)).IsError() ||
            (st = allocZero(sizeof(uint64_t), &d)).IsError() ||
            (st = allocZero(sizeof(uint32_t), &e)).IsError() ||
            (st = allocZero(sizeof(uint32_t), &f)).IsError()) {
            cleanupOnFail();
            return st.WithContext("BuildAndDownloadAivInfo: alloc PI/CI buffers");
        }
        sqPi[i] = reinterpret_cast<uint64_t>(a);
        sqCi[i] = reinterpret_cast<uint64_t>(b);
        wqeCnt[i] = reinterpret_cast<uint64_t>(c);
        amo[i] = reinterpret_cast<uint64_t>(d);
        cqPi[i] = reinterpret_cast<uint64_t>(e);
        cqCi[i] = reinterpret_cast<uint64_t>(f);
    }

    void* eidTableDev = nullptr;
    {
        auto st = acl::DlAclRt::Malloc(&eidTableDev, std::size_t(peerCount) * sizeof(UdmaEid));
        if (st.IsError()) {
            cleanupOnFail();
            return st.WithContext("alloc eid table");
        }
        bufs.push_back(eidTableDev);
        std::vector<UdmaEid> eidHost(peerCount);
        for (uint32_t p = 0; p < peerCount; ++p) {
            eidHost[p] = ToImportedEid(spec.remotes[std::size_t(p) * qpNum]->remoteEid);
        }
        st = acl::DlAclRt::Memcpy(eidTableDev, std::size_t(peerCount) * sizeof(UdmaEid),
                                  eidHost.data(), std::size_t(peerCount) * sizeof(UdmaEid),
                                  acl::MemcpyKind::HostToDevice);
        if (st.IsError()) {
            cleanupOnFail();
            return st.WithContext("H2D eid table");
        }
    }

    std::vector<uint8_t> blob(total, 0);
    auto* hdr = reinterpret_cast<UdmaAivInfo*>(blob.data());
    auto* sqArr = reinterpret_cast<UdmaWqCtx*>(blob.data() + sqOff);
    auto* rqArr = reinterpret_cast<UdmaWqCtx*>(blob.data() + rqOff);
    auto* scqA = reinterpret_cast<UdmaCqCtx*>(blob.data() + scqOff);
    auto* rcqA = reinterpret_cast<UdmaCqCtx*>(blob.data() + rcqOff);
    auto* segA = reinterpret_cast<UdmaSegInfo*>(blob.data() + memOff);
    hdr->qpNum = qpNum;
    hdr->peerCount = peerCount;

    for (uint32_t p = 0; p < peerCount; ++p) {
        for (uint32_t q = 0; q < qpNum; ++q) {
            const std::size_t idx = std::size_t(p) * qpNum + q;
            const LocalJettyHandle* lj = spec.locals[idx];
            const RemoteJettyHandle* rj = spec.remotes[idx];
            const std::size_t physical = aliasPlan.logicalToPhysical[idx];

            UdmaWqCtx& sq = sqArr[idx];
            sq.wqn = 0;
            sq.bufAddr = lj->sqRingVa;
            sq.wqeShiftSize = IntLog2(lj->wqebbSize ? lj->wqebbSize : 64);
            sq.depth = lj->sqRingDepth;
            sq.headAddr = sqPi[physical];
            sq.tailAddr = sqCi[physical];
            sq.dbMode = UdmaDbMode::SwDb;
            sq.dbAddr = lj->dbAddr;
            sq.sl = 0;
            sq.wqeCntAddr = wqeCnt[physical];
            sq.amoAddr = amo[physical];
            rqArr[idx] = sq;

            UdmaCqCtx& cq = scqA[idx];
            cq.cqn = 0;
            cq.bufAddr = lj->cqRingVa;
            cq.cqeShiftSize = IntLog2(lj->cqeSize ? lj->cqeSize : 64);
            cq.depth = lj->cqDepth;
            cq.headAddr = cqPi[physical];
            cq.tailAddr = cqCi[physical];
            cq.dbMode = UdmaDbMode::SwDb;
            cq.dbAddr = lj->cqDbAddr;
            rcqA[idx] = cq;

            UdmaSegInfo& sg = segA[idx];
            sg.tokenValueValid = (rj->remoteTokenValue != 0) ? 1 : 0;
            sg.rmtJettyType = 1;
            sg.targetHint = 0;  // JETTY_GRP_POLICY_RR
            sg.tpn = rj->tpn;
            sg.tid = rj->jettyId;
            sg.rmtTokenValue = rj->remoteTokenValue;
            sg.len = rj->remoteSize;
            sg.addr = rj->remoteAddr;
            sg.eidAddr = reinterpret_cast<uint64_t>(eidTableDev) + std::size_t(p) * sizeof(UdmaEid);

            UB_LOG_DEBUG(
                "AIV queue map: peer={} qp={} logical={} physical={} "
                "localJetty={} remoteJetty={} sq=0x{:x} cq=0x{:x} "
                "sqPi=0x{:x} sqCi=0x{:x} cqPi=0x{:x} cqCi=0x{:x} tpn={}",
                p, q, idx, physical, lj->jettyId, rj->jettyId,
                static_cast<unsigned long>(lj->sqRingVa), static_cast<unsigned long>(lj->cqRingVa),
                static_cast<unsigned long>(sq.headAddr), static_cast<unsigned long>(sq.tailAddr),
                static_cast<unsigned long>(cq.headAddr), static_cast<unsigned long>(cq.tailAddr),
                rj->tpn);
        }
    }

    void* devBase = nullptr;
    {
        auto st = acl::DlAclRt::Malloc(&devBase, total);
        if (st.IsError()) {
            cleanupOnFail();
            return st.WithContext("alloc aivInfo blob");
        }
        bufs.push_back(devBase);
    }
    const uint64_t base = reinterpret_cast<uint64_t>(devBase);
    hdr->sqPtr = base + sqOff;
    hdr->rqPtr = base + rqOff;
    hdr->scqPtr = base + scqOff;
    hdr->rcqPtr = base + rcqOff;
    hdr->memPtr = base + memOff;
    hdr->signalSlotPtr = base + sigOff;
    hdr->flagSlotPtr = base + flagOff;
    {
        auto st =
            acl::DlAclRt::Memcpy(devBase, total, blob.data(), total, acl::MemcpyKind::HostToDevice);
        if (st.IsError()) {
            cleanupOnFail();
            return st.WithContext("H2D aivInfo blob");
        }
    }

    const uint64_t handleId = impl_->nextDeviceHandleId++;
    impl_->deviceInfoBufs.emplace(handleId, std::move(bufs));
    out->aivInfoDevVa = base;
    out->signalSlotDevVa = base + sigOff;
    out->flagSlotDevVa = base + flagOff;
    out->totalBytes = total;
    out->handleId = handleId;

    UB_LOG_DEBUG(
        "BuildAndDownloadAivInfo ok: peers={} qpNum={} logicalQueues={} "
        "physicalQueues={} stripe={} total={}B devVa=0x{:x} flagVa=0x{:x}",
        peerCount, qpNum, N, physicalQueueCount, stripe, total, static_cast<unsigned long>(base),
        static_cast<unsigned long>(out->flagSlotDevVa));
    return UbStatus::Ok();
}

UbStatus UbV2ResourceManager::ReleaseDeviceAivInfo(const DeviceAivInfo& info)
{
    if (info.handleId == 0) return UbStatus::Ok();
    if (auto bc = BindThreadContext(); bc.IsError()) {
        UB_LOG_WARN("ReleaseDeviceAivInfo: BindThreadContext failed: {}", bc.Message().c_str());
    }
    return impl_ ? impl_->FreeDeviceInfo(info.handleId) : UbStatus::Ok();
}

UbStatus UbV2ResourceManager::Teardown()
{
    UbStatus firstError = UbStatus::Ok();
    auto recordStatus = [&](const UbStatus& st, const char* operation) {
        if (st.IsOk()) return;
        UB_LOG_ERROR("Teardown {} failed: {}", operation, st.Message().c_str());
        if (firstError.IsOk()) firstError = st.WithContext(operation);
    };
    auto recordCode = [&](UbErrorCode code, const char* operation) {
        if (code == UbErrorCode::Ok) return;
        UB_LOG_ERROR("Teardown {} failed: {}", operation, UbErrorCodeToString(code));
        if (firstError.IsOk()) { firstError = UbStatus(code, std::string(operation) + " failed"); }
    };

    if (auto bc = BindThreadContext(); bc.IsError()) {
        UB_LOG_WARN("Teardown: BindThreadContext failed: {}", bc.Message().c_str());
        recordStatus(bc, "BindThreadContext");
    }

    const auto deviceInfoStatus = impl_->FreeAllDeviceInfo();
    recordStatus(deviceInfoStatus, "FreeAllDeviceInfo");
    if (deviceInfoStatus.IsError()) { return firstError; }

    recordCode(impl_->token.Reset(), "RaCtxTokenIdFree");
    if (!impl_->token) {
        recordCode(impl_->ctx.Reset(), "RaCtxDeinit");
    } else {
        UB_LOG_WARN("skip RaCtxDeinit: token is still alive");
    }

    const bool contextReleased = !impl_->token && !impl_->ctx;
    if (contextReleased) {
        std::lock_guard<std::mutex> cacheLock(impl_->tpCacheMutex);
        impl_->tpCache.clear();
    }
    if (contextReleased && impl_->raInitCalled) {
        int rc = v2::DlHccpV2Api::RaDeinit(&impl_->raInitCfg);
        if (rc != 0) {
            UB_LOG_ERROR("RaDeinit returned rc={}", rc);
            if (firstError.IsOk()) {
                firstError = UbStatus(UbErrorCode::HccpV2RaInitFailed,
                                      "RaDeinit failed rc=" + std::to_string(rc));
            }
        } else {
            impl_->raInitCalled = false;
        }
    }
    if (contextReleased && !impl_->raInitCalled) {
        recordCode(impl_->tsdProc.Reset(), "TsdProcessClose");
    }

    const bool hccpReleased = contextReleased && !impl_->raInitCalled && !impl_->tsdProc;
    if (impl_->deviceSet && hccpReleased) {
        // The ACL device is process-wide and may be shared by the UCM host.
        // This manager only selected it; ownership remains with the host process.
        impl_->deviceSet = false;
    }
    if (!impl_->deviceSet) impl_->externalRaSession = false;
    impl_->initialized = false;

    return firstError;
}

UbStatus AllocDeviceMemory(void** devPtr, uint64_t size)
{
    if (devPtr == nullptr) { return UbStatus(UbErrorCode::InvalidArgument, "devPtr null"); }
    if (size == 0) { return UbStatus(UbErrorCode::InvalidArgument, "size == 0"); }
    *devPtr = nullptr;
    const uint64_t allocSize = AlignUrmaSegSize(size);
    auto st = acl::DlAclRt::Malloc(devPtr, allocSize, acl::MallocPolicy::HugeFirst);
    if (st.IsError()) return st;
    if (*devPtr == nullptr) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclrtMalloc returned a null pointer");
    }
    st = acl::DlAclRt::Memset(*devPtr, allocSize, 0, allocSize);
    if (st.IsError()) {
        auto freeStatus = acl::DlAclRt::Free(*devPtr);
        if (freeStatus.IsError()) {
            UB_LOG_ERROR("AllocDeviceMemory cleanup failed: {}", freeStatus.Message().c_str());
        }
        *devPtr = nullptr;
        return st.WithContext("failed to initialize device memory");
    }
    return UbStatus::Ok();
}

UbStatus FreeDeviceMemory(void* devPtr) { return acl::DlAclRt::Free(devPtr); }

}  // namespace umc::comm
