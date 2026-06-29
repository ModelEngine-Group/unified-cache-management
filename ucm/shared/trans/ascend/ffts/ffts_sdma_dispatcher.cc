/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "ffts_sdma_dispatcher.h"
#include <algorithm>
#include <limits>
#include "logger/logger.h"

namespace UC::Trans {

namespace {
constexpr uint32_t kFftsSdmaFp32AtomicMoveSqe = 0x1E70;
constexpr uint16_t kFftsContextMaxNum = 128;
constexpr uint8_t kFftsCommunicationTask = 0x5A;

static_assert(sizeof(rtFftsPlusComCtx_t) == 128, "rtFftsPlusComCtx_t must be 128 bytes");
static_assert(sizeof(rtFftsPlusSdmaCtx_t) == 128, "rtFftsPlusSdmaCtx_t must be 128 bytes");

uint64_t PtrToU64(const void* ptr)
{
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
}
}  // namespace

void FftsSdmaDispatcher::Reset()
{
    contexts_.clear();
    completed_ = false;
}

Status FftsSdmaDispatcher::AddMemcpy(void* dst, const void* src, size_t size)
{
    if (completed_) { return Status::Error("FFTS dispatcher already launched"); }
    if (dst == nullptr || src == nullptr || size == 0) {
        return Status::InvalidParam("invalid FFTS copy spec");
    }
    if (size > std::numeric_limits<uint32_t>::max()) {
        return Status::InvalidParam("too large FFTS copy size({})", size);
    }
    if (contexts_.size() >= std::numeric_limits<uint16_t>::max()) {
        return Status::InvalidParam("too many FFTS contexts({})", contexts_.size());
    }

    rtFftsPlusComCtx_t comCtx{};
    auto* sdmaCtx = reinterpret_cast<rtFftsPlusSdmaCtx_t*>(&comCtx);
    BuildSdmaCtx(dst, src, size, sdmaCtx);
    contexts_.push_back(comCtx);
    return Status::OK();
}

Status FftsSdmaDispatcher::AddDependency(uint32_t predecessorId, uint32_t successorId)
{
    if (predecessorId >= contexts_.size() || successorId >= contexts_.size()) {
        return Status::InvalidParam("invalid FFTS dependency({},{})", predecessorId, successorId);
    }
    auto& predecessor = contexts_[predecessorId];
    auto& successor = contexts_[successorId];
    if (predecessor.successorNum >= RT_CTX_SUCCESSOR_NUM ||
        successor.predCntInit >= std::numeric_limits<uint8_t>::max()) {
        return Status::InvalidParam("too many FFTS dependencies");
    }

    predecessor.successorList[predecessor.successorNum] = static_cast<uint16_t>(successorId);
    predecessor.successorNum++;
    successor.predCntInit++;
    successor.predCnt++;
    return Status::OK();
}

Status FftsSdmaDispatcher::BuildCopies(const std::vector<AscendFftsCopySpec>& copies,
                                       uint16_t maxReadyLanes, uint16_t& readyContextNum)
{
    readyContextNum = 0;
    Reset();
    if (copies.empty()) { return Status::InvalidParam("empty FFTS copy specs"); }
    if (maxReadyLanes == 0) { return Status::InvalidParam("invalid FFTS max ready lanes"); }

    contexts_.reserve(copies.size());
    const auto laneCount = static_cast<uint16_t>(std::min<size_t>(copies.size(), maxReadyLanes));
    std::vector<int32_t> lastTaskId(laneCount, -1);

    for (size_t i = 0; i < copies.size(); ++i) {
        auto s = AddMemcpy(copies[i].dst, copies[i].src, copies[i].size);
        if (s.Failure()) { return s; }

        const size_t lane = i % laneCount;
        const auto taskId = static_cast<uint32_t>(contexts_.size() - 1);
        if (lastTaskId[lane] >= 0) {
            s = AddDependency(static_cast<uint32_t>(lastTaskId[lane]), taskId);
            if (s.Failure()) { return s; }
        }
        lastTaskId[lane] = static_cast<int32_t>(taskId);
    }

    readyContextNum = laneCount;
    return Status::OK();
}

Status FftsSdmaDispatcher::Launch(aclrtStream stream, uint16_t readyContextNum)
{
    if (stream == nullptr) { return Status::InvalidParam("invalid FFTS stream"); }
    if (contexts_.empty()) { return Status::InvalidParam("empty FFTS contexts"); }
    if (readyContextNum == 0 || readyContextNum > contexts_.size()) {
        return Status::InvalidParam("invalid FFTS ready context number({})", readyContextNum);
    }

    rtFftsPlusSqe_t sqe{};
    sqe.fftsType = RT_FFTS_PLUS_TYPE;
    sqe.totalContextNum = static_cast<uint16_t>(contexts_.size());
    sqe.readyContextNum = readyContextNum;
    sqe.preloadContextNum = std::min<uint16_t>(readyContextNum, kFftsContextMaxNum);
    sqe.timeout = 0;
    sqe.subType = kFftsCommunicationTask;

    rtFftsPlusTaskInfo_t task{};
    task.fftsPlusSqe = &sqe;
    task.descBuf = contexts_.data();
    task.descBufLen = sizeof(rtFftsPlusComCtx_t) * contexts_.size();
    task.descAddrType = RT_FFTS_PLUS_CTX_DESC_ADDR_TYPE_HOST;
    task.argsHandleInfoNum = 0;
    task.argsHandleInfoPtr = nullptr;

    completed_ = true;
    auto ret = rtFftsPlusTaskLaunchWithFlag(&task, reinterpret_cast<rtStream_t>(stream), 0);
    if (ret == RT_ERROR_NONE) { return Status::OK(); }
    UC_ERROR("Failed({}) to launch FFTS plus task.", static_cast<int32_t>(ret));
    return Status{static_cast<int32_t>(ret), "rtFftsPlusTaskLaunchWithFlag failed"};
}

void FftsSdmaDispatcher::BuildSdmaCtx(void* dst, const void* src, size_t size,
                                      rtFftsPlusSdmaCtx_t* ctx)
{
    constexpr uint32_t kShift = 32;
    constexpr uint64_t kLowMask = 0xFFFFFFFFULL;

    const uint64_t srcAddr = PtrToU64(src);
    const uint64_t dstAddr = PtrToU64(dst);

    ctx->contextType = RT_CTX_TYPE_SDMA;
    ctx->threadDim = 1;
    ctx->sdmaSqeHeader = kFftsSdmaFp32AtomicMoveSqe;
    ctx->sourceAddressBaseL = static_cast<uint32_t>(srcAddr & kLowMask);
    ctx->sourceAddressBaseH = static_cast<uint32_t>(srcAddr >> kShift);
    ctx->sourceAddressOffset = 0;
    ctx->destinationAddressBaseL = static_cast<uint32_t>(dstAddr & kLowMask);
    ctx->destinationAddressBaseH = static_cast<uint32_t>(dstAddr >> kShift);
    ctx->destinationAddressOffset = 0;
    ctx->nonTailDataLength = static_cast<uint32_t>(size);
    ctx->tailDataLength = static_cast<uint32_t>(size);
}

}  // namespace UC::Trans
