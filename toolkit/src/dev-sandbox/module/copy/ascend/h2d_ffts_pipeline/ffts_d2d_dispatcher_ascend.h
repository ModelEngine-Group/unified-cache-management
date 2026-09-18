/**
 * MIT License
 *
 * Copyright (c) 2026 Mag1c.H
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
#ifndef FFTS_D2D_DISPATCHER_ASCEND_H
#define FFTS_D2D_DISPATCHER_ASCEND_H

#include <acl/acl.h>
#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>
#include "error_handle.h"

#if __has_include("runtime/rt_ffts_plus.h")
#include "runtime/rt_ffts_plus.h"
#elif __has_include("rt_external_ffts.h")
#include "rt_external_ffts.h"
#else
#error "FFTS Plus header was not found. Configure Ascend FFTS include directories in CMake."
#endif

constexpr uint32_t kFftsSdmaFp32AtomicMoveSqe = 0x1E70;
constexpr uint16_t kFftsContextMaxNum = 128;
constexpr uint16_t kDefaultFftsMaxReadyLanes = 8;
constexpr uint8_t kFftsCommunicationTask = 0x5A;
constexpr const char* kFftsMaxReadyLanesEnv = "FFTS_MAX_READY_LANES";

static_assert(sizeof(rtFftsPlusComCtx_t) == 128, "rtFftsPlusComCtx_t must be 128 bytes");
static_assert(sizeof(rtFftsPlusSdmaCtx_t) == 128, "rtFftsPlusSdmaCtx_t must be 128 bytes");

#define ASCEND_RT_ASSERT(expr)                                             \
    do {                                                                   \
        auto __rtErr = (expr);                                             \
        if ((__rtErr) != RT_ERROR_NONE) {                                  \
            fprintf(stderr, "[RT Error %d] in expression %s at %s:%d\n",   \
                    static_cast<int>(__rtErr), #expr, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

struct AscendFftsCopySpec {
    void* dst;
    const void* src;
    size_t size;
};

class FftsD2DDispatcher {
public:
    void Reserve(size_t count) { contexts_.reserve(count); }

    void Reset()
    {
        contexts_.clear();
        completed_ = false;
    }

    void AddMemcpy(void* dst, const void* src, size_t size)
    {
        ASSERT(!completed_);
        ASSERT(dst != nullptr);
        ASSERT(src != nullptr);
        ASSERT(size > 0);
        ASSERT(size <= std::numeric_limits<uint32_t>::max());
        ASSERT(contexts_.size() < std::numeric_limits<uint16_t>::max());

        rtFftsPlusComCtx_t comCtx{};
        auto* sdmaCtx = reinterpret_cast<rtFftsPlusSdmaCtx_t*>(&comCtx);
        BuildSdmaCtx(dst, src, size, sdmaCtx);
        contexts_.push_back(comCtx);
    }

    void AddDependency(uint32_t predecessorId, uint32_t successorId)
    {
        ASSERT(predecessorId < contexts_.size());
        ASSERT(successorId < contexts_.size());

        auto& predecessor = contexts_[predecessorId];
        auto& successor = contexts_[successorId];
        ASSERT(predecessor.successorNum < RT_CTX_SUCCESSOR_NUM);
        ASSERT(successor.predCntInit < std::numeric_limits<uint8_t>::max());

        predecessor.successorList[predecessor.successorNum] = static_cast<uint16_t>(successorId);
        predecessor.successorNum++;
        successor.predCntInit++;
        successor.predCnt++;
    }

    uint16_t BuildCopies(const std::vector<AscendFftsCopySpec>& copies)
    {
        Reset();
        if (copies.empty()) { return 0; }

        Reserve(copies.size());
        const uint16_t maxReadyLanes = MaxReadyLanes();
        const uint16_t laneCount =
            static_cast<uint16_t>(std::min<size_t>(copies.size(), maxReadyLanes));
        std::vector<int32_t> lastTaskId(laneCount, -1);

        for (size_t i = 0; i < copies.size(); ++i) {
            AddMemcpy(copies[i].dst, copies[i].src, copies[i].size);

            const size_t lane = i % laneCount;
            const uint32_t taskId = static_cast<uint32_t>(contexts_.size() - 1);
            if (lastTaskId[lane] >= 0) {
                AddDependency(static_cast<uint32_t>(lastTaskId[lane]), taskId);
            }
            lastTaskId[lane] = static_cast<int32_t>(taskId);
        }

        return laneCount;
    }

    void Launch(aclrtStream stream, uint16_t readyContextNum)
    {
        ASSERT(!contexts_.empty());
        ASSERT(readyContextNum > 0);
        ASSERT(readyContextNum <= contexts_.size());

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
        ASCEND_RT_ASSERT(
            rtFftsPlusTaskLaunchWithFlag(&task, reinterpret_cast<rtStream_t>(stream), 0));
    }

private:
    static uint16_t MaxReadyLanes()
    {
        const char* value = std::getenv(kFftsMaxReadyLanesEnv);
        if (value == nullptr || value[0] == '\0') { return kDefaultFftsMaxReadyLanes; }

        char* end = nullptr;
        errno = 0;
        const unsigned long parsed = std::strtoul(value, &end, 10);
        if (errno != 0 || end == value || *end != '\0' || parsed == 0) {
            return kDefaultFftsMaxReadyLanes;
        }

        const auto maxValue = static_cast<unsigned long>(std::numeric_limits<uint16_t>::max());
        if (parsed > maxValue) { return std::numeric_limits<uint16_t>::max(); }
        return static_cast<uint16_t>(parsed);
    }

    static uint64_t PtrToU64(const void* ptr)
    {
        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
    }

    static void BuildSdmaCtx(void* dst, const void* src, size_t size, rtFftsPlusSdmaCtx_t* ctx)
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

    std::vector<rtFftsPlusComCtx_t> contexts_;
    bool completed_ = false;
};

#endif  // FFTS_D2D_DISPATCHER_ASCEND_H
