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
#include "detail/ffts_engine.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <sstream>

namespace UC::Transport::Ffts {
// Visible within the current .cpp file
namespace {
constexpr size_t kMaxReadyContexts = 128;
constexpr uint16_t kFftsTimeout = std::numeric_limits<uint16_t>::max();
constexpr uint8_t kCommunicationSubType = 0x5A;
constexpr uint8_t kLabelMarker = 0x5A;

Status AclStatus(aclError code, const char* call)
{
    if (code == ACL_SUCCESS) { return Status::OK(); }
    std::ostringstream oss;
    oss << call << " failed: " << code;
    return Status{static_cast<int32_t>(code), oss.str()};
}

Status RtStatus(rtError_t code, const char* call)
{
    if (code == RT_ERROR_NONE) { return Status::OK(); }
    std::ostringstream oss;
    oss << call << " failed: " << code;
    return Status{static_cast<int32_t>(code), oss.str()};
}

uint32_t Low32(uint64_t value) { return static_cast<uint32_t>(value & 0xFFFFFFFFULL); }

uint32_t High32(uint64_t value) { return static_cast<uint32_t>((value >> 32U) & 0xFFFFFFFFULL); }

uint32_t BuildSdmaMoveHeader()
{
    constexpr uint32_t kDataTypeFp32 = 7U;
    constexpr uint32_t kSourceSubstreamValid = 1U << 9U;
    constexpr uint32_t kDestinationSubstreamValid = 1U << 10U;
    constexpr uint32_t kSourceNonSecure = 1U << 11U;
    constexpr uint32_t kDestinationNonSecure = 1U << 12U;
    return (kDataTypeFp32 << 4U) | kSourceSubstreamValid | kDestinationSubstreamValid |
           kSourceNonSecure | kDestinationNonSecure;
}

void FillSdmaContext(rtFftsPlusComCtx_t& storage, const CopyDesc& copy)
{
    static_assert(sizeof(rtFftsPlusSdmaCtx_t) <= sizeof(rtFftsPlusComCtx_t),
                  "FFTS SDMA context must fit in common context storage");
    std::memset(&storage, 0, sizeof(storage));

    auto& ctx = *reinterpret_cast<rtFftsPlusSdmaCtx_t*>(&storage);
    ctx.contextType = RT_CTX_TYPE_SDMA;
    ctx.threadDim = 1;
    ctx.res3 = kLabelMarker;
    ctx.sdmaSqeHeader = BuildSdmaMoveHeader();

    const auto src = reinterpret_cast<uint64_t>(copy.src);
    const auto dst = reinterpret_cast<uint64_t>(copy.dst);
    ctx.sourceAddressBaseL = Low32(src);
    ctx.sourceAddressBaseH = High32(src);
    ctx.destinationAddressBaseL = Low32(dst);
    ctx.destinationAddressBaseH = High32(dst);

    const auto bytes = static_cast<uint32_t>(copy.size);
    ctx.nonTailDataLength = bytes;
    ctx.tailDataLength = bytes;
}

rtFftsPlusSqe_t BuildSqe(uint16_t contextCount)
{
    rtFftsPlusSqe_t sqe{};
    sqe.fftsType = RT_FFTS_PLUS_TYPE;
    sqe.totalContextNum = contextCount;
    sqe.readyContextNum = contextCount;
    sqe.preloadContextNum = contextCount;
    sqe.timeout = kFftsTimeout;
    sqe.subType = kCommunicationSubType;
    return sqe;
}
}  // namespace

FftsEngine::FftsEngine() = default;

FftsEngine::~FftsEngine()
{
    if (stream_ != nullptr) {
        (void)aclrtSynchronizeStream(stream_);
        pendingContexts_.clear();
        (void)aclrtDestroyStream(stream_);
        stream_ = nullptr;
    }
}

/**
 * @brief Initializes the engine for the given Ascend device and creates the ACL stream.
 *
 * @param deviceId Ascend device ID used by this engine.
 * @return Status::OK() on success, otherwise an error status.
 */
Status FftsEngine::Setup(int32_t deviceId)
{
    if (ready_) {
        if (deviceId_ == deviceId) { return Status::OK(); }
        return Status::InvalidParam("FFTS transport is already setup with a different device");
    }

    auto status = AclStatus(aclrtSetDevice(deviceId), "aclrtSetDevice");
    if (status.Failure()) { return status; }

    status = AclStatus(
        aclrtCreateStreamWithConfig(&stream_, 0, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC),
        "aclrtCreateStreamWithConfig");
    if (status.Failure()) {
        stream_ = nullptr;
        return status;
    }

    deviceId_ = deviceId;
    ready_ = true;
    return Status::OK();
}

/**
 * @brief Waits for an ACL event on the engine stream.
 *
 * @param event ACL event handle to wait for. nullptr is treated as no-op.
 * @return Status::OK() on success, otherwise an error status.
 */
Status FftsEngine::WaitEvent(void* event)
{
    if (event == nullptr) { return Status::OK(); }
    auto status = EnsureReady();
    if (status.Failure()) { return status; }
    return AclStatus(aclrtStreamWaitEvent(stream_, static_cast<aclrtEvent>(event)),
                     "aclrtStreamWaitEvent");
}

/**
 * @brief Submits copy descriptors in chunks to the FFTS engine.
 *
 * @param copies Copy descriptor array to submit.
 * @param count Number of descriptors in the array.
 * @return Status::OK() on success, otherwise an error status.
 */
Status FftsEngine::Submit(const CopyDesc* copies, size_t count)
{
    auto status = EnsureReady();
    if (status.Failure()) { return status; }
    if (count == 0) { return Status::OK(); }
    if (copies == nullptr) { return Status::InvalidParam("FFTS copy list is null"); }

    size_t offset = 0;
    while (offset < count) {
        const auto chunk = std::min(kMaxReadyContexts, count - offset);
        status = SubmitChunk(copies + offset, chunk);
        if (status.Failure()) { return status; }
        offset += chunk;
    }
    return Status::OK();
}

Status FftsEngine::Synchronize()
{
    auto status = EnsureReady();
    if (status.Failure()) { return status; }
    status = AclStatus(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream");
    if (status.Success()) { ClearCompletedGraphs(); }
    return status;
}

Status FftsEngine::EnsureReady() const
{
    if (!ready_ || stream_ == nullptr) { return Status::Error("FFTS transport is not setup"); }
    return Status::OK();
}

/**
 * @brief Converts valid copy descriptors into SDMA contexts and submits them as one FFTS Plus task.
 *
 * @param copies Copy descriptors to submit.
 * @param count Number of descriptors in this chunk.
 * @return Status::OK() on success, otherwise an error status.
 */
Status FftsEngine::SubmitChunk(const CopyDesc* copies, size_t count)
{
    std::vector<CopyDesc> activeCopies;

    // Filter out no-op copies and validate the remaining descriptors.
    activeCopies.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const auto& copy = copies[i];
        if (copy.size == 0 || copy.dst == copy.src) { continue; }
        if (copy.dst == nullptr || copy.src == nullptr) {
            return Status::InvalidParam("FFTS copy source and destination must not be null");
        }
        // A single SDMA context stores the copy size as uint32_t.
        if (copy.size > std::numeric_limits<uint32_t>::max()) {
            return Status::InvalidParam("FFTS copy size exceeds single SDMA context limit");
        }
        activeCopies.push_back(copy);
    }

    if (activeCopies.empty()) { return Status::OK(); }

    // Build one SDMA context for each valid copy descriptor.
    auto contexts = std::make_shared<ContextBuffer>(activeCopies.size());
    for (size_t i = 0; i < activeCopies.size(); ++i) {
        FillSdmaContext((*contexts)[i], activeCopies[i]);
    }

    // Build the FFTS Plus SQE using the number of generated contexts.
    const auto contextCount = static_cast<uint16_t>(contexts->size());
    auto sqe = BuildSqe(contextCount);

    // Describe the task to launch. The context buffer is stored in host memory.
    rtFftsPlusTaskInfo_t task{};
    task.fftsPlusSqe = &sqe;
    task.descBuf = contexts->data();
    task.descBufLen = sizeof(rtFftsPlusComCtx_t) * contexts->size();
    task.descAddrType = RT_FFTS_PLUS_CTX_DESC_ADDR_TYPE_HOST;

    // Launch the FFTS Plus task on the engine stream.
    auto status =
        RtStatus(rtFftsPlusTaskLaunchWithFlag(&task, stream_, 0), "rtFftsPlusTaskLaunchWithFlag");
    if (status.Failure()) { return status; }

    // Keep the context buffer alive until the stream is synchronized.
    KeepAlive(std::move(contexts));
    return Status::OK();
}

void FftsEngine::KeepAlive(std::shared_ptr<ContextBuffer> contexts)
{
    pendingContexts_.emplace_back(std::move(contexts));
}

void FftsEngine::ClearCompletedGraphs() { pendingContexts_.clear(); }

}  // namespace UC::Transport::Ffts
