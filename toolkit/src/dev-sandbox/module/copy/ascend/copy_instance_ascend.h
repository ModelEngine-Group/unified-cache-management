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
#ifndef COPY_INSTANCE_ASCEND_H
#define COPY_INSTANCE_ASCEND_H

#include <chrono>
#include <cstring>
#include <utility>
#include <vector>
#include "copy_buffer.h"
#include "copy_instance.h"
#include "error_handle_ascend.h"

struct AscendStreamContext {
    size_t deviceId;
    aclrtStream stream;
    aclrtEvent endEvent;
    size_t size;
    std::vector<void*> src;
    std::vector<void*> dst;
};

class AscendCopyInstanceBase : public CopyInstance {
protected:
    std::vector<AscendStreamContext> contexts_;
    aclrtEvent totalStart_;
    aclrtEvent totalEnd_;

    void Prepare(const std::vector<const CopyBuffer*>& srcBuffers,
                 const std::vector<const CopyBuffer*>& dstBuffers) override
    {
        contexts_.clear();
        const auto bufferNumber = srcBuffers.size();
        for (size_t i = 0; i < bufferNumber; i++) {
            auto& src = *srcBuffers[i];
            auto& dst = *dstBuffers[i];
            ASSERT(src.Number() == dst.Number());
            ASSERT(src.Size() == dst.Size());

            AscendStreamContext ctx;
            ctx.deviceId = AffinityDeviceId(src, dst);
            ctx.size = src.Size();
            ASCEND_ASSERT(aclrtSetDevice(ctx.deviceId));
            ASCEND_ASSERT(aclrtCreateStream(&ctx.stream));
            ASCEND_ASSERT(aclrtCreateEvent(&ctx.endEvent));
            ctx.src.reserve(src.Number());
            ctx.dst.reserve(dst.Number());
            for (size_t j = 0; j < src.Number(); j++) {
                ctx.src.push_back(src[j]);
                ctx.dst.push_back(dst[j]);
            }
            contexts_.push_back(std::move(ctx));
        }

        ASCEND_ASSERT(aclrtSetDevice(contexts_[0].deviceId));
        ASCEND_ASSERT(aclrtCreateEvent(&totalStart_));
        ASCEND_ASSERT(aclrtCreateEvent(&totalEnd_));
    }

    void Cleanup() override
    {
        for (auto& ctx : contexts_) {
            ASCEND_ASSERT(aclrtSetDevice(ctx.deviceId));
            ASCEND_ASSERT(aclrtDestroyEvent(ctx.endEvent));
            ASCEND_ASSERT(aclrtDestroyStream(ctx.stream));
        }
        ASCEND_ASSERT(aclrtSetDevice(contexts_[0].deviceId));
        ASCEND_ASSERT(aclrtDestroyEvent(totalStart_));
        ASCEND_ASSERT(aclrtDestroyEvent(totalEnd_));
        contexts_.clear();
    }

    std::pair<size_t, size_t> DoCopyOnce() override
    {
        using namespace std::chrono;

        ASCEND_ASSERT(aclrtSetDevice(contexts_[0].deviceId));
        ASCEND_ASSERT(aclrtRecordEvent(totalStart_, contexts_[0].stream));

        for (size_t i = 1; i < contexts_.size(); i++) {
            ASCEND_ASSERT(aclrtSetDevice(contexts_[i].deviceId));
            ASCEND_ASSERT(aclrtStreamWaitEvent(contexts_[i].stream, totalStart_));
        }

        auto submitStart = steady_clock::now();
        for (auto& ctx : contexts_) {
            ASCEND_ASSERT(aclrtSetDevice(ctx.deviceId));
            CopyInternal(ctx);
        }
        auto submitCost = duration_cast<microseconds>(steady_clock::now() - submitStart).count();

        for (size_t i = 1; i < contexts_.size(); i++) {
            ASCEND_ASSERT(aclrtSetDevice(contexts_[i].deviceId));
            ASCEND_ASSERT(aclrtRecordEvent(contexts_[i].endEvent, contexts_[i].stream));
            ASCEND_ASSERT(aclrtSetDevice(contexts_[0].deviceId));
            ASCEND_ASSERT(aclrtStreamWaitEvent(contexts_[0].stream, contexts_[i].endEvent));
        }

        ASCEND_ASSERT(aclrtSetDevice(contexts_[0].deviceId));
        ASCEND_ASSERT(aclrtRecordEvent(totalEnd_, contexts_[0].stream));
        SynchronizeInternal(contexts_[0]);

        float copyCostMs = 0.f;
        ASCEND_ASSERT(aclrtEventElapsedTime(&copyCostMs, totalStart_, totalEnd_));
        size_t copyCost = static_cast<size_t>(copyCostMs * 1000);

        return {copyCost, submitCost};
    }

    virtual void CopyInternal(const AscendStreamContext& ctx) = 0;
    virtual void SynchronizeInternal(const AscendStreamContext& ctx) = 0;

public:
    AscendCopyInstanceBase(size_t iterations, bool affinitySrc)
        : CopyInstance(iterations, affinitySrc)
    {
    }
};

class H2DCECopyInstance : public AscendCopyInstanceBase {
protected:
    void CopyInternal(const AscendStreamContext& ctx) override
    {
        for (size_t i = 0; i < ctx.src.size(); i++) {
            ASCEND_ASSERT(aclrtMemcpyAsync(ctx.dst[i], ctx.size, ctx.src[i], ctx.size,
                                           ACL_MEMCPY_HOST_TO_DEVICE, ctx.stream));
        }
    }

    void SynchronizeInternal(const AscendStreamContext& ctx) override
    {
        ASCEND_ASSERT(aclrtSynchronizeStream(ctx.stream));
    }

public:
    H2DCECopyInstance(size_t iterations, bool affinitySrc)
        : AscendCopyInstanceBase(iterations, affinitySrc)
    {
    }

    std::string Name() const override { return "CE"; }
};

class H2DBatchCECopyInstance : public AscendCopyInstanceBase {
protected:
    size_t targetDevice_;

    void CopyInternal(const AscendStreamContext& ctx) override
    {
        aclrtMemcpyBatchAttr attr;
        std::memset(&attr, 0, sizeof(attr));
        attr.srcLoc.type = ACL_MEM_LOCATION_TYPE_HOST;
        attr.dstLoc.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        attr.dstLoc.id = targetDevice_;

        std::vector<aclrtMemcpyBatchAttr> attrArray{attr};
        std::vector<size_t> attrIdxArray(ctx.src.size(), 0);
        std::vector<size_t> sizeArray(ctx.src.size(), ctx.size);
        size_t failureIdx = 0;

        ASCEND_ASSERT(aclrtMemcpyBatchAsync(const_cast<void**>(ctx.dst.data()), sizeArray.data(),
                                            const_cast<void**>(ctx.src.data()), sizeArray.data(),
                                            ctx.src.size(), attrArray.data(), attrIdxArray.data(),
                                            attrArray.size(), &failureIdx, ctx.stream));
    }

    void SynchronizeInternal(const AscendStreamContext& ctx) override
    {
        ASCEND_ASSERT(aclrtSynchronizeStream(ctx.stream));
    }

public:
    H2DBatchCECopyInstance(size_t iterations, bool affinitySrc, size_t targetDevice)
        : AscendCopyInstanceBase(iterations, affinitySrc), targetDevice_(targetDevice)
    {
    }

    std::string Name() const override { return "BatchCE"; }
};

class D2DCECopyInstance : public AscendCopyInstanceBase {
protected:
    void CopyInternal(const AscendStreamContext& ctx) override
    {
        for (size_t i = 0; i < ctx.src.size(); i++) {
            ASCEND_ASSERT(aclrtMemcpyAsync(ctx.dst[i], ctx.size, ctx.src[i], ctx.size,
                                           ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream));
        }
    }

    void SynchronizeInternal(const AscendStreamContext& ctx) override
    {
        ASCEND_ASSERT(aclrtSynchronizeStream(ctx.stream));
    }

public:
    D2DCECopyInstance(size_t iterations, bool affinitySrc)
        : AscendCopyInstanceBase(iterations, affinitySrc)
    {
    }

    std::string Name() const override { return "CE"; }
};

class H2DCEMultiStreamCopyInstance : public CopyInstance {
protected:
    std::vector<AscendStreamContext> contexts_;
    aclrtEvent totalStart_;
    aclrtEvent totalEnd_;
    size_t streamCount_;

    void Prepare(const std::vector<const CopyBuffer*>& srcBuffers,
                 const std::vector<const CopyBuffer*>& dstBuffers) override
    {
        contexts_.clear();
        const auto bufferNumber = srcBuffers.size();
        contexts_.reserve(bufferNumber * streamCount_);

        for (size_t i = 0; i < bufferNumber; i++) {
            auto& src = *srcBuffers[i];
            auto& dst = *dstBuffers[i];
            ASSERT(src.Number() == dst.Number());
            ASSERT(src.Size() == dst.Size());

            size_t bufferCount = src.Number();
            size_t base = bufferCount / streamCount_;
            size_t remainder = bufferCount % streamCount_;
            size_t deviceId = AffinityDeviceId(src, dst);
            ASCEND_ASSERT(aclrtSetDevice(deviceId));

            size_t offset = 0;
            for (size_t s = 0; s < streamCount_; s++) {
                size_t count = base + (s < remainder ? 1 : 0);
                if (count == 0) continue;

                AscendStreamContext ctx;
                ctx.deviceId = deviceId;
                ctx.size = src.Size();
                ASCEND_ASSERT(aclrtCreateStream(&ctx.stream));
                ASCEND_ASSERT(aclrtCreateEvent(&ctx.endEvent));
                ctx.src.reserve(count);
                ctx.dst.reserve(count);
                for (size_t j = 0; j < count; j++) {
                    ctx.src.push_back(src[offset + j]);
                    ctx.dst.push_back(dst[offset + j]);
                }
                contexts_.push_back(std::move(ctx));
                offset += count;
            }
        }

        ASCEND_ASSERT(aclrtSetDevice(contexts_[0].deviceId));
        ASCEND_ASSERT(aclrtCreateEvent(&totalStart_));
        ASCEND_ASSERT(aclrtCreateEvent(&totalEnd_));
    }

    void Cleanup() override
    {
        for (auto& ctx : contexts_) {
            ASCEND_ASSERT(aclrtSetDevice(ctx.deviceId));
            ASCEND_ASSERT(aclrtDestroyEvent(ctx.endEvent));
            ASCEND_ASSERT(aclrtDestroyStream(ctx.stream));
        }
        ASCEND_ASSERT(aclrtSetDevice(contexts_[0].deviceId));
        ASCEND_ASSERT(aclrtDestroyEvent(totalStart_));
        ASCEND_ASSERT(aclrtDestroyEvent(totalEnd_));
        contexts_.clear();
    }

    std::pair<size_t, size_t> DoCopyOnce() override
    {
        using namespace std::chrono;

        ASCEND_ASSERT(aclrtSetDevice(contexts_[0].deviceId));
        ASCEND_ASSERT(aclrtRecordEvent(totalStart_, contexts_[0].stream));

        for (size_t i = 1; i < contexts_.size(); i++) {
            ASCEND_ASSERT(aclrtSetDevice(contexts_[i].deviceId));
            ASCEND_ASSERT(aclrtStreamWaitEvent(contexts_[i].stream, totalStart_));
        }

        auto submitStart = steady_clock::now();
        for (auto& ctx : contexts_) {
            ASCEND_ASSERT(aclrtSetDevice(ctx.deviceId));
            for (size_t i = 0; i < ctx.src.size(); i++) {
                ASCEND_ASSERT(aclrtMemcpyAsync(ctx.dst[i], ctx.size, ctx.src[i], ctx.size,
                                               ACL_MEMCPY_HOST_TO_DEVICE, ctx.stream));
            }
        }
        auto submitCost = duration_cast<microseconds>(steady_clock::now() - submitStart).count();

        for (size_t i = 1; i < contexts_.size(); i++) {
            ASCEND_ASSERT(aclrtSetDevice(contexts_[i].deviceId));
            ASCEND_ASSERT(aclrtRecordEvent(contexts_[i].endEvent, contexts_[i].stream));
            ASCEND_ASSERT(aclrtSetDevice(contexts_[0].deviceId));
            ASCEND_ASSERT(aclrtStreamWaitEvent(contexts_[0].stream, contexts_[i].endEvent));
        }

        ASCEND_ASSERT(aclrtSetDevice(contexts_[0].deviceId));
        ASCEND_ASSERT(aclrtRecordEvent(totalEnd_, contexts_[0].stream));
        ASCEND_ASSERT(aclrtSynchronizeStream(contexts_[0].stream));

        float copyCostMs = 0.f;
        ASCEND_ASSERT(aclrtEventElapsedTime(&copyCostMs, totalStart_, totalEnd_));
        size_t copyCost = static_cast<size_t>(copyCostMs * 1000);

        return {copyCost, submitCost};
    }

public:
    H2DCEMultiStreamCopyInstance(size_t iterations, bool affinitySrc, size_t streamCount)
        : CopyInstance(iterations, affinitySrc), streamCount_(streamCount)
    {
    }

    std::string Name() const override { return "CE"; }
};

#endif  // COPY_INSTANCE_ASCEND_H
