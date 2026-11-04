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
#ifndef COPY_INSTANCE_CUDA_H
#define COPY_INSTANCE_CUDA_H

#include <chrono>
#include <cstring>
#include <utility>
#include <vector>
#include "copy_buffer.h"
#include "copy_instance.h"
#include "error_handle_cuda.h"

extern cudaError_t CudaSMCopyBatchAsync(void* src[], void* dst[], size_t size, size_t number,
                                        cudaStream_t stream);

struct CopyContext {
    size_t deviceId;
    cudaStream_t stream;
    cudaEvent_t endEvent;
    size_t size;
    std::vector<void*> src;
    std::vector<void*> dst;
};

class BaseCopyInstance : public CopyInstance {
protected:
    std::vector<CopyContext> contexts_;
    cudaEvent_t totalStart_;
    cudaEvent_t totalEnd_;

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

            CopyContext ctx;
            ctx.deviceId = AffinityDeviceId(src, dst);
            ctx.size = src.Size();
            CUDA_ASSERT(cudaSetDevice(ctx.deviceId));
            CUDA_ASSERT(cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking));
            CUDA_ASSERT(cudaEventCreate(&ctx.endEvent, cudaEventDefault));
            ctx.src.reserve(src.Number());
            ctx.dst.reserve(dst.Number());
            for (size_t j = 0; j < src.Number(); j++) {
                ctx.src.push_back(src[j]);
                ctx.dst.push_back(dst[j]);
            }
            contexts_.push_back(std::move(ctx));
        }

        CUDA_ASSERT(cudaSetDevice(contexts_[0].deviceId));
        CUDA_ASSERT(cudaEventCreate(&totalStart_));
        CUDA_ASSERT(cudaEventCreate(&totalEnd_));
    }

    void Cleanup() override
    {
        for (auto& ctx : contexts_) {
            CUDA_ASSERT(cudaSetDevice(ctx.deviceId));
            CUDA_ASSERT(cudaEventDestroy(ctx.endEvent));
            CUDA_ASSERT(cudaStreamDestroy(ctx.stream));
        }
        CUDA_ASSERT(cudaSetDevice(contexts_[0].deviceId));
        CUDA_ASSERT(cudaEventDestroy(totalStart_));
        CUDA_ASSERT(cudaEventDestroy(totalEnd_));
        contexts_.clear();
    }

    std::pair<size_t, size_t> DoCopyOnce() override
    {
        using namespace std::chrono;

        CUDA_ASSERT(cudaSetDevice(contexts_[0].deviceId));
        CUDA_ASSERT(cudaEventRecord(totalStart_, contexts_[0].stream));

        for (size_t i = 1; i < contexts_.size(); i++) {
            CUDA_ASSERT(cudaSetDevice(contexts_[i].deviceId));
            CUDA_ASSERT(cudaStreamWaitEvent(contexts_[i].stream, totalStart_));
        }

        auto submitStart = steady_clock::now();
        for (auto& ctx : contexts_) {
            CUDA_ASSERT(cudaSetDevice(ctx.deviceId));
            CopyInternal(ctx);
        }
        auto submitCost = duration_cast<microseconds>(steady_clock::now() - submitStart).count();

        for (size_t i = 1; i < contexts_.size(); i++) {
            CUDA_ASSERT(cudaSetDevice(contexts_[i].deviceId));
            CUDA_ASSERT(cudaEventRecord(contexts_[i].endEvent, contexts_[i].stream));
            CUDA_ASSERT(cudaSetDevice(contexts_[0].deviceId));
            CUDA_ASSERT(cudaStreamWaitEvent(contexts_[0].stream, contexts_[i].endEvent));
        }

        CUDA_ASSERT(cudaSetDevice(contexts_[0].deviceId));
        CUDA_ASSERT(cudaEventRecord(totalEnd_, contexts_[0].stream));
        SynchronizeInternal(contexts_[0]);

        float copyCostMs = 0.f;
        CUDA_ASSERT(cudaEventElapsedTime(&copyCostMs, totalStart_, totalEnd_));
        size_t copyCost = static_cast<size_t>(copyCostMs * 1000);

        return {copyCost, submitCost};
    }

    virtual void CopyInternal(const CopyContext& ctx) = 0;
    virtual void SynchronizeInternal(const CopyContext& ctx) = 0;

public:
    BaseCopyInstance(size_t iterations, bool affinitySrc) : CopyInstance(iterations, affinitySrc) {}
};

class CECopyInstance : public BaseCopyInstance {
protected:
    void SynchronizeInternal(const CopyContext& ctx) override
    {
        CUDA_ASSERT(cudaStreamSynchronize(ctx.stream));
    }

public:
    CECopyInstance(size_t iterations, bool affinitySrc) : BaseCopyInstance(iterations, affinitySrc)
    {
    }

    std::string Name() const override { return "CE"; }
};

class H2DCECopyInstance : public CECopyInstance {
protected:
    void CopyInternal(const CopyContext& ctx) override
    {
        for (size_t i = 0; i < ctx.src.size(); i++) {
            CUDA_ASSERT(cudaMemcpyAsync(ctx.dst[i], ctx.src[i], ctx.size, cudaMemcpyHostToDevice,
                                        ctx.stream));
        }
    }

public:
    H2DCECopyInstance(size_t iterations, bool affinitySrc) : CECopyInstance(iterations, affinitySrc)
    {
    }
};

class D2HCECopyInstance : public CECopyInstance {
protected:
    void CopyInternal(const CopyContext& ctx) override
    {
        for (size_t i = 0; i < ctx.src.size(); i++) {
            CUDA_ASSERT(cudaMemcpyAsync(ctx.dst[i], ctx.src[i], ctx.size, cudaMemcpyDeviceToHost,
                                        ctx.stream));
        }
    }

public:
    D2HCECopyInstance(size_t iterations, bool affinitySrc) : CECopyInstance(iterations, affinitySrc)
    {
    }
};

class BatchCECopyInstance : public CECopyInstance {
protected:
    size_t device_;

public:
    BatchCECopyInstance(size_t iterations, bool affinitySrc, size_t device)
        : CECopyInstance(iterations, affinitySrc), device_(device)
    {
    }

    std::string Name() const override { return "BatchCE"; }
};

class H2DBatchCECopyInstance : public BatchCECopyInstance {
protected:
    void CopyInternal(const CopyContext& ctx) override
    {
        cudaMemcpyAttributes attr;
        std::memset(&attr, 0, sizeof(attr));
        attr.srcAccessOrder = cudaMemcpySrcAccessOrderAny;
        attr.srcLocHint.type = cudaMemLocationTypeHost;
        attr.dstLocHint.type = cudaMemLocationTypeDevice;
        attr.dstLocHint.id = device_;
        attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;

        std::vector<cudaMemcpyAttributes> attrArray{attr};
        std::vector<size_t> attrIdxArray(ctx.src.size(), 0);
        std::vector<size_t> sizeArray(ctx.src.size(), ctx.size);
        size_t failureIdx = 0;

        CUDA_ASSERT(cudaMemcpyBatchAsync(const_cast<void**>(ctx.dst.data()),
                                         const_cast<void**>(ctx.src.data()), sizeArray.data(),
                                         ctx.src.size(), attrArray.data(), attrIdxArray.data(),
                                         attrArray.size(), &failureIdx, ctx.stream));
    }

public:
    H2DBatchCECopyInstance(size_t iterations, bool affinitySrc, size_t device)
        : BatchCECopyInstance(iterations, affinitySrc, device)
    {
    }
};

class D2HBatchCECopyInstance : public BatchCECopyInstance {
protected:
    void CopyInternal(const CopyContext& ctx) override
    {
        cudaMemcpyAttributes attr;
        std::memset(&attr, 0, sizeof(attr));
        attr.srcAccessOrder = cudaMemcpySrcAccessOrderAny;
        attr.srcLocHint.type = cudaMemLocationTypeDevice;
        attr.srcLocHint.id = device_;
        attr.dstLocHint.type = cudaMemLocationTypeHost;
        attr.dstLocHint.id = device_;
        attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;

        std::vector<cudaMemcpyAttributes> attrArray{attr};
        std::vector<size_t> attrIdxArray(ctx.src.size(), 0);
        std::vector<size_t> sizeArray(ctx.src.size(), ctx.size);
        size_t failureIdx = 0;

        CUDA_ASSERT(cudaMemcpyBatchAsync(const_cast<void**>(ctx.dst.data()),
                                         const_cast<void**>(ctx.src.data()), sizeArray.data(),
                                         ctx.src.size(), attrArray.data(), attrIdxArray.data(),
                                         attrArray.size(), &failureIdx, ctx.stream));
    }

public:
    D2HBatchCECopyInstance(size_t iterations, bool affinitySrc, size_t device)
        : BatchCECopyInstance(iterations, affinitySrc, device)
    {
    }
};

class SMCopyInstance : public BaseCopyInstance {
protected:
    size_t device_;
    size_t number_;
    void* dSrc_;
    void* dDst_;

    void CopyInternal(const CopyContext& ctx) override
    {
        size_t ptrSize = sizeof(void*) * ctx.src.size();
        CUDA_ASSERT(
            cudaMemcpyAsync(dSrc_, ctx.src.data(), ptrSize, cudaMemcpyHostToDevice, ctx.stream));
        CUDA_ASSERT(
            cudaMemcpyAsync(dDst_, ctx.dst.data(), ptrSize, cudaMemcpyHostToDevice, ctx.stream));
        CUDA_ASSERT(CudaSMCopyBatchAsync(static_cast<void**>(dSrc_), static_cast<void**>(dDst_),
                                         ctx.size, ctx.src.size(), ctx.stream));
    }

    void SynchronizeInternal(const CopyContext& ctx) override
    {
        CUDA_ASSERT(cudaStreamSynchronize(ctx.stream));
    }

public:
    SMCopyInstance(size_t iterations, bool affinitySrc, size_t device, size_t number)
        : BaseCopyInstance(iterations, affinitySrc),
          device_(device),
          number_(number),
          dSrc_(nullptr),
          dDst_(nullptr)
    {
        CUDA_ASSERT(cudaSetDevice(device_));
        CUDA_ASSERT(cudaMalloc(&dSrc_, sizeof(void*) * number_));
        CUDA_ASSERT(cudaMalloc(&dDst_, sizeof(void*) * number_));
    }

    ~SMCopyInstance()
    {
        CUDA_ASSERT(cudaSetDevice(device_));
        if (dSrc_) CUDA_ASSERT(cudaFree(dSrc_));
        if (dDst_) CUDA_ASSERT(cudaFree(dDst_));
    }

    std::string Name() const override { return "SM"; }
};

class D2DCECopyInstance : public CECopyInstance {
protected:
    void CopyInternal(const CopyContext& ctx) override
    {
        for (size_t i = 0; i < ctx.src.size(); i++) {
            CUDA_ASSERT(cudaMemcpyAsync(ctx.dst[i], ctx.src[i], ctx.size, cudaMemcpyDeviceToDevice,
                                        ctx.stream));
        }
    }

public:
    D2DCECopyInstance(size_t iterations, bool affinitySrc) : CECopyInstance(iterations, affinitySrc)
    {
    }
};

#endif  // COPY_INSTANCE_CUDA_H
