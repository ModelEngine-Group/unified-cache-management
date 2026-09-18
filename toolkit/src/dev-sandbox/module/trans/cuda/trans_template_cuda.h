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
#ifndef TRANS_TEMPLATE_CUDA_H
#define TRANS_TEMPLATE_CUDA_H

#include "trans_assert_cuda.h"
#include "trans_kernel_cuda.h"
#include "trans_stopwatch.h"
#include "trans_template.h"

class TransStreamTemplate : public TransTemplate {
protected:
    struct TransTask {
        std::size_t device;
        cudaStream_t stream;
        cudaEvent_t finish;
        std::size_t size;
        std::vector<void*> src;
        std::vector<void*> dst;
    };
    std::vector<TransTask> tasks_;

    void OnTransPre(const std::vector<const TransBuffer*>& srcBuffers,
                    const std::vector<const TransBuffer*>& dstBuffers) override
    {
        tasks_.clear();
        TRANS_ASSERT(srcBuffers.size() == dstBuffers.size());
        for (std::size_t i = 0; i < srcBuffers.size(); i++) {
            auto& srcBuffer = *srcBuffers[i];
            auto& dstBuffer = *dstBuffers[i];
            TRANS_ASSERT(srcBuffer.Number() == dstBuffer.Number());
            TRANS_ASSERT(srcBuffer.Size() == dstBuffer.Size());
            TransTask task;
            task.device = AffinityDeviceId(srcBuffer, dstBuffer);
            CUDA_ASSERT(cudaSetDevice(task.device));
            CUDA_ASSERT(cudaStreamCreateWithFlags(&task.stream, cudaStreamNonBlocking));
            CUDA_ASSERT(cudaEventCreate(&task.finish));
            task.size = srcBuffer.Size();
            for (std::size_t j = 0; j < srcBuffer.Number(); j++) {
                task.src.push_back(srcBuffer[j]);
                task.dst.push_back(dstBuffer[j]);
            }
            tasks_.push_back(std::move(task));
        }
    }
    virtual void OnTransSubmit(TransTask& task) const = 0;
    std::pair<std::size_t, std::size_t> OnTrans() override
    {
        cudaEvent_t start;
        cudaEvent_t end;
        auto& first = tasks_[0];
        CUDA_ASSERT(cudaSetDevice(first.device));
        CUDA_ASSERT(cudaEventCreate(&start));
        CUDA_ASSERT(cudaEventCreate(&end));

        TransStopwatch watch;
        CUDA_ASSERT(cudaEventRecord(start, first.stream));
        for (std::size_t i = 1; i < tasks_.size(); i++) {
            CUDA_ASSERT(cudaSetDevice(tasks_[i].device));
            CUDA_ASSERT(cudaStreamWaitEvent(tasks_[i].stream, start));
        }
        for (auto& task : tasks_) { OnTransSubmit(task); }
        for (std::size_t i = 1; i < tasks_.size(); i++) {
            CUDA_ASSERT(cudaSetDevice(first.device));
            CUDA_ASSERT(cudaStreamWaitEvent(first.stream, tasks_[i].finish));
        }
        CUDA_ASSERT(cudaEventRecord(end, first.stream));
        auto costSubmit = watch.Elapse();

        CUDA_ASSERT(cudaSetDevice(first.device));
        CUDA_ASSERT(cudaStreamSynchronize(first.stream));
        auto costMs = 0.0f;
        CUDA_ASSERT(cudaEventElapsedTime(&costMs, start, end));
        CUDA_ASSERT(cudaEventDestroy(start));
        CUDA_ASSERT(cudaEventDestroy(end));
        return {costMs * 1e3, costSubmit};
    }
    void OnTransPost() override
    {
        for (auto& task : tasks_) {
            CUDA_ASSERT(cudaSetDevice(task.device));
            CUDA_ASSERT(cudaEventDestroy(task.finish));
            CUDA_ASSERT(cudaStreamDestroy(task.stream));
        }
        tasks_.clear();
    }

public:
    TransStreamTemplate(std::size_t iterations, bool affinitySrc)
        : TransTemplate{iterations, affinitySrc}
    {
    }
};

class TransH2DCETemplate : public TransStreamTemplate {
protected:
    void OnTransSubmit(TransTask& task) const override
    {
        CUDA_ASSERT(cudaSetDevice(task.device));
        for (std::size_t i = 0; i < task.src.size(); i++) {
            CUDA_ASSERT(cudaMemcpyAsync(task.dst[i], task.src[i], task.size, cudaMemcpyHostToDevice,
                                        task.stream));
        }
        CUDA_ASSERT(cudaEventRecord(task.finish, task.stream));
    }

public:
    TransH2DCETemplate(std::size_t iterations, bool affinitySrc)
        : TransStreamTemplate{iterations, affinitySrc}
    {
    }
    std::string Name() const override { return "ce"; }
};

class TransH2DBatchCETemplate : public TransStreamTemplate {
protected:
    std::size_t device_;

    void OnTransSubmit(TransTask& task) const override
    {
        CUDA_ASSERT(cudaSetDevice(task.device));
        cudaMemcpyAttributes attr;
        std::memset(&attr, 0, sizeof(attr));
        attr.srcAccessOrder = cudaMemcpySrcAccessOrderAny;
        attr.srcLocHint.type = cudaMemLocationTypeHost;
        attr.dstLocHint.type = cudaMemLocationTypeDevice;
        attr.dstLocHint.id = device_;
        attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;
        std::vector<cudaMemcpyAttributes> attrArray{attr};
        std::vector<size_t> attrIdxArray(task.src.size(), 0);
        std::vector<size_t> sizeArray(task.src.size(), task.size);
        size_t failureIdx = 0;
        CUDA_ASSERT(cudaMemcpyBatchAsync(const_cast<void**>(task.dst.data()),
                                         const_cast<void**>(task.src.data()), sizeArray.data(),
                                         task.src.size(), attrArray.data(), attrIdxArray.data(),
                                         attrArray.size(), &failureIdx, task.stream));
        CUDA_ASSERT(cudaEventRecord(task.finish, task.stream));
    }

public:
    TransH2DBatchCETemplate(std::size_t device, std::size_t iterations, bool affinitySrc)
        : TransStreamTemplate{iterations, affinitySrc}, device_{device}
    {
    }
    std::string Name() const override { return "batch-ce"; }
};

class TransD2HCETemplate : public TransStreamTemplate {
protected:
    void OnTransSubmit(TransTask& task) const override
    {
        CUDA_ASSERT(cudaSetDevice(task.device));
        for (std::size_t i = 0; i < task.src.size(); i++) {
            CUDA_ASSERT(cudaMemcpyAsync(task.dst[i], task.src[i], task.size, cudaMemcpyDeviceToHost,
                                        task.stream));
        }
        CUDA_ASSERT(cudaEventRecord(task.finish, task.stream));
    }

public:
    TransD2HCETemplate(std::size_t iterations, bool affinitySrc)
        : TransStreamTemplate{iterations, affinitySrc}
    {
    }
    std::string Name() const override { return "ce"; }
};

class TransD2HBatchCETemplate : public TransStreamTemplate {
protected:
    std::size_t device_;

    void OnTransSubmit(TransTask& task) const override
    {
        CUDA_ASSERT(cudaSetDevice(task.device));
        cudaMemcpyAttributes attr;
        std::memset(&attr, 0, sizeof(attr));
        attr.srcAccessOrder = cudaMemcpySrcAccessOrderAny;
        attr.srcLocHint.type = cudaMemLocationTypeDevice;
        attr.srcLocHint.id = device_;
        attr.dstLocHint.type = cudaMemLocationTypeHost;
        attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;
        std::vector<cudaMemcpyAttributes> attrArray{attr};
        std::vector<size_t> attrIdxArray(task.src.size(), 0);
        std::vector<size_t> sizeArray(task.src.size(), task.size);
        size_t failureIdx = 0;
        CUDA_ASSERT(cudaMemcpyBatchAsync(const_cast<void**>(task.dst.data()),
                                         const_cast<void**>(task.src.data()), sizeArray.data(),
                                         task.src.size(), attrArray.data(), attrIdxArray.data(),
                                         attrArray.size(), &failureIdx, task.stream));
        CUDA_ASSERT(cudaEventRecord(task.finish, task.stream));
    }

public:
    TransD2HBatchCETemplate(std::size_t device, std::size_t iterations, bool affinitySrc)
        : TransStreamTemplate{iterations, affinitySrc}, device_{device}
    {
    }
    std::string Name() const override { return "batch-ce"; }
};

class TransSMTemplate : public TransStreamTemplate {
protected:
    std::size_t device_;
    std::size_t number_;
    void* dSrc_;
    void* dDst_;

    void OnTransSubmit(TransTask& task) const override
    {
        TRANS_ASSERT(task.src.size() <= number_);
        const auto ptrSize = task.src.size() * sizeof(void*);
        CUDA_ASSERT(cudaSetDevice(task.device));
        CUDA_ASSERT(
            cudaMemcpyAsync(dSrc_, task.src.data(), ptrSize, cudaMemcpyHostToDevice, task.stream));
        CUDA_ASSERT(
            cudaMemcpyAsync(dDst_, task.dst.data(), ptrSize, cudaMemcpyHostToDevice, task.stream));
        CUDA_ASSERT(CudaSMTransBatchAsync(static_cast<void**>(dSrc_), static_cast<void**>(dDst_),
                                          task.size, task.src.size(), task.stream));
        CUDA_ASSERT(cudaEventRecord(task.finish, task.stream));
    }

public:
    TransSMTemplate(std::size_t device, std::size_t num, std::size_t iterations, bool affinitySrc)
        : TransStreamTemplate{iterations, affinitySrc},
          device_{device},
          number_{num},
          dSrc_{nullptr},
          dDst_{nullptr}
    {
        const auto ptrSize = num * sizeof(void*);
        CUDA_ASSERT(cudaSetDevice(device));
        CUDA_ASSERT(cudaMalloc(&dSrc_, ptrSize));
        CUDA_ASSERT(cudaMalloc(&dDst_, ptrSize));
    }
    ~TransSMTemplate() override
    {
        CUDA_ASSERT(cudaSetDevice(device_));
        if (dSrc_) { CUDA_ASSERT(cudaFree(dSrc_)); }
        if (dDst_) { CUDA_ASSERT(cudaFree(dDst_)); }
    }
    std::string Name() const override { return "sm"; }
};

#endif
