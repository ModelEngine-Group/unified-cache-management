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
#ifndef TRANS_TEMPLATE_ASCEND_H
#define TRANS_TEMPLATE_ASCEND_H

#include "trans_assert_ascend.h"
#include "trans_stopwatch.h"
#include "trans_template.h"

class TransStreamTemplate : public TransTemplate {
protected:
    struct TransTask {
        std::size_t device;
        aclrtStream stream;
        aclrtEvent finish;
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
            ASCEND_ASSERT(aclrtSetDevice(task.device));
            ASCEND_ASSERT(aclrtCreateStream(&task.stream));
            ASCEND_ASSERT(aclrtCreateEvent(&task.finish));
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
        aclrtEvent start;
        aclrtEvent end;
        auto& first = tasks_[0];
        ASCEND_ASSERT(aclrtSetDevice(first.device));
        ASCEND_ASSERT(aclrtCreateEvent(&start));
        ASCEND_ASSERT(aclrtCreateEvent(&end));

        TransStopwatch watch;
        ASCEND_ASSERT(aclrtRecordEvent(start, first.stream));
        for (std::size_t i = 1; i < tasks_.size(); i++) {
            ASCEND_ASSERT(aclrtSetDevice(tasks_[i].device));
            ASCEND_ASSERT(aclrtStreamWaitEvent(tasks_[i].stream, start));
        }
        for (auto& task : tasks_) { OnTransSubmit(task); }
        for (std::size_t i = 1; i < tasks_.size(); i++) {
            ASCEND_ASSERT(aclrtSetDevice(first.device));
            ASCEND_ASSERT(aclrtStreamWaitEvent(first.stream, tasks_[i].finish));
        }
        ASCEND_ASSERT(aclrtRecordEvent(end, first.stream));
        auto costSubmit = watch.Elapse();

        ASCEND_ASSERT(aclrtSetDevice(first.device));
        ASCEND_ASSERT(aclrtSynchronizeStream(first.stream));
        auto costMs = 0.0f;
        ASCEND_ASSERT(aclrtEventElapsedTime(&costMs, start, end));
        ASCEND_ASSERT(aclrtDestroyEvent(start));
        ASCEND_ASSERT(aclrtDestroyEvent(end));
        return {costMs * 1e3, costSubmit};
    }
    void OnTransPost() override
    {
        for (auto& task : tasks_) {
            ASCEND_ASSERT(aclrtSetDevice(task.device));
            ASCEND_ASSERT(aclrtDestroyEvent(task.finish));
            ASCEND_ASSERT(aclrtDestroyStream(task.stream));
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
        ASCEND_ASSERT(aclrtSetDevice(task.device));
        for (std::size_t i = 0; i < task.src.size(); i++) {
            ASCEND_ASSERT(aclrtMemcpyAsync(task.dst[i], task.size, task.src[i], task.size,
                                           ACL_MEMCPY_HOST_TO_DEVICE, task.stream));
        }
        ASCEND_ASSERT(aclrtRecordEvent(task.finish, task.stream));
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
        ASCEND_ASSERT(aclrtSetDevice(task.device));
        aclrtMemcpyBatchAttr attr;
        std::memset(&attr, 0, sizeof(attr));
        attr.srcLoc.type = ACL_MEM_LOCATION_TYPE_HOST;
        attr.dstLoc.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        attr.dstLoc.id = device_;
        std::vector<aclrtMemcpyBatchAttr> attrArray{attr};
        std::vector<size_t> attrIdxArray(task.src.size(), 0);
        std::vector<size_t> sizeArray(task.src.size(), task.size);
        size_t failureIdx = 0;
        ASCEND_ASSERT(aclrtMemcpyBatchAsync(const_cast<void**>(task.dst.data()), sizeArray.data(),
                                            const_cast<void**>(task.src.data()), sizeArray.data(),
                                            task.src.size(), attrArray.data(), attrIdxArray.data(),
                                            attrArray.size(), &failureIdx, task.stream));
        ASCEND_ASSERT(aclrtRecordEvent(task.finish, task.stream));
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
        ASCEND_ASSERT(aclrtSetDevice(task.device));
        for (std::size_t i = 0; i < task.src.size(); i++) {
            ASCEND_ASSERT(aclrtMemcpyAsync(task.dst[i], task.size, task.src[i], task.size,
                                           ACL_MEMCPY_DEVICE_TO_HOST, task.stream));
        }
        ASCEND_ASSERT(aclrtRecordEvent(task.finish, task.stream));
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
        ASCEND_ASSERT(aclrtSetDevice(task.device));
        aclrtMemcpyBatchAttr attr;
        std::memset(&attr, 0, sizeof(attr));
        attr.srcLoc.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        attr.srcLoc.id = device_;
        attr.dstLoc.type = ACL_MEM_LOCATION_TYPE_HOST;
        std::vector<aclrtMemcpyBatchAttr> attrArray{attr};
        std::vector<size_t> attrIdxArray(task.src.size(), 0);
        std::vector<size_t> sizeArray(task.src.size(), task.size);
        size_t failureIdx = 0;
        ASCEND_ASSERT(aclrtMemcpyBatchAsync(const_cast<void**>(task.dst.data()), sizeArray.data(),
                                            const_cast<void**>(task.src.data()), sizeArray.data(),
                                            task.src.size(), attrArray.data(), attrIdxArray.data(),
                                            attrArray.size(), &failureIdx, task.stream));
        ASCEND_ASSERT(aclrtRecordEvent(task.finish, task.stream));
    }

public:
    TransD2HBatchCETemplate(std::size_t device, std::size_t iterations, bool affinitySrc)
        : TransStreamTemplate{iterations, affinitySrc}, device_{device}
    {
    }
    std::string Name() const override { return "batch-ce"; }
};

class TransMultiStreamTemplate : public TransStreamTemplate {
protected:
    std::size_t streamCount_;
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
            auto bufferCount = srcBuffer.Number();
            auto base = bufferCount / streamCount_;
            auto remainder = bufferCount % streamCount_;
            auto device = AffinityDeviceId(srcBuffer, dstBuffer);
            ASCEND_ASSERT(aclrtSetDevice(device));
            std::size_t offset = 0;
            for (std::size_t s = 0; s < streamCount_; s++) {
                size_t count = base + (s < remainder ? 1 : 0);
                if (count == 0) { continue; }
                TransTask task;
                task.device = device;
                task.size = srcBuffer.Size();
                ASCEND_ASSERT(aclrtCreateStream(&task.stream));
                ASCEND_ASSERT(aclrtCreateEvent(&task.finish));
                for (size_t j = 0; j < count; j++) {
                    task.src.push_back(srcBuffer[offset + j]);
                    task.dst.push_back(dstBuffer[offset + j]);
                }
                tasks_.push_back(std::move(task));
                offset += count;
            }
        }
    }

public:
    TransMultiStreamTemplate(std::size_t streamCount, std::size_t iterations, bool affinitySrc)
        : TransStreamTemplate{iterations, affinitySrc}, streamCount_(streamCount)
    {
    }
    std::string Name() const override { return std::to_string(streamCount_) + "-stream"; }
};

class TransH2DMultiStreamTemplate : public TransMultiStreamTemplate {
protected:
    void OnTransSubmit(TransTask& task) const override
    {
        ASCEND_ASSERT(aclrtSetDevice(task.device));
        for (std::size_t i = 0; i < task.src.size(); i++) {
            ASCEND_ASSERT(aclrtMemcpyAsync(task.dst[i], task.size, task.src[i], task.size,
                                           ACL_MEMCPY_HOST_TO_DEVICE, task.stream));
        }
        ASCEND_ASSERT(aclrtRecordEvent(task.finish, task.stream));
    }

public:
    TransH2DMultiStreamTemplate(std::size_t streamCount, std::size_t iterations, bool affinitySrc)
        : TransMultiStreamTemplate{streamCount, iterations, affinitySrc}
    {
    }
};

class TransD2HMultiStreamTemplate : public TransMultiStreamTemplate {
protected:
    void OnTransSubmit(TransTask& task) const override
    {
        ASCEND_ASSERT(aclrtSetDevice(task.device));
        for (std::size_t i = 0; i < task.src.size(); i++) {
            ASCEND_ASSERT(aclrtMemcpyAsync(task.dst[i], task.size, task.src[i], task.size,
                                           ACL_MEMCPY_DEVICE_TO_HOST, task.stream));
        }
        ASCEND_ASSERT(aclrtRecordEvent(task.finish, task.stream));
    }

public:
    TransD2HMultiStreamTemplate(std::size_t streamCount, std::size_t iterations, bool affinitySrc)
        : TransMultiStreamTemplate{streamCount, iterations, affinitySrc}
    {
    }
};

#endif
