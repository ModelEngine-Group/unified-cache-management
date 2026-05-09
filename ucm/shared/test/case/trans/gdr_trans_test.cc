/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#include <gtest/gtest.h>

#if defined(UCM_ENABLE_GDR_STREAM)
#include <chrono>
#include <memory>
#include <new>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "trans/cuda/gdr/gdr_mr_buffer.h"
#include "trans/device.h"

namespace {

struct DelayedEventPayload {
    int delayMs;
};

void CUDART_CB DelayEventCallback(cudaStream_t stream, cudaError_t status, void* data)
{
    (void)stream;
    (void)status;
    std::unique_ptr<DelayedEventPayload> payload{static_cast<DelayedEventPayload*>(data)};
    std::this_thread::sleep_for(std::chrono::milliseconds(payload->delayMs));
}

UC::Status ScheduleDelayedEvent(cudaStream_t stream, cudaEvent_t event, int delayMs)
{
    auto* payload = new (std::nothrow) DelayedEventPayload{delayMs};
    if (!payload) { return UC::Status::OutOfMemory(); }

    auto ret = cudaStreamAddCallback(stream, DelayEventCallback, payload, 0);
    if (ret != cudaSuccess) {
        delete payload;
        return UC::Status{ret, cudaGetErrorString(ret)};
    }

    ret = cudaEventRecord(event, stream);
    if (ret != cudaSuccess) { return UC::Status{ret, cudaGetErrorString(ret)}; }
    return UC::Status::OK();
}

template <class Fn>
bool WaitUntil(Fn&& fn, std::chrono::milliseconds timeout)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (fn()) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return fn();
}

}  // namespace

class UCGdrTransUnitTest : public ::testing::Test {};

TEST_F(UCGdrTransUnitTest, CopyDataWithGDR)
{
    const auto ok = UC::Status::OK();
    constexpr int32_t deviceId = 0;
    constexpr size_t size = 36 * 1024;
    constexpr size_t number = 64 * 61;
    UC::Trans::Device device;
    ASSERT_EQ(device.Setup(deviceId), ok);
    auto buffer = device.MakeBuffer();
    auto stream = device.MakeGdrStream();
    if (!stream) { GTEST_SKIP() << "GDR stream is unavailable in current environment."; }
    auto hPtr1 = buffer->MakeHostBuffer(size * number);
    ASSERT_NE(hPtr1, nullptr);
    ASSERT_EQ(buffer->MakeDeviceBuffers(size, number), ok);
    std::vector<std::shared_ptr<void>> ptrHolder;
    ptrHolder.reserve(number);
    void* dPtrArr[number];
    for (size_t i = 0; i < number; i++) {
        *(size_t*)(((char*)hPtr1.get()) + size * i) = i;
        auto ptr = buffer->GetDeviceBuffer(size);
        dPtrArr[i] = ptr.get();
        ptrHolder.emplace_back(ptr);
    }
    auto hPtr2 = buffer->MakeHostBuffer(size * number);
    ASSERT_NE(hPtr2, nullptr);
    ASSERT_EQ(stream->HostToDeviceAsync(hPtr1.get(), dPtrArr, size, number), ok);
    ASSERT_EQ(stream->DeviceToHostAsync(dPtrArr, hPtr2.get(), size, number), ok);
    ASSERT_EQ(stream->Synchronized(), ok);
    for (size_t i = 0; i < number; i++) {
        ASSERT_EQ(*(size_t*)(((char*)hPtr2.get()) + size * i), i);
    }
}

TEST_F(UCGdrTransUnitTest, ResolveDeviceBufferRegistry)
{
    char buffer[1024] {};
    const auto* basePtr = static_cast<void*>(buffer);
    auto* offsetPtr = static_cast<void*>(buffer + 128);
    uint64_t resolvedBase = 0;
    size_t resolvedSize = 0;

    UC::Trans::DeviceBufferRegistry::Register(basePtr, sizeof(buffer));
    ASSERT_TRUE(UC::Trans::DeviceBufferRegistry::Resolve(offsetPtr, 64, &resolvedBase,
                                                         &resolvedSize));
    ASSERT_EQ(resolvedBase, reinterpret_cast<uint64_t>(basePtr));
    ASSERT_EQ(resolvedSize, sizeof(buffer));

    UC::Trans::DeviceBufferRegistry::Unregister(basePtr);
    ASSERT_FALSE(UC::Trans::DeviceBufferRegistry::Resolve(offsetPtr, 64, &resolvedBase,
                                                          &resolvedSize));
}

TEST_F(UCGdrTransUnitTest, GdrWaitEventProgressesInBackground)
{
    const auto ok = UC::Status::OK();
    constexpr int32_t deviceId = 0;
    constexpr uint64_t expected = 0x123456789abcdef0ULL;
    UC::Trans::Device device;
    ASSERT_EQ(device.Setup(deviceId), ok);
    auto buffer = device.MakeBuffer();
    auto stream = device.MakeGdrStream();
    if (!stream) { GTEST_SKIP() << "GDR stream is unavailable in current environment."; }

    ASSERT_EQ(buffer->MakeDeviceBuffers(sizeof(expected), 1), ok);
    auto dPtr = buffer->GetDeviceBuffer(sizeof(expected));
    auto hPtr = buffer->MakeHostBuffer(sizeof(expected));
    ASSERT_NE(dPtr, nullptr);
    ASSERT_NE(hPtr, nullptr);
    *static_cast<uint64_t*>(hPtr.get()) = 0;
    ASSERT_EQ(cudaMemcpy(dPtr.get(), &expected, sizeof(expected), cudaMemcpyHostToDevice),
              cudaSuccess);

    cudaStream_t gateStream = nullptr;
    cudaEvent_t gateEvent = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&gateStream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&gateEvent, cudaEventDisableTiming), cudaSuccess);
    ASSERT_EQ(ScheduleDelayedEvent(gateStream, gateEvent, 50), ok);

    ASSERT_EQ(stream->WaitEvent(static_cast<void*>(gateEvent)), ok);
    ASSERT_EQ(stream->DeviceToHostAsync(dPtr.get(), hPtr.get(), sizeof(expected)), ok);
    ASSERT_TRUE(WaitUntil([&] { return *static_cast<uint64_t*>(hPtr.get()) == expected; },
                          std::chrono::milliseconds(500)));
    ASSERT_EQ(stream->Synchronized(), ok);

    ASSERT_EQ(cudaEventDestroy(gateEvent), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(gateStream), cudaSuccess);
}

TEST_F(UCGdrTransUnitTest, GdrWaitEventKeepsLaterBarrierFromBlockingEarlierCopy)
{
    const auto ok = UC::Status::OK();
    constexpr int32_t deviceId = 0;
    constexpr uint64_t expected1 = 0x1111111111111111ULL;
    constexpr uint64_t expected2 = 0x2222222222222222ULL;
    UC::Trans::Device device;
    ASSERT_EQ(device.Setup(deviceId), ok);
    auto buffer = device.MakeBuffer();
    auto stream = device.MakeGdrStream();
    if (!stream) { GTEST_SKIP() << "GDR stream is unavailable in current environment."; }

    ASSERT_EQ(buffer->MakeDeviceBuffers(sizeof(expected1), 2), ok);
    auto dPtr1 = buffer->GetDeviceBuffer(sizeof(expected1));
    auto dPtr2 = buffer->GetDeviceBuffer(sizeof(expected2));
    auto hPtr1 = buffer->MakeHostBuffer(sizeof(expected1));
    auto hPtr2 = buffer->MakeHostBuffer(sizeof(expected2));
    ASSERT_NE(dPtr1, nullptr);
    ASSERT_NE(dPtr2, nullptr);
    ASSERT_NE(hPtr1, nullptr);
    ASSERT_NE(hPtr2, nullptr);
    *static_cast<uint64_t*>(hPtr1.get()) = 0;
    *static_cast<uint64_t*>(hPtr2.get()) = 0;
    ASSERT_EQ(cudaMemcpy(dPtr1.get(), &expected1, sizeof(expected1), cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dPtr2.get(), &expected2, sizeof(expected2), cudaMemcpyHostToDevice),
              cudaSuccess);

    cudaStream_t gateStream1 = nullptr;
    cudaStream_t gateStream2 = nullptr;
    cudaEvent_t gateEvent1 = nullptr;
    cudaEvent_t gateEvent2 = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&gateStream1, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&gateStream2, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&gateEvent1, cudaEventDisableTiming), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&gateEvent2, cudaEventDisableTiming), cudaSuccess);
    ASSERT_EQ(ScheduleDelayedEvent(gateStream1, gateEvent1, 50), ok);
    ASSERT_EQ(ScheduleDelayedEvent(gateStream2, gateEvent2, 500), ok);

    ASSERT_EQ(stream->WaitEvent(static_cast<void*>(gateEvent1)), ok);
    ASSERT_EQ(stream->DeviceToHostAsync(dPtr1.get(), hPtr1.get(), sizeof(expected1)), ok);
    ASSERT_EQ(stream->WaitEvent(static_cast<void*>(gateEvent2)), ok);
    ASSERT_EQ(stream->DeviceToHostAsync(dPtr2.get(), hPtr2.get(), sizeof(expected2)), ok);

    ASSERT_TRUE(WaitUntil([&] { return *static_cast<uint64_t*>(hPtr1.get()) == expected1; },
                          std::chrono::milliseconds(250)));
    ASSERT_NE(*static_cast<uint64_t*>(hPtr2.get()), expected2);
    ASSERT_EQ(stream->Synchronized(), ok);
    ASSERT_EQ(*static_cast<uint64_t*>(hPtr2.get()), expected2);

    ASSERT_EQ(cudaEventDestroy(gateEvent1), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(gateEvent2), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(gateStream1), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(gateStream2), cudaSuccess);
}

TEST_F(UCGdrTransUnitTest, CopyDataWithGdrPressure)
{
    const auto ok = UC::Status::OK();
    constexpr int32_t deviceId = 0;
    constexpr size_t size = sizeof(uint64_t);
    constexpr size_t number = 5000;
    UC::Trans::Device device;
    ASSERT_EQ(device.Setup(deviceId), ok);
    auto buffer = device.MakeBuffer();
    auto stream = device.MakeGdrStream();
    if (!stream) { GTEST_SKIP() << "GDR stream is unavailable in current environment."; }

    auto hPtr1 = buffer->MakeHostBuffer(size * number);
    ASSERT_NE(hPtr1, nullptr);
    ASSERT_EQ(buffer->MakeDeviceBuffers(size, number), ok);
    std::vector<std::shared_ptr<void>> ptrHolder;
    std::vector<void*> dPtrArr(number);
    ptrHolder.reserve(number);
    for (size_t i = 0; i < number; ++i) {
        *(static_cast<uint64_t*>(hPtr1.get()) + i) = i + 1;
        auto ptr = buffer->GetDeviceBuffer(size);
        dPtrArr[i] = ptr.get();
        ptrHolder.emplace_back(ptr);
    }

    auto hPtr2 = buffer->MakeHostBuffer(size * number);
    ASSERT_NE(hPtr2, nullptr);
    ASSERT_EQ(stream->HostToDeviceAsync(hPtr1.get(), dPtrArr.data(), size, number), ok);
    ASSERT_EQ(stream->DeviceToHostAsync(dPtrArr.data(), hPtr2.get(), size, number), ok);
    ASSERT_EQ(stream->Synchronized(), ok);
    for (size_t i = 0; i < number; ++i) {
        ASSERT_EQ(*(static_cast<uint64_t*>(hPtr2.get()) + i), i + 1);
    }
}

TEST_F(UCGdrTransUnitTest, GdrStreamContinuesAfterSingleCopyFailure)
{
    const auto ok = UC::Status::OK();
    constexpr int32_t deviceId = 0;
    constexpr uint64_t value = 7;
    UC::Trans::Device device;
    ASSERT_EQ(device.Setup(deviceId), ok);
    auto buffer = device.MakeBuffer();
    auto stream = device.MakeGdrStream();
    if (!stream) { GTEST_SKIP() << "GDR stream is unavailable in current environment."; }

    auto hPtr = buffer->MakeHostBuffer(sizeof(value));
    auto dPtr = buffer->MakeDeviceBuffer(sizeof(value));
    ASSERT_NE(hPtr, nullptr);
    ASSERT_NE(dPtr, nullptr);
    *static_cast<uint64_t*>(hPtr.get()) = value;

    ASSERT_EQ(stream->HostToDeviceAsync(hPtr.get(), nullptr, sizeof(value)), ok);
    const auto failedStatus = stream->Synchronized();
    ASSERT_TRUE(failedStatus.Failure());

    ASSERT_EQ(stream->HostToDeviceAsync(hPtr.get(), dPtr.get(), sizeof(value)), ok);
    ASSERT_EQ(stream->Synchronized(), ok);

    *static_cast<uint64_t*>(hPtr.get()) = 0;
    ASSERT_EQ(stream->DeviceToHostAsync(dPtr.get(), hPtr.get(), sizeof(value)), ok);
    ASSERT_EQ(stream->Synchronized(), ok);
    ASSERT_EQ(*static_cast<uint64_t*>(hPtr.get()), value);
}
#endif
