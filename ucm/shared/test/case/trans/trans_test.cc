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
#include "trans/device.h"

class UCTransUnitTest : public ::testing::Test {};

TEST_F(UCTransUnitTest, CopyDataWithCE)
{
    const auto ok = UC::Trans::Status::OK();
    constexpr int32_t deviceId = 0;
    constexpr size_t size = 36 * 1024;
    constexpr size_t number = 64 * 61;
    UC::Trans::Device device;
    ASSERT_EQ(device.Setup(deviceId), ok);
    auto buffer = device.MakeBuffer();
    auto stream = device.MakeStream();
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

TEST_F(UCTransUnitTest, CopyDataWithSM)
{
    const auto ok = UC::Trans::Status::OK();
    constexpr int32_t deviceId = 0;
    constexpr size_t size = 36 * 1024;
    constexpr size_t number = 64 * 61;
    UC::Trans::Device device;
    ASSERT_EQ(device.Setup(deviceId), ok);
    auto buffer = device.MakeBuffer();
    auto stream = device.MakeSMStream();
    if (!stream) { return; }
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
    auto dPtrArrOnDev = buffer->MakeDeviceBuffer(sizeof(dPtrArr));
    ASSERT_EQ(stream->HostToDevice((void*)dPtrArr, dPtrArrOnDev.get(), sizeof(dPtrArr)), ok);
    auto hPtr2 = buffer->MakeHostBuffer(size * number);
    ASSERT_NE(hPtr2, nullptr);
    ASSERT_EQ(stream->HostToDeviceAsync(hPtr1.get(), (void**)dPtrArrOnDev.get(), size, number), ok);
    ASSERT_EQ(stream->DeviceToHostAsync((void**)dPtrArrOnDev.get(), hPtr2.get(), size, number), ok);
    ASSERT_EQ(stream->Synchronized(), ok);
    for (size_t i = 0; i < number; i++) {
        ASSERT_EQ(*(size_t*)(((char*)hPtr2.get()) + size * i), i);
    }
}
