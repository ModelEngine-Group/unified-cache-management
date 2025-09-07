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
#include "device/idevice.h"

class UCDeviceTest : public ::testing::Test {};

TEST_F(UCDeviceTest, TransData)
{
    using T = size_t;
    constexpr size_t number = 2048;
    constexpr size_t size = sizeof(T);
    auto device = UC::DeviceFactory::Make(-1, size, number);
    ASSERT_NE(device, nullptr);
    ASSERT_TRUE(device->Setup().Success());
    uintptr_t src[number] = {0};
    uintptr_t dst[number] = {0};
    for (size_t i = 0; i < number; i++) {
        src[i] = (uintptr_t)device->GetBuffer(i);
        *(T*)(src[i]) = 100;
        dst[i] = (uintptr_t)device->GetBuffer(number + i);
        *(T*)(dst[i]) = 200;
    }
    ASSERT_TRUE(device->H2DBatch(src, dst, number, size).Success());
    for (size_t i = 0; i < number; i++) {
        ASSERT_EQ(*(T*)(src[i]), *(T*)(dst[i]));
        device->PutBuffer(i, (void*)src[i]);
        device->PutBuffer(number + i, (void*)dst[i]);
    }
}
