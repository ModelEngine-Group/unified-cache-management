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
#include "trans/cuda/gdr/gdr_config.h"
#include "trans/cuda/gdr/gdr_mr_buffer.h"

TEST(UCGdrMrBufferTest, GdrKVBufferConfigAddsAndRemovesDeviceRanges)
{
    char buffer1[256] {};
    char buffer2[512] {};
    uint64_t resolvedBase = 0;
    size_t resolvedSize = 0;
    {
        UC::Trans::GdrKVBufferConfig registration;
        auto status = registration.Register(
            {reinterpret_cast<uintptr_t>(buffer1), reinterpret_cast<uintptr_t>(buffer2)},
            {sizeof(buffer1), sizeof(buffer2)});
        ASSERT_EQ(status, UC::Status::OK());
        ASSERT_TRUE(UC::Trans::DeviceBufferRegistry::Resolve(buffer1 + 32, 64, &resolvedBase,
                                                             &resolvedSize));
        ASSERT_EQ(resolvedBase, reinterpret_cast<uint64_t>(buffer1));
        ASSERT_EQ(resolvedSize, sizeof(buffer1));
        ASSERT_TRUE(UC::Trans::DeviceBufferRegistry::Resolve(buffer2 + 64, 128, &resolvedBase,
                                                             &resolvedSize));
        ASSERT_EQ(resolvedBase, reinterpret_cast<uint64_t>(buffer2));
        ASSERT_EQ(resolvedSize, sizeof(buffer2));
    }
    ASSERT_FALSE(UC::Trans::DeviceBufferRegistry::Resolve(buffer1 + 32, 64, &resolvedBase,
                                                          &resolvedSize));
    ASSERT_FALSE(UC::Trans::DeviceBufferRegistry::Resolve(buffer2 + 64, 128, &resolvedBase,
                                                          &resolvedSize));
}

TEST(UCGdrMrBufferTest, GdrKVBufferConfigRejectsInvalidBufferConfig)
{
    char buffer[256] {};
    ASSERT_EQ(UC::Trans::GdrKVBufferConfig::Validate(
                  {reinterpret_cast<uintptr_t>(buffer)}, {}),
              UC::Status::InvalidParam());
    ASSERT_EQ(UC::Trans::GdrKVBufferConfig::Validate({0}, {sizeof(buffer)}),
              UC::Status::InvalidParam());
    ASSERT_EQ(UC::Trans::GdrKVBufferConfig::Validate(
                  {reinterpret_cast<uintptr_t>(buffer)}, {0}),
              UC::Status::InvalidParam());
}
