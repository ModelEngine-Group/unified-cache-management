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

class UCGdrNicConfigTest : public testing::Test {
protected:
    void SetUp() override { UC::Trans::GdrNicConfig::ClearForTest(); }
    void TearDown() override { UC::Trans::GdrNicConfig::ClearForTest(); }
};

TEST_F(UCGdrNicConfigTest, ResolveUsesConfiguredDeviceNicNames)
{
    ASSERT_EQ(UC::Trans::GdrNicConfig::SetDeviceNicNames({"mlx5_0", "mlx5_1"}),
              UC::Status::OK());

    auto nic = UC::Trans::GdrNicConfig::ResolveNicName(1);
    ASSERT_TRUE(nic);
    ASSERT_EQ(nic.Value(), "mlx5_1");
}

TEST_F(UCGdrNicConfigTest, AllowsSameMappingAndRejectsConflictingMapping)
{
    ASSERT_EQ(UC::Trans::GdrNicConfig::SetDeviceNicNames({"mlx5_0", "mlx5_1"}),
              UC::Status::OK());
    ASSERT_EQ(UC::Trans::GdrNicConfig::SetDeviceNicNames({"mlx5_0", "mlx5_1"}),
              UC::Status::OK());
    ASSERT_EQ(UC::Trans::GdrNicConfig::SetDeviceNicNames({"mlx5_0", "mlx5_2"}),
              UC::Status::InvalidParam());
}

TEST_F(UCGdrNicConfigTest, RejectsInvalidDeviceNicNames)
{
    ASSERT_EQ(UC::Trans::GdrNicConfig::ValidateDeviceNicNames({"mlx5_0"}, 1),
              UC::Status::InvalidParam());
    ASSERT_EQ(UC::Trans::GdrNicConfig::ValidateDeviceNicNames({""}, 0),
              UC::Status::InvalidParam());
}

TEST_F(UCGdrNicConfigTest, ResolveFallsBackWhenNoMappingIsConfigured)
{
    auto nic = UC::Trans::GdrNicConfig::ResolveNicName(123);
    ASSERT_TRUE(nic);
    ASSERT_FALSE(nic.Value().empty());
}
