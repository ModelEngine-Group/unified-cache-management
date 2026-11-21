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
#ifdef UCM_ASU_ENABLE_HCCP_KV_OVER_ROCE_PROVIDER

#include "hccp_kv_over_roce/hccp_acl_context.h"
#include <acl/acl.h>
#include <cstdint>
#include <gtest/gtest.h>

namespace kv::test {
namespace {

class HccpAclContextTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        ASSERT_EQ(aclrtGetDevice(&device_), ACL_SUCCESS);
        ASSERT_EQ(aclrtGetCurrentContext(&callerContext_), ACL_SUCCESS);
        ASSERT_NE(callerContext_, nullptr);
    }

    void TearDown() override
    {
        // A test may have switched devices; bind the shared environment's device
        // again so the following tests start from a known ACL state.
        EXPECT_EQ(aclrtSetDevice(device_), ACL_SUCCESS);
    }

    TransportConfig MakeConfig(std::int32_t deviceId) const
    {
        TransportConfig config;
        config.deviceId = deviceId;
        return config;
    }

    std::int32_t device_{0};
    aclrtContext callerContext_{nullptr};
};

TEST_F(HccpAclContextTest, ResolvePrefersConfigDeviceId)
{
    auto config = MakeConfig(device_);
    config.attrs["device_id"] = std::to_string(device_);
    HccpAclBinding binding;
    ASSERT_TRUE(ResolveHccpAclBinding(config, binding).ok());
    EXPECT_EQ(binding.deviceId, device_);
    EXPECT_EQ(binding.source, "config.deviceId");
    EXPECT_NE(binding.context, nullptr);
}

TEST_F(HccpAclContextTest, ResolveUsesAttributeKeys)
{
    for (const char* key : {"device_id", "deviceId", "logical_device_id"}) {
        auto config = MakeConfig(-1);
        config.attrs[key] = std::to_string(device_);
        HccpAclBinding binding;
        ASSERT_TRUE(ResolveHccpAclBinding(config, binding).ok()) << key;
        EXPECT_EQ(binding.deviceId, device_) << key;
        EXPECT_EQ(binding.source, "config.attr") << key;
    }
}

TEST_F(HccpAclContextTest, ResolveInvalidAttributeUsesCurrentDevice)
{
    for (const char* value : {"not-a-number", "-1", "99999999999999999999"}) {
        auto config = MakeConfig(-1);
        config.attrs["device_id"] = value;
        HccpAclBinding binding;
        ASSERT_TRUE(ResolveHccpAclBinding(config, binding).ok()) << value;
        EXPECT_EQ(binding.deviceId, device_) << value;
        EXPECT_EQ(binding.source, "current_acl_device") << value;
        EXPECT_NE(binding.context, nullptr) << value;
    }
}

TEST_F(HccpAclContextTest, ResolveOverflowingAttributeUsesCurrentDevice)
{
    auto config = MakeConfig(-1);
    config.attrs["device_id"] = "4294967295";
    HccpAclBinding binding;
    ASSERT_TRUE(ResolveHccpAclBinding(config, binding).ok());
    EXPECT_EQ(binding.deviceId, device_);
    EXPECT_EQ(binding.source, "current_acl_device");
}

TEST_F(HccpAclContextTest, ScopedContextRejectsNegativeDevice)
{
    HccpAclBinding binding;
    binding.deviceId = -1;
    ScopedHccpAclContext scope("test", binding);
    EXPECT_EQ(scope.status().code, StatusCode::INVALID_ARGUMENT);
}

TEST_F(HccpAclContextTest, ScopedContextKeepsCallerContextWhenAlreadyBound)
{
    HccpAclBinding binding;
    ASSERT_TRUE(ResolveHccpAclBinding(MakeConfig(device_), binding).ok());
    ASSERT_NE(binding.context, nullptr);

    {
        ScopedHccpAclContext scope("test", binding);
        ASSERT_TRUE(scope.status().ok());
        aclrtContext inside = nullptr;
        ASSERT_EQ(aclrtGetCurrentContext(&inside), ACL_SUCCESS);
        EXPECT_EQ(inside, binding.context);
    }

    aclrtContext after = nullptr;
    ASSERT_EQ(aclrtGetCurrentContext(&after), ACL_SUCCESS);
    EXPECT_EQ(after, callerContext_);
}

TEST_F(HccpAclContextTest, ScopedContextRejectsUnresolvedBinding)
{
    HccpAclBinding binding;
    binding.deviceId = device_;
    binding.context = nullptr;
    binding.source = "test";

    ScopedHccpAclContext scope("test", binding);
    EXPECT_EQ(scope.status().code, StatusCode::NOT_INITIALIZED);
    EXPECT_EQ(binding.context, nullptr);
}

TEST_F(HccpAclContextTest, ScopedContextSwitchesAndRestoresAcrossDevices)
{
    std::uint32_t deviceCount = 0;
    ASSERT_EQ(aclrtGetDeviceCount(&deviceCount), ACL_SUCCESS);
    if (deviceCount < 2U) { GTEST_SKIP() << "requires at least two ACL devices"; }

    ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
    aclrtContext context0 = nullptr;
    ASSERT_EQ(aclrtGetCurrentContext(&context0), ACL_SUCCESS);
    ASSERT_NE(context0, nullptr);

    if (aclrtSetDevice(1) != ACL_SUCCESS) { GTEST_SKIP() << "device 1 is not usable"; }
    aclrtContext callerContext1 = nullptr;
    ASSERT_EQ(aclrtGetCurrentContext(&callerContext1), ACL_SUCCESS);
    ASSERT_NE(callerContext1, nullptr);
    ASSERT_NE(callerContext1, context0);

    HccpAclBinding binding;
    binding.deviceId = 0;
    binding.context = context0;
    binding.source = "test";
    {
        ScopedHccpAclContext scope("multi", binding);
        ASSERT_TRUE(scope.status().ok());
        aclrtContext inside = nullptr;
        ASSERT_EQ(aclrtGetCurrentContext(&inside), ACL_SUCCESS);
        EXPECT_EQ(inside, context0);
    }

    aclrtContext after = nullptr;
    ASSERT_EQ(aclrtGetCurrentContext(&after), ACL_SUCCESS);
    EXPECT_EQ(after, callerContext1);
}

TEST_F(HccpAclContextTest, ResolveRestoresCallerDeviceAndContext)
{
    std::uint32_t deviceCount = 0;
    ASSERT_EQ(aclrtGetDeviceCount(&deviceCount), ACL_SUCCESS);
    if (deviceCount < 2U) { GTEST_SKIP() << "requires at least two ACL devices"; }

    ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
    if (aclrtSetDevice(1) != ACL_SUCCESS) { GTEST_SKIP() << "device 1 is not usable"; }
    std::int32_t callerDevice = -1;
    ASSERT_EQ(aclrtGetDevice(&callerDevice), ACL_SUCCESS);
    ASSERT_EQ(callerDevice, 1);
    aclrtContext callerContext = nullptr;
    ASSERT_EQ(aclrtGetCurrentContext(&callerContext), ACL_SUCCESS);
    ASSERT_NE(callerContext, nullptr);

    HccpAclBinding binding;
    ASSERT_TRUE(ResolveHccpAclBinding(MakeConfig(0), binding).ok());
    EXPECT_EQ(binding.deviceId, 0);
    ASSERT_NE(binding.context, nullptr);

    // Capture must leave the constructing thread on the device and context it
    // started from.
    std::int32_t restoredDevice = -1;
    ASSERT_EQ(aclrtGetDevice(&restoredDevice), ACL_SUCCESS);
    EXPECT_EQ(restoredDevice, callerDevice);
    aclrtContext restoredContext = nullptr;
    ASSERT_EQ(aclrtGetCurrentContext(&restoredContext), ACL_SUCCESS);
    EXPECT_EQ(restoredContext, callerContext);

    // The captured context stays usable after the device was restored, and
    // binding it restores the caller again on scope exit.
    {
        ScopedHccpAclContext scope("after-resolve", binding);
        ASSERT_TRUE(scope.status().ok());
        aclrtContext inside = nullptr;
        ASSERT_EQ(aclrtGetCurrentContext(&inside), ACL_SUCCESS);
        EXPECT_EQ(inside, binding.context);
    }
    ASSERT_EQ(aclrtGetDevice(&restoredDevice), ACL_SUCCESS);
    EXPECT_EQ(restoredDevice, callerDevice);
    ASSERT_EQ(aclrtGetCurrentContext(&restoredContext), ACL_SUCCESS);
    EXPECT_EQ(restoredContext, callerContext);
}

}  // namespace
}  // namespace kv::test

#endif  // UCM_ASU_ENABLE_HCCP_KV_OVER_ROCE_PROVIDER
