#include <gtest/gtest.h>
#include "runtime/device.h"

namespace kv::test {
namespace {

class AsuDeviceTestEnvironment final : public ::testing::Environment {
public:
    void SetUp() override
    {
        const auto initStatus = device_.Init();
        ASSERT_TRUE(initStatus.ok()) << "Device::Init failed: " << initStatus.message;
        ASSERT_TRUE(device_.Setup(0).ok());
    }

    void TearDown() override
    {
        EXPECT_TRUE(device_.Reset(0).ok());
        EXPECT_TRUE(device_.Finalize().ok());
    }

private:
    runtime::Device device_;
};

}  // namespace
}  // namespace kv::test

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new kv::test::AsuDeviceTestEnvironment);
    return RUN_ALL_TESTS();
}
