#include <gtest/gtest.h>
#include "trans/device.h"

namespace UC::ASU {
namespace {

class AsuDeviceTestEnvironment final : public ::testing::Environment {
public:
    void SetUp() override
    {
        const auto initStatus = device_.Init();
        ASSERT_TRUE(initStatus.Success() || initStatus == UC::Status::DuplicateKey())
            << "Device::Init failed: " << initStatus.ToString();
        ASSERT_TRUE(device_.Setup(0).Success());
    }

    void TearDown() override
    {
        EXPECT_TRUE(device_.Reset(0).Success());
        EXPECT_TRUE(device_.Finalize().Success());
    }

private:
    Trans::Device device_;
};

}  // namespace
}  // namespace UC::ASU

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new UC::ASU::AsuDeviceTestEnvironment);
    return RUN_ALL_TESTS();
}
