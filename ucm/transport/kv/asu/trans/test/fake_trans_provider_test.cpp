#include "fake_trans_provider.h"
#include <gtest/gtest.h>
#include <unordered_set>

namespace UC::ASU {
namespace {

TEST(FakeTransProviderTest, RegisterMemoryReturnsUniqueHandlesAcrossCalls)
{
    FakeTransProvider provider(FakeTransProviderConfig{});
    const std::vector<TransProvider::RegisterMemoryDesc> descs{
        {TransProvider::MemType::MEM_DEVICE, 0x1000, 4096},
        {TransProvider::MemType::MEM_DEVICE, 0x2000, 4096}
    };

    std::vector<TransProvider::MemHandle> firstHandles;
    ASSERT_TRUE(provider.RegisterMemory(nullptr, descs, firstHandles).ok());
    std::vector<TransProvider::MemHandle> secondHandles;
    ASSERT_TRUE(provider.RegisterMemory(nullptr, descs, secondHandles).ok());

    std::unordered_set<TransProvider::MemHandle> uniqueHandles;
    uniqueHandles.insert(firstHandles.begin(), firstHandles.end());
    uniqueHandles.insert(secondHandles.begin(), secondHandles.end());
    EXPECT_EQ(uniqueHandles.size(), firstHandles.size() + secondHandles.size());
    EXPECT_EQ(uniqueHandles.count(nullptr), 0);
}

}  // namespace
}  // namespace UC::ASU
