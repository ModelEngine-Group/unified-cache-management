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

    std::vector<MRHandle> firstHandles;
    ASSERT_TRUE(provider.RegisterMemory(descs, firstHandles).ok());
    std::vector<MRHandle> secondHandles;
    ASSERT_TRUE(provider.RegisterMemory(descs, secondHandles).ok());

    std::unordered_set<MRHandle> uniqueHandles;
    uniqueHandles.insert(firstHandles.begin(), firstHandles.end());
    uniqueHandles.insert(secondHandles.begin(), secondHandles.end());
    EXPECT_EQ(uniqueHandles.size(), firstHandles.size() + secondHandles.size());
    EXPECT_EQ(uniqueHandles.count(kInvalidMRHandle), 0);
}

TEST(FakeTransProviderTest, BindMemoryCreatesProviderLocalHandles)
{
    FakeTransProvider provider(FakeTransProviderConfig{});
    const std::vector<TransProvider::BindMemoryDesc> descs{
        {TransProvider::MemType::MEM_DEVICE, 0x1000, 4096, 1},
        {TransProvider::MemType::MEM_DEVICE, 0x2000, 4096, 1}
    };

    std::vector<MRHandle> handles;
    ASSERT_TRUE(provider.BindMemory(descs, handles).ok());

    ASSERT_EQ(handles.size(), descs.size());
    EXPECT_NE(handles[0], kInvalidMRHandle);
    EXPECT_NE(handles[1], kInvalidMRHandle);
    EXPECT_NE(handles[0], handles[1]);
}

}  // namespace
}  // namespace UC::ASU
