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

TEST(FakeTransProviderTest, BindMemoryAcceptsRegisteredRegions)
{
    FakeTransProvider provider(FakeTransProviderConfig{});
    std::vector<RegisteredMemory> regions(2);
    regions[0].handle = reinterpret_cast<MRHandle>(std::uintptr_t{101});
    regions[1].handle = reinterpret_cast<MRHandle>(std::uintptr_t{102});

    std::vector<MRHandle> handles;
    ASSERT_TRUE(provider.BindMemory(regions, handles).ok());
    ASSERT_EQ(handles.size(), regions.size());
    EXPECT_NE(handles[0], regions[0].handle);
    EXPECT_NE(handles[1], regions[1].handle);

    std::vector<TransProvider::UnbindMemoryDesc> descs;
    for (auto handle : handles) { descs.push_back({handle}); }
    const auto statuses = provider.UnbindMemory(descs);
    ASSERT_EQ(statuses.size(), regions.size());
    EXPECT_TRUE(statuses[0].ok());
    EXPECT_TRUE(statuses[1].ok());
}

}  // namespace
}  // namespace UC::ASU
