#define private public
#include "fake_trans_provider.h"
#undef private
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <sstream>
#include <thread>
#include <unordered_set>
#include "proto/kv_protocol.h"

namespace kv::test {
namespace {

std::string FakeBackendKeyFileName(const CacheKey& key)
{
    std::uint64_t hash = 1469598103934665603ULL;
    for (auto byte : key) {
        hash ^= std::to_integer<unsigned char>(byte);
        hash *= 1099511628211ULL;
    }

    std::ostringstream stream;
    stream << std::hex << std::setw(16) << std::setfill('0') << hash << ".bin";
    return stream.str();
}

std::vector<std::uint32_t> BuildExistRequest(const std::vector<CacheKey>& keys, bool sc)
{
    std::vector<std::uint32_t> request(kSqeDwordCount + keys.size() * kKeyEntryDwordCount, 0);
    request[0] = static_cast<std::uint32_t>(KvOpcode::Exist);
    request[1] = 1;
    request[10] = static_cast<std::uint32_t>(keys.size());
    if (sc) { request[10] |= 1U << 16; }
    for (std::size_t index = 0; index < keys.size(); ++index) {
        std::memcpy(request.data() + kSqeDwordCount + index * kKeyEntryDwordCount,
                    keys[index].data(), kCacheKeySizeBytes);
    }
    return request;
}

std::array<std::uint32_t, kSqeDwordCount> BuildKeepAliveRequest(std::uint16_t cid)
{
    std::array<std::uint32_t, kSqeDwordCount> request{};
    request[0] =
        static_cast<std::uint32_t>(KvOpcode::KeepAlive) | (static_cast<std::uint32_t>(cid) << 16);
    return request;
}

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

TEST(FakeTransProviderTest, SendQueuesCompletionForWorker)
{
    FakeTransProviderConfig config;
    config.latencyMs = 20;
    config.workerThreads = 2;
    FakeTransProvider provider(config);

    constexpr std::uint16_t cid = 7;
    auto request = BuildKeepAliveRequest(cid);
    std::array<std::uint32_t, kCqeDwordCount> completion{};
    std::vector<MRHandle> handles;
    ASSERT_TRUE(
        provider
            .RegisterMemory(
                {
                    {TransProvider::MemType::MEM_HOST,
                     reinterpret_cast<std::uintptr_t>(request.data()),    sizeof(request),
                     reinterpret_cast<std::uintptr_t>(request.data())   },
                    {TransProvider::MemType::MEM_HOST,
                     reinterpret_cast<std::uintptr_t>(completion.data()), sizeof(completion),
                     reinterpret_cast<std::uintptr_t>(completion.data())}
    },
                handles)
            .ok());

    const auto statuses = provider.Send(
        {
            {nullptr, request.data(), completion.data(), sizeof(request)}
    },
        0, 0);
    ASSERT_EQ(statuses.size(), 1U);
    ASSERT_TRUE(statuses[0].ok()) << statuses[0].message;
    EXPECT_EQ(__atomic_load_n(completion.data() + 3, __ATOMIC_ACQUIRE), 0U);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while ((__atomic_load_n(completion.data() + 3, __ATOMIC_ACQUIRE) & 0xFFFF) == 0 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(__atomic_load_n(completion.data() + 3, __ATOMIC_ACQUIRE) & 0xFFFF, cid);
}

TEST(FakeTransProviderTest, ParsesWorkerThreadCount)
{
    TransportConfig config;
    config.attrs["fake_backend.worker_threads"] = "6";
    EXPECT_EQ(MakeFakeTransProviderConfig(config).workerThreads, 6U);
}

TEST(FakeTransProviderTest, ExistHonorsSeekControl)
{
    const auto storePath =
        std::filesystem::temp_directory_path() /
        ("asu-fake-sc-test-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto asuPath = storePath / "asu-1";
    std::filesystem::create_directories(asuPath);

    CacheKey first{};
    first[0] = std::byte{1};
    CacheKey missing{};
    missing[0] = std::byte{2};
    CacheKey third{};
    third[0] = std::byte{3};
    std::ofstream{asuPath / FakeBackendKeyFileName(first)}.put('\0');
    std::ofstream{asuPath / FakeBackendKeyFileName(third)}.put('\0');

    FakeTransProviderConfig config;
    config.storePath = storePath.string();
    config.latencyMs = 0;
    const std::vector<CacheKey> keys{first, missing, third};
    FakeTransProvider provider{config};

    auto request = BuildExistRequest(keys, false);
    std::vector<std::uint32_t> completion;
    auto status = provider.CompleteFakeBackendRequest(
        request.data(), request.size() * sizeof(std::uint32_t), completion);
    EXPECT_TRUE(status.ok()) << status.message;
    ASSERT_GT(completion.size(), kCqeDwordCount);
    EXPECT_EQ(completion[0] & 0xFFFF, 1U);

    request = BuildExistRequest(keys, true);
    status = provider.CompleteFakeBackendRequest(
        request.data(), request.size() * sizeof(std::uint32_t), completion);
    EXPECT_TRUE(status.ok()) << status.message;
    ASSERT_GT(completion.size(), kCqeDwordCount);
    EXPECT_EQ(completion[0] & 0xFFFF, 2U);
    EXPECT_EQ(completion[kCqeDwordCount] & 0x7, 0x5U);

    std::error_code errorCode;
    std::filesystem::remove_all(storePath, errorCode);
}

}  // namespace
}  // namespace kv::test
