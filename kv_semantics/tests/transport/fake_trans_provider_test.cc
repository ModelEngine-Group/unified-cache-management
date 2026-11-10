#define private public
#include "fake/fake_trans_provider.h"
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

constexpr std::uint16_t kCqeSuccess = 0x000;
constexpr std::uint16_t kCqeCheckResultBuffer = 0x732;

std::string FakeBackendKeyFileName(const CacheKey& key)
{
    std::uint64_t hash = 1469598103934665603ULL;
    for (auto byte : CacheKeyView(key)) {
        hash ^= static_cast<unsigned char>(byte);
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
                    CacheKeyView(keys[index]).data(), kCacheKeySizeBytes);
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
    config.latencyUs = 20000;
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

TEST(FakeTransProviderTest, ParsesImmediateCompletionMode)
{
    TransportConfig config;
    config.attrs["fake_backend.complete_immediately"] = "true";
    EXPECT_TRUE(MakeFakeTransProviderConfig(config).completeImmediately);
}

TEST(FakeTransProviderTest, ParsesLatencyInMicroseconds)
{
    TransportConfig config;
    config.attrs["fake_backend.latency_us"] = "250";
    EXPECT_EQ(MakeFakeTransProviderConfig(config).latencyUs, 250U);
}

TEST(FakeTransProviderTest, ImmediateCompletionBypassesBackendExecution)
{
    FakeTransProviderConfig config;
    config.completeImmediately = true;
    config.latencyUs = 1000000;
    FakeTransProvider provider(config);

    constexpr std::uint16_t cid = 11;
    std::array<std::uint32_t, kSqeDwordCount> request{};
    request[0] = 0xFFU | (static_cast<std::uint32_t>(cid) << 16);
    std::vector<std::uint32_t> completion;

    const auto status = provider.CompleteFakeBackendRequest(
        request.data(), request.size() * sizeof(std::uint32_t), completion);

    EXPECT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(completion.size(), kCqeDwordCount);
    EXPECT_EQ(completion[3] & 0xFFFF, cid);
    EXPECT_EQ(completion[3] >> 17, kCqeSuccess);
}

TEST(FakeTransProviderTest, ImmediateCompletionReportsAllQueriedKeysMissing)
{
    FakeTransProviderConfig config;
    config.completeImmediately = true;
    FakeTransProvider provider(config);

    BinaryKey first{};
    first[0] = std::byte{1};
    BinaryKey second{};
    second[0] = std::byte{2};
    for (const bool useSeekControl : {false, true}) {
        const auto request = BuildExistRequest({first, second}, useSeekControl);
        std::vector<std::uint32_t> completion;

        const auto status = provider.CompleteFakeBackendRequest(
            request.data(), request.size() * sizeof(std::uint32_t), completion);

        EXPECT_TRUE(status.ok()) << status.message;
        ASSERT_EQ(completion.size(), kCqeDwordCount + 1);
        EXPECT_EQ(completion[0] & 0xFFFF, 0U);
        EXPECT_EQ(completion[3] >> 17, kCqeCheckResultBuffer);
        EXPECT_EQ(completion[kCqeDwordCount], 0U);
    }
}

TEST(FakeTransProviderTest, NamespaceIsolatesStoreLoadQueryAndDeleteForTheSameKey)
{
    const auto path = std::filesystem::temp_directory_path() /
                      ("asu-fake-namespace-" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    FakeTransProviderConfig config;
    config.storePath = path.string();
    config.latencyUs = 0;
    FakeTransProvider provider(config);
    BinaryKey bytes{};
    bytes[0] = std::byte{1};
    const CacheKey key = bytes;
    alignas(512) std::array<char, 512> first;
    alignas(512) std::array<char, 512> second;
    alignas(512) std::array<char, 512> target{};
    first.fill('a');
    second.fill('b');
    alignas(64) std::array<std::uint32_t, 8> flag{};
    auto execute = [&](auto&& protocol, const auto& request) {
        std::vector<std::uint32_t> buffer(64, 0);
        const auto packed = protocol.PackSqe(request, buffer.data());
        ASSERT_TRUE(packed.ok()) << packed.message;
        std::vector<std::uint32_t> completion;
        ASSERT_TRUE(provider
                        .CompleteFakeBackendRequest(
                            buffer.data(), buffer.size() * sizeof(std::uint32_t), completion)
                        .ok());
        ASSERT_GE(completion.size(), kCqeDwordCount);
        EXPECT_EQ((completion[3] >> 17) & 0x7ff, kCqeSuccess);
    };
    KvStoreRequest store;
    store.key = key;
    store.buffer_length = store.length = 512;
    store.mr_key = 1;
    store.kv_ns_id = 7;
    store.buffer_addr = reinterpret_cast<std::uint64_t>(first.data());
    execute(KvStoreProtocol{}, store);
    store.kv_ns_id = 9;
    store.buffer_addr = reinterpret_cast<std::uint64_t>(second.data());
    execute(KvStoreProtocol{}, store);
    KvRetrieveRequest load;
    load.key = key;
    load.buffer_addr = reinterpret_cast<std::uint64_t>(target.data());
    load.buffer_length = load.length = 512;
    load.mr_key = 1;
    load.kv_ns_id = 7;
    execute(KvRetrieveProtocol{}, load);
    EXPECT_EQ(target, first);
    load.kv_ns_id = 9;
    execute(KvRetrieveProtocol{}, load);
    EXPECT_EQ(target, second);
    KvDeleteRequest remove;
    remove.kv_ns_id = 7;
    remove.keys = {key};
    remove.batch_number = 1;
    remove.response_buffer_addr = reinterpret_cast<std::uint64_t>(flag.data());
    remove.response_mr_key = 1;
    remove.rflag = true;
    execute(KvDeleteProtocol{}, remove);
    for (auto id : {7U, 9U}) {
        auto query = BuildExistRequest({key}, true);
        query[1] = id;
        std::vector<std::uint32_t> completion;
        ASSERT_TRUE(provider
                        .CompleteFakeBackendRequest(
                            query.data(), query.size() * sizeof(std::uint32_t), completion)
                        .ok());
        EXPECT_EQ(completion[0] & 0xffff, id == 7 ? 0U : 1U);
    }
    target.fill(0);
    execute(KvRetrieveProtocol{}, load);
    EXPECT_EQ(target, second);
    std::error_code errorCode;
    std::filesystem::remove_all(path, errorCode);
}

TEST(FakeTransProviderTest, ExistHonorsSeekControl)
{
    const auto storePath =
        std::filesystem::temp_directory_path() /
        ("asu-fake-sc-test-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto asuPath = storePath / "asu-1";
    std::filesystem::create_directories(asuPath);

    BinaryKey first{};
    first[0] = std::byte{1};
    BinaryKey missing{};
    missing[0] = std::byte{2};
    BinaryKey third{};
    third[0] = std::byte{3};
    std::ofstream{asuPath / FakeBackendKeyFileName(first)}.put('\0');
    std::ofstream{asuPath / FakeBackendKeyFileName(third)}.put('\0');

    FakeTransProviderConfig config;
    config.storePath = storePath.string();
    config.latencyUs = 0;
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
