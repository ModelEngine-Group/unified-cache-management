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
#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "yuanrong_helper.h"

namespace {

UC::Detail::BlockId MakeBlock(std::initializer_list<uint8_t> bytes)
{
    UC::Detail::BlockId block{};
    size_t index = 0;
    for (auto byte : bytes) { block[index++] = static_cast<std::byte>(byte); }
    return block;
}

}  // namespace

TEST(YuanRongConfigTest, UsesConservativeDefaults)
{
    UC::YuanRongStore::Config config;

    EXPECT_TRUE(config.host.empty());
    EXPECT_EQ(config.port, 0);
    EXPECT_EQ(config.dumpPrerequisiteWorkerCount, 2);
    EXPECT_EQ(config.memoryAlignment, UC::YuanRongStore::kDefaultMemoryAlignment);
    EXPECT_EQ(config.posixDumpBatchSize, 0);
    EXPECT_EQ(config.posixMaxInflightGb, 1);
}

TEST(YuanRongHelperTest, DerivePosixDumpBatchSizeTargetsBoundedByteBatches)
{
    using namespace UC::YuanRongStore;
    constexpr uint64_t oneGb = 1ULL << 30;
    constexpr size_t oneMb = 1ULL << 20;

    EXPECT_EQ(DeriveYuanRongPosixDumpBatchSize(16 * oneMb, oneGb), 16);
    EXPECT_EQ(DeriveYuanRongPosixDumpBatchSize(64 * oneMb, oneGb), 4);
    EXPECT_EQ(DeriveYuanRongPosixDumpBatchSize(300 * oneMb, oneGb), 1);
    EXPECT_EQ(DeriveYuanRongPosixDumpBatchSize(oneMb, oneGb), 32);
    EXPECT_EQ(DeriveYuanRongPosixDumpBatchSize(0, oneGb), 0);
    EXPECT_EQ(DeriveYuanRongPosixDumpBatchSize(oneGb + 1, oneGb), 0);
}

TEST(YuanRongHelperTest, DeviceMemoryPreRegistrationRequiresRemoteH2DAndHccs)
{
    using namespace UC::YuanRongStore;

    Config config;
    config.enableRemoteH2D = false;
    EXPECT_TRUE(ResolveDeviceMemoryPreRegistration(config, "HCCS").Success());
    EXPECT_FALSE(config.enableDeviceMemoryPreRegistration);
    EXPECT_TRUE(ResolveDeviceMemoryPreRegistration(config, "invalid").Success());
    EXPECT_FALSE(config.enableDeviceMemoryPreRegistration);

    config.enableRemoteH2D = true;
    EXPECT_TRUE(ResolveDeviceMemoryPreRegistration(config, nullptr).Success());
    EXPECT_FALSE(config.enableDeviceMemoryPreRegistration);
    EXPECT_TRUE(ResolveDeviceMemoryPreRegistration(config, "").Success());
    EXPECT_FALSE(config.enableDeviceMemoryPreRegistration);
    EXPECT_TRUE(ResolveDeviceMemoryPreRegistration(config, "ROCE").Success());
    EXPECT_FALSE(config.enableDeviceMemoryPreRegistration);
    EXPECT_TRUE(ResolveDeviceMemoryPreRegistration(config, "HCCS").Success());
    EXPECT_TRUE(config.enableDeviceMemoryPreRegistration);
}

TEST(YuanRongHelperTest, DeviceMemoryPreRegistrationRejectsInvalidLinkType)
{
    using namespace UC::YuanRongStore;

    Config config;
    config.enableRemoteH2D = true;
    auto status = ResolveDeviceMemoryPreRegistration(config, "invalid");

    EXPECT_TRUE(status.Failure());
    EXPECT_FALSE(config.enableDeviceMemoryPreRegistration);
}

TEST(YuanRongHelperTest, BuildKeysAndBlobsMapsShardToOneContiguousYuanRongObject)
{
    using namespace UC::YuanRongStore;

    Config config;
    config.nameSpace = "ns";
    config.deviceId = 3;
    config.tensorSizes = {64, 96};

    auto block = MakeBlock({0xab, 0xcd});
    std::array<char, 64> tensor0{};
    std::array<char, 96> tensor1{};
    UC::Detail::TaskDesc desc{
        UC::Detail::Shard{block, 7, {tensor0.data(), tensor1.data()}}
    };

    std::vector<std::string> keys;
    std::vector<datasystem::DeviceBlobList> blobLists;
    auto status = BuildKeysAndBlobs(config, desc, keys, blobLists);

    ASSERT_TRUE(status.Success()) << status.ToString();
    ASSERT_EQ(keys.size(), 1);
    EXPECT_EQ(keys[0], "ucm_ns_abcd0000000000000000000000000000_7");
    ASSERT_EQ(blobLists.size(), 1);
    EXPECT_EQ(blobLists[0].deviceIdx, 3);
    EXPECT_EQ(blobLists[0].srcOffset, 0);
    ASSERT_EQ(blobLists[0].blobs.size(), 2);
    EXPECT_EQ(blobLists[0].blobs[0].pointer, tensor0.data());
    EXPECT_EQ(blobLists[0].blobs[0].size, 64);
    EXPECT_EQ(blobLists[0].blobs[1].pointer, tensor1.data());
    EXPECT_EQ(blobLists[0].blobs[1].size, 96);
}

TEST(YuanRongHelperTest, BuildKeysAndBlobsRejectsAddressCountMismatch)
{
    using namespace UC::YuanRongStore;

    Config config;
    config.nameSpace = "ns";
    config.deviceId = 0;
    config.tensorSizes = {64, 96};
    auto block = MakeBlock({0x01});
    std::array<char, 64> tensor0{};
    UC::Detail::TaskDesc desc{
        UC::Detail::Shard{block, 0, {tensor0.data()}}
    };

    std::vector<std::string> keys;
    std::vector<datasystem::DeviceBlobList> blobLists;
    auto status = BuildKeysAndBlobs(config, desc, keys, blobLists);

    EXPECT_TRUE(status.Failure());
}

TEST(YuanRongHelperTest, DeduplicateYuanRongObjectsKeepsFirstShardForEachKey)
{
    using namespace UC::YuanRongStore;

    Config config;
    config.nameSpace = "ns";
    config.deviceId = 0;
    config.tensorSizes = {64};
    auto block0 = MakeBlock({0x01});
    auto block1 = MakeBlock({0x02});
    std::array<char, 64> tensor0{};
    std::array<char, 64> tensor1{};
    std::array<char, 64> tensor2{};
    UC::Detail::TaskDesc desc{
        UC::Detail::Shard{block0, 0, {tensor0.data()}},
        UC::Detail::Shard{block1, 0, {tensor1.data()}},
        UC::Detail::Shard{block0, 0, {tensor2.data()}},
    };

    std::vector<std::string> keys;
    std::vector<datasystem::DeviceBlobList> blobLists;
    auto status = BuildKeysAndBlobs(config, desc, keys, blobLists);
    ASSERT_TRUE(status.Success()) << status.ToString();

    DeduplicateYuanRongObjects(keys, blobLists, &desc);

    ASSERT_EQ(keys.size(), 2);
    ASSERT_EQ(blobLists.size(), 2);
    ASSERT_EQ(desc.size(), 2);
    EXPECT_EQ(keys[0], "ucm_ns_01000000000000000000000000000000_0");
    EXPECT_EQ(keys[1], "ucm_ns_02000000000000000000000000000000_0");
    EXPECT_EQ(blobLists[0].blobs[0].pointer, tensor0.data());
    EXPECT_EQ(blobLists[1].blobs[0].pointer, tensor1.data());
    EXPECT_EQ(desc[0].addrs[0], tensor0.data());
    EXPECT_EQ(desc[1].addrs[0], tensor1.data());
}

TEST(YuanRongHelperTest, FilterH2dFailedIndexesHandlesTotalFailureAndPartialFailure)
{
    using namespace UC::YuanRongStore;

    std::vector<std::string> keys{"k0", "k1", "k2"};

    EXPECT_TRUE(FilterH2dFailedIndexes(keys, {}, false).empty());

    auto all = FilterH2dFailedIndexes(keys, {}, true);
    ASSERT_EQ(all.size(), 3);
    EXPECT_EQ(all[0], 0);
    EXPECT_EQ(all[1], 1);
    EXPECT_EQ(all[2], 2);

    auto partial = FilterH2dFailedIndexes(keys, {"k2", "k0"}, false);
    ASSERT_EQ(partial.size(), 2);
    EXPECT_EQ(partial[0], 0);
    EXPECT_EQ(partial[1], 2);
}

TEST(YuanRongHelperTest, SplitByExistencePreservesOriginalIndexes)
{
    using namespace UC::YuanRongStore;

    std::vector<size_t> hitIndexes;
    std::vector<size_t> missIndexes;
    SplitByExistence({true, false, true, false, false}, hitIndexes, missIndexes);

    EXPECT_EQ(hitIndexes, (std::vector<size_t>{0, 2}));
    EXPECT_EQ(missIndexes, (std::vector<size_t>{1, 3, 4}));

    SplitByExistence({}, hitIndexes, missIndexes);
    EXPECT_TRUE(hitIndexes.empty());
    EXPECT_TRUE(missIndexes.empty());
}

TEST(YuanRongHelperTest, TieredPrefixMapsBackendMissToOriginalIndex)
{
    using namespace UC::YuanRongStore;

    const std::vector<size_t> missIndexes{1, 4, 7};
    ssize_t result = -1;

    EXPECT_TRUE(ResolveTieredPrefixHit(10, missIndexes, -1, result).Success());
    EXPECT_EQ(result, 0);
    EXPECT_TRUE(ResolveTieredPrefixHit(10, missIndexes, 0, result).Success());
    EXPECT_EQ(result, 3);
    EXPECT_TRUE(ResolveTieredPrefixHit(10, missIndexes, 2, result).Success());
    EXPECT_EQ(result, 9);
    EXPECT_TRUE(ResolveTieredPrefixHit(10, missIndexes, 3, result).Failure());
}

TEST(YuanRongHelperTest, TieredReverseMapsBackendHitToOriginalIndex)
{
    using namespace UC::YuanRongStore;

    ssize_t result = -1;

    EXPECT_TRUE(ResolveTieredReverseHit(10, -1, -1, result).Success());
    EXPECT_EQ(result, -1);
    EXPECT_TRUE(ResolveTieredReverseHit(10, -1, 7, result).Success());
    EXPECT_EQ(result, 7);
    EXPECT_TRUE(ResolveTieredReverseHit(10, 3, -1, result).Success());
    EXPECT_EQ(result, 3);
    EXPECT_TRUE(ResolveTieredReverseHit(10, 3, 4, result).Success());
    EXPECT_EQ(result, 8);
}

TEST(YuanRongHelperTest, TieredReverseRejectsUnexpectedIndexes)
{
    using namespace UC::YuanRongStore;

    ssize_t result = -1;

    EXPECT_TRUE(ResolveTieredReverseHit(10, -2, -1, result).Failure());
    EXPECT_TRUE(ResolveTieredReverseHit(10, 10, -1, result).Failure());
    EXPECT_TRUE(ResolveTieredReverseHit(10, -1, 10, result).Failure());
    EXPECT_TRUE(ResolveTieredReverseHit(10, 3, 6, result).Failure());
}

TEST(YuanRongHelperTest, FilterKeysByLocalSetKeysKeepsInputOrderAndAlignedShards)
{
    using namespace UC::YuanRongStore;

    std::array<char, 1> tensor0{};
    std::array<char, 1> tensor1{};
    std::array<char, 1> tensor2{};
    std::vector<std::string> keys{"key-a", "key-b", "key-c"};
    UC::Detail::TaskDesc desc;
    desc.push_back(UC::Detail::Shard{MakeBlock({0x01}), 10, {tensor0.data()}});
    desc.push_back(UC::Detail::Shard{MakeBlock({0x02}), 20, {tensor1.data()}});
    desc.push_back(UC::Detail::Shard{MakeBlock({0x03}), 30, {tensor2.data()}});

    auto status = FilterKeysByLocalSetKeys({"key-c", "key-a"}, keys, desc);

    ASSERT_TRUE(status.Success()) << status.ToString();
    ASSERT_EQ(keys.size(), 2);
    EXPECT_EQ(keys[0], "key-a");
    EXPECT_EQ(keys[1], "key-c");
    ASSERT_EQ(desc.size(), 2);
    EXPECT_EQ(desc[0].index, 10);
    EXPECT_EQ(desc[1].index, 30);
    EXPECT_EQ(desc[0].addrs[0], tensor0.data());
    EXPECT_EQ(desc[1].addrs[0], tensor2.data());
}

TEST(YuanRongHelperTest, FilterKeysByLocalSetKeysRejectsInvalidOutput)
{
    using namespace UC::YuanRongStore;

    std::vector<std::string> keys{"key-a", "key-b"};
    UC::Detail::TaskDesc desc{
        UC::Detail::Shard{MakeBlock({0x01}), 10, {}},
        UC::Detail::Shard{MakeBlock({0x02}), 20, {}},
    };

    auto duplicateKeys = keys;
    auto duplicateDesc = desc;
    EXPECT_TRUE(
        FilterKeysByLocalSetKeys({"key-a", "key-a"}, duplicateKeys, duplicateDesc).Failure());

    auto unknownKeys = keys;
    auto unknownDesc = desc;
    EXPECT_TRUE(FilterKeysByLocalSetKeys({"key-c"}, unknownKeys, unknownDesc).Failure());
}

TEST(YuanRongHelperTest, RecoveryBatchRangesCoverAllIndexesWithoutOverlap)
{
    using namespace UC::YuanRongStore;

    auto ranges = RecoveryBatchRanges(119, 32);

    ASSERT_EQ(ranges.size(), 4);
    EXPECT_EQ(ranges[0], std::make_pair(size_t{0}, size_t{32}));
    EXPECT_EQ(ranges[1], std::make_pair(size_t{32}, size_t{64}));
    EXPECT_EQ(ranges[2], std::make_pair(size_t{64}, size_t{96}));
    EXPECT_EQ(ranges[3], std::make_pair(size_t{96}, size_t{119}));
    EXPECT_TRUE(RecoveryBatchRanges(0, 32).empty());
    EXPECT_TRUE(RecoveryBatchRanges(10, 0).empty());
}

TEST(YuanRongHelperTest, HostBufferCountUsesPipelineConcurrencyAndCompleteBatches)
{
    using namespace UC::YuanRongStore;

    constexpr uint64_t capacityBytes = 8ULL << 30;
    EXPECT_EQ(DeriveYuanRongHostBufferCount(4ULL << 20, 32, 4, 1, capacityBytes), 288);
    EXPECT_EQ(DeriveYuanRongHostBufferCount(64ULL << 20, 32, 4, 1, capacityBytes), 128);
}

TEST(YuanRongHelperTest, HostBufferCountRejectsCapacitySmallerThanOneBatch)
{
    using namespace UC::YuanRongStore;

    constexpr uint64_t capacityBytes = 8ULL << 30;
    EXPECT_EQ(DeriveYuanRongHostBufferCount(1ULL << 30, 32, 4, 1, capacityBytes), 0);
    EXPECT_EQ(DeriveYuanRongHostBufferCount(0, 32, 4, 1, capacityBytes), 0);
}

TEST(YuanRongHelperTest, ComposedBufferHeaderMapsPayloadAfterYuanRongHeader)
{
    using namespace UC::YuanRongStore;

    constexpr size_t memoryAlignment = 4096;
    std::vector<size_t> tensorSizes{64, 96, 128};
    std::vector<uint8_t> buffer(YuanRongComposedObjectSize(tensorSizes, memoryAlignment));

    void* payloadAddress = nullptr;
    auto initStatus = InitYuanRongComposedBuffer("key-a", buffer.data(), buffer.size(), tensorSizes,
                                                 memoryAlignment, payloadAddress);
    ASSERT_TRUE(initStatus.Success()) << initStatus.ToString();

    const auto headerSize = YuanRongHeaderSize(tensorSizes.size(), memoryAlignment);
    EXPECT_EQ(headerSize, memoryAlignment);
    EXPECT_EQ(payloadAddress, buffer.data() + headerSize);

    auto* offsets = reinterpret_cast<uint64_t*>(buffer.data());
    EXPECT_EQ(offsets[0], tensorSizes.size());
    EXPECT_EQ(offsets[1], headerSize);
    EXPECT_EQ(offsets[2], headerSize + tensorSizes[0]);
    EXPECT_EQ(offsets[3], headerSize + tensorSizes[0] + tensorSizes[1]);
    EXPECT_EQ(offsets[4], buffer.size());

    datasystem::MetaInfo metaInfo;
    metaInfo.blobSizeList = {64, 96, 128};
    const void* readPayloadAddress = nullptr;
    auto getStatus = GetYuanRongPayloadAddress("key-a", buffer.data(), buffer.size(), metaInfo,
                                               tensorSizes, memoryAlignment, readPayloadAddress);
    ASSERT_TRUE(getStatus.Success()) << getStatus.ToString();
    EXPECT_EQ(readPayloadAddress, payloadAddress);
}

TEST(YuanRongHelperTest, PayloadAddressRejectsInvalidComposedHeader)
{
    using namespace UC::YuanRongStore;

    constexpr size_t memoryAlignment = 64;
    std::vector<size_t> tensorSizes{64, 96};
    std::vector<uint8_t> buffer(YuanRongComposedObjectSize(tensorSizes, memoryAlignment));
    void* payloadAddress = nullptr;
    ASSERT_TRUE(InitYuanRongComposedBuffer("key-a", buffer.data(), buffer.size(), tensorSizes,
                                           memoryAlignment, payloadAddress)
                    .Success());

    auto* offsets = reinterpret_cast<uint64_t*>(buffer.data());
    offsets[2] += 1;

    datasystem::MetaInfo metaInfo;
    metaInfo.blobSizeList = {64, 96};
    const void* readPayloadAddress = nullptr;
    auto status = GetYuanRongPayloadAddress("key-a", buffer.data(), buffer.size(), metaInfo,
                                            tensorSizes, memoryAlignment, readPayloadAddress);
    EXPECT_TRUE(status.Failure());
    EXPECT_EQ(readPayloadAddress, nullptr);
}

TEST(YuanRongHelperTest, DirectIoPayloadRequiresAlignedAddressAndSize)
{
    using namespace UC::YuanRongStore;

    constexpr size_t alignment = 4096;
    alignas(alignment) std::array<uint8_t, alignment * 2> buffer{};

    EXPECT_TRUE(
        ValidateYuanRongDirectIoPayload("key-a", buffer.data(), alignment, alignment).Success());
    EXPECT_TRUE(ValidateYuanRongDirectIoPayload("key-a", buffer.data() + 1, alignment, alignment)
                    .Failure());
    EXPECT_TRUE(ValidateYuanRongDirectIoPayload("key-a", buffer.data(), alignment - 1, alignment)
                    .Failure());
    EXPECT_TRUE(ValidateYuanRongDirectIoPayload("key-a", buffer.data(), alignment, 0).Failure());
}
