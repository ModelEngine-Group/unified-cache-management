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
#ifndef UNIFIEDCACHE_YUANRONG_STORE_CC_YUANRONG_HELPER_H
#define UNIFIEDCACHE_YUANRONG_STORE_CC_YUANRONG_HELPER_H

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "datasystem/hetero/device_common.h"
#include "datasystem/hetero_client.h"
#include "datasystem/utils/status.h"
#include "status/status.h"
#include "trans_task.h"
#include "yuanrong_config.h"

namespace UC::YuanRongStore {

inline Status ResolveDeviceMemoryPreRegistration(Config& config, const char* linkTypeEnv)
{
    config.enableDeviceMemoryPreRegistration = false;
    if (!config.enableRemoteH2D) { return Status::OK(); }

    const std::string linkType =
        linkTypeEnv == nullptr || *linkTypeEnv == '\0' ? "ROCE" : linkTypeEnv;
    if (linkType != "ROCE" && linkType != "HCCS") {
        return Status::InvalidParam("DS_RH2D_LINK_TYPE must be ROCE or HCCS");
    }

    config.enableDeviceMemoryPreRegistration = linkType == "HCCS";
    return Status::OK();
}

inline std::string BlockIdToHex(const Detail::BlockId& block)
{
    constexpr char hex[] = "0123456789abcdef";
    std::string result(block.size() * 2, '0');
    for (size_t i = 0; i < block.size(); ++i) {
        auto value = static_cast<uint8_t>(block[i]);
        result[i * 2] = hex[value >> 4];
        result[i * 2 + 1] = hex[value & 0x0f];
    }
    return result;
}

inline std::string MakeKey(const Config& config, const Detail::Shard& shard)
{
    return "ucm_" + config.nameSpace + "_" + BlockIdToHex(shard.owner) + "_" +
           std::to_string(shard.index);
}

inline std::string MakeLookupKey(const Config& config, const Detail::BlockId& block)
{
    Detail::Shard shard{block, 0, {}};
    return MakeKey(config, shard);
}

inline Status BuildKeysAndBlobs(const Config& config, const Detail::TaskDesc& desc,
                                std::vector<std::string>& keys,
                                std::vector<datasystem::DeviceBlobList>& blobLists)
{
    keys.clear();
    blobLists.clear();
    keys.reserve(desc.size());
    blobLists.reserve(desc.size());
    for (const auto& shard : desc) {
        if (shard.addrs.size() != config.tensorSizes.size()) {
            return Status::InvalidParam("address count({}) does not match tensor count({})",
                                        shard.addrs.size(), config.tensorSizes.size());
        }
        datasystem::DeviceBlobList blobList;
        blobList.deviceIdx = config.deviceId;
        blobList.srcOffset = 0;
        blobList.blobs.reserve(shard.addrs.size());
        for (size_t i = 0; i < shard.addrs.size(); ++i) {
            if (shard.addrs[i] == nullptr) {
                return Status::InvalidParam("null device address at tensor({})", i);
            }
            blobList.blobs.push_back({shard.addrs[i], config.tensorSizes[i]});
        }
        keys.push_back(MakeKey(config, shard));
        blobLists.push_back(std::move(blobList));
    }
    return Status::OK();
}

inline void DeduplicateYuanRongObjects(std::vector<std::string>& keys,
                                       std::vector<datasystem::DeviceBlobList>& blobLists,
                                       Detail::TaskDesc* desc = nullptr)
{
    std::unordered_set<std::string> seen;
    seen.reserve(keys.size());
    std::vector<std::string> uniqueKeys;
    std::vector<datasystem::DeviceBlobList> uniqueBlobLists;
    Detail::TaskDesc uniqueDesc;
    uniqueKeys.reserve(keys.size());
    uniqueBlobLists.reserve(blobLists.size());
    if (desc != nullptr) { uniqueDesc.reserve(desc->size()); }

    for (size_t i = 0; i < keys.size(); ++i) {
        if (!seen.insert(keys[i]).second) { continue; }
        uniqueKeys.push_back(std::move(keys[i]));
        uniqueBlobLists.push_back(std::move(blobLists[i]));
        if (desc != nullptr) { uniqueDesc.push_back(std::move((*desc)[i])); }
    }
    keys = std::move(uniqueKeys);
    blobLists = std::move(uniqueBlobLists);
    if (desc != nullptr) { *desc = std::move(uniqueDesc); }
}

inline size_t DeriveYuanRongHostBufferCount(size_t objectSize, size_t recoveryBatchSize,
                                            size_t loadWorkerCount, size_t backfillWorkerCount,
                                            uint64_t capacityBytes)
{
    if (objectSize == 0 || recoveryBatchSize == 0 || loadWorkerCount == 0 ||
        backfillWorkerCount == 0 || capacityBytes == 0) {
        return 0;
    }

    const auto maxValue = std::numeric_limits<uint64_t>::max();
    const auto loadWorkers = static_cast<uint64_t>(loadWorkerCount);
    const auto backfillWorkers = static_cast<uint64_t>(backfillWorkerCount);
    const auto targetBatchCount = loadWorkers > (maxValue - backfillWorkers) / 2
                                      ? maxValue
                                      : loadWorkers * 2 + backfillWorkers;
    const auto buffersByCapacity = capacityBytes / static_cast<uint64_t>(objectSize);
    const auto batchesByCapacity = buffersByCapacity / static_cast<uint64_t>(recoveryBatchSize);
    const auto selectedBatchCount = std::min(targetBatchCount, batchesByCapacity);
    const auto selectedCount = selectedBatchCount * static_cast<uint64_t>(recoveryBatchSize);
    if (selectedCount > std::numeric_limits<size_t>::max()) { return 0; }
    return static_cast<size_t>(selectedCount);
}

inline size_t DeriveYuanRongPosixDumpBatchSize(size_t objectSize, uint64_t maxInflightBytes)
{
    constexpr uint64_t targetBatchBytes = 256ULL << 20;
    constexpr size_t maxBatchKeys = 32;
    constexpr size_t targetInflightBatches = 4;
    if (objectSize == 0 || maxInflightBytes < objectSize) { return 0; }

    const auto bytesPerBatch = std::min(targetBatchBytes, maxInflightBytes / targetInflightBatches);
    const auto keys = bytesPerBatch / static_cast<uint64_t>(objectSize);
    return std::clamp<size_t>(static_cast<size_t>(keys), 1, maxBatchKeys);
}

inline Status FilterKeysByLocalSetKeys(const std::vector<std::string>& selectedKeys,
                                       std::vector<std::string>& keys, Detail::TaskDesc& desc)
{
    if (desc.size() != keys.size()) {
        return Status::InvalidParam("YuanRong object selection size mismatch: keys({}), shards({})",
                                    keys.size(), desc.size());
    }

    std::unordered_set<std::string> inputKeys(keys.begin(), keys.end());
    std::unordered_set<std::string> selected;
    selected.reserve(selectedKeys.size());
    for (const auto& key : selectedKeys) {
        if (inputKeys.count(key) == 0) {
            return Status::InvalidParam("YuanRong MSetD2H returned an unknown local key({})", key);
        }
        if (!selected.insert(key).second) {
            return Status::InvalidParam("YuanRong MSetD2H returned a duplicated local key({})",
                                        key);
        }
    }

    std::vector<std::string> filteredKeys;
    Detail::TaskDesc filteredDesc;
    filteredKeys.reserve(selectedKeys.size());
    filteredDesc.reserve(selectedKeys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (selected.count(keys[i]) == 0) { continue; }
        filteredKeys.push_back(std::move(keys[i]));
        filteredDesc.push_back(std::move(desc[i]));
    }
    keys = std::move(filteredKeys);
    desc = std::move(filteredDesc);
    return Status::OK();
}

inline std::vector<size_t> FilterH2dFailedIndexes(const std::vector<std::string>& keys,
                                                  const std::vector<std::string>& failedKeys,
                                                  bool requestFailed)
{
    if (failedKeys.empty()) {
        if (!requestFailed) { return {}; }
        std::vector<size_t> all(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) { all[i] = i; }
        return all;
    }
    std::unordered_set<std::string> failed(failedKeys.begin(), failedKeys.end());
    std::vector<size_t> result;
    result.reserve(failed.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (failed.count(keys[i]) != 0) { result.push_back(i); }
    }
    return result;
}

inline void SplitByExistence(const std::vector<bool>& exists, std::vector<size_t>& hitIndexes,
                             std::vector<size_t>& missIndexes)
{
    hitIndexes.clear();
    missIndexes.clear();
    hitIndexes.reserve(exists.size());
    missIndexes.reserve(exists.size());
    for (size_t i = 0; i < exists.size(); ++i) {
        (exists[i] ? hitIndexes : missIndexes).push_back(i);
    }
}

inline Status ResolveTieredPrefixHit(size_t totalCount, const std::vector<size_t>& missIndexes,
                                     ssize_t backendHit, ssize_t& result)
{
    if (backendHit < -1 || static_cast<size_t>(backendHit + 1) > missIndexes.size()) {
        return Status::Error("backend LookupOnPrefix returned an unexpected result index");
    }
    if (static_cast<size_t>(backendHit + 1) == missIndexes.size()) {
        result = static_cast<ssize_t>(totalCount) - 1;
    } else {
        result = static_cast<ssize_t>(missIndexes[backendHit + 1]) - 1;
    }
    return Status::OK();
}

inline Status ResolveTieredReverseHit(size_t totalCount, ssize_t yuanRongHit, ssize_t backendHit,
                                      ssize_t& result)
{
    if (yuanRongHit < -1 || (yuanRongHit >= 0 && static_cast<size_t>(yuanRongHit) >= totalCount)) {
        return Status::Error("YuanRong LookupOnReverse returned an unexpected result index");
    }
    const auto backendStart = static_cast<size_t>(yuanRongHit + 1);
    const auto backendCount = totalCount - backendStart;
    if (backendHit < -1 || (backendHit >= 0 && static_cast<size_t>(backendHit) >= backendCount)) {
        return Status::Error("backend LookupOnReverse returned an unexpected result index");
    }
    result = backendHit < 0 ? yuanRongHit : static_cast<ssize_t>(backendStart) + backendHit;
    return Status::OK();
}

inline std::vector<std::pair<size_t, size_t>> RecoveryBatchRanges(size_t count, size_t batchSize)
{
    std::vector<std::pair<size_t, size_t>> ranges;
    if (count == 0 || batchSize == 0) { return ranges; }
    ranges.reserve(count / batchSize + (count % batchSize != 0));
    for (size_t begin = 0; begin < count; begin += batchSize) {
        ranges.emplace_back(begin, std::min(count, begin + batchSize));
    }
    return ranges;
}

inline Status FromYuanRongStatus(const datasystem::Status& status)
{
    return status.IsOk() ? Status::OK() : Status::Error(status.ToString());
}

inline bool YuanRongBufferHasEnoughPayload(int64_t bufferSize, size_t objectSize)
{
    return bufferSize >= 0 && static_cast<size_t>(bufferSize) >= objectSize;
}

inline size_t YuanRongHeaderSize(size_t blobCount, size_t memoryAlignment)
{
    auto size = sizeof(uint64_t) * (blobCount + 2);
    return (size + memoryAlignment - 1) / memoryAlignment * memoryAlignment;
}

inline size_t YuanRongComposedObjectSize(const std::vector<size_t>& tensorSizes,
                                         size_t memoryAlignment)
{
    size_t size = YuanRongHeaderSize(tensorSizes.size(), memoryAlignment);
    for (auto tensorSize : tensorSizes) { size += tensorSize; }
    return size;
}

inline Status ValidateYuanRongBlobSizes(const std::string& key,
                                        const datasystem::MetaInfo& metaInfo,
                                        const std::vector<size_t>& tensorSizes)
{
    if (metaInfo.blobSizeList.size() != tensorSizes.size()) {
        return Status::Error(
            fmt::format("YuanRong blob count({}) does not match tensor count({}) "
                        "for key({})",
                        metaInfo.blobSizeList.size(), tensorSizes.size(), key));
    }
    for (size_t i = 0; i < tensorSizes.size(); ++i) {
        if (metaInfo.blobSizeList[i] != tensorSizes[i]) {
            return Status::Error(
                fmt::format("YuanRong blob size({}) does not match tensor "
                            "size({}) for key({}) at blob({})",
                            metaInfo.blobSizeList[i], tensorSizes[i], key, i));
        }
    }
    return Status::OK();
}

inline Status GetYuanRongPayloadAddress(const std::string& key, const void* address,
                                        int64_t bufferSize, const datasystem::MetaInfo& metaInfo,
                                        const std::vector<size_t>& tensorSizes,
                                        size_t memoryAlignment, const void*& payloadAddress)
{
    payloadAddress = nullptr;
    auto metaCheck = ValidateYuanRongBlobSizes(key, metaInfo, tensorSizes);
    if (metaCheck.Failure()) { return metaCheck; }
    if (address == nullptr) {
        return Status::Error(fmt::format("YuanRong buffer has no address for key({})", key));
    }
    if (bufferSize < 0) {
        return Status::Error(
            fmt::format("YuanRong buffer size({}) is invalid for key({})", bufferSize, key));
    }
    const auto size = static_cast<size_t>(bufferSize);
    const auto count = tensorSizes.size();
    const auto headerSize = YuanRongHeaderSize(count, memoryAlignment);
    const auto composedSize = YuanRongComposedObjectSize(tensorSizes, memoryAlignment);
    if (size < composedSize) {
        return Status::Error(
            fmt::format("YuanRong buffer size({}) is smaller than composed "
                        "object size({}) for key({})",
                        size, composedSize, key));
    }

    const auto* offsets = reinterpret_cast<const uint64_t*>(address);
    if (offsets[0] != count) {
        return Status::Error(
            fmt::format("YuanRong blob count({}) does not match tensor count({}) "
                        "in buffer for key({})",
                        offsets[0], count, key));
    }
    if (offsets[1] != headerSize) {
        return Status::Error(
            fmt::format("YuanRong payload offset({}) does not match expected "
                        "offset({}) for key({})",
                        offsets[1], headerSize, key));
    }
    for (size_t i = 0; i < count; ++i) {
        const auto expected = offsets[i + 1] + tensorSizes[i];
        if (offsets[i + 2] != expected) {
            return Status::Error(
                fmt::format("YuanRong payload offset({}) does not match "
                            "expected offset({}) for key({}) at blob({})",
                            offsets[i + 2], expected, key, i));
        }
    }
    if (offsets[count + 1] != composedSize) {
        return Status::Error(
            fmt::format("YuanRong composed size({}) does not match expected "
                        "size({}) for key({})",
                        offsets[count + 1], composedSize, key));
    }
    payloadAddress = static_cast<const uint8_t*>(address) + offsets[1];
    return Status::OK();
}

inline Status InitYuanRongComposedBuffer(const std::string& key, void* address, int64_t bufferSize,
                                         const std::vector<size_t>& tensorSizes,
                                         size_t memoryAlignment, void*& payloadAddress)
{
    payloadAddress = nullptr;
    if (address == nullptr) {
        return Status::Error(
            fmt::format("YuanRong buffer has no mutable address for key({})", key));
    }
    if (bufferSize < 0) {
        return Status::Error(
            fmt::format("YuanRong buffer size({}) is invalid for key({})", bufferSize, key));
    }
    const auto size = static_cast<size_t>(bufferSize);
    const auto count = tensorSizes.size();
    const auto headerSize = YuanRongHeaderSize(count, memoryAlignment);
    const auto composedSize = YuanRongComposedObjectSize(tensorSizes, memoryAlignment);
    if (size < composedSize) {
        return Status::Error(
            fmt::format("YuanRong buffer size({}) is smaller than composed "
                        "object size({}) for key({})",
                        size, composedSize, key));
    }

    auto* offsets = reinterpret_cast<uint64_t*>(address);
    offsets[0] = count;
    offsets[1] = headerSize;
    for (size_t i = 0; i < count; ++i) { offsets[i + 2] = offsets[i + 1] + tensorSizes[i]; }
    payloadAddress = static_cast<uint8_t*>(address) + headerSize;
    return Status::OK();
}

inline Status ValidateYuanRongDirectIoPayload(const std::string& key, const void* payloadAddress,
                                              size_t payloadSize, size_t memoryAlignment)
{
    if (memoryAlignment == 0) { return Status::InvalidParam("invalid YuanRong memory alignment"); }
    if (payloadAddress == nullptr ||
        reinterpret_cast<uintptr_t>(payloadAddress) % memoryAlignment != 0) {
        return Status::Error(
            fmt::format("YuanRong payload address is not aligned to {} bytes for key({})",
                        memoryAlignment, key));
    }
    if (payloadSize % memoryAlignment != 0) {
        return Status::Error(
            fmt::format("YuanRong payload size({}) is not aligned to {} bytes for key({})",
                        payloadSize, memoryAlignment, key));
    }
    return Status::OK();
}

}  // namespace UC::YuanRongStore

#endif
