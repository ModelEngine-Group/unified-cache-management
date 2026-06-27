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
#include "fake_trans_provider.h"
#include <acl/acl.h>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>
#include "kv_protocol.h"
#include "logger.h"

namespace UC::ASU {
namespace {

constexpr std::uint16_t kCqeSuccess = 0x000;
constexpr std::uint16_t kCqeCheckResultBuffer = 0x732;
constexpr std::uint8_t kBatchEntryOk = 0x0;
constexpr std::uint8_t kBatchEntryKeyNotFound = 0x3;
constexpr std::uint8_t kDeleteEntryOk = 0x0;
constexpr std::uint8_t kDeleteEntryFailed = 0x1;
constexpr std::uint8_t kExistEntryNotExist = 0x0;
constexpr std::uint8_t kExistEntryExist = 0x1;

std::uint64_t ReadU64(std::uint32_t low, std::uint32_t high)
{
    return static_cast<std::uint64_t>(low) | (static_cast<std::uint64_t>(high) << 32);
}

std::uint32_t RequestCid(const std::uint32_t* request) { return request[0] >> 16; }

AsuId RequestAsuId(const std::uint32_t* request) { return request[1]; }

KvOpcode RequestOpcode(const std::uint32_t* request)
{
    return static_cast<KvOpcode>(request[0] & 0xFF);
}

CacheKey ReadKey(const std::uint32_t* data)
{
    CacheKey key{};
    std::memcpy(key.data(), data, kCacheKeySizeBytes);
    return key;
}

std::string KeyFileName(const CacheKey& key)
{
    std::uint64_t hash = 1469598103934665603ULL;
    for (auto byte : key) {
        const auto ch = std::to_integer<unsigned char>(byte);
        hash ^= ch;
        hash *= 1099511628211ULL;
    }

    std::ostringstream stream;
    stream << std::hex << std::setw(16) << std::setfill('0') << hash << ".bin";
    return stream.str();
}

std::filesystem::path AsuRoot(const FakeTransProviderConfig& config, AsuId asuId)
{
    return std::filesystem::path(config.storePath) / ("asu-" + std::to_string(asuId));
}

std::filesystem::path KeyPath(const FakeTransProviderConfig& config, AsuId asuId,
                              const CacheKey& key)
{
    return AsuRoot(config, asuId) / KeyFileName(key);
}

bool StoreBytes(const FakeTransProviderConfig& config, AsuId asuId, const CacheKey& key,
                std::uint64_t addr, std::uint32_t length)
{
    std::vector<char> buffer(length);
    auto ret = aclrtMemcpy(buffer.data(), buffer.size(), reinterpret_cast<const void*>(addr),
                           length, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        UC_ERROR(
            "ASU fake backend device-to-host copy failed asuId={} key={} addr={} length={} "
            "ret={}.",
            asuId, CacheKeyToHex(key), addr, length, ret);
        return false;
    }

    std::filesystem::create_directories(AsuRoot(config, asuId));
    std::ofstream output(KeyPath(config, asuId, key), std::ios::binary | std::ios::trunc);
    if (!output) {
        UC_ERROR("ASU fake backend failed to open store file asuId={} key={} path={}.", asuId,
                 CacheKeyToHex(key), KeyPath(config, asuId, key).string());
        return false;
    }
    output.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    return output.good();
}

bool LoadBytes(const FakeTransProviderConfig& config, AsuId asuId, const CacheKey& key,
               std::uint64_t addr, std::uint32_t length)
{
    std::ifstream input(KeyPath(config, asuId, key), std::ios::binary);
    if (!input) {
        UC_ERROR("ASU fake backend failed to open load file asuId={} key={} path={}.", asuId,
                 CacheKeyToHex(key), KeyPath(config, asuId, key).string());
        return false;
    }
    std::vector<char> buffer(length, 0);
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto readCount = input.gcount();
    if (readCount < static_cast<std::streamsize>(length)) {
        std::fill(buffer.begin() + readCount, buffer.end(), 0);
    }
    auto ret = aclrtMemcpy(reinterpret_cast<void*>(addr), length, buffer.data(), buffer.size(),
                           ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        UC_ERROR(
            "ASU fake backend host-to-device copy failed asuId={} key={} addr={} length={} "
            "ret={}.",
            asuId, CacheKeyToHex(key), addr, length, ret);
        return false;
    }
    return true;
}

bool DeleteKey(const FakeTransProviderConfig& config, AsuId asuId, const CacheKey& key)
{
    std::error_code errorCode;
    std::filesystem::remove(KeyPath(config, asuId, key), errorCode);
    return !errorCode;
}

bool ExistsKey(const FakeTransProviderConfig& config, AsuId asuId, const CacheKey& key)
{
    std::error_code errorCode;
    return std::filesystem::exists(KeyPath(config, asuId, key), errorCode);
}

void PackCqeHeader(std::uint32_t* flagBuffer, std::uint16_t cid, std::uint16_t status)
{
    flagBuffer[0] = 0;
    flagBuffer[1] = 0;
    flagBuffer[2] = 0;
    flagBuffer[3] = static_cast<std::uint32_t>(cid) | (static_cast<std::uint32_t>(status) << 17);
}

void PackResultBuffer4Bit(std::uint32_t* resultData, const std::vector<std::uint8_t>& results)
{
    const auto dwordCount = (results.size() + 7) / 8;
    std::fill(resultData, resultData + dwordCount, 0);
    for (std::size_t index = 0; index < results.size(); ++index) {
        resultData[index / 8] |= static_cast<std::uint32_t>(results[index] & 0xF)
                                 << ((index % 8) * 4);
    }
}

void PackResultBuffer1Bit(std::uint32_t* resultData, const std::vector<std::uint8_t>& results)
{
    const auto dwordCount = (results.size() + 31) / 32;
    std::fill(resultData, resultData + dwordCount, 0);
    for (std::size_t index = 0; index < results.size(); ++index) {
        resultData[index / 32] |= static_cast<std::uint32_t>(results[index] & 0x1) << (index % 32);
    }
}

struct BatchEntry {
    CacheKey key{};
    std::uint64_t bufferAddr{0};
    std::uint32_t length{0};
};

std::vector<BatchEntry> ReadBatchEntries(const std::uint32_t* request, std::uint16_t batchNumber)
{
    std::vector<BatchEntry> entries;
    entries.reserve(batchNumber);
    for (std::uint16_t index = 0; index < batchNumber; ++index) {
        const auto* entry = request + kSqeDwordCount + index * kBatchEntryDwordCount;
        BatchEntry parsed;
        parsed.key = ReadKey(entry + 1);
        parsed.bufferAddr = ReadU64(entry[5], entry[6]);
        parsed.length = entry[7] & 0xFFFFFF;
        entries.emplace_back(std::move(parsed));
    }
    return entries;
}

std::vector<CacheKey> ReadKeyEntries(const std::uint32_t* request, std::uint16_t batchNumber)
{
    std::vector<CacheKey> keys;
    keys.reserve(batchNumber);
    for (std::uint16_t index = 0; index < batchNumber; ++index) {
        const auto* entry = request + kSqeDwordCount + index * kKeyEntryDwordCount;
        keys.emplace_back(ReadKey(entry));
    }
    return keys;
}

Status CompleteBatchStore(const FakeTransProviderConfig& config, AsuId asuId,
                          const std::uint32_t* request)
{
    const auto cid = static_cast<std::uint16_t>(RequestCid(request));
    const auto responseBufferAddr = ReadU64(request[3], request[4]);
    const auto batchNumber = static_cast<std::uint16_t>(request[10] & 0xFFFF);
    auto* flagBuffer = reinterpret_cast<std::uint32_t*>(responseBufferAddr);
    std::vector<std::uint8_t> results(batchNumber, kBatchEntryOk);

    const auto entries = ReadBatchEntries(request, batchNumber);
    for (std::size_t index = 0; index < entries.size(); ++index) {
        const auto& entry = entries[index];
        if (!StoreBytes(config, asuId, entry.key, entry.bufferAddr, entry.length)) {
            results[index] = kBatchEntryKeyNotFound;
        }
    }

    const auto allOk = std::all_of(results.begin(), results.end(),
                                   [](std::uint8_t result) { return result == kBatchEntryOk; });
    const auto cqeStatus = allOk ? kCqeSuccess : kCqeCheckResultBuffer;
    PackCqeHeader(flagBuffer, cid, cqeStatus);
    if (!allOk) { PackResultBuffer4Bit(flagBuffer + kCqeDwordCount, results); }
    return Status::OK();
}

Status CompleteBatchRetrieve(const FakeTransProviderConfig& config, AsuId asuId,
                             const std::uint32_t* request)
{
    const auto cid = static_cast<std::uint16_t>(RequestCid(request));
    const auto responseBufferAddr = ReadU64(request[3], request[4]);
    const auto batchNumber = static_cast<std::uint16_t>(request[10] & 0xFFFF);
    auto* flagBuffer = reinterpret_cast<std::uint32_t*>(responseBufferAddr);
    std::vector<std::uint8_t> results(batchNumber, kBatchEntryOk);

    const auto entries = ReadBatchEntries(request, batchNumber);
    for (std::size_t index = 0; index < entries.size(); ++index) {
        const auto& entry = entries[index];
        if (!LoadBytes(config, asuId, entry.key, entry.bufferAddr, entry.length)) {
            results[index] = kBatchEntryKeyNotFound;
        }
    }

    const auto allOk = std::all_of(results.begin(), results.end(),
                                   [](std::uint8_t result) { return result == kBatchEntryOk; });
    const auto cqeStatus = allOk ? kCqeSuccess : kCqeCheckResultBuffer;
    PackCqeHeader(flagBuffer, cid, cqeStatus);
    if (!allOk) { PackResultBuffer4Bit(flagBuffer + kCqeDwordCount, results); }
    return Status::OK();
}

Status CompleteDelete(const FakeTransProviderConfig& config, AsuId asuId,
                      const std::uint32_t* request)
{
    const auto cid = static_cast<std::uint16_t>(RequestCid(request));
    const auto responseBufferAddr = ReadU64(request[3], request[4]);
    const auto batchNumber = static_cast<std::uint16_t>(request[10] & 0xFFFF);
    auto* flagBuffer = reinterpret_cast<std::uint32_t*>(responseBufferAddr);
    std::vector<std::uint8_t> results(batchNumber, kDeleteEntryOk);

    const auto keys = ReadKeyEntries(request, batchNumber);
    for (std::size_t index = 0; index < keys.size(); ++index) {
        if (!DeleteKey(config, asuId, keys[index])) { results[index] = kDeleteEntryFailed; }
    }

    const auto allOk = std::all_of(results.begin(), results.end(),
                                   [](std::uint8_t result) { return result == kDeleteEntryOk; });
    const auto cqeStatus = allOk ? kCqeSuccess : kCqeCheckResultBuffer;
    PackCqeHeader(flagBuffer, cid, cqeStatus);
    if (!allOk) { PackResultBuffer1Bit(flagBuffer + kCqeDwordCount, results); }
    return Status::OK();
}

Status CompleteExist(const FakeTransProviderConfig& config, AsuId asuId,
                     const std::uint32_t* request)
{
    const auto cid = static_cast<std::uint16_t>(RequestCid(request));
    const auto responseBufferAddr = ReadU64(request[3], request[4]);
    const auto batchNumber = static_cast<std::uint16_t>(request[10] & 0xFFFF);
    auto* flagBuffer = reinterpret_cast<std::uint32_t*>(responseBufferAddr);
    std::vector<std::uint8_t> results(batchNumber, kExistEntryNotExist);
    std::uint16_t existingKeyNumber = 0;

    const auto keys = ReadKeyEntries(request, batchNumber);
    for (std::size_t index = 0; index < keys.size(); ++index) {
        if (ExistsKey(config, asuId, keys[index])) {
            results[index] = kExistEntryExist;
            ++existingKeyNumber;
        }
    }

    const auto allExist = existingKeyNumber == batchNumber;
    const auto cqeStatus = allExist ? kCqeSuccess : kCqeCheckResultBuffer;
    PackCqeHeader(flagBuffer, cid, cqeStatus);
    flagBuffer[0] = existingKeyNumber;
    if (!allExist) { PackResultBuffer1Bit(flagBuffer + kCqeDwordCount, results); }
    return Status::OK();
}

Status CompleteFakeBackendRequest(const FakeTransProviderConfig& config, const void* sendBuffer,
                                  std::uint64_t len)
{
    if (config.latencyMs > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(config.latencyMs));
    }

    if (sendBuffer == nullptr || len < sizeof(std::uint32_t)) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "fake backend send buffer is empty");
    }

    const auto* request = reinterpret_cast<const std::uint32_t*>(sendBuffer);
    const auto asuId = RequestAsuId(request);
    switch (RequestOpcode(request)) {
        case KvOpcode::BatchStore: return CompleteBatchStore(config, asuId, request);
        case KvOpcode::BatchRetrieve: return CompleteBatchRetrieve(config, asuId, request);
        case KvOpcode::Delete: return CompleteDelete(config, asuId, request);
        case KvOpcode::Exist: return CompleteExist(config, asuId, request);
        case KvOpcode::KeepAlive: {
            auto* flagBuffer = reinterpret_cast<std::uint32_t*>(ReadU64(request[3], request[4]));
            PackCqeHeader(flagBuffer, static_cast<std::uint16_t>(RequestCid(request)), kCqeSuccess);
            return Status::OK();
        }
        default:
            return Status::Error(StatusCode::UNSUPPORTED,
                                 "fake backend only supports batch ASU operations");
    }
}

}  // namespace

FakeTransProviderConfig MakeFakeTransProviderConfig(const TransportConfig& config)
{
    FakeTransProviderConfig fakeConfig;
    fakeConfig.deviceId = config.endpoints.empty() ? 0 : config.endpoints.front().deviceId;
    auto pathIter = config.attrs.find("fake_backend.path");
    if (pathIter != config.attrs.end() && !pathIter->second.empty()) {
        fakeConfig.storePath = pathIter->second;
    }
    auto latencyIter = config.attrs.find("fake_backend.latency_ms");
    if (latencyIter != config.attrs.end()) {
        fakeConfig.latencyMs = static_cast<std::uint64_t>(std::stoull(latencyIter->second));
    }
    auto deviceIter = config.attrs.find("fake_backend.device_id");
    if (deviceIter != config.attrs.end()) {
        fakeConfig.deviceId = static_cast<std::int32_t>(std::stol(deviceIter->second));
    }
    return fakeConfig;
}

FakeTransProvider::FakeTransProvider(FakeTransProviderConfig config) : config_(std::move(config)) {}

Status FakeTransProvider::SetUpAclRuntime()
{
    if (aclReady_) { return Status::OK(); }
    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
        return Status::Error(StatusCode::INTERNAL_ERROR,
                             "ASU fake backend aclInit failed: " + std::to_string(ret));
    }

    const auto deviceId = config_.deviceId < 0 ? 0 : config_.deviceId;
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        return Status::Error(StatusCode::INTERNAL_ERROR,
                             "ASU fake backend aclrtSetDevice failed: device_id=" +
                                 std::to_string(deviceId) + " ret=" + std::to_string(ret));
    }
    aclReady_ = true;
    return Status::OK();
}

Status FakeTransProvider::CreateConnection(const std::string&, const std::string&, uint32_t,
                                           uint32_t qpNum, uint32_t,
                                           std::vector<ConnectionHandle>& handles)
{
    auto status = SetUpAclRuntime();
    if (!status.ok()) { return status; }
    handles.clear();
    handles.reserve(qpNum);
    for (uint32_t index = 0; index < qpNum; ++index) {
        handles.push_back(reinterpret_cast<ConnectionHandle>(static_cast<std::uintptr_t>(index) +
                                                             static_cast<std::uintptr_t>(1)));
    }
    return Status::OK();
}

std::vector<Status> FakeTransProvider::DeleteConnections(
    const std::vector<ConnectionHandle>& handles)
{
    return std::vector<Status>(handles.size(), Status::OK());
}

std::vector<Status> FakeTransProvider::Send(const std::vector<SendIoBatch>& ioBatches,
                                            uint32_t kernelCount, uint32_t quietCount)
{
    (void)kernelCount;
    (void)quietCount;
    std::vector<Status> statuses;
    statuses.reserve(ioBatches.size());
    for (const auto& ioBatch : ioBatches) {
        statuses.emplace_back(CompleteFakeBackendRequest(config_, ioBatch.sendBuffer, ioBatch.len));
    }
    return statuses;
}

Status FakeTransProvider::RegisterMemory(ConnectionHandle,
                                         const std::vector<RegisterMemoryDesc>& memoryDescs,
                                         std::vector<MemHandle>& memoryHandles)
{
    memoryHandles.clear();
    memoryHandles.reserve(memoryDescs.size());
    for (std::size_t index = 0; index < memoryDescs.size(); ++index) {
        memoryHandles.push_back(reinterpret_cast<MemHandle>(static_cast<std::uintptr_t>(index) +
                                                            static_cast<std::uintptr_t>(1)));
    }
    return Status::OK();
}

std::vector<Status> FakeTransProvider::UnregisterMemory(
    const std::vector<UnregisterMemoryDesc>& handles)
{
    return std::vector<Status>(handles.size(), Status::OK());
}

Status FakeTransProvider::AllocThread(uint32_t, const std::vector<uint32_t>&,
                                      std::vector<ThreadHandle>&)
{
    return Status::OK();
}

std::vector<Status> FakeTransProvider::FreeThread(const std::vector<ThreadHandle>& threads)
{
    return std::vector<Status>(threads.size(), Status::OK());
}

Status FakeTransProvider::GetMemTokenId(MemHandle, uint32_t& tokenId)
{
    tokenId = 1;
    return Status::OK();
}

}  // namespace UC::ASU
