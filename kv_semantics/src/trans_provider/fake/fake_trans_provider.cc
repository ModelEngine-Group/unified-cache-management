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
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#include "proto/kv_protocol.h"
#include "logger.h"

namespace UC::ASU {
namespace {

constexpr std::uint16_t kCqeSuccess = 0x000;
constexpr std::uint16_t kCqeInternalError = 0x006;
constexpr std::uint16_t kCqeCheckResultBuffer = 0x732;
constexpr std::uint8_t kBatchEntryOk = 0x0;
constexpr std::uint8_t kBatchEntryKeyNotFound = 0x3;
constexpr std::uint8_t kDeleteEntryOk = 0x0;
constexpr std::uint8_t kDeleteEntryFailed = 0x1;
constexpr std::uint8_t kExistEntryNotExist = 0x0;
constexpr std::uint8_t kExistEntryExist = 0x1;
constexpr std::uint32_t kExistSeekControlMask = 1U << 16;
constexpr std::size_t kKeyLockCount = 256;

std::array<std::shared_mutex, kKeyLockCount> g_keyMutexes;

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

std::uint64_t KeyHash(const CacheKey& key)
{
    std::uint64_t hash = 1469598103934665603ULL;
    for (auto byte : key) {
        const auto ch = std::to_integer<unsigned char>(byte);
        hash ^= ch;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string KeyFileName(const CacheKey& key)
{
    std::ostringstream stream;
    stream << std::hex << std::setw(16) << std::setfill('0') << KeyHash(key) << ".bin";
    return stream.str();
}

std::shared_mutex& KeyMutex(const CacheKey& key)
{
    return g_keyMutexes[KeyHash(key) % kKeyLockCount];
}

std::filesystem::path AsuRoot(const std::string& storePath, AsuId asuId)
{
    return std::filesystem::path(storePath) / ("asu-" + std::to_string(asuId));
}

std::filesystem::path KeyPath(const std::string& storePath, AsuId asuId, const CacheKey& key)
{
    return AsuRoot(storePath, asuId) / KeyFileName(key);
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
    std::uint32_t offset{0};
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
        parsed.offset = entry[0];
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

std::size_t CompletionDwordCount(const std::uint32_t* request)
{
    const auto opcode = RequestOpcode(request);
    if (opcode == KvOpcode::KeepAlive) { return kCqeDwordCount; }
    if (opcode == KvOpcode::Store || opcode == KvOpcode::Retrieve) { return kCqeDwordCount; }

    const auto batchNumber = static_cast<std::uint16_t>(request[10] & 0xFFFF);
    const auto resultDwordCount =
        opcode == KvOpcode::BatchStore || opcode == KvOpcode::BatchRetrieve
            ? (static_cast<std::size_t>(batchNumber) + 7) / 8
            : (static_cast<std::size_t>(batchNumber) + 31) / 32;
    return kCqeDwordCount + resultDwordCount;
}

std::size_t RequestDwordCount(const std::uint32_t* request)
{
    const auto opcode = RequestOpcode(request);
    if (opcode == KvOpcode::Store || opcode == KvOpcode::Retrieve ||
        opcode == KvOpcode::KeepAlive) {
        return kSqeDwordCount;
    }

    const auto batchNumber = static_cast<std::uint16_t>(request[10] & 0xFFFF);
    const auto entryDwordCount = opcode == KvOpcode::BatchStore || opcode == KvOpcode::BatchRetrieve
                                     ? kBatchEntryDwordCount
                                     : kKeyEntryDwordCount;
    return kSqeDwordCount + static_cast<std::size_t>(batchNumber) * entryDwordCount;
}

void PublishCompletion(std::uint32_t* flagBuffer, const std::vector<std::uint32_t>& completion)
{
    std::memcpy(flagBuffer, completion.data(), 3 * sizeof(std::uint32_t));
    if (completion.size() > kCqeDwordCount) {
        std::memcpy(flagBuffer + kCqeDwordCount, completion.data() + kCqeDwordCount,
                    (completion.size() - kCqeDwordCount) * sizeof(std::uint32_t));
    }
    __atomic_store_n(flagBuffer + 3, completion[3], __ATOMIC_RELEASE);
}

}  // namespace

class FakeTransProvider::WorkerPool {
public:
    WorkerPool(FakeTransProvider& provider, std::size_t workerCount) : provider_(provider)
    {
        if (workerCount == 0) { throw std::invalid_argument("fake backend worker count is zero"); }
        workers_.reserve(workerCount);
        try {
            for (std::size_t index = 0; index < workerCount; ++index) {
                workers_.emplace_back(&WorkerPool::WorkerLoop, this);
            }
        } catch (...) {
            StopAndJoin();
            throw;
        }
    }

    ~WorkerPool() { StopAndJoin(); }

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    bool Push(IoTask task)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) { return false; }
            tasks_.emplace_back(std::move(task));
            ++pendingTaskCount_;
        }
        workReady_.notify_one();
        return true;
    }

    void WaitUntilIdle()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        idle_.wait(lock, [this] { return pendingTaskCount_ == 0; });
    }

private:
    void WorkerLoop()
    {
        while (true) {
            IoTask task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                workReady_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
                if (stopping_ && tasks_.empty()) { return; }
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }

            provider_.ProcessIoTask(task);

            {
                std::lock_guard<std::mutex> lock(mutex_);
                --pendingTaskCount_;
                if (pendingTaskCount_ == 0) { idle_.notify_all(); }
            }
        }
    }

    void StopAndJoin()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        workReady_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) { worker.join(); }
        }
        workers_.clear();
    }

    FakeTransProvider& provider_;
    std::mutex mutex_;
    std::condition_variable workReady_;
    std::condition_variable idle_;
    std::deque<IoTask> tasks_;
    std::vector<std::thread> workers_;
    std::size_t pendingTaskCount_{0};
    bool stopping_{false};
};

FakeTransProviderConfig MakeFakeTransProviderConfig(const TransportConfig& config)
{
    FakeTransProviderConfig fakeConfig;
    fakeConfig.deviceId = config.deviceId < 0 ? 0 : config.deviceId;
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
    auto workerIter = config.attrs.find("fake_backend.worker_threads");
    if (workerIter != config.attrs.end()) {
        fakeConfig.workerThreads = static_cast<std::size_t>(std::stoull(workerIter->second));
    }
    return fakeConfig;
}

FakeTransProvider::FakeTransProvider(FakeTransProviderConfig config)
    : config_(std::move(config)),
      workerPool_(std::make_unique<WorkerPool>(*this, config_.workerThreads))
{
    if (SetupDeviceRuntime().ok()) { stream_ = device_.MakeStream(); }
}

FakeTransProvider::~FakeTransProvider() = default;

Status FakeTransProvider::SetupDeviceRuntime()
{
    const auto deviceId = config_.deviceId < 0 ? 0 : config_.deviceId;
    thread_local std::int32_t readyDeviceId = -1;
    if (readyDeviceId == deviceId) { return Status::OK(); }

    const auto initStatus = device_.Init();
    if (initStatus.Failure() && initStatus != UC::Status::DuplicateKey()) {
        return Status::Error(StatusCode::INTERNAL_ERROR,
                             "ASU fake backend Device::Init failed: " + initStatus.ToString());
    }

    const auto setupStatus = device_.Setup(deviceId);
    if (setupStatus.Failure()) {
        return Status::Error(StatusCode::INTERNAL_ERROR,
                             "ASU fake backend Device::Setup failed: device_id=" +
                                 std::to_string(deviceId) + " message=" + setupStatus.ToString());
    }
    readyDeviceId = deviceId;
    return Status::OK();
}

Status FakeTransProvider::ResolveLocalAddress(const void* providerAddr, std::size_t size,
                                              void*& localAddr)
{
    const auto address = reinterpret_cast<std::uintptr_t>(providerAddr);
    std::lock_guard<std::mutex> lock(registeredMemoryMu_);
    for (const auto& item : registeredMemories_) {
        const auto& memory = item.second;
        if (address < memory.providerAddr) { continue; }
        const auto offset = address - memory.providerAddr;
        if (offset > memory.size || size > memory.size - offset) { continue; }
        localAddr = reinterpret_cast<void*>(memory.localAddr + offset);
        return Status::OK();
    }
    localAddr = nullptr;
    return Status::Error(StatusCode::BUFFER_NOT_REGISTERED,
                         "fake backend IO buffer is not registered");
}

bool FakeTransProvider::StoreBytes(AsuId asuId, const CacheKey& key, std::uint32_t offset,
                                   std::uint64_t addr, std::uint32_t length)
{
    std::unique_lock<std::shared_mutex> keyLock(KeyMutex(key));
    if (stream_ == nullptr) {
        UC_ERROR("ASU fake backend stream not initialized asuId={} key={} addr={} length={}.",
                 asuId, CacheKeyToHex(key), addr, length);
        return false;
    }
    std::vector<char> buffer(length);
    const auto copyStatus =
        stream_->DeviceToHost(reinterpret_cast<void*>(addr), buffer.data(), length);
    if (copyStatus.Failure()) {
        UC_ERROR(
            "ASU fake backend device-to-host copy failed asuId={} key={} addr={} length={} "
            "message={}.",
            asuId, CacheKeyToHex(key), addr, length, copyStatus.ToString());
        return false;
    }

    std::filesystem::create_directories(AsuRoot(config_.storePath, asuId));
    const auto path = KeyPath(config_.storePath, asuId, key);
    std::fstream output(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!output) {
        std::ofstream create(path, std::ios::binary);
        create.close();
        output.open(path, std::ios::binary | std::ios::in | std::ios::out);
    }
    if (!output) {
        UC_ERROR("ASU fake backend failed to open store file asuId={} key={} path={}.", asuId,
                 CacheKeyToHex(key), path.string());
        return false;
    }
    output.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!output) {
        UC_ERROR("ASU fake backend failed to seek store file asuId={} key={} path={} offset={}.",
                 asuId, CacheKeyToHex(key), path.string(), offset);
        return false;
    }
    output.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    return output.good();
}

bool FakeTransProvider::LoadBytes(AsuId asuId, const CacheKey& key, std::uint32_t offset,
                                  std::uint64_t addr, std::uint32_t length)
{
    std::shared_lock<std::shared_mutex> keyLock(KeyMutex(key));
    if (stream_ == nullptr) {
        UC_ERROR("ASU fake backend stream not initialized asuId={} key={} addr={} length={}.",
                 asuId, CacheKeyToHex(key), addr, length);
        return false;
    }
    const auto path = KeyPath(config_.storePath, asuId, key);
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        UC_ERROR("ASU fake backend failed to open load file asuId={} key={} path={}.", asuId,
                 CacheKeyToHex(key), path.string());
        return false;
    }
    std::vector<char> buffer(length, 0);
    input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto readCount = input.gcount();
        if (readCount < static_cast<std::streamsize>(length)) {
            std::fill(buffer.begin() + readCount, buffer.end(), 0);
        }
    }
    const auto copyStatus =
        stream_->HostToDevice(buffer.data(), reinterpret_cast<void*>(addr), length);
    if (copyStatus.Failure()) {
        UC_ERROR(
            "ASU fake backend host-to-device copy failed asuId={} key={} addr={} length={} "
            "message={}.",
            asuId, CacheKeyToHex(key), addr, length, copyStatus.ToString());
        return false;
    }
    return true;
}

bool FakeTransProvider::DeleteKey(AsuId asuId, const CacheKey& key)
{
    std::unique_lock<std::shared_mutex> keyLock(KeyMutex(key));
    std::error_code errorCode;
    std::filesystem::remove(KeyPath(config_.storePath, asuId, key), errorCode);
    return !errorCode;
}

bool FakeTransProvider::ExistsKey(AsuId asuId, const CacheKey& key)
{
    std::shared_lock<std::shared_mutex> keyLock(KeyMutex(key));
    std::error_code errorCode;
    return std::filesystem::exists(KeyPath(config_.storePath, asuId, key), errorCode);
}

Status FakeTransProvider::CompleteStore(AsuId asuId, const std::uint32_t* request,
                                        std::uint32_t* flagBuffer)
{
    const auto cid = static_cast<std::uint16_t>(RequestCid(request));
    const auto bufferAddr = ReadU64(request[6], request[7]);
    const auto bufferLength = request[8] & 0xFFFFFF;
    const auto offset = request[10];
    const auto key = ReadKey(request + 12);
    const auto status = StoreBytes(asuId, key, offset, bufferAddr, bufferLength)
                            ? kCqeSuccess
                            : kCqeCheckResultBuffer;
    PackCqeHeader(flagBuffer, cid, status);
    return Status::OK();
}

Status FakeTransProvider::CompleteRetrieve(AsuId asuId, const std::uint32_t* request,
                                           std::uint32_t* flagBuffer)
{
    const auto cid = static_cast<std::uint16_t>(RequestCid(request));
    const auto bufferAddr = ReadU64(request[6], request[7]);
    const auto bufferLength = request[8] & 0xFFFFFF;
    const auto offset = request[10];
    const auto key = ReadKey(request + 12);
    const auto status = LoadBytes(asuId, key, offset, bufferAddr, bufferLength)
                            ? kCqeSuccess
                            : kCqeCheckResultBuffer;
    PackCqeHeader(flagBuffer, cid, status);
    return Status::OK();
}

Status FakeTransProvider::CompleteBatchStore(AsuId asuId, const std::uint32_t* request,
                                             std::uint32_t* flagBuffer)
{
    const auto cid = static_cast<std::uint16_t>(RequestCid(request));
    const auto batchNumber = static_cast<std::uint16_t>(request[10] & 0xFFFF);
    std::vector<std::uint8_t> results(batchNumber, kBatchEntryOk);

    const auto entries = ReadBatchEntries(request, batchNumber);
    for (std::size_t index = 0; index < entries.size(); ++index) {
        const auto& entry = entries[index];
        if (!StoreBytes(asuId, entry.key, entry.offset, entry.bufferAddr, entry.length)) {
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

Status FakeTransProvider::CompleteBatchRetrieve(AsuId asuId, const std::uint32_t* request,
                                                std::uint32_t* flagBuffer)
{
    const auto cid = static_cast<std::uint16_t>(RequestCid(request));
    const auto batchNumber = static_cast<std::uint16_t>(request[10] & 0xFFFF);
    std::vector<std::uint8_t> results(batchNumber, kBatchEntryOk);

    const auto entries = ReadBatchEntries(request, batchNumber);
    for (std::size_t index = 0; index < entries.size(); ++index) {
        const auto& entry = entries[index];
        if (!LoadBytes(asuId, entry.key, entry.offset, entry.bufferAddr, entry.length)) {
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

Status FakeTransProvider::CompleteDelete(AsuId asuId, const std::uint32_t* request,
                                         std::uint32_t* flagBuffer)
{
    const auto cid = static_cast<std::uint16_t>(RequestCid(request));
    const auto batchNumber = static_cast<std::uint16_t>(request[10] & 0xFFFF);
    std::vector<std::uint8_t> results(batchNumber, kDeleteEntryOk);

    const auto keys = ReadKeyEntries(request, batchNumber);
    for (std::size_t index = 0; index < keys.size(); ++index) {
        if (!DeleteKey(asuId, keys[index])) { results[index] = kDeleteEntryFailed; }
    }

    const auto allOk = std::all_of(results.begin(), results.end(),
                                   [](std::uint8_t result) { return result == kDeleteEntryOk; });
    const auto cqeStatus = allOk ? kCqeSuccess : kCqeCheckResultBuffer;
    PackCqeHeader(flagBuffer, cid, cqeStatus);
    if (!allOk) { PackResultBuffer1Bit(flagBuffer + kCqeDwordCount, results); }
    return Status::OK();
}

Status FakeTransProvider::CompleteExist(AsuId asuId, const std::uint32_t* request,
                                        std::uint32_t* flagBuffer)
{
    const auto cid = static_cast<std::uint16_t>(RequestCid(request));
    const auto batchNumber = static_cast<std::uint16_t>(request[10] & 0xFFFF);
    std::vector<std::uint8_t> results(batchNumber, kExistEntryNotExist);
    std::uint16_t existingKeyNumber = 0;
    const bool useSeekControl = (request[10] & kExistSeekControlMask) != 0;

    const auto keys = ReadKeyEntries(request, batchNumber);
    for (std::size_t index = 0; index < keys.size(); ++index) {
        if (ExistsKey(asuId, keys[index])) {
            results[index] = kExistEntryExist;
            ++existingKeyNumber;
        } else if (!useSeekControl) {
            break;
        }
    }

    const auto allExist = existingKeyNumber == batchNumber;
    const auto cqeStatus = allExist ? kCqeSuccess : kCqeCheckResultBuffer;
    PackCqeHeader(flagBuffer, cid, cqeStatus);
    flagBuffer[0] = existingKeyNumber;
    if (!allExist) { PackResultBuffer1Bit(flagBuffer + kCqeDwordCount, results); }
    return Status::OK();
}

Status FakeTransProvider::CompleteFakeBackendRequest(const void* sendBuffer, std::uint64_t len,
                                                     std::vector<std::uint32_t>& completion)
{
    if (config_.latencyMs > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(config_.latencyMs));
    }

    if (sendBuffer == nullptr || len < kSqeDwordCount * sizeof(std::uint32_t)) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "fake backend send buffer is empty");
    }

    const auto* request = reinterpret_cast<const std::uint32_t*>(sendBuffer);
    completion.assign(CompletionDwordCount(request), 0);
    auto* flagBuffer = completion.data();
    const auto asuId = RequestAsuId(request);
    switch (RequestOpcode(request)) {
        case KvOpcode::Store: return CompleteStore(asuId, request, flagBuffer);
        case KvOpcode::Retrieve: return CompleteRetrieve(asuId, request, flagBuffer);
        case KvOpcode::BatchStore: return CompleteBatchStore(asuId, request, flagBuffer);
        case KvOpcode::BatchRetrieve: return CompleteBatchRetrieve(asuId, request, flagBuffer);
        case KvOpcode::Delete: return CompleteDelete(asuId, request, flagBuffer);
        case KvOpcode::Exist: return CompleteExist(asuId, request, flagBuffer);
        case KvOpcode::KeepAlive: {
            PackCqeHeader(flagBuffer, static_cast<std::uint16_t>(RequestCid(request)), kCqeSuccess);
            return Status::OK();
        }
        default:
            return Status::Error(StatusCode::UNSUPPORTED,
                                 "fake backend does not support this ASU operation");
    }
}

Status FakeTransProvider::CreateConnection(const std::string&, const std::string&, uint32_t,
                                           uint32_t qpNum, uint32_t,
                                           std::vector<ConnectionHandle>& handles)
{
    auto status = SetupDeviceRuntime();
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
        if (ioBatch.sendBuffer == nullptr || ioBatch.flagBuffer == nullptr ||
            ioBatch.len < kSqeDwordCount * sizeof(std::uint32_t) ||
            ioBatch.len > std::numeric_limits<std::size_t>::max()) {
            statuses.emplace_back(
                Status::Error(StatusCode::INVALID_ARGUMENT, "fake backend IO buffer is invalid"));
            continue;
        }

        void* localSendBuffer = nullptr;
        auto status = ResolveLocalAddress(ioBatch.sendBuffer, static_cast<std::size_t>(ioBatch.len),
                                          localSendBuffer);
        if (!status.ok()) {
            statuses.emplace_back(std::move(status));
            continue;
        }

        const auto* request = static_cast<const std::uint32_t*>(localSendBuffer);
        const auto requestSize = RequestDwordCount(request) * sizeof(std::uint32_t);
        if (requestSize > ioBatch.len) {
            statuses.emplace_back(Status::Error(StatusCode::INVALID_ARGUMENT,
                                                "fake backend send buffer is truncated"));
            continue;
        }

        const auto completionSize = CompletionDwordCount(request) * sizeof(std::uint32_t);
        void* localFlagBuffer = nullptr;
        status = ResolveLocalAddress(ioBatch.flagBuffer, completionSize, localFlagBuffer);
        if (!status.ok()) {
            statuses.emplace_back(std::move(status));
            continue;
        }

        IoTask task;
        task.request.resize(ioBatch.len / sizeof(std::uint32_t) +
                            (ioBatch.len % sizeof(std::uint32_t) != 0));
        std::memcpy(task.request.data(), localSendBuffer, static_cast<std::size_t>(ioBatch.len));
        task.requestLength = ioBatch.len;
        task.flagBuffer = static_cast<std::uint32_t*>(localFlagBuffer);
        if (!workerPool_->Push(std::move(task))) {
            statuses.emplace_back(
                Status::Error(StatusCode::NOT_INITIALIZED, "fake backend worker pool is stopping"));
            continue;
        }
        statuses.emplace_back(Status::OK());
    }
    return statuses;
}

void FakeTransProvider::ProcessIoTask(IoTask& task)
{
    std::vector<std::uint32_t> completion;
    Status status;
    try {
        status = SetupDeviceRuntime();
        if (status.ok()) {
            status =
                CompleteFakeBackendRequest(task.request.data(), task.requestLength, completion);
        }
    } catch (const std::exception& error) {
        status = Status::Error(StatusCode::INTERNAL_ERROR,
                               "fake backend worker exception: " + std::string(error.what()));
    } catch (...) {
        status = Status::Error(StatusCode::INTERNAL_ERROR,
                               "fake backend worker raised an unknown exception");
    }
    if (!status.ok()) {
        completion.assign(kCqeDwordCount, 0);
        PackCqeHeader(completion.data(),
                      static_cast<std::uint16_t>(RequestCid(task.request.data())),
                      kCqeInternalError);
        UC_ERROR("ASU fake backend worker failed code={} message={}", static_cast<int>(status.code),
                 status.message);
    }
    PublishCompletion(task.flagBuffer, completion);
}

Status FakeTransProvider::RegisterMemory(const std::vector<RegisterMemoryDesc>& memoryDescs,
                                         std::vector<MRHandle>& mrHandles)
{
    mrHandles.clear();
    mrHandles.reserve(memoryDescs.size());
    std::lock_guard<std::mutex> lock(registeredMemoryMu_);
    for (const auto& desc : memoryDescs) {
        auto handle = nextMrHandle_.fetch_add(1, std::memory_order_relaxed);
        if (handle == 0) { handle = nextMrHandle_.fetch_add(1, std::memory_order_relaxed); }
        auto mrHandle = reinterpret_cast<MRHandle>(handle);
        if (desc.localAddr != 0) {
            registeredMemories_[mrHandle] = RegisteredMemory{desc.addr, desc.localAddr, desc.size};
        }
        mrHandles.push_back(mrHandle);
    }
    return Status::OK();
}

Status FakeTransProvider::BindMemory(const std::vector<BindMemoryDesc>& memoryDescs,
                                     std::vector<MRHandle>& mrHandles)
{
    mrHandles.clear();
    mrHandles.reserve(memoryDescs.size());
    std::lock_guard<std::mutex> lock(registeredMemoryMu_);
    for (const auto& desc : memoryDescs) {
        auto handle = nextMrHandle_.fetch_add(1, std::memory_order_relaxed);
        if (handle == 0) { handle = nextMrHandle_.fetch_add(1, std::memory_order_relaxed); }
        const auto mrHandle = reinterpret_cast<MRHandle>(handle);
        registeredMemories_[mrHandle] = RegisteredMemory{desc.addr, desc.addr, desc.size};
        mrHandles.emplace_back(mrHandle);
    }
    return Status::OK();
}

std::vector<Status> FakeTransProvider::UnregisterMemory(
    const std::vector<UnregisterMemoryDesc>& handles)
{
    workerPool_->WaitUntilIdle();
    std::lock_guard<std::mutex> lock(registeredMemoryMu_);
    for (const auto& desc : handles) { registeredMemories_.erase(desc.mrHandle); }
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

Status FakeTransProvider::GetMemTokenId(MRHandle, uint32_t& tokenId)
{
    tokenId = 1;
    return Status::OK();
}

}  // namespace UC::ASU
