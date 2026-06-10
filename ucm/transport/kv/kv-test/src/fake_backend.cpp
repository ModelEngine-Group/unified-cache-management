#include "kv_test/fake_backend.h"
#include <acl/acl.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <thread>
#include "buffer_manager.h"
#include "connection_manager.h"
#include "kv_protocol.h"

namespace UC::KVTest {
namespace {

constexpr std::uint16_t kCqeSuccess = 0x000;
constexpr std::uint16_t kCqeCheckResultBuffer = 0x732;
constexpr std::uint8_t kBatchEntryOk = 0x0;
constexpr std::uint8_t kBatchEntryKeyNotFound = 0x3;
constexpr std::uint8_t kDeleteEntryOk = 0x0;
constexpr std::uint8_t kDeleteEntryFailed = 0x1;
constexpr std::uint8_t kExistEntryNotExist = 0x0;
constexpr std::uint8_t kExistEntryExist = 0x1;
constexpr int kExitInvalidArgument = 1;
constexpr int kFakeBackendAclDeviceId = 0;

std::mutex g_fakeBackendMu;
std::mutex g_traceMu;
FakeBackendConfig g_fakeBackendConfig;
bool g_fakeBackendEnabled = false;

std::string NormalizeMode(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::filesystem::path StoreRoot(const FakeBackendConfig& config)
{
    if (!config.storePath.empty()) { return config.storePath; }
    return "./kv-test-fake-backend-store";
}

std::uint64_t ReadU64(std::uint32_t low, std::uint32_t high)
{
    return static_cast<std::uint64_t>(low) | (static_cast<std::uint64_t>(high) << 32);
}

std::uint32_t RequestCid(const std::uint32_t* request) { return request[0] >> 16; }

UC::ASU::AsuId RequestAsuId(const std::uint32_t* request)
{
    // kv-test fake backend temporarily reuses kv_ns_id as the ASU store namespace.
    return request[1];
}

UC::ASU::KvOpcode RequestOpcode(const std::uint32_t* request)
{
    return static_cast<UC::ASU::KvOpcode>(request[0] & 0xFF);
}

const char* OpcodeName(UC::ASU::KvOpcode opcode)
{
    switch (opcode) {
        case UC::ASU::KvOpcode::BatchStore: return "BatchStore";
        case UC::ASU::KvOpcode::BatchRetrieve: return "BatchRetrieve";
        case UC::ASU::KvOpcode::Delete: return "Delete";
        case UC::ASU::KvOpcode::Exist: return "Exist";
        case UC::ASU::KvOpcode::KeepAlive: return "KeepAlive";
        default: return "Unknown";
    }
}

void TraceCompletion(UC::ASU::KvOpcode opcode, UC::ASU::AsuId asuId, std::uint16_t cid,
                     std::uint16_t status, bool resultBuffer, std::uint16_t batchNumber,
                     std::uint16_t existingKeyNumber = 0)
{
    const auto* tracePath = std::getenv("KV_TEST_FAKE_BACKEND_TRACE");
    if (tracePath == nullptr || tracePath[0] == '\0') { return; }

    std::lock_guard<std::mutex> lock(g_traceMu);
    std::ofstream trace(tracePath, std::ios::app);
    if (!trace) { return; }

    trace << "opcode=" << OpcodeName(opcode) << " asu_id=" << asuId << " cid=" << cid
          << " status=0x" << std::hex << std::setw(3) << std::setfill('0') << status << std::dec
          << " result_buffer=" << (resultBuffer ? 1 : 0) << " batch_number=" << batchNumber
          << " existing_key_number=" << existingKeyNumber << '\n';
}

std::string ReadKey(const std::uint32_t* data)
{
    char key[17] = {};
    std::memcpy(key, data, 16);
    const auto keyLen = std::find(key, key + 16, '\0') - key;
    return std::string(key, static_cast<std::size_t>(keyLen));
}

std::string KeyFileName(const std::string& key)
{
    std::uint64_t hash = 1469598103934665603ULL;
    for (unsigned char ch : key) {
        hash ^= ch;
        hash *= 1099511628211ULL;
    }

    std::ostringstream stream;
    stream << std::hex << std::setw(16) << std::setfill('0') << hash << ".bin";
    return stream.str();
}

std::filesystem::path AsuRoot(const FakeBackendConfig& config, UC::ASU::AsuId asuId)
{
    return StoreRoot(config) / ("asu-" + std::to_string(asuId));
}

std::filesystem::path KeyPath(const FakeBackendConfig& config, UC::ASU::AsuId asuId,
                              const std::string& key)
{
    return AsuRoot(config, asuId) / KeyFileName(key);
}

bool StoreBytes(const FakeBackendConfig& config, UC::ASU::AsuId asuId, const std::string& key,
                std::uint64_t addr, std::uint32_t length)
{
    std::filesystem::create_directories(AsuRoot(config, asuId));
    std::ofstream output(KeyPath(config, asuId, key), std::ios::binary | std::ios::trunc);
    if (!output) { return false; }
    output.write(reinterpret_cast<const char*>(addr), length);
    return output.good();
}

bool LoadBytes(const FakeBackendConfig& config, UC::ASU::AsuId asuId, const std::string& key,
               std::uint64_t addr, std::uint32_t length)
{
    std::ifstream input(KeyPath(config, asuId, key), std::ios::binary);
    if (!input) { return false; }
    input.read(reinterpret_cast<char*>(addr), length);
    const auto readCount = input.gcount();
    if (readCount < static_cast<std::streamsize>(length)) {
        std::memset(reinterpret_cast<char*>(addr) + readCount, 0,
                    length - static_cast<std::uint32_t>(readCount));
    }
    return true;
}

bool DeleteKey(const FakeBackendConfig& config, UC::ASU::AsuId asuId, const std::string& key)
{
    std::error_code errorCode;
    std::filesystem::remove(KeyPath(config, asuId, key), errorCode);
    // Delete result buffer uses 0 for success. A missing key is still a successful delete.
    return !errorCode;
}

bool ExistsKey(const FakeBackendConfig& config, UC::ASU::AsuId asuId, const std::string& key)
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
    std::string key;
    std::uint64_t bufferAddr{0};
    std::uint32_t length{0};
};

std::vector<BatchEntry> ReadBatchEntries(const std::uint32_t* request, std::uint16_t batchNumber)
{
    std::vector<BatchEntry> entries;
    entries.reserve(batchNumber);
    for (std::uint16_t index = 0; index < batchNumber; ++index) {
        const auto* entry =
            request + UC::ASU::kSqeDwordCount + index * UC::ASU::kBatchEntryDwordCount;
        BatchEntry parsed;
        parsed.key = ReadKey(entry + 1);
        parsed.bufferAddr = ReadU64(entry[5], entry[6]);
        parsed.length = entry[7] & 0xFFFFFF;
        entries.emplace_back(std::move(parsed));
    }
    return entries;
}

std::vector<std::string> ReadKeyEntries(const std::uint32_t* request, std::uint16_t batchNumber)
{
    std::vector<std::string> keys;
    keys.reserve(batchNumber);
    for (std::uint16_t index = 0; index < batchNumber; ++index) {
        const auto* entry =
            request + UC::ASU::kSqeDwordCount + index * UC::ASU::kKeyEntryDwordCount;
        keys.emplace_back(ReadKey(entry));
    }
    return keys;
}

UC::ASU::Status CompleteBatchStore(const FakeBackendConfig& config, UC::ASU::AsuId asuId,
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
    if (!allOk) { PackResultBuffer4Bit(flagBuffer + UC::ASU::kCqeDwordCount, results); }
    TraceCompletion(UC::ASU::KvOpcode::BatchStore, asuId, cid, cqeStatus, !allOk, batchNumber);
    return UC::ASU::Status::OK();
}

UC::ASU::Status CompleteBatchRetrieve(const FakeBackendConfig& config, UC::ASU::AsuId asuId,
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
    if (!allOk) { PackResultBuffer4Bit(flagBuffer + UC::ASU::kCqeDwordCount, results); }
    TraceCompletion(UC::ASU::KvOpcode::BatchRetrieve, asuId, cid, cqeStatus, !allOk, batchNumber);
    return UC::ASU::Status::OK();
}

UC::ASU::Status CompleteDelete(const FakeBackendConfig& config, UC::ASU::AsuId asuId,
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
    if (!allOk) { PackResultBuffer1Bit(flagBuffer + UC::ASU::kCqeDwordCount, results); }
    TraceCompletion(UC::ASU::KvOpcode::Delete, asuId, cid, cqeStatus, !allOk, batchNumber);
    return UC::ASU::Status::OK();
}

UC::ASU::Status CompleteExist(const FakeBackendConfig& config, UC::ASU::AsuId asuId,
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
        if (!ExistsKey(config, asuId, keys[index])) { continue; }
        results[index] = kExistEntryExist;
        ++existingKeyNumber;
    }

    const auto allExist = std::all_of(results.begin(), results.end(), [](std::uint8_t result) {
        return result == kExistEntryExist;
    });
    const auto cqeStatus = allExist ? kCqeSuccess : kCqeCheckResultBuffer;
    PackCqeHeader(flagBuffer, cid, cqeStatus);
    flagBuffer[0] = existingKeyNumber;
    if (!allExist) { PackResultBuffer1Bit(flagBuffer + UC::ASU::kCqeDwordCount, results); }
    TraceCompletion(UC::ASU::KvOpcode::Exist, asuId, cid, cqeStatus, !allExist, batchNumber,
                    existingKeyNumber);
    return UC::ASU::Status::OK();
}

UC::ASU::Status CompleteFakeBackendRequest(FakeBackendConfig config,
                                           const UC::ASU::ScatterGatherEntry& sendSge)
{
    if (config.latencyMs > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(config.latencyMs));
    }

    const auto* request = reinterpret_cast<const std::uint32_t*>(sendSge.addr);
    const auto asuId = RequestAsuId(request);
    switch (RequestOpcode(request)) {
        case UC::ASU::KvOpcode::BatchStore: return CompleteBatchStore(config, asuId, request);
        case UC::ASU::KvOpcode::BatchRetrieve: return CompleteBatchRetrieve(config, asuId, request);
        case UC::ASU::KvOpcode::Delete: return CompleteDelete(config, asuId, request);
        case UC::ASU::KvOpcode::Exist: return CompleteExist(config, asuId, request);
        case UC::ASU::KvOpcode::KeepAlive: {
            auto* flagBuffer = reinterpret_cast<std::uint32_t*>(ReadU64(request[3], request[4]));
            PackCqeHeader(flagBuffer, static_cast<std::uint16_t>(RequestCid(request)), kCqeSuccess);
            TraceCompletion(UC::ASU::KvOpcode::KeepAlive, asuId,
                            static_cast<std::uint16_t>(RequestCid(request)), kCqeSuccess, false, 0);
            return UC::ASU::Status::OK();
        }
        default:
            return UC::ASU::Status::Error(UC::ASU::StatusCode::UNSUPPORTED,
                                          "fake backend only supports batch ASU operations");
    }
}

FakeBackendConfig GetFakeBackendConfig(bool& enabled)
{
    std::lock_guard<std::mutex> lock(g_fakeBackendMu);
    enabled = g_fakeBackendEnabled;
    return g_fakeBackendConfig;
}

void SetFakeBackendConfig(FakeBackendConfig config)
{
    std::lock_guard<std::mutex> lock(g_fakeBackendMu);
    g_fakeBackendConfig = std::move(config);
    g_fakeBackendEnabled = true;
}

void DisableFakeBackend()
{
    std::lock_guard<std::mutex> lock(g_fakeBackendMu);
    g_fakeBackendConfig = FakeBackendConfig{};
    g_fakeBackendEnabled = false;
}

void PatchTransportConfig(UC::ASU::TransportConfig& config)
{
    config.attrs.try_emplace("kernel_count", "1");
    config.attrs.try_emplace("quiet_count", "1");
    // kv-test fake backend has no direct TransportConfig context in Send, so this temporary
    // test-only mapping lets the mock recover the ASU store namespace from the packed SQE.
    config.attrs["kv_ns_id"] = std::to_string(config.asuId);
    config.attrs.try_emplace("dtype", "0");
    config.attrs.try_emplace("dspec", "0");
    config.attrs.try_emplace("lr", "false");
    config.attrs["sc"] = "true";
    if (config.endpoints.empty()) {
        UC::ASU::AsuEndpoint endpoint;
        endpoint.ip = "fake_backend";
        endpoint.port = 19001;
        endpoint.protocol = UC::ASU::Protocol::TCP;
        config.endpoints.emplace_back(std::move(endpoint));
    }
}

}  // namespace

FakeBackendAclRuntime::~FakeBackendAclRuntime() { TearDown(); }

Status FakeBackendAclRuntime::MaybeSetUp(const KvTestConfig& config)
{
    if (!IsFakeBackendMode(config)) { return Status::Success(); }

    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        return Status::Error(kExitInvalidArgument,
                             "fake_backend aclInit failed: " + std::to_string(ret));
    }
    initialized_ = true;

    // kv-test fake_backend is a temporary standalone test path. Use device 0 until the
    // ASU client/transport runtime contract is formalized.
    ret = aclrtSetDevice(kFakeBackendAclDeviceId);
    if (ret != ACL_SUCCESS) {
        return Status::Error(kExitInvalidArgument,
                             "fake_backend aclrtSetDevice failed: device_id=" +
                                 std::to_string(kFakeBackendAclDeviceId) +
                                 " ret=" + std::to_string(ret));
    }
    deviceSet_ = true;
    return Status::Success();
}

void FakeBackendAclRuntime::TearDown()
{
    if (deviceSet_) {
        (void)aclrtResetDevice(kFakeBackendAclDeviceId);
        deviceSet_ = false;
    }
    if (initialized_) {
        (void)aclFinalize();
        initialized_ = false;
    }
}

bool IsFakeBackendMode(const KvTestConfig& config)
{
    const auto mode = NormalizeMode(config.asuClientMode);
    return mode == "fake_backend" || mode == "fakebackend";
}

void MaybePrepareFakeBackend(KvTestConfig& config)
{
    if (!IsFakeBackendMode(config)) {
        DisableFakeBackend();
        return;
    }

    if (config.fakeBackend.storePath.empty()) {
        config.fakeBackend.storePath =
            config.localStorePath.empty() ? "./kv-test-fake-backend-store" : config.localStorePath;
    }
    SetFakeBackendConfig(config.fakeBackend);

    config.asuClientConfig.attrs.try_emplace("hash_table.type", "RING_HASH");
    config.asuClientConfig.attrs.try_emplace("ring_hash.virtual_node_count", "128");
    if (config.asuClientConfig.transportConfigs.empty()) {
        UC::ASU::TransportConfig transportConfig;
        transportConfig.asuId = 1;
        config.asuClientConfig.transportConfigs.emplace_back(std::move(transportConfig));
    }
    for (auto& transportConfig : config.asuClientConfig.transportConfigs) {
        PatchTransportConfig(transportConfig);
    }
}

}  // namespace UC::KVTest

namespace UC::ASU {

std::vector<Status> MockSend(const std::vector<SendIoBatch>& ioBatches, std::uint32_t kernelCount,
                             std::uint32_t quietCount)
{
    (void)kernelCount;
    (void)quietCount;

    bool enabled = false;
    const auto config = UC::KVTest::GetFakeBackendConfig(enabled);
    if (!enabled) {
        return std::vector<Status>(
            ioBatches.size(),
            Status::Error(StatusCode::UNSUPPORTED, "kv-test fake backend Send is not enabled"));
    }

    std::vector<Status> statuses;
    statuses.reserve(ioBatches.size());
    for (const auto& ioBatch : ioBatches) {
        if (ioBatch.sendSge == nullptr || ioBatch.sendSge->addr == 0) {
            statuses.emplace_back(
                Status::Error(StatusCode::INVALID_ARGUMENT, "fake backend send SGE is empty"));
            continue;
        }

        // kv-test fake backend temporarily completes the CQE before Send returns. The production
        // path still observes completion through Transport polling, while the mock avoids detached
        // threads racing with sub-batch buffer lifetime in multi sub-batch tests.
        statuses.emplace_back(UC::KVTest::CompleteFakeBackendRequest(config, *ioBatch.sendSge));
    }
    return statuses;
}

std::vector<Status> Send(const std::vector<SendIoBatch>& ioBatches, std::uint32_t kernelCount,
                         std::uint32_t quietCount)
{
    return MockSend(ioBatches, kernelCount, quietCount);
}

}  // namespace UC::ASU
