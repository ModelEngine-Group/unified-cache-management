#include "kv_test/local_asu_transport.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace UC::KVTest {
namespace {

class LocalAsuTransport final : public UC::ASU::AsuTransport {
public:
    explicit LocalAsuTransport(std::string storeRoot) : storeRoot_(std::move(storeRoot)) {}

    UC::ASU::Status Init(const UC::ASU::TransportConfig& config) override
    {
        auto status = ValidateTransportAddress(config);
        if (!status.ok()) { return status; }

        config_ = config;
        std::error_code errorCode;
        std::filesystem::create_directories(AsuRoot(), errorCode);
        if (errorCode) {
            return UC::ASU::Status::Error(UC::ASU::StatusCode::IO_ERROR,
                                          "failed to create local ASU store path=" +
                                              AsuRoot().string() + " error=" + errorCode.message());
        }
        initialized_ = true;
        return UC::ASU::Status::OK();
    }

    UC::ASU::Status Init(const std::string&) override
    {
        return UC::ASU::Status::Error(UC::ASU::StatusCode::UNSUPPORTED,
                                      "local ASU transport does not load config files directly");
    }

    UC::ASU::Status Shutdown() override
    {
        initialized_ = false;
        std::lock_guard<std::mutex> lock(taskResultsMu_);
        taskResults_.clear();
        return UC::ASU::Status::OK();
    }

    UC::ASU::Status CheckHealth() override
    {
        return initialized_ ? UC::ASU::Status::OK() : NotInitialized();
    }

    UC::ASU::Status Query(const std::vector<UC::ASU::CacheKey>& keys,
                          const UC::ASU::QueryOptions& options,
                          UC::ASU::QueryResult& result) override
    {
        if (!initialized_) { return NotInitialized(); }

        result = UC::ASU::QueryResult{};
        if (options.mode == UC::ASU::QueryMode::PREFIX) {
            result.prefixHitKeys = CountPrefixHits(keys.empty() ? "" : keys.front());
            return UC::ASU::Status::OK();
        }

        result.exists.reserve(keys.size());
        for (const auto& key : keys) {
            result.exists.emplace_back(std::filesystem::exists(KeyPath(key)) ? 1 : 0);
        }
        return UC::ASU::Status::OK();
    }

    UC::ASU::Status QueryAsync(const std::vector<UC::ASU::CacheKey>& keys,
                               const UC::ASU::QueryOptions& options,
                               UC::ASU::TaskId& taskId) override
    {
        UC::ASU::QueryResult queryResult;
        auto status = Query(keys, options, queryResult);
        UC::ASU::TaskResult result;
        result.status = status;
        result.queryResult = queryResult;
        taskId = SaveTaskResult(result);
        return status;
    }

    UC::ASU::Status LoadAsync(const std::vector<UC::ASU::KVBuffer>& entries,
                              UC::ASU::TaskId& taskId) override
    {
        if (!initialized_) { return NotInitialized(); }

        UC::ASU::TaskResult result;
        result.status = UC::ASU::Status::OK();
        result.entryStatus.reserve(entries.size());
        for (const auto& entry : entries) { result.entryStatus.emplace_back(LoadEntry(entry)); }
        FinalizeEntryTask(result);
        taskId = SaveTaskResult(result);
        return UC::ASU::Status::OK();
    }

    UC::ASU::Status StoreAsync(const std::vector<UC::ASU::KVBuffer>& entries,
                               UC::ASU::TaskId& taskId) override
    {
        if (!initialized_) { return NotInitialized(); }

        UC::ASU::TaskResult result;
        result.status = UC::ASU::Status::OK();
        result.entryStatus.reserve(entries.size());
        for (const auto& entry : entries) { result.entryStatus.emplace_back(StoreEntry(entry)); }
        FinalizeEntryTask(result);
        taskId = SaveTaskResult(result);
        return UC::ASU::Status::OK();
    }

    UC::ASU::Status DeleteAsync(const std::vector<UC::ASU::CacheKey>& keys,
                                UC::ASU::TaskId& taskId) override
    {
        if (!initialized_) { return NotInitialized(); }

        UC::ASU::TaskResult result;
        result.status = UC::ASU::Status::OK();
        result.entryStatus.reserve(keys.size());
        for (const auto& key : keys) {
            std::error_code errorCode;
            (void)std::filesystem::remove(KeyPath(key), errorCode);
            if (errorCode) {
                result.entryStatus.emplace_back(UC::ASU::Status::Error(
                    UC::ASU::StatusCode::IO_ERROR,
                    "failed to delete local key=" + key + " error=" + errorCode.message()));
            } else {
                result.entryStatus.emplace_back(UC::ASU::Status::OK());
            }
        }
        FinalizeEntryTask(result);
        taskId = SaveTaskResult(result);
        return UC::ASU::Status::OK();
    }

    UC::ASU::Status Cancel(UC::ASU::TaskId taskId) override
    {
        std::lock_guard<std::mutex> lock(taskResultsMu_);
        auto iter = taskResults_.find(taskId);
        if (iter == taskResults_.end()) {
            return UC::ASU::Status::Error(UC::ASU::StatusCode::TASK_NOT_FOUND,
                                          "local task not found");
        }
        iter->second.status =
            UC::ASU::Status::Error(UC::ASU::StatusCode::CANCELED, "local task canceled");
        return UC::ASU::Status::OK();
    }

    UC::ASU::Status Check(UC::ASU::TaskId taskId, UC::ASU::TaskResult& result) override
    {
        std::lock_guard<std::mutex> lock(taskResultsMu_);
        auto iter = taskResults_.find(taskId);
        if (iter == taskResults_.end()) {
            return UC::ASU::Status::Error(UC::ASU::StatusCode::TASK_NOT_FOUND,
                                          "local task not found");
        }
        result = iter->second;
        return UC::ASU::Status::OK();
    }

    UC::ASU::Status Wait(UC::ASU::TaskId taskId, std::uint64_t,
                         UC::ASU::TaskResult& result) override
    {
        return Check(taskId, result);
    }

    UC::ASU::Status RegisterRegions(const std::vector<UC::ASU::MemoryRegion>& regions,
                                    std::vector<UC::ASU::RegisterResult>& results) override
    {
        results.clear();
        results.reserve(regions.size());
        std::lock_guard<std::mutex> lock(taskResultsMu_);
        for (std::size_t index = 0; index < regions.size(); ++index) {
            results.emplace_back(UC::ASU::RegisterResult{UC::ASU::Status::OK(), nextMrHandle_++});
        }
        return UC::ASU::Status::OK();
    }

    UC::ASU::Status BindRegisteredRegions(const std::vector<UC::ASU::RegisteredMemory>& regions,
                                          std::vector<UC::ASU::RegisterResult>& results) override
    {
        results.clear();
        results.reserve(regions.size());
        for (const auto& region : regions) {
            results.emplace_back(UC::ASU::RegisterResult{UC::ASU::Status::OK(), region.handle});
        }
        return UC::ASU::Status::OK();
    }

    UC::ASU::Status UnregisterRegions(const std::vector<UC::ASU::MRHandle>&) override
    {
        return UC::ASU::Status::OK();
    }

private:
    static UC::ASU::Status NotInitialized()
    {
        return UC::ASU::Status::Error(UC::ASU::StatusCode::NOT_INITIALIZED,
                                      "local ASU transport is not initialized");
    }

    static UC::ASU::Status ValidateTransportAddress(const UC::ASU::TransportConfig& config)
    {
        if (config.endpoints.empty()) {
            return UC::ASU::Status::Error(
                UC::ASU::StatusCode::INVALID_ARGUMENT,
                "local ASU transport endpoint is required, asuId=" + std::to_string(config.asuId));
        }

        for (std::size_t index = 0; index < config.endpoints.size(); ++index) {
            const auto& endpoint = config.endpoints[index];
            if (endpoint.ip.empty()) {
                return UC::ASU::Status::Error(
                    UC::ASU::StatusCode::INVALID_ARGUMENT,
                    "local ASU endpoint local.comm_id is required, asuId=" +
                        std::to_string(config.asuId) + ", endpointIndex=" + std::to_string(index));
            }
            if (endpoint.port == 0) {
                return UC::ASU::Status::Error(
                    UC::ASU::StatusCode::INVALID_ARGUMENT,
                    "local ASU endpoint port is required, asuId=" + std::to_string(config.asuId) +
                        ", endpointIndex=" + std::to_string(index));
            }
        }
        return UC::ASU::Status::OK();
    }

    static void FinalizeEntryTask(UC::ASU::TaskResult& result)
    {
        const auto failed =
            std::find_if(result.entryStatus.begin(), result.entryStatus.end(),
                         [](const UC::ASU::Status& status) { return !status.ok(); });
        if (failed != result.entryStatus.end()) {
            result.status = UC::ASU::Status::Error(UC::ASU::StatusCode::PARTIAL_FAILED,
                                                   "one or more local entries failed");
        }
    }

    static std::string HexEncode(const std::string& value)
    {
        constexpr char kHex[] = "0123456789abcdef";
        std::string output;
        output.reserve(value.size() * 2);
        for (unsigned char ch : value) {
            output.push_back(kHex[ch >> 4]);
            output.push_back(kHex[ch & 0x0F]);
        }
        return output;
    }

    std::filesystem::path AsuRoot() const
    {
        return std::filesystem::path(storeRoot_) / ("asu-" + std::to_string(config_.asuId));
    }

    std::filesystem::path KeyPath(const UC::ASU::CacheKey& key) const
    {
        return AsuRoot() / (HexEncode(key) + ".bin");
    }

    UC::ASU::Status StoreEntry(const UC::ASU::KVBuffer& entry)
    {
        std::ofstream file{KeyPath(entry.key), std::ios::binary | std::ios::trunc};
        if (!file.is_open()) {
            return UC::ASU::Status::Error(UC::ASU::StatusCode::IO_ERROR,
                                          "failed to open local key for store=" + entry.key);
        }
        const auto* data = reinterpret_cast<const char*>(entry.buffer.region.addr);
        file.write(data, static_cast<std::streamsize>(entry.buffer.region.size));
        if (!file.good()) {
            return UC::ASU::Status::Error(UC::ASU::StatusCode::IO_ERROR,
                                          "failed to write local key=" + entry.key);
        }
        return UC::ASU::Status::OK();
    }

    UC::ASU::Status LoadEntry(const UC::ASU::KVBuffer& entry)
    {
        std::ifstream file{KeyPath(entry.key), std::ios::binary};
        if (!file.is_open()) {
            return UC::ASU::Status::Error(UC::ASU::StatusCode::NOT_FOUND,
                                          "local key not found=" + entry.key);
        }

        file.seekg(0, std::ios::end);
        const auto size = file.tellg();
        file.seekg(0, std::ios::beg);
        if (size < 0 || static_cast<std::uint64_t>(size) >
                            static_cast<std::uint64_t>(entry.buffer.region.size)) {
            return UC::ASU::Status::Error(UC::ASU::StatusCode::BUFFER_NOT_SUPPORTED,
                                          "local value does not fit destination key=" + entry.key);
        }

        auto* data = reinterpret_cast<char*>(entry.buffer.region.addr);
        file.read(data, size);
        if (!file.good() && !file.eof()) {
            return UC::ASU::Status::Error(UC::ASU::StatusCode::IO_ERROR,
                                          "failed to read local key=" + entry.key);
        }
        return UC::ASU::Status::OK();
    }

    std::uint32_t CountPrefixHits(const std::string& prefix) const
    {
        std::uint32_t hits = 0;
        std::error_code errorCode;
        for (const auto& entry : std::filesystem::directory_iterator(AsuRoot(), errorCode)) {
            if (errorCode) { break; }
            if (entry.path().filename().string().rfind(HexEncode(prefix), 0) == 0) { ++hits; }
        }
        return hits;
    }

    UC::ASU::TaskId SaveTaskResult(UC::ASU::TaskResult result)
    {
        std::lock_guard<std::mutex> lock(taskResultsMu_);
        const auto taskId = nextTaskId_++;
        taskResults_[taskId] = std::move(result);
        return taskId;
    }

    std::string storeRoot_;
    UC::ASU::TransportConfig config_;
    bool initialized_{false};
    UC::ASU::TaskId nextTaskId_{1};
    UC::ASU::MRHandle nextMrHandle_{1};
    std::mutex taskResultsMu_;
    std::unordered_map<UC::ASU::TaskId, UC::ASU::TaskResult> taskResults_;
};

}  // namespace

UC::ASU::TransportFactory CreateLocalAsuTransportFactory(std::string storeRoot)
{
    if (storeRoot.empty()) { storeRoot = "./kv-test-local-store"; }
    return [storeRoot = std::move(storeRoot)] {
        return std::make_unique<LocalAsuTransport>(storeRoot);
    };
}

}  // namespace UC::KVTest
