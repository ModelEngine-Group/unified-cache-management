#include "asu_client_runner.h"
#include <algorithm>
#include <unordered_set>
#include "kv_client.h"

namespace kv::bench {

kv::TaskResult BuildEmptyTaskResult()
{
    kv::TaskResult result;
    result.status = kv::Status::OK();
    return result;
}

namespace {

constexpr int kExitInvalidArgument = 1;
constexpr int kExitAsuStatusBase = 100;

Status ToKvTestStatus(const kv::Status& status, const std::string& operation)
{
    if (status.ok()) { return Status::Success(); }
    return Status::Error(kExitAsuStatusBase + static_cast<int>(status.code),
                         operation + " failed: " + status.message);
}

kv::TaskResult BuildFailedTaskResult(const kv::Status& status, std::size_t entryCount)
{
    kv::TaskResult result;
    result.status = status;
    result.entryStatus.assign(entryCount, status);
    return result;
}

Status FinalizeTaskResult(CommandResult& result)
{
    result.status = ToKvTestStatus(result.taskResult.status, "asu task");
    return result.status;
}

Status FinalizeQueryResult(CommandResult& result)
{
    result.status = Status::Success();
    return result.status;
}

using EntrySubmitMethod = kv::Status (kv::AsuClient::*)(
    const std::vector<kv::KVBuffer>&, kv::TaskId&);

Status SubmitAndWaitEntries(kv::AsuClient& client,
                            const std::vector<kv::KVBuffer>& entries,
                            EntrySubmitMethod submitMethod, std::uint64_t timeoutMs,
                            const std::string& operation, CommandResult& result)
{
    kv::TaskId taskId{kv::kInvalidTaskId};
    auto status = (client.*submitMethod)(entries, taskId);
    if (!status.ok()) {
        result.taskResult = BuildFailedTaskResult(status, entries.size());
        return FinalizeTaskResult(result);
    }

    status = client.Wait(taskId, timeoutMs, result.taskResult);
    if (!status.ok()) {
        if (result.taskResult.status.ok()) { result.taskResult.status = status; }
        result.status = ToKvTestStatus(status, operation);
        return result.status;
    }

    return FinalizeTaskResult(result);
}

Status SubmitAndWaitKeys(kv::AsuClient& client, const std::vector<kv::CacheKey>& keys,
                         std::uint64_t timeoutMs, CommandResult& result)
{
    kv::TaskId taskId{kv::kInvalidTaskId};
    auto status = client.DeleteAsync(keys, taskId);
    if (!status.ok()) {
        result.taskResult = BuildFailedTaskResult(status, keys.size());
        return FinalizeTaskResult(result);
    }

    status = client.Wait(taskId, timeoutMs, result.taskResult);
    if (!status.ok()) {
        if (result.taskResult.status.ok()) { result.taskResult.status = status; }
        result.status = ToKvTestStatus(status, "delete");
        return result.status;
    }

    return FinalizeTaskResult(result);
}

Status SubmitEntriesOneByOne(kv::AsuClient& client,
                             const std::vector<kv::KVBuffer>& entries,
                             EntrySubmitMethod submitMethod, std::uint64_t timeoutMs,
                             const std::string& operation, CommandResult& result)
{
    result.taskResult = BuildEmptyTaskResult();
    result.taskResult.entryStatus.reserve(entries.size());

    kv::Status firstFailure = kv::Status::OK();
    for (const auto& entry : entries) {
        std::vector<kv::KVBuffer> singleEntry{entry};
        CommandResult singleResult;
        auto status = SubmitAndWaitEntries(client, singleEntry, submitMethod, timeoutMs, operation,
                                           singleResult);
        if (singleResult.taskResult.entryStatus.empty()) {
            result.taskResult.entryStatus.push_back(singleResult.taskResult.status);
        } else {
            result.taskResult.entryStatus.push_back(singleResult.taskResult.entryStatus.front());
        }

        if (!status.Ok() && firstFailure.ok()) {
            firstFailure =
                singleResult.taskResult.status.ok()
                    ? kv::Status::Error(kv::StatusCode::INTERNAL_ERROR, status.message)
                    : singleResult.taskResult.status;
        }
    }

    if (!firstFailure.ok()) {
        result.taskResult.status = kv::Status::Error(
            kv::StatusCode::PARTIAL_FAILED, operation + " failed for one or more entries");
        return FinalizeTaskResult(result);
    }

    return FinalizeTaskResult(result);
}

}  // namespace

AsuClientRunner::AsuClientRunner(std::unique_ptr<kv::AsuClient> client)
    : client_(std::move(client))
{
}

AsuClientRunner::~AsuClientRunner() = default;

Status AsuClientRunner::Init(const KvTestConfig& config)
{
    if (client_ == nullptr) { return Status::Error(kExitInvalidArgument, "asu client is null"); }

    auto status = client_->Init(config.asuClientConfig);
    return ToKvTestStatus(status, "asu client init");
}

Status AsuClientRunner::Shutdown()
{
    if (client_ == nullptr) { return Status::Success(); }

    auto status = client_->Shutdown();
    return ToKvTestStatus(status, "asu client shutdown");
}

Status AsuClientRunner::RegisterBuffers(BufferSet& buffers)
{
    if (client_ == nullptr) { return Status::Error(kExitInvalidArgument, "asu client is null"); }
    if (!buffers.entryRegionIndexes.empty() &&
        buffers.entryRegionIndexes.size() != buffers.entries.size()) {
        return Status::Error(kExitInvalidArgument,
                             "buffer entry region index count does not match entry count");
    }
    if (buffers.entryRegionIndexes.empty() && buffers.regions.size() != buffers.entries.size()) {
        return Status::Error(kExitInvalidArgument,
                             "buffer region count does not match entry count");
    }
    for (const auto regionIndex : buffers.entryRegionIndexes) {
        if (regionIndex >= buffers.regions.size()) {
            return Status::Error(kExitInvalidArgument, "buffer entry region index out of range");
        }
    }

    buffers.registeredRegions.clear();
    auto status = client_->RegisterRegions(buffers.regions, buffers.registeredRegions);
    if (!status.ok()) { return ToKvTestStatus(status, "register buffers"); }
    if (buffers.registeredRegions.size() != buffers.regions.size()) {
        return Status::Error(kExitInvalidArgument,
                             "register buffer result count does not match region count");
    }

    for (std::size_t entryIndex = 0; entryIndex < buffers.entries.size(); ++entryIndex) {
        const auto regionIndex = buffers.entryRegionIndexes.empty()
                                     ? entryIndex
                                     : buffers.entryRegionIndexes[entryIndex];
        buffers.entries[entryIndex].buffer.handle = buffers.registeredRegions[regionIndex].handle;
    }

    return Status::Success();
}

Status AsuClientRunner::UnregisterBuffers(const BufferSet& buffers)
{
    if (client_ == nullptr) { return Status::Success(); }

    std::vector<kv::MRHandle> handles;
    handles.reserve(buffers.registeredRegions.size());
    std::unordered_set<kv::MRHandle> seen;
    for (const auto& registeredRegion : buffers.registeredRegions) {
        if (registeredRegion.handle != kv::kInvalidMRHandle &&
            seen.insert(registeredRegion.handle).second) {
            handles.push_back(registeredRegion.handle);
        }
    }
    if (handles.empty()) { return Status::Success(); }

    auto status = client_->UnregisterRegions(handles);
    return ToKvTestStatus(status, "unregister buffers");
}

Status AsuClientRunner::Store(const BufferSet& buffers, SubmitMode submitMode,
                              std::uint64_t timeoutMs, CommandResult& result)
{
    if (client_ == nullptr) { return Status::Error(kExitInvalidArgument, "asu client is null"); }
    if (buffers.entries.empty()) {
        result.taskResult = BuildEmptyTaskResult();
        return FinalizeTaskResult(result);
    }

    if (submitMode == SubmitMode::SINGLE_ENTRY_PER_CALL) {
        return SubmitEntriesOneByOne(*client_, buffers.entries, &kv::AsuClient::StoreAsync,
                                     timeoutMs, "store", result);
    }

    return SubmitAndWaitEntries(*client_, buffers.entries, &kv::AsuClient::BatchStoreAsync,
                                timeoutMs, "store", result);
}

Status AsuClientRunner::Retrieve(const BufferSet& buffers, SubmitMode submitMode,
                                 std::uint64_t timeoutMs, CommandResult& result)
{
    if (client_ == nullptr) { return Status::Error(kExitInvalidArgument, "asu client is null"); }
    if (buffers.entries.empty()) {
        result.taskResult = BuildEmptyTaskResult();
        return FinalizeTaskResult(result);
    }

    if (submitMode == SubmitMode::SINGLE_ENTRY_PER_CALL) {
        return SubmitEntriesOneByOne(*client_, buffers.entries, &kv::AsuClient::LoadAsync,
                                     timeoutMs, "retrieve", result);
    }

    return SubmitAndWaitEntries(*client_, buffers.entries, &kv::AsuClient::BatchLoadAsync,
                                timeoutMs, "retrieve", result);
}

Status AsuClientRunner::Delete(const std::vector<kv::CacheKey>& keys, std::uint64_t timeoutMs,
                               CommandResult& result)
{
    if (client_ == nullptr) { return Status::Error(kExitInvalidArgument, "asu client is null"); }
    if (keys.empty()) {
        result.taskResult = BuildEmptyTaskResult();
        return FinalizeTaskResult(result);
    }

    return SubmitAndWaitKeys(*client_, keys, timeoutMs, result);
}

Status AsuClientRunner::Exist(const std::vector<kv::CacheKey>& keys, std::uint64_t timeoutMs,
                              CommandResult& result)
{
    if (client_ == nullptr) { return Status::Error(kExitInvalidArgument, "asu client is null"); }

    kv::TaskId taskId = kv::kInvalidTaskId;
    auto status = client_->QueryAsync(keys, taskId);
    if (status.ok()) {
        kv::TaskResult taskResult;
        status = client_->Wait(taskId, timeoutMs, taskResult);
        if (taskResult.queryResult.has_value()) {
            result.queryResult = std::move(*taskResult.queryResult);
        } else if (status.ok()) {
            status = kv::Status::Error(kv::StatusCode::INTERNAL_ERROR,
                                            "client query result is missing");
        }
    }
    if (!status.ok()) {
        result.status = ToKvTestStatus(status, "exist");
        return result.status;
    }

    return FinalizeQueryResult(result);
}

}  // namespace kv::bench
