#include "consistency_checker.h"
#include <string>
#include "key_value_generator.h"

namespace UC::KVTest {

using UC::ASU::CacheKeyToHex;

std::string AsuStatusCodeName(UC::ASU::StatusCode code)
{
    switch (code) {
        case UC::ASU::StatusCode::OK: return "OK";
        case UC::ASU::StatusCode::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case UC::ASU::StatusCode::NOT_INITIALIZED: return "NOT_INITIALIZED";
        case UC::ASU::StatusCode::TIMEOUT: return "TIMEOUT";
        case UC::ASU::StatusCode::NOT_FOUND: return "NOT_FOUND";
        case UC::ASU::StatusCode::PARTIAL_FAILED: return "PARTIAL_FAILED";
        case UC::ASU::StatusCode::CONNECTION_ERROR: return "CONNECTION_ERROR";
        case UC::ASU::StatusCode::IO_ERROR: return "IO_ERROR";
        case UC::ASU::StatusCode::BUFFER_NOT_REGISTERED: return "BUFFER_NOT_REGISTERED";
        case UC::ASU::StatusCode::BUFFER_NOT_SUPPORTED: return "BUFFER_NOT_SUPPORTED";
        case UC::ASU::StatusCode::TASK_NOT_FOUND: return "TASK_NOT_FOUND";
        case UC::ASU::StatusCode::RESOURCE_BUSY: return "RESOURCE_BUSY";
        case UC::ASU::StatusCode::UNSUPPORTED: return "UNSUPPORTED";
        case UC::ASU::StatusCode::IN_PROGRESS: return "IN_PROGRESS";
        case UC::ASU::StatusCode::INTERNAL_ERROR: return "INTERNAL_ERROR";
        case UC::ASU::StatusCode::CANCELED: return "CANCELED";
        default: return "UNKNOWN";
    }
}

namespace {

constexpr int kExitInvalidArgument = 1;
constexpr int kExitConsistencyFailed = 4;

Status ConsistencyError(const std::string& operation, const UC::ASU::CacheKey& key,
                        const std::string& reason)
{
    return Status::Error(kExitConsistencyFailed,
                         "consistency check failed: operation=" + operation +
                             " key=" + CacheKeyToHex(key) + " reason=" + reason);
}

Status ValidateTaskForConsistency(const CommandResult& result, const std::string& operation,
                                  std::size_t expectedCount)
{
    if (!result.status.Ok()) { return result.status; }
    if (!result.taskResult.status.ok()) {
        return Status::Error(kExitConsistencyFailed,
                             "consistency check failed: operation=" + operation + " task_status=" +
                                 AsuStatusCodeName(result.taskResult.status.code) +
                                 " message=" + result.taskResult.status.message);
    }
    if (!result.taskResult.entryStatus.empty() &&
        result.taskResult.entryStatus.size() != expectedCount) {
        return Status::Error(
            kExitConsistencyFailed,
            "consistency check failed: operation=" + operation +
                " entry_status_count=" + std::to_string(result.taskResult.entryStatus.size()) +
                " expected=" + std::to_string(expectedCount));
    }
    for (std::size_t index = 0; index < result.taskResult.entryStatus.size(); ++index) {
        const auto& entryStatus = result.taskResult.entryStatus[index];
        if (!entryStatus.ok()) {
            return Status::Error(kExitConsistencyFailed,
                                 "consistency check failed: operation=" + operation +
                                     " entry_index=" + std::to_string(index) +
                                     " entry_status=" + AsuStatusCodeName(entryStatus.code) +
                                     " message=" + entryStatus.message);
        }
    }
    return Status::Success();
}

Status ValidateExpectedBuffers(const GeneratedData& expected, const BufferSet& retrieved,
                               const std::string& operation)
{
    auto status = ValidateGeneratedData(expected, operation);
    if (!status.Ok()) { return status; }
    if (retrieved.ownedBuffers.size() != expected.values.size()) {
        return Status::Error(kExitInvalidArgument, operation + " retrieve buffer count mismatch");
    }
    return Status::Success();
}

Status DigestValue(const std::vector<std::uint8_t>& value, std::string& digest)
{
    KeyValueGenerator generator;
    return generator.Digest(value, digest);
}

void SetValueComparison(ConsistencySummary& summary, const UC::ASU::CacheKey& key,
                        const std::string& expectedDigest, const std::string& actualDigest)
{
    summary.key = CacheKeyToHex(key);
    summary.expected = "digest=" + expectedDigest;
    summary.actual = "digest=" + actualDigest;
}

void SetExistComparison(ConsistencySummary& summary, const UC::ASU::CacheKey& key,
                        bool expectedExists, bool actualExists)
{
    summary.key = CacheKeyToHex(key);
    summary.expected = expectedExists ? "exists=true" : "exists=false";
    summary.actual = actualExists ? "exists=true" : "exists=false";
}

Status CheckRetrievedValues(const GeneratedData& expected, const BufferSet& retrieved,
                            const std::string& operation, ConsistencySummary& summary)
{
    auto status = ValidateExpectedBuffers(expected, retrieved, operation);
    if (!status.Ok()) { return status; }

    summary.enabled = true;
    summary.checked = expected.values.size();
    for (std::size_t index = 0; index < expected.values.size(); ++index) {
        const auto& expectedValue = expected.values[index];
        const auto& actualValue = retrieved.ownedBuffers[index];

        std::string expectedDigest;
        std::string actualDigest;
        status = DigestValue(expectedValue, expectedDigest);
        if (!status.Ok()) { return status; }
        status = DigestValue(actualValue, actualDigest);
        if (!status.Ok()) { return status; }
        SetValueComparison(summary, expected.keys[index], expectedDigest, actualDigest);

        if (actualValue == expectedValue) {
            ++summary.passed;
            continue;
        }

        summary.failed = summary.checked - summary.passed;
        return ConsistencyError(operation, expected.keys[index],
                                "value mismatch expected_digest=" + expectedDigest +
                                    " actual_digest=" + actualDigest +
                                    " expected_size=" + std::to_string(expectedValue.size()) +
                                    " actual_size=" + std::to_string(actualValue.size()));
    }

    summary.failed = 0;
    return Status::Success();
}

Status ValidateQueryResultForConsistency(const std::vector<UC::ASU::CacheKey>& keys,
                                         const CommandResult& result, const std::string& operation)
{
    if (!result.status.Ok()) { return result.status; }
    if (result.queryResult.exists.size() != keys.size()) {
        return Status::Error(kExitConsistencyFailed,
                             "consistency check failed: operation=" + operation + " exists_count=" +
                                 std::to_string(result.queryResult.exists.size()) +
                                 " expected=" + std::to_string(keys.size()));
    }
    return Status::Success();
}

}  // namespace

Status ConsistencyChecker::CheckStoreResult(const GeneratedData& expected,
                                            const BufferSet& retrieved, const CommandResult& result,
                                            ConsistencySummary& summary) const
{
    auto status = ValidateTaskForConsistency(result, "store", expected.keys.size());
    if (!status.Ok()) { return status; }
    return CheckRetrievedValues(expected, retrieved, "store", summary);
}

Status ConsistencyChecker::CheckRetrieveResult(const GeneratedData& expected,
                                               const BufferSet& retrieved,
                                               const CommandResult& result,
                                               ConsistencySummary& summary) const
{
    auto status = ValidateTaskForConsistency(result, "retrieve", expected.keys.size());
    if (!status.Ok()) { return status; }
    return CheckRetrievedValues(expected, retrieved, "retrieve", summary);
}

Status ConsistencyChecker::CheckDeleteResult(const std::vector<UC::ASU::CacheKey>& keys,
                                             const CommandResult& deleteResult,
                                             const CommandResult& existResult,
                                             ConsistencySummary& summary) const
{
    auto status = ValidateTaskForConsistency(deleteResult, "delete", keys.size());
    if (!status.Ok()) { return status; }

    status = ValidateQueryResultForConsistency(keys, existResult, "delete-exist");
    if (!status.Ok()) { return status; }

    summary.enabled = true;
    summary.checked = keys.size();
    for (std::size_t index = 0; index < keys.size(); ++index) {
        const bool actualExists = existResult.queryResult.exists[index] != 0;
        SetExistComparison(summary, keys[index], false, actualExists);
        if (actualExists) {
            summary.failed = summary.checked - summary.passed;
            return ConsistencyError("delete", keys[index],
                                    "expected_exists=false actual_exists=true");
        }
        ++summary.passed;
    }
    summary.failed = 0;
    return Status::Success();
}

Status ConsistencyChecker::CheckExistResult(const std::vector<UC::ASU::CacheKey>& keys,
                                            const CommandResult& result, bool expectedExists,
                                            ConsistencySummary& summary) const
{
    auto status = ValidateQueryResultForConsistency(keys, result, "exist");
    if (!status.Ok()) { return status; }

    const std::uint8_t expectedValue = expectedExists ? 1 : 0;
    summary.enabled = true;
    summary.checked = keys.size();
    for (std::size_t index = 0; index < keys.size(); ++index) {
        const bool actualExists = result.queryResult.exists[index] != 0;
        SetExistComparison(summary, keys[index], expectedExists, actualExists);
        if (actualExists != expectedExists) {
            summary.failed = summary.checked - summary.passed;
            return ConsistencyError(
                "exist", keys[index],
                "expected_exists=" + std::to_string(expectedValue) +
                    " actual_exists=" + std::to_string(result.queryResult.exists[index]));
        }
        ++summary.passed;
    }
    summary.failed = 0;
    return Status::Success();
}

}  // namespace UC::KVTest
