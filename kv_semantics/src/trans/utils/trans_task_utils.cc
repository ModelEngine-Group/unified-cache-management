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
#include "trans_task_utils.h"
#include <algorithm>
#include <string>

namespace kv {
namespace {

constexpr int kAsuBatchEntryStatusBase = 0x0100;
constexpr int kAsuDeleteEntryStatusBase = 0x0200;
constexpr int kAsuExistEntryStatusBase = 0x0300;
constexpr int kAsuCqeStatusBase = 0x10000;

StatusCode EntryStatusCode(AsuOpType opType, std::uint8_t rawResult)
{
    if (opType == AsuOpType::BATCH_STORE || opType == AsuOpType::BATCH_LOAD) {
        return static_cast<StatusCode>(kAsuBatchEntryStatusBase | rawResult);
    }
    if (opType == AsuOpType::DELETE) {
        return static_cast<StatusCode>(kAsuDeleteEntryStatusBase | rawResult);
    }
    if (opType == AsuOpType::QUERY) {
        return static_cast<StatusCode>(kAsuExistEntryStatusBase | rawResult);
    }
    return rawResult == 0 ? StatusCode::OK : StatusCode::IO_ERROR;
}

StatusCode CqeStatusCode(std::uint16_t rawStatus)
{
    return static_cast<StatusCode>(kAsuCqeStatusBase | rawStatus);
}

Status ResultBufferEntryToStatus(AsuOpType opType, std::uint8_t rawResult)
{
    if (opType != AsuOpType::QUERY && rawResult == 0x00) { return Status::OK(); }

    if (opType == AsuOpType::BATCH_STORE || opType == AsuOpType::BATCH_LOAD) {
        switch (EntryStatusCode(opType, rawResult)) {
            case StatusCode::ASU_ENTRY_RETRY_ADVISED:
                return Status::Error(StatusCode::ASU_ENTRY_RETRY_ADVISED,
                                     "general error, retry advised");
            case StatusCode::ASU_ENTRY_NO_RETRY_ADVISED:
                return Status::Error(StatusCode::ASU_ENTRY_NO_RETRY_ADVISED,
                                     "general error, no retry advised");
            case StatusCode::ASU_ENTRY_KEY_NOT_FOUND:
                return Status::Error(StatusCode::ASU_ENTRY_KEY_NOT_FOUND, "key not found");
            case StatusCode::ASU_ENTRY_DATA_NOT_EXIST:
                return Status::Error(StatusCode::ASU_ENTRY_DATA_NOT_EXIST, "data not exist");
            default:
                return Status::Error(StatusCode::IO_ERROR, "unknown ASU entry key result is " +
                                                               std::to_string(rawResult));
        }
    }

    if (opType == AsuOpType::DELETE) {
        switch (EntryStatusCode(opType, rawResult)) {
            case StatusCode::ASU_ENTRY_DELETE_FAILED:
                return Status::Error(StatusCode::ASU_ENTRY_DELETE_FAILED, "delete failed");
            default:
                return Status::Error(StatusCode::IO_ERROR,
                                     "unknown ASU delete result is " + std::to_string(rawResult));
        }
    }

    if (opType == AsuOpType::QUERY) {
        switch (EntryStatusCode(opType, rawResult)) {
            case StatusCode::ASU_ENTRY_KEY_NOT_EXIST:
                return Status::Error(StatusCode::ASU_ENTRY_KEY_NOT_EXIST, "key not exist");
            case StatusCode::ASU_ENTRY_KEY_EXIST:
                return Status::Error(StatusCode::ASU_ENTRY_KEY_EXIST, "key exist");
            default:
                return Status::Error(StatusCode::IO_ERROR,
                                     "unknown ASU exist result is " + std::to_string(rawResult));
        }
    }

    return Status::Error(StatusCode::IO_ERROR, "entry CQE status is " + std::to_string(rawResult));
}

}  // namespace

Status KvResponseStatusToSubBatchStatus(std::uint16_t rawStatus)
{
    if (rawStatus == 0x00) { return Status::OK(); }

    switch (CqeStatusCode(rawStatus)) {
        case StatusCode::ASU_CQE_INVALID_COMMAND_OPCODE:
            return Status::Error(StatusCode::ASU_CQE_INVALID_COMMAND_OPCODE,
                                 "Invalid Command Opcode");
        case StatusCode::ASU_CQE_INVALID_FIELD_IN_COMMAND:
            return Status::Error(StatusCode::ASU_CQE_INVALID_FIELD_IN_COMMAND,
                                 "Invalid Field in Command");
        case StatusCode::ASU_CQE_INTERNAL_ERROR:
            return Status::Error(StatusCode::ASU_CQE_INTERNAL_ERROR, "Internal Error");
        case StatusCode::ASU_CQE_WRITE_FAULT:
            return Status::Error(StatusCode::ASU_CQE_WRITE_FAULT, "Write fault");
        case StatusCode::ASU_CQE_UNRECOVERED_READ_ERROR:
            return Status::Error(StatusCode::ASU_CQE_UNRECOVERED_READ_ERROR,
                                 "Unrecovered Read Error");
        case StatusCode::ASU_CQE_KEY_NOT_EXIST:
            return Status::Error(StatusCode::ASU_CQE_KEY_NOT_EXIST, "Key Not Exist");
        case StatusCode::ASU_CQE_OUT_OF_CREATE_SIZE:
            return Status::Error(StatusCode::ASU_CQE_OUT_OF_CREATE_SIZE, "Out of Create Size");
        case StatusCode::ASU_CQE_IO_TIMEOUT:
            return Status::Error(StatusCode::ASU_CQE_IO_TIMEOUT, "IO TimeOut");
        case StatusCode::ASU_CQE_KEY_ALREADY_EXISTED:
            return Status::Error(StatusCode::ASU_CQE_KEY_ALREADY_EXISTED, "Key already Existed");
        case StatusCode::ASU_CQE_RESOURCE_BUSY:
            return Status::Error(StatusCode::ASU_CQE_RESOURCE_BUSY, "Resource Busy");
        case StatusCode::ASU_CQE_CHECK_RESULT_BUFFER:
            return Status::Error(StatusCode::ASU_CQE_CHECK_RESULT_BUFFER,
                                 "Batched Result, check result buffer for entry errors");
        default:
            return Status::Error(StatusCode::IO_ERROR,
                                 "unknown ASU sub-batch status is " + std::to_string(rawStatus));
    }
}

void FillEntryStatusFromCqeResult(const KvResponse& response, AsuOpType opType,
                                  bool useSeekControl, Status& status,
                                  std::vector<Status>& entryStatus)
{
    const auto keyExist = Status::Error(StatusCode::ASU_ENTRY_KEY_EXIST, "key exist");
    const auto keyNotExist = Status::Error(StatusCode::ASU_ENTRY_KEY_NOT_EXIST, "key not exist");
    const bool isQuery = opType == AsuOpType::QUERY;

    if (status.ok()) {
        std::fill(entryStatus.begin(), entryStatus.end(), isQuery ? keyExist : Status::OK());
        return;
    }

    if (isQuery && !useSeekControl && status.code == StatusCode::ASU_CQE_CHECK_RESULT_BUFFER) {
        const auto existingKeyCount =
            std::min(entryStatus.size(), static_cast<std::size_t>(response.existing_key_number));
        std::fill(entryStatus.begin(), entryStatus.end(), keyNotExist);
        std::fill_n(entryStatus.begin(), existingKeyCount, keyExist);
        return;
    }

    if (status.code != StatusCode::ASU_CQE_CHECK_RESULT_BUFFER ||
        response.result_buffer.empty()) {
        std::fill(entryStatus.begin(), entryStatus.end(), status);
        return;
    }

    for (std::size_t index = 0; index < entryStatus.size(); ++index) {
        entryStatus[index] = ResultBufferEntryToStatus(opType, response.result_buffer[index]);
    }
}

QueryResult BuildQueryResultFromEntryStatus(const std::vector<Status>& entryStatus)
{
    QueryResult queryResult;
    queryResult.exists.assign(entryStatus.size(), 0);
    for (std::size_t index = 0; index < entryStatus.size(); ++index) {
        queryResult.exists[index] = entryStatus[index].code == StatusCode::ASU_ENTRY_KEY_EXIST;
        if (queryResult.prefixHitKeys == index && queryResult.exists[index] != 0) {
            ++queryResult.prefixHitKeys;
        }
    }
    return queryResult;
}

}  // namespace kv