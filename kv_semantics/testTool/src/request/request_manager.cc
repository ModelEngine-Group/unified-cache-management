/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "request/request_manager.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "log/logger_manager.h"
#include "utils/file_util.h"

namespace {

/** 十进制字符串解析为无符号 64 位整数；含非数字字符或溢出时返回 false */
bool ParseUint64(const std::string &text, std::uint64_t &value)
{
    if (text.empty() || text.size() > 20) {
        return false;
    }
    for (char c : text) {
        if (c < '0' || c > '9') {
            return false;
        }
    }
    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
    if (errno != 0 || end != text.c_str() + text.size()) {
        return false;
    }
    value = static_cast<std::uint64_t>(parsed);
    return true;
}

/** 十进制字符串解析为无符号 32 位整数（0 合法，offset 等字段可为 0）；含非数字字符或越界时返回 false */
bool ParseUint32(const std::string &text, std::uint32_t &value)
{
    std::uint64_t parsed = 0;
    if (!ParseUint64(text, parsed) || parsed > 0xFFFFFFFFULL) {
        return false;
    }
    value = static_cast<std::uint32_t>(parsed);
    return true;
}

/**
 * key 十进制字符串解析为 128 位无符号整数（GCC 内建类型，OS 层 Linux 编译豁免范围内）；
 * 含非数字字符、超过 39 位十进制或溢出时返回 false
 */
bool ParseKey128(const std::string &text, unsigned __int128 &value)
{
    if (text.empty() || text.size() > 39) {
        return false;
    }
    unsigned __int128 result = 0;
    for (char c : text) {
        if (c < '0' || c > '9') {
            return false;
        }
        const unsigned digit = static_cast<unsigned>(c - '0');
        // 溢出判定：result * 10 + digit 会超出 128 位上限
        if (result > (~static_cast<unsigned __int128>(0) - digit) / 10) {
            return false;
        }
        result = result * 10 + digit;
    }
    value = result;
    return true;
}

/** optype 文本到枚举（大小写敏感，get/put/exist/delete）；不识别返回 false */
bool ParseOrcOp(const std::string &text, OrcOp &op)
{
    if (text == "get") {
        op = OrcOp::GET;
    } else if (text == "put") {
        op = OrcOp::PUT;
    } else if (text == "exist") {
        op = OrcOp::EXIST;
    } else if (text == "delete") {
        op = OrcOp::DELETE;
    } else {
        return false;
    }
    return true;
}

/** 行首尾 trim（字符集为空格与 \t，readme 规则） */
std::string TrimLine(const std::string &line)
{
    const std::size_t begin = line.find_first_not_of(" \t");
    if (begin == std::string::npos) {
        return std::string();
    }
    const std::size_t end = line.find_last_not_of(" \t");
    return line.substr(begin, end - begin + 1);
}

/** 空白分词（空格与 \t） */
std::vector<std::string> Tokenize(const std::string &line)
{
    std::vector<std::string> tokens;
    std::size_t pos = 0;
    while (pos < line.size()) {
        const std::size_t start = line.find_first_not_of(" \t", pos);
        if (start == std::string::npos) {
            break;
        }
        const std::size_t stop = line.find_first_of(" \t", start);
        if (stop == std::string::npos) {
            tokens.emplace_back(line.substr(start));
            break;
        }
        tokens.emplace_back(line.substr(start, stop - start));
        pos = stop + 1;
    }
    return tokens;
}

/** key 128 位整数的低 64 位分量 */
std::uint64_t KeyLow(const unsigned __int128 key)
{
    return static_cast<std::uint64_t>(key);
}

/** key 128 位整数右移 64 位后的低 16 位分量（bit64-79；testmode=1 的 value 生成规则使用） */
std::uint16_t KeyHigh16(const unsigned __int128 key)
{
    return static_cast<std::uint16_t>(key >> 64);
}

/**
 * 解析单 key 行（op 已识别，fields 为 op 之后的全部字段）：
 * 先试四整数（nsid key offset length [value]），再试三整数（key offset length [value]，nsid 缺省 0）；
 * length 须为正；失败返回 false
 */
bool ParseSingleEntry(const std::vector<std::string> &fields, RequestOrc &out)
{
    if (fields.size() >= 4 && fields.size() <= 5) {
        std::uint64_t nsid = 0;
        std::uint32_t offset = 0;
        std::uint32_t length = 0;
        unsigned __int128 key = 0;
        if (ParseUint64(fields[0], nsid) && ParseKey128(fields[1], key) &&
            ParseUint32(fields[2], offset) && ParseUint32(fields[3], length) && length != 0) {
            out.nsid = nsid;
            out.keyLow64 = KeyLow(key);
            out.keyHigh16 = KeyHigh16(key);
            out.offset = offset;
            out.length = length;
            if (fields.size() == 5) {
                out.value = fields[4];
            }
            return true;
        }
    }
    if (fields.size() >= 3 && fields.size() <= 4) {
        std::uint32_t offset = 0;
        std::uint32_t length = 0;
        unsigned __int128 key = 0;
        if (ParseKey128(fields[0], key) && ParseUint32(fields[1], offset) &&
            ParseUint32(fields[2], length) && length != 0) {
            out.nsid = 0;
            out.keyLow64 = KeyLow(key);
            out.keyHigh16 = KeyHigh16(key);
            out.offset = offset;
            out.length = length;
            if (fields.size() == 4) {
                out.value = fields[3];
            }
            return true;
        }
    }
    return false;
}

/**
 * 解析 batch 头字段（op 已识别，fields[0] 为 "batch"）：batch、batch nsid=x、batch nsid x
 * 三种形态；nsid 可省略，缺省 0；失败返回 false
 */
bool ParseBatchHeader(const std::vector<std::string> &fields, std::uint64_t &nsid)
{
    if (fields.size() == 1) {
        nsid = 0;
        return true;
    }
    if (fields.size() == 2 && fields[1].rfind("nsid=", 0) == 0) {
        return ParseUint64(fields[1].substr(5), nsid);
    }
    if (fields.size() == 3 && fields[1] == "nsid") {
        return ParseUint64(fields[2], nsid);
    }
    return false;
}

/** 解析 batch 内行：key offset length [value]（nsid 与 op 由批头继承）；length 须为正；失败返回 false */
bool ParseBatchEntry(const std::vector<std::string> &fields, RequestOrc &out)
{
    if (fields.size() < 3 || fields.size() > 4) {
        return false;
    }
    std::uint32_t offset = 0;
    std::uint32_t length = 0;
    unsigned __int128 key = 0;
    if (!ParseKey128(fields[0], key) || !ParseUint32(fields[1], offset) ||
        !ParseUint32(fields[2], length) || length == 0) {
        return false;
    }
    out.keyLow64 = KeyLow(key);
    out.keyHigh16 = KeyHigh16(key);
    out.offset = offset;
    out.length = length;
    if (fields.size() == 4) {
        out.value = fields[3];
    }
    return true;
}

} // namespace

bool RequestManager::Load(const std::string &inputPath, std::string &err)
{
    std::vector<std::string> lines;
    if (!FileUtil::ReadLines(inputPath, lines, err)) {
        err = "input file read failed: " + err;
        return false;
    }

    steps_.clear();
    bool inBatch = false;
    OrcTask batchTask;

    for (std::size_t lineNo = 1; lineNo <= lines.size(); ++lineNo) {
        const std::string line = TrimLine(lines[lineNo - 1]);
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> tokens = Tokenize(line);

        // batch 内：/batch 闭合，其余为批内行
        if (inBatch) {
            if (line == "/batch") {
                if (!batchTask.entries.empty()) {
                    steps_.emplace_back(std::move(batchTask));
                } else {
                    LOG_WARN("RequestManager: empty batch at line {} ignored", lineNo);
                }
                batchTask = OrcTask{};
                inBatch = false;
                continue;
            }
            RequestOrc entry;
            if (!ParseBatchEntry(tokens, entry)) {
                LOG_WARN("RequestManager: skip invalid batch entry at line {}: {}", lineNo, line);
                continue;
            }
            entry.op = batchTask.op;
            entry.nsid = batchTask.nsid;
            batchTask.entries.emplace_back(entry);
            continue;
        }

        // sleep 行：独立暂停步骤（非 kv 任务，不带任务包装）
        if (tokens[0] == "sleep") {
            std::uint32_t seconds = 0;
            if (tokens.size() != 2 || !ParseUint32(tokens[1], seconds)) {
                LOG_WARN("RequestManager: skip invalid sleep at line {}: {}", lineNo, line);
                continue;
            }
            steps_.emplace_back(SleepStep{seconds});
            continue;
        }

        // 行首操作符识别
        OrcOp op;
        if (!ParseOrcOp(tokens[0], op)) {
            LOG_WARN("RequestManager: skip unknown op at line {}: {}", lineNo, line);
            continue;
        }

        // batch 头（op 后接 batch）：开始聚合
        std::vector<std::string> fields(tokens.begin() + 1, tokens.end());
        if (!fields.empty() && fields[0] == "batch") {
            std::uint64_t nsid = 0;
            if (!ParseBatchHeader(fields, nsid)) {
                LOG_WARN("RequestManager: skip invalid batch header at line {}: {}", lineNo, line);
                continue;
            }
            batchTask = OrcTask{};
            batchTask.op = op;
            batchTask.nsid = nsid;
            inBatch = true;
            continue;
        }

        // 单 key 行：一个独立任务
        RequestOrc entry;
        entry.op = op;
        if (!ParseSingleEntry(fields, entry)) {
            LOG_WARN("RequestManager: skip invalid single entry at line {}: {}", lineNo, line);
            continue;
        }
        OrcTask task;
        task.op = op;
        task.nsid = entry.nsid;
        task.entries.emplace_back(entry);
        steps_.emplace_back(task);
    }

    // 文件结束时批未闭合：警告并将已有内容保留执行
    if (inBatch) {
        LOG_WARN("RequestManager: batch not closed at end of file, {} entries kept",
                 batchTask.entries.size());
        if (!batchTask.entries.empty()) {
            steps_.emplace_back(std::move(batchTask));
        }
    }

    if (steps_.empty()) {
        err = "no valid step in input file: " + inputPath;
        return false;
    }
    LOG_INFO("RequestManager: {} steps parsed from {}", steps_.size(), inputPath);
    return true;
}

std::size_t RequestManager::StepCount() const
{
    return steps_.size();
}

const FlowStep &RequestManager::GetStep(std::size_t index) const
{
    return steps_[index];
}
