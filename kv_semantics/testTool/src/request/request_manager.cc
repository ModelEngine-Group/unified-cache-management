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

#include "request/request_manager.h"
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include "kv_types.h"
#include "log/logger_manager.h"
#include "utils/file_util.h"

namespace {

// ---- token 级原子解析 ----

/** 十进制字符串解析为无符号 64 位整数；含非数字字符或溢出时返回 false */
bool ParseUint64(const std::string &text, std::uint64_t &value)
{
    if (text.empty() || text.size() > 20) { return false; }
    for (char c : text) {
        if (c < '0' || c > '9') { return false; }
    }
    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
    if (errno != 0 || end != text.c_str() + text.size()) { return false; }
    value = static_cast<std::uint64_t>(parsed);
    return true;
}

/** 十进制字符串解析为无符号 32 位整数（0 合法，offset 等字段可为 0）；含非数字字符或越界时返回
 * false */
bool ParseUint32(const std::string &text, std::uint32_t &value)
{
    std::uint64_t parsed = 0;
    if (!ParseUint64(text, parsed) || parsed > 0xFFFFFFFFULL) { return false; }
    value = static_cast<std::uint32_t>(parsed);
    return true;
}

/**
 * key 十进制字符串解析为 128 位无符号整数（GCC 内建类型，OS 层 Linux 编译豁免范围内）；
 * 含非数字字符、超过 39 位十进制或溢出时返回 false
 */
bool ParseKey128(const std::string &text, unsigned __int128 &value)
{
    if (text.empty() || text.size() > 39) { return false; }
    unsigned __int128 result = 0;
    for (char c : text) {
        if (c < '0' || c > '9') { return false; }
        const unsigned digit = static_cast<unsigned>(c - '0');
        // 溢出判定：result * 10 + digit 会超出 128 位上限
        if (result > (~static_cast<unsigned __int128>(0) - digit) / 10) { return false; }
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

// ---- 行工具 ----

/** 行首尾 trim（字符集为空格与 \t，readme 规则） */
std::string TrimLine(const std::string &line)
{
    const std::size_t begin = line.find_first_not_of(" \t");
    if (begin == std::string::npos) { return std::string(); }
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
        if (start == std::string::npos) { break; }
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

/**
 * 跳过 text 行首 tokenCount 个空白分隔 token，返回剩余原文（左侧空白已去）；
 * token 不足时返回空串。用于取"固定字段后的全部剩余内容"型字段（value 允许含空格）
 */
std::string TextAfterTokens(const std::string &text, std::size_t tokenCount)
{
    std::size_t pos = 0;
    for (std::size_t i = 0; i < tokenCount; ++i) {
        const std::size_t start = text.find_first_not_of(" \t", pos);
        if (start == std::string::npos) { return std::string(); }
        const std::size_t stop = text.find_first_of(" \t", start);
        if (stop == std::string::npos) { return std::string(); }
        pos = stop;
    }
    const std::size_t rest = text.find_first_not_of(" \t", pos);
    return (rest == std::string::npos) ? std::string() : text.substr(rest);
}

/** key 128 位整数的低 64 位分量 */
std::uint64_t KeyLow(const unsigned __int128 key) { return static_cast<std::uint64_t>(key); }

/** key 128 位整数右移 64 位后的低 16 位分量（bit64-79；testmode=1 的 value 生成规则使用） */
std::uint16_t KeyHigh16(const unsigned __int128 key)
{
    return static_cast<std::uint16_t>(key >> 64);
}

/** 组装行级错误信息：input line <N> <what>: <line> */
std::string LineErr(std::size_t lineNo, const std::string &what, const std::string &line)
{
    return "input line " + std::to_string(lineNo) + " " + what + ": " + line;
}

// ---- entry 组装与校验 ----

/**
 * 解析 key offset length 三 token 并填充 entry 的 key 分量 / offset / length / rawKey 与
 * value：value 取 textAfter 跳过 fixedCount 个 token 后的剩余原文（允许含空格，不足时置空）；
 * 失败返回 false
 */
bool FillEntry(const std::string &keyToken, const std::string &offsetToken,
               const std::string &lengthToken, const std::string &textAfter, std::size_t fixedCount,
               RequestOrc &out)
{
    unsigned __int128 key = 0;
    std::uint32_t offset = 0;
    std::uint32_t length = 0;
    if (!ParseKey128(keyToken, key) || !ParseUint32(offsetToken, offset) ||
        !ParseUint32(lengthToken, length)) {
        return false;
    }
    out.keyLow64 = KeyLow(key);
    out.keyHigh16 = KeyHigh16(key);
    out.offset = offset;
    out.length = length;
    out.rawKey = keyToken;
    out.value = TextAfterTokens(textAfter, fixedCount);
    return true;
}

/**
 * 解析单 key 行（fields 为 op 之后 token，textAfterOp 为 op 之后原文）：
 * 先试四整数（nsid key offset length [value...]），失败回退三整数（key offset length
 * [value...]，nsid 缺省 0）
 */
bool ParseSingleEntry(const std::vector<std::string> &fields, const std::string &textAfterOp,
                      RequestOrc &out)
{
    std::uint64_t nsid = 0;
    if (fields.size() >= 4 && ParseUint64(fields[0], nsid) &&
        FillEntry(fields[1], fields[2], fields[3], textAfterOp, 4, out)) {
        out.nsid = nsid;
        return true;
    }
    if (fields.size() >= 3 && FillEntry(fields[0], fields[1], fields[2], textAfterOp, 3, out)) {
        out.nsid = 0;
        return true;
    }
    return false;
}

/** 解析 batch 头字段（fields[0] 为 "batch"）：batch、batch nsid=x、batch nsid x 三种形态，nsid
 * 缺省 0 */
bool ParseBatchHeader(const std::vector<std::string> &fields, std::uint64_t &nsid)
{
    if (fields.size() == 1) {
        nsid = 0;
        return true;
    }
    if (fields.size() == 2 && fields[1].rfind("nsid=", 0) == 0) {
        return ParseUint64(fields[1].substr(5), nsid);
    }
    if (fields.size() == 3 && fields[1] == "nsid") { return ParseUint64(fields[2], nsid); }
    return false;
}

/** 解析 batch 内行（fields 为行 token，line 为行原文）：key offset length [value...]（nsid 与
 * op 由批头继承） */
bool ParseBatchEntry(const std::vector<std::string> &fields, const std::string &line,
                     RequestOrc &out)
{
    if (fields.size() < 3) { return false; }
    return FillEntry(fields[0], fields[1], fields[2], line, 3, out);
}

/**
 * entry 级 kv 协议约束前置校验（对齐协议层 BatchStore/BatchRetrieve 的 INVALID_ARGUMENT
 * 校验，让非法输入在启动期报错而非运行期以任务失败暴露）：key 低 64 位非全 0（CacheKey
 * 全 0 协议拒绝）、offset 与 length 512 字节对齐、length 非 0 且不超 24 位上限；
 * 违规返回 false 并经 reason 带回具体原因
 */
bool ValidateEntry(const RequestOrc &entry, std::string &reason)
{
    if (entry.keyLow64 == 0) {
        reason = "key low 64 bits are all zero (protocol rejects all-zero cache key)";
        return false;
    }
    if (entry.offset % kv::kAlignmentBytes != 0) {
        reason = "offset " + std::to_string(entry.offset) + " not 512B aligned";
        return false;
    }
    if (entry.length == 0) {
        reason = "length is zero";
        return false;
    }
    if (entry.length % kv::kAlignmentBytes != 0) {
        reason = "length " + std::to_string(entry.length) + " not 512B aligned";
        return false;
    }
    if (entry.length > 0xFFFFFF) {
        reason = "length " + std::to_string(entry.length) + " exceeds 24-bit limit 0xFFFFFF";
        return false;
    }
    return true;
}

}  // namespace

bool RequestManager::Load(const std::string &inputPath, std::string &err)
{
    std::vector<std::string> lines;
    if (!FileUtil::ReadLines(inputPath, lines, err)) {
        err = "input file read failed: " + err;
        return false;
    }

    steps_.clear();
    inBatch_ = false;
    batchTask_ = OrcTask{};

    for (std::size_t lineNo = 1; lineNo <= lines.size(); ++lineNo) {
        const std::string line = TrimLine(lines[lineNo - 1]);
        if (line.empty()) { continue; }
        if (!ProcessLine(lineNo, line, err)) { return false; }
    }

    // 文件结束批未闭合（缺 /batch）：批结构不完整属非法输入，报错退出提示管理员补全
    if (inBatch_) {
        err = "input batch not closed: missing /batch before end of file, " +
              std::to_string(batchTask_.entries.size()) + " entries pending";
        return false;
    }
    if (steps_.empty()) {
        err = "no valid step in input file: " + inputPath;
        return false;
    }
    LOG_INFO("RequestManager: {} steps parsed from {}", steps_.size(), inputPath);
    return true;
}

bool RequestManager::ProcessLine(std::size_t lineNo, const std::string &line, std::string &err)
{
    const std::vector<std::string> tokens = Tokenize(line);
    if (inBatch_) { return HandleBatchBody(lineNo, line, tokens, err); }
    return HandleTopLevel(lineNo, line, tokens, err);
}

bool RequestManager::HandleBatchBody(std::size_t lineNo, const std::string &line,
                                     const std::vector<std::string> &tokens, std::string &err)
{
    if (line == "/batch") {
        CloseBatch(lineNo);
        return true;
    }
    RequestOrc entry;
    if (!ParseBatchEntry(tokens, line, entry)) {
        err = LineErr(lineNo, "invalid batch entry", line);
        return false;
    }
    if (!CheckEntry(entry, lineNo, line, err)) { return false; }
    entry.op = batchTask_.op;
    entry.nsid = batchTask_.nsid;
    batchTask_.entries.emplace_back(entry);
    return true;
}

void RequestManager::CloseBatch(std::size_t lineNo)
{
    if (batchTask_.entries.empty()) {
        LOG_WARN("RequestManager: empty batch closed at line {} ignored", lineNo);
    } else {
        steps_.emplace_back(std::move(batchTask_));
    }
    batchTask_ = OrcTask{};
    inBatch_ = false;
}

bool RequestManager::HandleTopLevel(std::size_t lineNo, const std::string &line,
                                    const std::vector<std::string> &tokens, std::string &err)
{
    // sleep 行：独立暂停步骤（非 kv 任务）
    if (tokens[0] == "sleep") {
        std::uint32_t seconds = 0;
        if (tokens.size() != 2 || !ParseUint32(tokens[1], seconds)) {
            err = LineErr(lineNo, "invalid sleep", line);
            return false;
        }
        steps_.emplace_back(SleepStep{seconds});
        return true;
    }

    OrcOp op;
    if (!ParseOrcOp(tokens[0], op)) {
        err = LineErr(lineNo, "unknown op", line);
        return false;
    }

    // op 之后 token（batch 头判断与单 key 行解析共用）
    std::vector<std::string> fields(tokens.begin() + 1, tokens.end());
    if (!fields.empty() && fields[0] == "batch") {
        return BeginBatch(op, fields, lineNo, line, err);
    }
    return AppendSingleTask(op, fields, line, lineNo, err);
}

bool RequestManager::BeginBatch(OrcOp op, const std::vector<std::string> &fields,
                                std::size_t lineNo, const std::string &line, std::string &err)
{
    std::uint64_t nsid = 0;
    if (!ParseBatchHeader(fields, nsid)) {
        err = LineErr(lineNo, "invalid batch header", line);
        return false;
    }
    batchTask_ = OrcTask{};
    batchTask_.op = op;
    batchTask_.nsid = nsid;
    inBatch_ = true;
    return true;
}

bool RequestManager::AppendSingleTask(OrcOp op, const std::vector<std::string> &fields,
                                      const std::string &line, std::size_t lineNo, std::string &err)
{
    // value 取 op 之后固定字段外的剩余原文（传入原文供含空格 value 解析）
    RequestOrc entry;
    entry.op = op;
    if (!ParseSingleEntry(fields, TextAfterTokens(line, 1), entry)) {
        err = LineErr(lineNo, "invalid single entry", line);
        return false;
    }
    if (!CheckEntry(entry, lineNo, line, err)) { return false; }
    OrcTask task;
    task.op = op;
    task.nsid = entry.nsid;
    task.entries.emplace_back(entry);
    steps_.emplace_back(task);
    return true;
}

bool RequestManager::CheckEntry(const RequestOrc &entry, std::size_t lineNo,
                                const std::string &line, std::string &err)
{
    std::string reason;
    if (ValidateEntry(entry, reason)) { return true; }
    err = LineErr(lineNo, "violates kv protocol constraint (" + reason + ")", line);
    return false;
}

std::size_t RequestManager::StepCount() const { return steps_.size(); }

const FlowStep &RequestManager::GetStep(std::size_t index) const { return steps_[index]; }
