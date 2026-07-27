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
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include "drampool_config.h"
#include "parse_utils.h"

namespace UC::DramPool {
namespace {

// Must match the fixed slot layout used by BufferManager.
constexpr std::size_t kSlotAlignment = 64;

bool IsLongOption(const std::string& value)
{
    return value.size() > 2 && value.rfind("--", 0) == 0;
}

Status ReadSingleValue(const std::string& option, bool hasInlineValue,
                       const std::string& inlineValue, int argc, char** argv, int& index,
                       std::string& value)
{
    if (hasInlineValue) {
        if (inlineValue.empty()) { return Status::InvalidParam("{} requires a value", option); }
        value = inlineValue;
        return Status::OK();
    }

    if (++index >= argc || argv[index] == nullptr || IsLongOption(argv[index])) {
        return Status::InvalidParam("{} requires a value", option);
    }
    value = argv[index];
    if (value.empty()) { return Status::InvalidParam("{} requires a value", option); }
    return Status::OK();
}

Status ReadListValues(const std::string& option, bool hasInlineValue,
                      const std::string& inlineValue, int argc, char** argv, int& index,
                      std::vector<std::string>& values)
{
    values.clear();
    if (hasInlineValue) {
        if (inlineValue.empty()) {
            return Status::InvalidParam("{} requires at least one value", option);
        }
        values.emplace_back(inlineValue);
    }

    while (index + 1 < argc && argv[index + 1] != nullptr && !IsLongOption(argv[index + 1])) {
        const std::string value = argv[++index];
        if (value.empty()) { return Status::InvalidParam("{} has an empty value", option); }
        values.emplace_back(value);
    }
    if (values.empty()) { return Status::InvalidParam("{} requires at least one value", option); }
    return Status::OK();
}

Status ParseBlockSizes(const std::vector<std::string>& values, std::vector<std::uint64_t>& sizes)
{
    sizes.clear();
    sizes.reserve(values.size());
    for (const auto& value : values) {
        std::uint64_t size;
        if (Dram::ParseUint64(value, size).Failure()) {
            return Status::InvalidParam("invalid --kvcache-block-sizes value: {}", value);
        }
        sizes.push_back(size);
    }
    return Status::OK();
}

Status ParseBlockProportions(const std::vector<std::string>& values,
                             std::vector<std::uint32_t>& proportions)
{
    proportions.clear();
    proportions.reserve(values.size());
    for (const auto& value : values) {
        std::uint32_t proportion;
        if (Dram::ParseUint32(value, proportion).Failure()) {
            return Status::InvalidParam("invalid --kvcache-block-proportions value: {}", value);
        }
        proportions.push_back(proportion);
    }
    return Status::OK();
}

Status ValidateNics(const std::vector<std::string>& nics)
{
    for (const auto& nic : nics) {
        if (Dram::Trim(nic).empty()) {
            return Status::InvalidParam("--nics contains an empty name");
        }
    }
    return Status::OK();
}

Status ValidatePoolSize(std::uint64_t poolSizeGb)
{
    if (poolSizeGb == 0) {
        return Status::InvalidParam("--pool-size-gb must be greater than zero");
    }
    if (poolSizeGb > std::numeric_limits<std::uint64_t>::max() / kBytesPerGiB) {
        return Status::InvalidParam("--pool-size-gb is too large");
    }
    const auto totalBytes = poolSizeGb * kBytesPerGiB;
    if (static_cast<std::uint64_t>(static_cast<std::size_t>(totalBytes)) != totalBytes) {
        return Status::InvalidParam("--pool-size-gb exceeds addressable process memory");
    }
    return Status::OK();
}

Status ValidateBlockSizes(const std::vector<std::uint64_t>& blockSizes)
{
    for (const auto blockSize : blockSizes) {
        const auto slotCapacity = static_cast<std::size_t>(blockSize);
        if (blockSize == 0 || blockSize > std::numeric_limits<std::uint32_t>::max() ||
            static_cast<std::uint64_t>(slotCapacity) != blockSize) {
            return Status::InvalidParam("--kvcache-block-sizes item is unsupported");
        }
        if (slotCapacity > std::numeric_limits<std::size_t>::max() - (kSlotAlignment - 1)) {
            return Status::InvalidParam("--kvcache-block-sizes item overflows slot alignment");
        }
    }
    return Status::OK();
}

Status ValidateBlockProportions(const std::vector<std::uint32_t>& proportions)
{
    std::uint64_t totalProportion = 0;
    for (const auto proportion : proportions) {
        if (proportion == 0 ||
            totalProportion > std::numeric_limits<std::uint32_t>::max() - proportion) {
            return Status::InvalidParam("sum of --kvcache-block-proportions must be in [1, {}]",
                                        std::numeric_limits<std::uint32_t>::max());
        }
        totalProportion += proportion;
    }
    return Status::OK();
}

Status ValidateBlockClassCount(const std::vector<std::uint64_t>& blockSizes,
                               const std::vector<std::uint32_t>& proportions)
{
    if (blockSizes.size() != proportions.size()) {
        return Status::InvalidParam(
            "--kvcache-block-sizes and --kvcache-block-proportions must have the same length");
    }
    return Status::OK();
}

Status CalculatePoolSlotCounts(DramPoolConfig& config)
{
    config.poolSlotCounts.clear();
    std::uint64_t totalProportion = 0;
    for (const auto proportion : config.poolBlockProportions) { totalProportion += proportion; }

    const auto totalBytes = config.poolSizeGb * kBytesPerGiB;
    std::vector<std::uint32_t> slotCounts;
    slotCounts.reserve(config.poolBlockSizes.size());
    const auto wholeShare = totalBytes / totalProportion;
    const auto remainder = totalBytes % totalProportion;
    for (std::size_t index = 0; index < config.poolBlockSizes.size(); ++index) {
        const auto slotCapacity = static_cast<std::size_t>(config.poolBlockSizes[index]);
        const auto slotStride =
            (slotCapacity + kSlotAlignment - 1) / kSlotAlignment * kSlotAlignment;
        const auto proportion = static_cast<std::uint64_t>(config.poolBlockProportions[index]);
        // totalProportion is bounded to uint32_t, so this remainder product cannot overflow.
        const auto classBytes = wholeShare * proportion + remainder * proportion / totalProportion;
        const auto slotCount = classBytes / slotStride;
        if (slotCount == 0 || slotCount >= std::numeric_limits<std::uint32_t>::max()) {
            return Status::InvalidParam(
                "pool capacity for --kvcache-block-sizes item {} cannot form a valid slot pool",
                index);
        }
        slotCounts.push_back(static_cast<std::uint32_t>(slotCount));
    }

    config.poolSlotCounts = std::move(slotCounts);
    return Status::OK();
}

}  // namespace

Status ParseDramPoolEndpoint(const std::string& name, const std::string& value,
                             transport::Endpoint& endpoint)
{
    const auto normalized = Dram::Trim(value);
    if (normalized.empty()) { return Status::InvalidParam("{} is required", name); }

    const auto separator = normalized.rfind(':');
    if (separator == std::string::npos || separator == 0 || separator + 1 >= normalized.size()) {
        return Status::InvalidParam("{} must be <IP>:<PORT>", name);
    }
    std::uint16_t port;
    if (Dram::ParseUint16(normalized.substr(separator + 1), port).Failure() || port == 0) {
        return Status::InvalidParam("{} must have a valid port", name);
    }
    endpoint.host = normalized.substr(0, separator);
    endpoint.port = port;
    return Status::OK();
}

std::string BuildUsage(const char* program)
{
    const std::string name = program == nullptr ? "drampool" : program;
    return "Usage: " + name +
           " --addr <IP>:<PORT> --nics <NAME>... --pool-size-gb <SIZE>"
           " --kvcache-block-sizes <SIZE>... [options]\n"
           "Required options:\n"
           "  --addr <IP>:<PORT>                 DramPool service address for KV control "
           "messages.\n"
           "  --nics <NAME>...                   RDMA NIC names for pinned memory registration.\n"
           "  --pool-size-gb <SIZE>              Local DRAM capacity contributed to the pool.\n"
           "  --kvcache-block-sizes <SIZE>...    Supported fixed KVCache block sizes.\n"
           "Optional options:\n"
           "  --config <PATH>                    Runtime YAML path; defaults to " +
           kDefaultDramPoolRuntimeConfigPath +
           ".\n"
           "  --kvcache-block-proportions <P>... Relative capacity per block size; defaults to "
           "1:1.\n"
           "  --ttl-minutes <MINUTES>            Absolute block lifetime; defaults to 120.\n";
}

Status ParseCommandLine(int argc, char** argv, DramPoolConfig& config)
{
    config = DramPoolConfig{};

    bool hasAddr = false;
    bool hasNics = false;
    bool hasPoolSize = false;
    bool hasBlockSizes = false;
    bool hasBlockProportions = false;
    bool hasTtl = false;
    bool hasConfig = false;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index] == nullptr ? "" : argv[index];
        if (!IsLongOption(argument)) {
            return Status::InvalidParam("unexpected argument: {}", argument);
        }

        const auto equals = argument.find('=');
        const std::string option = argument.substr(0, equals);
        const bool hasInlineValue = equals != std::string::npos;
        const std::string inlineValue =
            hasInlineValue ? argument.substr(equals + 1) : std::string{};
        std::string value;
        std::vector<std::string> values;
        auto status = Status::OK();

        if (option == "--config") {
            if (hasConfig) { return Status::InvalidParam("--config may be specified once"); }
            status = ReadSingleValue(option, hasInlineValue, inlineValue, argc, argv, index, value);
            if (status.Failure()) { return status; }
            if (Dram::Trim(value).empty()) {
                return Status::InvalidParam("--config must not be blank");
            }
            config.runtimeConfigPath = std::move(value);
            hasConfig = true;
            continue;
        }
        if (option == "--addr") {
            if (hasAddr) { return Status::InvalidParam("--addr may be specified once"); }
            status = ReadSingleValue(option, hasInlineValue, inlineValue, argc, argv, index, value);
            if (status.Failure()) { return status; }
            transport::Endpoint endpoint;
            status = ParseDramPoolEndpoint("--addr", value, endpoint);
            if (status.Failure()) { return status; }
            config.addr = std::move(endpoint);
            hasAddr = true;
            continue;
        }
        if (option == "--nics") {
            if (hasNics) { return Status::InvalidParam("--nics may be specified once"); }
            status = ReadListValues(option, hasInlineValue, inlineValue, argc, argv, index, values);
            if (status.Failure()) { return status; }
            config.nics = std::move(values);
            status = ValidateNics(config.nics);
            if (status.Failure()) { return status; }
            hasNics = true;
            continue;
        }
        if (option == "--pool-size-gb") {
            if (hasPoolSize) {
                return Status::InvalidParam("--pool-size-gb may be specified once");
            }
            status = ReadSingleValue(option, hasInlineValue, inlineValue, argc, argv, index, value);
            if (status.Failure()) { return status; }
            if (Dram::ParseUint64(value, config.poolSizeGb).Failure()) {
                return Status::InvalidParam("invalid {} value: {}", option, value);
            }
            status = ValidatePoolSize(config.poolSizeGb);
            if (status.Failure()) { return status; }
            hasPoolSize = true;
            continue;
        }
        if (option == "--kvcache-block-sizes") {
            if (hasBlockSizes) {
                return Status::InvalidParam("--kvcache-block-sizes may be specified once");
            }
            status = ReadListValues(option, hasInlineValue, inlineValue, argc, argv, index, values);
            if (status.Failure()) { return status; }
            status = ParseBlockSizes(values, config.poolBlockSizes);
            if (status.Failure()) { return status; }
            status = ValidateBlockSizes(config.poolBlockSizes);
            if (status.Failure()) { return status; }
            hasBlockSizes = true;
            continue;
        }
        if (option == "--kvcache-block-proportions") {
            if (hasBlockProportions) {
                return Status::InvalidParam("--kvcache-block-proportions may be specified once");
            }
            status = ReadListValues(option, hasInlineValue, inlineValue, argc, argv, index, values);
            if (status.Failure()) { return status; }
            status = ParseBlockProportions(values, config.poolBlockProportions);
            if (status.Failure()) { return status; }
            status = ValidateBlockProportions(config.poolBlockProportions);
            if (status.Failure()) { return status; }
            hasBlockProportions = true;
            continue;
        }
        if (option == "--ttl-minutes") {
            if (hasTtl) { return Status::InvalidParam("--ttl-minutes may be specified once"); }
            status = ReadSingleValue(option, hasInlineValue, inlineValue, argc, argv, index, value);
            if (status.Failure()) { return status; }
            std::uint64_t ttlMinutes;
            if (Dram::ParseUint64(value, ttlMinutes).Failure()) {
                return Status::InvalidParam("invalid {} value: {}", option, value);
            }
            if (ttlMinutes == 0 || ttlMinutes > std::numeric_limits<std::uint64_t>::max() /
                                                    kMillisecondsPerMinute) {
                return Status::InvalidParam("--ttl-minutes must be a positive value");
            }
            config.defaultDumpTtlMs = ttlMinutes * kMillisecondsPerMinute;
            hasTtl = true;
            continue;
        }
        return Status::InvalidParam("unknown argument: {}", argument);
    }

    if (!hasAddr || !hasNics || !hasPoolSize || !hasBlockSizes) {
        return Status::InvalidParam(
            "--addr, --nics, --pool-size-gb, and --kvcache-block-sizes are required");
    }
    if (!hasBlockProportions) {
        // Each configured block size receives an equal capacity share by default.
        config.poolBlockProportions.assign(config.poolBlockSizes.size(), 1);
    }
    if (const auto status =
            ValidateBlockClassCount(config.poolBlockSizes, config.poolBlockProportions);
        status.Failure()) {
        return status;
    }
    // Options are order-independent; all pool inputs are complete only here.
    if (const auto status = CalculatePoolSlotCounts(config); status.Failure()) { return status; }
    return Status::OK();
}

}  // namespace UC::DramPool
