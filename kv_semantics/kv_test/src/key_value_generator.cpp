#include "key_value_generator.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace UC::KVTest {

namespace {

constexpr int kExitInvalidArgument = 1;

}  // namespace

Status StringToCacheKey(const std::string& value, const std::string& source, UC::ASU::CacheKey& key)
{
    if (value.size() > key.size()) {
        return Status::Error(kExitInvalidArgument,
                             source + " key length exceeds " + std::to_string(key.size()) +
                                 " bytes: length=" + std::to_string(value.size()) +
                                 ", key=" + value);
    }
    key = UC::ASU::CacheKey{};
    if (!value.empty()) { std::memcpy(key.data(), value.data(), value.size()); }
    return Status::Success();
}

Status ValidateGeneratedData(const GeneratedData& data, const std::string& operation)
{
    if (data.keys.size() != data.values.size()) {
        return Status::Error(kExitInvalidArgument,
                             operation + " generated key/value count mismatch");
    }
    return Status::Success();
}

namespace {

constexpr std::uint64_t kFnvOffsetBasis64 = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime64 = 1099511628211ULL;
constexpr std::uint64_t kSplitMixIncrement = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t kCrc64EcmaPolynomial = 0x42F0E1EBA9EA3693ULL;

std::uint64_t HashByte(std::uint64_t hash, std::uint8_t value)
{
    hash ^= value;
    hash *= kFnvPrime64;
    return hash;
}

std::uint64_t HashUint64(std::uint64_t hash, std::uint64_t value)
{
    for (std::uint32_t index = 0; index < 8; ++index) {
        hash = HashByte(hash, static_cast<std::uint8_t>((value >> (index * 8)) & 0xFFU));
    }
    return hash;
}

std::uint64_t HashString(std::uint64_t hash, std::string_view value)
{
    for (const char byte : value) { hash = HashByte(hash, static_cast<std::uint8_t>(byte)); }
    return hash;
}

std::uint64_t SplitMix64Next(std::uint64_t& state)
{
    state += kSplitMixIncrement;
    std::uint64_t value = state;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

std::vector<std::uint8_t> GenerateValueBytes(const UC::ASU::CacheKey& key, std::uint64_t seed,
                                             std::uint64_t valueSize)
{
    std::vector<std::uint8_t> value(static_cast<std::size_t>(valueSize));
    std::uint64_t state =
        HashString(HashUint64(kFnvOffsetBasis64, seed), UC::ASU::CacheKeyView(key));
    for (std::size_t offset = 0; offset < value.size();) {
        const std::uint64_t random = SplitMix64Next(state);
        for (std::uint32_t byteIndex = 0; byteIndex < 8 && offset < value.size();
             ++byteIndex, ++offset) {
            value[offset] = static_cast<std::uint8_t>((random >> (byteIndex * 8)) & 0xFFU);
        }
    }
    return value;
}

std::string Trim(const std::string& value)
{
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) { return ""; }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

Status AddCommaSeparatedKeys(const std::string& value, const std::string& source,
                             std::vector<UC::ASU::CacheKey>& keys)
{
    std::string normalized = value;
    std::replace(normalized.begin(), normalized.end(), '\n', ',');
    std::replace(normalized.begin(), normalized.end(), '\r', ',');

    std::string item;
    std::stringstream stream(normalized);
    while (std::getline(stream, item, ',')) {
        item = Trim(item);
        if (item.empty()) {
            return Status::Error(kExitInvalidArgument, source + " contains an empty key");
        }
        UC::ASU::CacheKey key{};
        auto status = StringToCacheKey(item, source, key);
        if (!status.Ok()) { return status; }
        keys.push_back(key);
    }

    return Status::Success();
}

Status LoadKeysFile(const std::string& keysFile, std::vector<UC::ASU::CacheKey>& keys)
{
    std::ifstream input{keysFile};
    if (!input.is_open()) {
        return Status::Error(kExitInvalidArgument, "failed to open keys file: " + keysFile);
    }

    std::ostringstream content;
    content << input.rdbuf();
    auto status = AddCommaSeparatedKeys(content.str(), "--keys-file", keys);
    if (!status.Ok()) { return status; }
    if (keys.empty()) {
        return Status::Error(kExitInvalidArgument, "--keys-file does not contain any keys");
    }
    return Status::Success();
}

Status GenerateRangeKeys(const CommandOptions& options, std::vector<UC::ASU::CacheKey>& keys)
{
    if (!options.keyStartSet && !options.keyEndSet) { return Status::Success(); }

    const auto rangeSpan = options.keyEnd - options.keyStart;
    const auto maxCount = static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
    if (rangeSpan >= maxCount) {
        return Status::Error(kExitInvalidArgument, "key range exceeds addressable memory");
    }
    const auto rangeCount = rangeSpan + 1;

    keys.reserve(static_cast<std::size_t>(rangeCount));
    for (std::uint64_t index = options.keyStart; index <= options.keyEnd; ++index) {
        UC::ASU::CacheKey key{};
        auto status =
            StringToCacheKey(options.keyPrefix + std::to_string(index), "generated range", key);
        if (!status.Ok()) { return status; }
        keys.push_back(key);
        if (index == std::numeric_limits<std::uint64_t>::max()) { break; }
    }
    return Status::Success();
}

Status CheckValueMemoryLimit(std::uint64_t count, std::uint64_t valueSize,
                             std::uint64_t memoryMaxBytes)
{
    if (memoryMaxBytes == 0) {
        return Status::Error(kExitInvalidArgument,
                             "limits.memory_max_bytes must be greater than zero");
    }
    if (count == 0 || valueSize == 0) { return Status::Success(); }
    if (count > std::numeric_limits<std::uint64_t>::max() / valueSize) {
        return Status::Error(kExitInvalidArgument, "generated value bytes overflow uint64");
    }

    const auto requiredBytes = count * valueSize;
    if (requiredBytes > memoryMaxBytes) {
        return Status::Error(kExitInvalidArgument,
                             "generated value bytes exceed limits.memory_max_bytes: required=" +
                                 std::to_string(requiredBytes) +
                                 ", limit=" + std::to_string(memoryMaxBytes));
    }
    return Status::Success();
}

}  // namespace

Status KeyValueGenerator::Generate(const CommandOptions& options, const KvTestConfig& config,
                                   GeneratedData& data) const
{
    const std::uint64_t count = options.count == 0 ? config.count : options.count;
    const std::uint64_t seed = options.seed == 0 ? config.seed : options.seed;
    const std::uint64_t valueSize = options.valueSize == 0 ? config.valueSize : options.valueSize;

    if (valueSize > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return Status::Error(kExitInvalidArgument, "value-size exceeds addressable memory");
    }

    data.keys.clear();
    data.values.clear();

    if (!options.keys.empty()) {
        data.keys.reserve(options.keys.size());
        for (const auto& key : options.keys) {
            if (key.empty()) { return Status::Error(kExitInvalidArgument, "key cannot be empty"); }
            UC::ASU::CacheKey cacheKey{};
            auto status = StringToCacheKey(key, "--key/--keys", cacheKey);
            if (!status.Ok()) { return status; }
            data.keys.push_back(cacheKey);
        }
    } else if (!options.keysFile.empty()) {
        auto status = LoadKeysFile(options.keysFile, data.keys);
        if (!status.Ok()) { return status; }
    } else if (options.keyStartSet || options.keyEndSet) {
        auto status = GenerateRangeKeys(options, data.keys);
        if (!status.Ok()) { return status; }
    } else {
        if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            return Status::Error(kExitInvalidArgument, "count exceeds addressable memory");
        }
        data.keys.reserve(static_cast<std::size_t>(count));
        for (std::uint64_t index = 0; index < count; ++index) {
            UC::ASU::CacheKey key{};
            auto status =
                StringToCacheKey(config.keyPrefix + std::to_string(index), "generated config", key);
            if (!status.Ok()) { return status; }
            data.keys.push_back(key);
        }
    }

    if (options.command == CommandType::DELETE || options.command == CommandType::EXIST) {
        return Status::Success();
    }

    auto status = CheckValueMemoryLimit(static_cast<std::uint64_t>(data.keys.size()), valueSize,
                                        config.memoryMaxBytes);
    if (!status.Ok()) { return status; }

    data.values.reserve(data.keys.size());
    for (const auto& key : data.keys) {
        data.values.push_back(GenerateValueBytes(key, seed, valueSize));
    }
    return Status::Success();
}

Status KeyValueGenerator::Digest(const std::vector<std::uint8_t>& value, std::string& digest) const
{
    std::uint64_t crc = 0;
    for (const std::uint8_t byte : value) {
        crc ^= static_cast<std::uint64_t>(byte) << 56;
        for (std::uint32_t bit = 0; bit < 8; ++bit) {
            crc = (crc & 0x8000000000000000ULL) != 0 ? (crc << 1) ^ kCrc64EcmaPolynomial : crc << 1;
        }
    }

    std::ostringstream output;
    output << std::hex << std::nouppercase << std::setfill('0') << std::setw(16) << crc;
    digest = output.str();
    return Status::Success();
}

}  // namespace UC::KVTest
