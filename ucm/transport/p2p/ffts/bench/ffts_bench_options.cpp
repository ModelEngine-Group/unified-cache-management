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
#include "ffts_bench_options.h"

#include <cctype>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace UC::Transport::Ffts::Bench {
namespace {
size_t ParseBytes(const std::string& value)
{
    if (value.empty()) { throw std::invalid_argument("empty size value"); }

    std::string number = value;
    size_t multiplier = 1;
    const auto suffix = static_cast<char>(std::tolower(static_cast<unsigned char>(value.back())));
    if (suffix == 'k' || suffix == 'm' || suffix == 'g') {
        number = value.substr(0, value.size() - 1);
        multiplier = suffix == 'k' ? kKiB : (suffix == 'm' ? kMiB : kGiB);
    }

    size_t consumed = 0;
    const auto parsed = std::stoull(number, &consumed, 0);
    if (consumed != number.size()) { throw std::invalid_argument("invalid size value: " + value); }
    if (parsed > std::numeric_limits<size_t>::max() / multiplier) {
        throw std::overflow_error("size value overflows: " + value);
    }
    return static_cast<size_t>(parsed) * multiplier;
}

size_t ParseCount(const std::string& value)
{
    size_t consumed = 0;
    const auto parsed = std::stoull(value, &consumed, 0);
    if (consumed != value.size()) { throw std::invalid_argument("invalid count value: " + value); }
    return static_cast<size_t>(parsed);
}

std::vector<size_t> ParseList(const std::string& value, bool parseBytes)
{
    std::vector<size_t> result;
    size_t begin = 0;
    while (begin <= value.size()) {
        const auto end = value.find(',', begin);
        const auto token = value.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        if (token.empty()) { throw std::invalid_argument("empty list item: " + value); }
        result.push_back(parseBytes ? ParseBytes(token) : ParseCount(token));
        if (end == std::string::npos) { break; }
        begin = end + 1;
    }
    return result;
}

Scenario ParseScenario(const std::string& value)
{
    if (value == "all") { return Scenario::All; }
    if (value == "single") { return Scenario::Single; }
    if (value == "batch") { return Scenario::Batch; }
    throw std::invalid_argument("invalid scenario: " + value);
}

void PrintUsage(const char* program)
{
    std::cout << "Usage: " << program << " [options]\n"
              << "Options:\n"
              << "  --device N              Ascend device id, default 0\n"
              << "  --warmup N              Warmup iterations per case, default 10\n"
              << "  --iters N               Timed iterations per case, default 100\n"
              << "  --scenario NAME         all, single, or batch; default all\n"
              << "  --min-bytes SIZE        Minimum single-copy size, default 4K\n"
              << "  --max-bytes SIZE        Maximum single-copy size, default 256M\n"
              << "  --batch-counts LIST     Comma-separated batch counts, default 4,16,64,128\n"
              << "  --batch-chunks LIST     Comma-separated chunk sizes, default 4K,16K,64K,256K,1M\n"
              << "  --help                  Show this help text\n";
}
}  // namespace

Options ParseOptions(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto requireValue = [&](const char* name) {
            if (i + 1 >= argc) { throw std::invalid_argument(std::string(name) + " requires a value"); }
            return std::string(argv[++i]);
        };

        if (arg == "--help") {
            PrintUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--device") {
            options.deviceId = static_cast<int32_t>(ParseCount(requireValue("--device")));
        } else if (arg == "--warmup") {
            options.warmup = ParseCount(requireValue("--warmup"));
        } else if (arg == "--iters") {
            options.iterations = ParseCount(requireValue("--iters"));
        } else if (arg == "--scenario") {
            options.scenario = ParseScenario(requireValue("--scenario"));
        } else if (arg == "--min-bytes") {
            options.minBytes = ParseBytes(requireValue("--min-bytes"));
        } else if (arg == "--max-bytes") {
            options.maxBytes = ParseBytes(requireValue("--max-bytes"));
        } else if (arg == "--batch-counts") {
            options.batchCounts = ParseList(requireValue("--batch-counts"), false);
        } else if (arg == "--batch-chunks") {
            options.batchChunkBytes = ParseList(requireValue("--batch-chunks"), true);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (options.iterations == 0) { throw std::invalid_argument("--iters must be greater than 0"); }
    if (options.minBytes == 0 || options.maxBytes == 0 || options.minBytes > options.maxBytes) {
        throw std::invalid_argument("invalid min/max bytes");
    }
    return options;
}

}  // namespace UC::Transport::Ffts::Bench
