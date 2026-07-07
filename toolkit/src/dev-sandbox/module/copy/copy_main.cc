/**
 * MIT License
 *
 * Copyright (c) 2026 Mag1c.H
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
#include <charconv>
#include <fmt/format.h>
#include <unordered_set>
#include "copy_case.h"
#include "copy_runtime.h"

struct ArgsParser {
    std::unordered_set<std::string> names;
    CopyCase::Context ctx{.size = 512ull * 1024ull * 1024ull, .num = 8, .iter = 128, .nDevice = 8};

    static void Help(std::string_view proc)
    {
        fmt::println("Usage: {} [options]", proc);
        fmt::println("Options:");
        fmt::println("  -t <name>        Case name");
        fmt::println("  -s <size>        Data size in KB/MB (e.g., 4K, 16K, 1M, default: 512MB)");
        fmt::println("  -n <count>       Data number (default: 8)");
        fmt::println("  -i <count>       Iteration count (default: 128)");
        fmt::println("  -d <count>       Number of devices (default: 8)");
    }
    static std::size_t ParseUnsigned(std::string_view text, std::string_view errorMessage)
    {
        std::size_t value = 0;
        const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
        if (ec != std::errc() || ptr != text.data() + text.size()) {
            fmt::println("{}", errorMessage);
            std::exit(EXIT_FAILURE);
        }
        return value;
    }
    static std::size_t ParseSize(std::string_view sizeStr)
    {
        if (sizeStr.empty()) {
            fmt::println("Invalid size unit. Use K for KB or M for MB.");
            std::exit(EXIT_FAILURE);
        }
        const auto unit = sizeStr.back();
        sizeStr.remove_suffix(1);
        const auto value = ParseUnsigned(sizeStr, "Invalid size value.");
        switch (unit) {
            case 'K':
            case 'k': return value * 1024ull;
            case 'M':
            case 'm': return value * 1024ull * 1024ull;
            default:
                fmt::println("Invalid size unit. Use K for KB or M for MB.");
                std::exit(EXIT_FAILURE);
        }
    }
    ArgsParser(int argc, char const* argv[])
    {
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg{argv[i]};
            if (arg == "-t" && i + 1 < argc) {
                names.emplace(argv[++i]);
            } else if (arg == "-s" && i + 1 < argc) {
                ctx.size = ParseSize(argv[++i]);
            } else if (arg == "-n" && i + 1 < argc) {
                ctx.num = ParseUnsigned(argv[++i], "Invalid data count.");
            } else if (arg == "-i" && i + 1 < argc) {
                ctx.iter = ParseUnsigned(argv[++i], "Invalid iteration count.");
            } else if (arg == "-d" && i + 1 < argc) {
                ctx.nDevice = ParseUnsigned(argv[++i], "Invalid device count.");
            } else {
                Help(argv[0]);
                std::exit(EXIT_FAILURE);
            }
        }
    }
};

int main(int argc, char const* argv[])
{
    ArgsParser args{argc, argv};
    if (args.names.empty()) {
        ArgsParser::Help(argv[0]);
        return -1;
    }
    CopyRuntime runtime;
    const auto cases = CopyCaseFactory::Instance().Filter(args.names);
    if (cases.empty()) {
        for (auto& c : CopyCaseFactory::Instance().AllCases()) {
            fmt::println("{:<32}: {}", c->Key(), c->Brief());
        }
        return -1;
    }
    for (auto& c : cases) { c->Run(args.ctx); }
    return 0;
}
