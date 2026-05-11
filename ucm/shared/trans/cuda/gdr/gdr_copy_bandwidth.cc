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
#include "gdr_copy.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

namespace {

struct Args {
    int32_t deviceId{0};
    std::string nicName{"mlx5_0"};
    std::string direction{"both"};
    size_t minSize{4 * 1024};
    size_t maxSize{64 * 1024 * 1024};
    size_t sizeMultiplier{4};
    size_t h2dWarmup{100};
    size_t h2dIters{1000};
    size_t d2hWarmup{100};
    size_t d2hIters{1000};
};

struct DirectionBench {
    std::string name;
    bool h2d;
    GdrCopyKind kind;
    size_t warmup;
    size_t iters;
};

struct Result {
    size_t bytes;
    size_t warmup;
    size_t iters;
    double totalUs;
    double bwGBs;
};

double NowUs()
{
    using namespace std::chrono;
    return duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count() /
        1e3;
}

void PrintUsage(const char* prog)
{
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --device-id N          CUDA visible device id, default 0\n"
              << "  --nic-name NAME        RDMA NIC name, default mlx5_0\n"
              << "  --direction DIR        h2d, d2h, or both, default both\n"
              << "  --min-size SIZE        Default 4KB\n"
              << "  --max-size SIZE        Default 64MB\n"
              << "  --size-multiplier N    Size sweep multiplier, default 4\n"
              << "  --warmup N             Set both H2D and D2H warmup count, default 100\n"
              << "  --iters N              Set both H2D and D2H measured issue count, default 1000\n"
              << "  --h2d-warmup N         H2D warmup count\n"
              << "  --h2d-iters N          H2D measured issue count\n"
              << "  --d2h-warmup N         D2H warmup count\n"
              << "  --d2h-iters N          D2H measured issue count\n";
}

std::string ToLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

size_t ParseSize(std::string value)
{
    value = ToLower(std::move(value));
    std::string number;
    std::string suffix;
    for (char ch : value) {
        if (std::isdigit(static_cast<unsigned char>(ch)) || ch == '.') {
            number.push_back(ch);
        } else if (!std::isspace(static_cast<unsigned char>(ch))) {
            suffix.push_back(ch);
        }
    }
    if (number.empty()) { throw std::invalid_argument("invalid size: " + value); }

    double factor = 1.0;
    if (suffix.empty() || suffix == "b") {
        factor = 1.0;
    } else if (suffix == "k" || suffix == "kb" || suffix == "kib") {
        factor = 1024.0;
    } else if (suffix == "m" || suffix == "mb" || suffix == "mib") {
        factor = 1024.0 * 1024.0;
    } else if (suffix == "g" || suffix == "gb" || suffix == "gib") {
        factor = 1024.0 * 1024.0 * 1024.0;
    } else {
        throw std::invalid_argument("invalid size suffix: " + value);
    }
    return static_cast<size_t>(std::stod(number) * factor);
}

size_t ParseNonNegative(const char* value)
{
    return static_cast<size_t>(std::stoull(value));
}

size_t ParsePositive(const char* value)
{
    const size_t parsed = ParseNonNegative(value);
    if (parsed == 0) { throw std::invalid_argument("numeric argument must be > 0"); }
    return parsed;
}

Args ParseArgs(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { throw std::invalid_argument("missing value for " + key); }
            return argv[++i];
        };
        if (key == "--help" || key == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        } else if (key == "--device-id") {
            args.deviceId = static_cast<int32_t>(std::stoi(next()));
        } else if (key == "--nic-name") {
            args.nicName = next();
        } else if (key == "--direction") {
            args.direction = ToLower(next());
        } else if (key == "--min-size") {
            args.minSize = ParseSize(next());
        } else if (key == "--max-size") {
            args.maxSize = ParseSize(next());
        } else if (key == "--size-multiplier") {
            args.sizeMultiplier = ParsePositive(next());
        } else if (key == "--warmup") {
            const size_t warmup = ParseNonNegative(next());
            args.h2dWarmup = warmup;
            args.d2hWarmup = warmup;
        } else if (key == "--iters") {
            const size_t iters = ParsePositive(next());
            args.h2dIters = iters;
            args.d2hIters = iters;
        } else if (key == "--h2d-warmup") {
            args.h2dWarmup = ParseNonNegative(next());
        } else if (key == "--h2d-iters") {
            args.h2dIters = ParsePositive(next());
        } else if (key == "--d2h-warmup") {
            args.d2hWarmup = ParseNonNegative(next());
        } else if (key == "--d2h-iters") {
            args.d2hIters = ParsePositive(next());
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    if (args.direction != "h2d" && args.direction != "d2h" && args.direction != "both") {
        throw std::invalid_argument("--direction must be h2d, d2h, or both");
    }
    if (args.minSize == 0 || args.maxSize < args.minSize) {
        throw std::invalid_argument("invalid size range");
    }
    if (args.sizeMultiplier < 2) {
        throw std::invalid_argument("--size-multiplier must be >= 2");
    }
    return args;
}

std::string FormatSize(size_t size)
{
    constexpr size_t kKiB = 1024;
    constexpr size_t kMiB = 1024 * kKiB;
    constexpr size_t kGiB = 1024 * kMiB;
    if (size >= kGiB && size % kGiB == 0) { return std::to_string(size / kGiB) + "GiB"; }
    if (size >= kMiB && size % kMiB == 0) { return std::to_string(size / kMiB) + "MiB"; }
    if (size >= kKiB && size % kKiB == 0) { return std::to_string(size / kKiB) + "KiB"; }
    return std::to_string(size) + "B";
}

void CheckCuda(cudaError_t ret, const char* op)
{
    if (ret != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(ret));
    }
}

void CheckStatus(const UC::Status& status, const char* op)
{
    if (status.Failure()) { throw std::runtime_error(std::string(op) + ": " + status.ToString()); }
}

std::vector<size_t> BuildSizes(const Args& args)
{
    std::vector<size_t> sizes;
    for (size_t size = args.minSize; size <= args.maxSize; size *= args.sizeMultiplier) {
        sizes.push_back(size);
        if (size > args.maxSize / args.sizeMultiplier) { break; }
    }
    return sizes;
}

std::vector<DirectionBench> BuildDirectionBenches(const Args& args)
{
    std::vector<DirectionBench> benches;
    if (args.direction == "both" || args.direction == "h2d") {
        benches.push_back(
            DirectionBench{"Host->Device", true, GdrMemcpyHostToDevice, args.h2dWarmup,
                           args.h2dIters});
    }
    if (args.direction == "both" || args.direction == "d2h") {
        benches.push_back(
            DirectionBench{"Device->Host", false, GdrMemcpyDeviceToHost, args.d2hWarmup,
                           args.d2hIters});
    }
    return benches;
}

void DrainOneCompletion(GdrCopyChannel& channel, size_t& done)
{
    for (;;) {
        uint64_t reqId = 0;
        const auto poll = channel.PollCompletion(&reqId);
        if (poll == GdrCompletionPollResult::Completed) {
            ++done;
            return;
        }
        if (poll == GdrCompletionPollResult::Empty) {
            std::this_thread::yield();
            continue;
        }
        if (poll == GdrCompletionPollResult::UnknownRequest) {
            throw std::runtime_error("PollCompletion got an unknown request");
        }
        throw std::runtime_error("PollCompletion failed");
    }
}

void IssueOne(GdrCopyChannel& channel, const DirectionBench& bench, void* host, void* device,
              size_t bytes, size_t& issued, size_t& done)
{
    void* dst = bench.h2d ? device : host;
    const void* src = bench.h2d ? host : device;
    uint64_t reqId = 0;
    const int rc = channel.GdrMemcpyAsync(dst, src, bytes, bench.kind, &reqId);
    if (rc == 0) {
        ++issued;
        if (reqId == 0) { ++done; }
        return;
    }
    if (rc == -EAGAIN) {
        DrainOneCompletion(channel, done);
        return;
    }
    throw std::runtime_error("GdrMemcpyAsync failed(" + std::to_string(rc) + ")");
}

double RunBatch(GdrCopyChannel& channel, const DirectionBench& bench, void* host, void* device,
                size_t bytes, size_t count, bool measure)
{
    if (count == 0) { return 0.0; }

    size_t issued = 0;
    size_t done = 0;
    const double t0 = measure ? NowUs() : 0.0;
    while (issued < count) { IssueOne(channel, bench, host, device, bytes, issued, done); }
    while (done < count) { DrainOneCompletion(channel, done); }
    if (!measure) { return 0.0; }
    return NowUs() - t0;
}

Result RunOne(GdrCopyChannel& channel, const DirectionBench& bench, void* host, void* device,
              size_t bytes)
{
    RunBatch(channel, bench, host, device, bytes, bench.warmup, false);
    const double totalUs = RunBatch(channel, bench, host, device, bytes, bench.iters, true);
    const double totalBytes = static_cast<double>(bytes) * static_cast<double>(bench.iters);
    const double bwGBs = totalUs > 0.0 ? (totalBytes / 1e9) / (totalUs / 1e6) : 0.0;
    return Result{bytes, bench.warmup, bench.iters, totalUs, bwGBs};
}

void PrintTable(const DirectionBench& bench, const std::vector<Result>& results)
{
    std::cout << "\n--- " << bench.name << " GdrCopy API Bandwidth ---\n";
    std::cout << "size          warmup     iters       total      time_us     GB/s\n";
    std::cout << "------------  -------  --------  ----------  ----------  -------\n";
    for (const auto& row : results) {
        std::cout << std::left << std::setw(12) << FormatSize(row.bytes) << std::right
                  << "  " << std::setw(7) << row.warmup << "  " << std::setw(8)
                  << row.iters << "  " << std::setw(10)
                  << FormatSize(row.bytes * row.iters) << "  " << std::setw(10)
                  << std::fixed << std::setprecision(2) << row.totalUs << "  "
                  << std::setw(7) << std::setprecision(2) << row.bwGBs << "\n";
    }
}

}  // namespace

int main(int argc, char** argv)
{
    void* host = nullptr;
    void* device = nullptr;
    bool hostRegistered = false;
    bool deviceRegistered = false;
    try {
        const Args args = ParseArgs(argc, argv);

        CheckCuda(cudaSetDevice(args.deviceId), "cudaSetDevice");
        cudaDeviceProp prop{};
        CheckCuda(cudaGetDeviceProperties(&prop, args.deviceId), "cudaGetDeviceProperties");

        auto channel = GdrCopyLib::Open(args.deviceId, args.nicName);
        if (!channel) { throw std::runtime_error("GdrCopyLib::Open returned nullptr"); }

        CheckCuda(cudaHostAlloc(&host, args.maxSize, cudaHostAllocPortable), "cudaHostAlloc");
        CheckCuda(cudaMalloc(&device, args.maxSize), "cudaMalloc");
        std::memset(host, 0xA5, args.maxSize);
        CheckCuda(cudaMemset(device, 0x5A, args.maxSize), "cudaMemset");

        GdrCopyLib::RegisterHostBuffer(host, args.maxSize);
        hostRegistered = true;
        CheckStatus(GdrCopyLib::RegisterDeviceBuffer(device, args.maxSize),
                    "RegisterDeviceBuffer");
        deviceRegistered = true;

        const auto sizes = BuildSizes(args);
        const auto benches = BuildDirectionBenches(args);

        std::cout << "=================================================================\n";
        std::cout << "  UCM GdrCopy API Bandwidth  --  GPU " << args.deviceId << "  NIC "
                  << args.nicName << "\n";
        std::cout << "=================================================================\n";
        std::cout << "GPU: " << prop.name << "\n";
        std::cout << "Mode: direct GdrMemcpyAsync batch + completion drain\n";
        std::cout << "Buffer MR window: " << FormatSize(args.maxSize) << "\n";

        for (const auto& bench : benches) {
            std::vector<Result> results;
            results.reserve(sizes.size());
            for (size_t bytes : sizes) {
                results.push_back(RunOne(*channel, bench, host, device, bytes));
            }
            PrintTable(bench, results);
        }

        if (deviceRegistered) { GdrCopyLib::UnregisterDeviceBuffer(device); }
        if (hostRegistered) { GdrCopyLib::UnregisterHostBuffer(host); }
        cudaFree(device);
        cudaFreeHost(host);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        if (deviceRegistered) { GdrCopyLib::UnregisterDeviceBuffer(device); }
        if (hostRegistered) { GdrCopyLib::UnregisterHostBuffer(host); }
        if (device) { cudaFree(device); }
        if (host) { cudaFreeHost(host); }
        return 1;
    }
}
