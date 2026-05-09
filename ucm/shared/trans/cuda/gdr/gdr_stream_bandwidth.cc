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
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "trans/device.h"

namespace {

struct Args {
    int32_t deviceId{0};
    std::string nicName{};
    std::string direction{"both"};
    size_t minSize{4 * 1024};
    size_t maxSize{64 * 1024 * 1024};
    size_t targetBytes{512 * 1024 * 1024};
    size_t minIters{8};
    size_t maxIters{8192};
    size_t fixedIters{0};
    size_t warmup{2};
    size_t repeats{5};
};

void PrintUsage(const char* prog)
{
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --device-id N        CUDA visible device id, default 0\n"
              << "  --nic-name NAME      Set UCM_GDR_NIC_NAME before opening GDR stream\n"
              << "  --direction DIR      h2d, d2h, or both, default both\n"
              << "  --min-size SIZE      Default 4KB\n"
              << "  --max-size SIZE      Default 64MB\n"
              << "  --target-bytes SIZE  Total bytes per size when --iters is not set, default 512MB\n"
              << "  --min-iters N        Default 8\n"
              << "  --max-iters N        Default 8192\n"
              << "  --iters N            Fixed async copy count for each size\n"
              << "  --warmup N           Default 2\n"
              << "  --repeats N          Default 5\n";
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

size_t ParseSizeT(const char* value)
{
    const auto parsed = std::stoull(value);
    if (parsed == 0) { throw std::invalid_argument("numeric argument must be > 0"); }
    return static_cast<size_t>(parsed);
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
        } else if (key == "--target-bytes") {
            args.targetBytes = ParseSize(next());
        } else if (key == "--min-iters") {
            args.minIters = ParseSizeT(next());
        } else if (key == "--max-iters") {
            args.maxIters = ParseSizeT(next());
        } else if (key == "--iters") {
            args.fixedIters = ParseSizeT(next());
        } else if (key == "--warmup") {
            args.warmup = ParseSizeT(next());
        } else if (key == "--repeats") {
            args.repeats = ParseSizeT(next());
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

void CheckStatus(const UC::Status& status, const char* op)
{
    if (status.Failure()) { throw std::runtime_error(std::string(op) + ": " + status.ToString()); }
}

void CheckCuda(cudaError_t ret, const char* op)
{
    if (ret != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(ret));
    }
}

size_t ChooseIters(size_t size, const Args& args)
{
    if (args.fixedIters != 0) { return args.fixedIters; }
    const size_t byTarget = (args.targetBytes + size - 1) / size;
    return std::max(args.minIters, std::min(args.maxIters, byTarget));
}

std::vector<void*> MakeDevicePtrs(void* base, size_t size, size_t number)
{
    std::vector<void*> ptrs;
    ptrs.reserve(number);
    auto* bytes = static_cast<uint8_t*>(base);
    for (size_t i = 0; i < number; ++i) { ptrs.push_back(bytes + i * size); }
    return ptrs;
}

void IssueCopy(UC::Trans::Stream& stream, const std::string& direction, void* host,
               std::vector<void*>& devicePtrs, size_t size)
{
    if (direction == "h2d") {
        CheckStatus(stream.HostToDeviceAsync(host, devicePtrs.data(), size, devicePtrs.size()),
                    "HostToDeviceAsync");
        return;
    }
    CheckStatus(stream.DeviceToHostAsync(devicePtrs.data(), host, size, devicePtrs.size()),
                "DeviceToHostAsync");
}

struct Result {
    std::string direction;
    size_t size;
    size_t iters;
    size_t totalBytes;
    double bestMs;
    double avgMs;
    double bestGBs;
    double avgGBs;
};

Result RunOne(UC::Trans::Device& device, UC::Trans::Stream& stream, const std::string& direction,
              size_t size, size_t iters, const Args& args)
{
    const size_t totalBytes = size * iters;
    auto buffer = device.MakeBuffer();
    if (!buffer) { throw std::runtime_error("MakeBuffer returned nullptr"); }
    auto host = buffer->MakeHostBuffer(totalBytes);
    auto gpu = buffer->MakeDeviceBuffer(totalBytes);
    if (!host || !gpu) { throw std::runtime_error("failed to allocate host/device buffer"); }

    CheckCuda(cudaMemset(gpu.get(), 0, totalBytes), "cudaMemset");
    std::vector<void*> devicePtrs = MakeDevicePtrs(gpu.get(), size, iters);

    for (size_t i = 0; i < args.warmup; ++i) {
        IssueCopy(stream, direction, host.get(), devicePtrs, size);
        CheckStatus(stream.Synchronized(), "Synchronized(warmup)");
    }

    std::vector<double> elapsedMs;
    elapsedMs.reserve(args.repeats);
    for (size_t i = 0; i < args.repeats; ++i) {
        const auto start = std::chrono::steady_clock::now();
        IssueCopy(stream, direction, host.get(), devicePtrs, size);
        CheckStatus(stream.Synchronized(), "Synchronized");
        const auto end = std::chrono::steady_clock::now();
        elapsedMs.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    const auto bestIt = std::min_element(elapsedMs.begin(), elapsedMs.end());
    double sum = 0.0;
    for (double value : elapsedMs) { sum += value; }
    const double bestMs = *bestIt;
    const double avgMs = sum / static_cast<double>(elapsedMs.size());
    return Result{direction,
                  size,
                  iters,
                  totalBytes,
                  bestMs,
                  avgMs,
                  static_cast<double>(totalBytes) / (bestMs / 1000.0) / 1e9,
                  static_cast<double>(totalBytes) / (avgMs / 1000.0) / 1e9};
}

void PrintResult(const Result& result)
{
    std::cout << std::setw(3) << result.direction << " " << std::setw(8)
              << FormatSize(result.size) << " " << std::setw(8) << result.iters << " "
              << std::setw(9) << FormatSize(result.totalBytes) << " " << std::setw(11)
              << std::fixed << std::setprecision(3) << result.bestMs << " " << std::setw(11)
              << result.avgMs << " " << std::setw(12) << result.bestGBs << " " << std::setw(12)
              << result.avgGBs << "\n";
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Args args = ParseArgs(argc, argv);
        if (!args.nicName.empty()) {
            if (setenv("UCM_GDR_NIC_NAME", args.nicName.c_str(), 1) != 0) {
                throw std::runtime_error("failed to set UCM_GDR_NIC_NAME");
            }
        }

        UC::Trans::Device device;
        CheckStatus(device.Setup(args.deviceId), "Device::Setup");
        auto stream = device.MakeGdrStream();
        if (!stream) {
            throw std::runtime_error(
                "MakeGdrStream returned nullptr; build with UCM_ENABLE_GDR_STREAM=ON");
        }

        std::vector<std::string> directions;
        if (args.direction == "both") {
            directions = {"h2d", "d2h"};
        } else {
            directions = {args.direction};
        }

        std::cout << "device: " << args.deviceId << ", nic: "
                  << (args.nicName.empty() ? "<default>" : args.nicName) << "\n";
        std::cout << "dir     size    iters     total     best_ms      avg_ms    best_GB/s     avg_GB/s\n";
        for (size_t size = args.minSize; size <= args.maxSize; size *= 2) {
            const size_t iters = ChooseIters(size, args);
            for (const auto& direction : directions) {
                PrintResult(RunOne(device, *stream, direction, size, iters, args));
            }
            if (size > args.maxSize / 2) { break; }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
