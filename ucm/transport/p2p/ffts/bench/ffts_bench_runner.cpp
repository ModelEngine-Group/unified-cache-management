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
#include "ffts_bench_runner.h"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <acl/acl.h>

#include "ffts_bench_runtime.h"
#include "ffts_transport.h"

namespace UC::Transport::Ffts::Bench {
namespace {
constexpr size_t kOutputWidth = 14;

void FillPattern(std::vector<uint8_t>& data)
{
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>((i * 131ULL + 17ULL) & 0xFFU);
    }
}

void* Offset(void* ptr, size_t bytes)
{
    return static_cast<void*>(static_cast<uint8_t*>(ptr) + bytes);
}

const void* Offset(const void* ptr, size_t bytes)
{
    return static_cast<const void*>(static_cast<const uint8_t*>(ptr) + bytes);
}

std::vector<UC::Transport::Ffts::CopyDesc> BuildCopies(void* dst, const void* src, size_t chunkBytes, size_t count)
{
    std::vector<UC::Transport::Ffts::CopyDesc> copies;
    copies.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const auto offset = i * chunkBytes;
        copies.push_back({Offset(dst, offset), Offset(src, offset), chunkBytes});
    }
    return copies;
}

void PrepareBuffers(DeviceBuffer& src, DeviceBuffer& dst, size_t bytes)
{
    std::vector<uint8_t> host(bytes);
    FillPattern(host);
    CheckAcl(aclrtMemcpy(src.Get(), src.Size(), host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE),
             "aclrtMemcpy H2D");
    CheckAcl(aclrtMemset(dst.Get(), dst.Size(), 0, bytes), "aclrtMemset");
}

void VerifyBuffer(DeviceBuffer& dst, size_t bytes)
{
    std::vector<uint8_t> expected(bytes);
    std::vector<uint8_t> actual(bytes, 0);
    FillPattern(expected);
    CheckAcl(aclrtMemcpy(actual.data(), bytes, dst.Get(), bytes, ACL_MEMCPY_DEVICE_TO_HOST),
             "aclrtMemcpy D2H");
    if (actual != expected) { throw std::runtime_error("copy verification failed"); }
}

template <typename Func>
Samples Measure(size_t warmup, size_t iterations, Func&& func)
{
    for (size_t i = 0; i < warmup; ++i) {
        func();
    }

    Samples samples;
    for (size_t i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        func();
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double, std::micro> elapsed = end - start;
        samples.Add(elapsed.count());
    }
    return samples;
}

void RunFftsCopy(UC::Transport::Ffts::FftsTransport& transport, void* dst, const void* src, size_t bytes)
{
    CheckStatus(transport.CopyAsync(dst, src, bytes), "FftsTransport::CopyAsync");
    CheckStatus(transport.Synchronize(), "FftsTransport::Synchronize");
}

void RunFftsBatch(UC::Transport::Ffts::FftsTransport& transport,
                  const std::vector<UC::Transport::Ffts::CopyDesc>& copies)
{
    CheckStatus(transport.Submit(copies), "FftsTransport::Submit");
    CheckStatus(transport.Synchronize(), "FftsTransport::Synchronize");
}

void RunAclCopy(aclrtStream stream, void* dst, const void* src, size_t bytes)
{
    CheckAcl(aclrtMemcpyAsync(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
             "aclrtMemcpyAsync D2D");
    CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
}

void RunAclBatch(aclrtStream stream, const std::vector<UC::Transport::Ffts::CopyDesc>& copies)
{
    for (const auto& copy : copies) {
        CheckAcl(aclrtMemcpyAsync(copy.dst, copy.size, copy.src, copy.size, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                 "aclrtMemcpyAsync D2D");
    }
    CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
}

void PrintHeader()
{
    std::cout << std::left << std::setw(10) << "scenario" << std::setw(10) << "method" << std::right
              << std::setw(kOutputWidth) << "bytes" << std::setw(kOutputWidth) << "chunk"
              << std::setw(kOutputWidth) << "count" << std::setw(kOutputWidth) << "avg_us"
              << std::setw(kOutputWidth) << "min_us" << std::setw(kOutputWidth) << "p50_us"
              << std::setw(kOutputWidth) << "GB/s" << '\n';
}

void PrintResult(const std::string& scenario, const std::string& method, size_t bytes, size_t chunkBytes,
                 size_t count, const Samples& samples)
{
    const auto avgUs = samples.Average();
    const auto gbps = avgUs == 0.0 ? 0.0 : (static_cast<double>(bytes) / (avgUs / 1'000'000.0)) / 1'000'000'000.0;
    std::cout << std::left << std::setw(10) << scenario << std::setw(10) << method << std::right
              << std::setw(kOutputWidth) << bytes << std::setw(kOutputWidth) << chunkBytes
              << std::setw(kOutputWidth) << count << std::setw(kOutputWidth) << std::fixed
              << std::setprecision(2) << avgUs << std::setw(kOutputWidth) << samples.Min()
              << std::setw(kOutputWidth) << samples.Median() << std::setw(kOutputWidth) << gbps << '\n';
}

void RunSingleCase(UC::Transport::Ffts::FftsTransport& transport, aclrtStream stream, size_t bytes,
                   const Options& options)
{
    DeviceBuffer src(bytes);
    DeviceBuffer dst(bytes);
    PrepareBuffers(src, dst, bytes);

    RunFftsCopy(transport, dst.Get(), src.Get(), bytes);
    VerifyBuffer(dst, bytes);
    CheckAcl(aclrtMemset(dst.Get(), dst.Size(), 0, bytes), "aclrtMemset");

    RunAclCopy(stream, dst.Get(), src.Get(), bytes);
    VerifyBuffer(dst, bytes);
    CheckAcl(aclrtMemset(dst.Get(), dst.Size(), 0, bytes), "aclrtMemset");

    const auto ffts = Measure(options.warmup, options.iterations, [&]() {
        RunFftsCopy(transport, dst.Get(), src.Get(), bytes);
    });
    const auto acl = Measure(options.warmup, options.iterations, [&]() {
        RunAclCopy(stream, dst.Get(), src.Get(), bytes);
    });

    PrintResult("single", "ffts", bytes, bytes, 1, ffts);
    PrintResult("single", "acl_async", bytes, bytes, 1, acl);
}

void RunBatchCase(UC::Transport::Ffts::FftsTransport& transport, aclrtStream stream, size_t chunkBytes, size_t count,
                  const Options& options)
{
    if (count > std::numeric_limits<size_t>::max() / chunkBytes) {
        throw std::overflow_error("batch bytes overflow");
    }
    const auto bytes = chunkBytes * count;
    DeviceBuffer src(bytes);
    DeviceBuffer dst(bytes);
    PrepareBuffers(src, dst, bytes);
    const auto copies = BuildCopies(dst.Get(), src.Get(), chunkBytes, count);

    RunFftsBatch(transport, copies);
    VerifyBuffer(dst, bytes);
    CheckAcl(aclrtMemset(dst.Get(), dst.Size(), 0, bytes), "aclrtMemset");

    RunAclBatch(stream, copies);
    VerifyBuffer(dst, bytes);
    CheckAcl(aclrtMemset(dst.Get(), dst.Size(), 0, bytes), "aclrtMemset");

    const auto ffts = Measure(options.warmup, options.iterations, [&]() {
        RunFftsBatch(transport, copies);
    });
    const auto acl = Measure(options.warmup, options.iterations, [&]() {
        RunAclBatch(stream, copies);
    });

    PrintResult("batch", "ffts", bytes, chunkBytes, count, ffts);
    PrintResult("batch", "acl_async", bytes, chunkBytes, count, acl);
}
}  // namespace

void RunBenchmark(const Options& options)
{
    AclSession session;
    session.Init(options.deviceId);

    AclStream stream;
    UC::Transport::Ffts::FftsTransport transport;
    CheckStatus(transport.Setup(options.deviceId), "FftsTransport::Setup");

    PrintHeader();
    if (options.scenario == Scenario::All || options.scenario == Scenario::Single) {
        for (size_t bytes = options.minBytes; bytes <= options.maxBytes; bytes *= 2) {
            RunSingleCase(transport, stream.Get(), bytes, options);
            if (bytes > options.maxBytes / 2) { break; }
        }
    }

    if (options.scenario == Scenario::All || options.scenario == Scenario::Batch) {
        for (const auto chunkBytes : options.batchChunkBytes) {
            for (const auto count : options.batchCounts) {
                RunBatchCase(transport, stream.Get(), chunkBytes, count, options);
            }
        }
    }
}

}  // namespace UC::Transport::Ffts::Bench
