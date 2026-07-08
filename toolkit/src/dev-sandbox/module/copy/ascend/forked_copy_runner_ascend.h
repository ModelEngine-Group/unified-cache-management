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
#ifndef FORKED_COPY_RUNNER_ASCEND_H
#define FORKED_COPY_RUNNER_ASCEND_H

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>
#include "copy_case.h"
#include "copy_result.h"
#include "copy_runtime.h"
#include "error_handle.h"

namespace ascend_copy {

using ForkedChildCopyFn = std::function<CopyResult::Result(size_t device)>;

struct ForkedChildProcess {
    pid_t pid = -1;
    int readFd = -1;
    size_t device = 0;
};

struct ResultWireHeader {
    std::uint64_t srcLen = 0;
    std::uint64_t dstLen = 0;
    std::uint64_t methodLen = 0;
    std::uint64_t size = 0;
    std::uint64_t count = 0;
    std::uint64_t submitCount = 0;
    std::uint64_t copyCount = 0;
};

inline bool WriteExact(int fd, const void* data, size_t size)
{
    const auto* cursor = static_cast<const char*>(data);
    while (size > 0) {
        const auto written = write(fd, cursor, size);
        if (written < 0) {
            if (errno == EINTR) { continue; }
            return false;
        }
        if (written == 0) { return false; }
        cursor += written;
        size -= static_cast<size_t>(written);
    }
    return true;
}

inline bool ReadExact(int fd, void* data, size_t size)
{
    auto* cursor = static_cast<char*>(data);
    while (size > 0) {
        const auto nread = read(fd, cursor, size);
        if (nread < 0) {
            if (errno == EINTR) { continue; }
            return false;
        }
        if (nread == 0) { return false; }
        cursor += nread;
        size -= static_cast<size_t>(nread);
    }
    return true;
}

inline bool WriteString(int fd, const std::string& value)
{
    return value.empty() || WriteExact(fd, value.data(), value.size());
}

inline bool ReadString(int fd, std::uint64_t size, std::string& value)
{
    value.assign(static_cast<size_t>(size), '\0');
    return value.empty() || ReadExact(fd, value.data(), value.size());
}

inline bool WriteCosts(int fd, const std::vector<size_t>& costs)
{
    for (const auto cost : costs) {
        const std::uint64_t wireCost = static_cast<std::uint64_t>(cost);
        if (!WriteExact(fd, &wireCost, sizeof(wireCost))) { return false; }
    }
    return true;
}

inline bool ReadCosts(int fd, std::uint64_t count, std::vector<size_t>& costs)
{
    costs.resize(static_cast<size_t>(count));
    for (auto& cost : costs) {
        std::uint64_t wireCost = 0;
        if (!ReadExact(fd, &wireCost, sizeof(wireCost))) { return false; }
        cost = static_cast<size_t>(wireCost);
    }
    return true;
}

inline bool WriteResult(int fd, const CopyResult::Result& result)
{
    ResultWireHeader header;
    header.srcLen = static_cast<std::uint64_t>(result.src.size());
    header.dstLen = static_cast<std::uint64_t>(result.dst.size());
    header.methodLen = static_cast<std::uint64_t>(result.method.size());
    header.size = static_cast<std::uint64_t>(result.size);
    header.count = static_cast<std::uint64_t>(result.count);
    header.submitCount = static_cast<std::uint64_t>(result.submitCosts.size());
    header.copyCount = static_cast<std::uint64_t>(result.copyCosts.size());

    return WriteExact(fd, &header, sizeof(header)) && WriteString(fd, result.src) &&
           WriteString(fd, result.dst) && WriteString(fd, result.method) &&
           WriteCosts(fd, result.submitCosts) && WriteCosts(fd, result.copyCosts);
}

inline bool ReadResult(int fd, CopyResult::Result& result)
{
    ResultWireHeader header;
    if (!ReadExact(fd, &header, sizeof(header))) { return false; }

    std::string src;
    std::string dst;
    std::string method;
    std::vector<size_t> submitCosts;
    std::vector<size_t> copyCosts;
    if (!ReadString(fd, header.srcLen, src) || !ReadString(fd, header.dstLen, dst) ||
        !ReadString(fd, header.methodLen, method) ||
        !ReadCosts(fd, header.submitCount, submitCosts) ||
        !ReadCosts(fd, header.copyCount, copyCosts)) {
        return false;
    }

    result = CopyResult::Result{std::move(src),
                                std::move(dst),
                                std::move(method),
                                static_cast<size_t>(header.size),
                                static_cast<size_t>(header.count),
                                std::move(submitCosts),
                                std::move(copyCosts)};
    return true;
}

[[noreturn]] inline void RunChildCopy(size_t device, int writeFd,
                                      const ForkedChildCopyFn& childCopy)
{
    int status = EXIT_FAILURE;
    {
        try {
            CopyRuntime runtime;
            auto result = childCopy(device);
            status = WriteResult(writeFd, result) ? EXIT_SUCCESS : EXIT_FAILURE;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[fork-copy] device %zu failed: %s\n", device, e.what());
        } catch (...) {
            std::fprintf(stderr, "[fork-copy] device %zu failed with unknown error\n", device);
        }
    }
    close(writeFd);
    std::_Exit(status);
}

inline std::vector<size_t> MergeMaxCosts(const std::vector<CopyResult::Result>& results,
                                         bool submit)
{
    ASSERT(!results.empty());
    const auto& first = submit ? results.front().submitCosts : results.front().copyCosts;
    std::vector<size_t> merged(first.size(), 0);
    for (const auto& result : results) {
        const auto& costs = submit ? result.submitCosts : result.copyCosts;
        ASSERT(costs.size() == merged.size());
        for (size_t i = 0; i < costs.size(); ++i) {
            merged[i] = std::max(merged[i], costs[i]);
        }
    }
    return merged;
}

inline CopyResult::Result MergeForkedResults(std::vector<CopyResult::Result>&& results,
                                             std::string srcName, std::string dstName,
                                             std::string methodName)
{
    ASSERT(!results.empty());
    size_t totalCount = 0;
    for (const auto& result : results) {
        ASSERT(result.size == results.front().size);
        totalCount += result.count;
    }

    return {std::move(srcName),
            std::move(dstName),
            std::move(methodName),
            results.front().size,
            totalCount,
            MergeMaxCosts(results, true),
            MergeMaxCosts(results, false)};
}

inline std::vector<CopyResult::Result> RunForkedCopyBatchPerDevice(
    const CopyCase::Context& ctx, const ForkedChildCopyFn& childCopy)
{
    ASSERT(ctx.nDevice > 0);
    std::vector<ForkedChildProcess> children;
    children.reserve(ctx.nDevice);

    for (size_t device = 0; device < ctx.nDevice; ++device) {
        int pipeFds[2];
        ASSERT(pipe(pipeFds) == 0);
        const auto pid = fork();
        ASSERT(pid != -1);
        if (pid == 0) {
            close(pipeFds[0]);
            RunChildCopy(device, pipeFds[1], childCopy);
        }

        close(pipeFds[1]);
        children.push_back({pid, pipeFds[0], device});
    }

    std::vector<CopyResult::Result> childResults;
    childResults.reserve(children.size());
    bool failed = false;
    for (auto& child : children) {
        CopyResult::Result result{"", "", "", 0, 0, {}, {}};
        if (!ReadResult(child.readFd, result)) {
            std::fprintf(stderr, "[fork-copy] failed to read result from device %zu\n",
                         child.device);
            failed = true;
        } else {
            childResults.push_back(std::move(result));
        }
        close(child.readFd);
    }

    for (const auto& child : children) {
        int status = 0;
        pid_t waited = -1;
        do {
            waited = waitpid(child.pid, &status, 0);
        } while (waited == -1 && errno == EINTR);
        ASSERT(waited == child.pid);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != EXIT_SUCCESS) {
            std::fprintf(stderr, "[fork-copy] child for device %zu exited abnormally\n",
                         child.device);
            failed = true;
        }
    }
    ASSERT(!failed);
    ASSERT(childResults.size() == ctx.nDevice);

    return childResults;
}

inline CopyResult::Result RunForkedCopyBatch(const CopyCase::Context& ctx, std::string srcName,
                                             std::string dstName, std::string methodName,
                                             const ForkedChildCopyFn& childCopy)
{
    auto childResults = RunForkedCopyBatchPerDevice(ctx, childCopy);
    return MergeForkedResults(std::move(childResults), std::move(srcName), std::move(dstName),
                              std::move(methodName));
}

}  // namespace ascend_copy

#endif  // FORKED_COPY_RUNNER_ASCEND_H
