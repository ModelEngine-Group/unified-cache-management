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
#include "numa_policy.h"
#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <fstream>
#include <linux/mempolicy.h>
#include <map>
#include <string_view>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>
#include "logger/logger.h"

namespace UC::Cache2::Numa {

namespace {

/* Nodes are re-read on every Resolve; the kernel keeps these files current. */
constexpr const char* kOnlinePath = "/sys/devices/system/node/online";
constexpr const char* kMemoryPath = "/sys/devices/system/node/has_memory";
constexpr const char* kProcStatusPath = "/proc/self/status";
constexpr const char* kAllowedPrefix = "Mems_allowed_list:";

/* Cap on the pages Verify samples, so a large segment still gets a bounded,
 * cheap placement check. */
constexpr size_t kMaxVerifyPages = 4096;

struct NodeMask {
    std::vector<unsigned long> words;
    unsigned long maxNode;
};

long PageSize() { return ::sysconf(_SC_PAGESIZE); }

bool ReadFirstLine(const char* path, std::string& out)
{
    std::ifstream file(path);
    return static_cast<bool>(std::getline(file, out));
}

/* Accepts "0-7,9" style lists, the form the kernel uses for these files. */
Expected<std::vector<size_t>> ParseNodes(std::string_view text)
{
    std::vector<size_t> nodes;
    auto parseOne = [](std::string_view token) -> Expected<size_t> {
        size_t value = 0;
        const auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), value);
        if (token.empty() || error != std::errc() || end != token.data() + token.size() ||
            value > 65535) {
            return Status::InvalidParam("cache2 numa: invalid node ID '{}'", token);
        }
        return value;
    };
    while (true) {
        const auto comma = text.find(',');
        const auto token = text.substr(0, comma);
        const auto dash = token.find('-');
        auto first = parseOne(token.substr(0, dash));
        if (!first) { return first.Error(); }
        size_t last = first.Value();
        if (dash != std::string_view::npos) {
            auto parsed = parseOne(token.substr(dash + 1));
            if (!parsed) { return parsed.Error(); }
            last = parsed.Value();
        }
        if (last < first.Value()) {
            return Status::InvalidParam("cache2 numa: descending node range '{}'", token);
        }
        for (size_t node = first.Value(); node <= last; ++node) {
            if (std::find(nodes.begin(), nodes.end(), node) != nodes.end()) {
                return Status::InvalidParam("cache2 numa: duplicated node({})", node);
            }
            nodes.push_back(node);
        }
        if (comma == std::string_view::npos) { break; }
        text.remove_prefix(comma + 1);
    }
    return nodes;
}

Expected<std::vector<size_t>> ReadNodes(const char* path)
{
    std::string text;
    if (!ReadFirstLine(path, text)) {
        return Status::Error(fmt::format("cache2 numa: cannot read {}", path));
    }
    return ParseNodes(text);
}

/* Nodes this process is actually allowed to allocate from. */
Expected<std::vector<size_t>> AllowedNodes()
{
    std::ifstream status(kProcStatusPath);
    std::string line;
    while (std::getline(status, line)) {
        if (line.compare(0, std::strlen(kAllowedPrefix), kAllowedPrefix) != 0) { continue; }
        const auto start = line.find_first_not_of(" \t", std::strlen(kAllowedPrefix));
        if (start == std::string::npos) { break; }
        return ParseNodes(line.substr(start));
    }
    return Status::Error("cache2 numa: cannot read Mems_allowed_list from /proc/self/status");
}

/* Online, has memory, and reachable by this process; ascending by node ID. */
Expected<std::vector<size_t>> AvailableNodes()
{
    auto online = ReadNodes(kOnlinePath);
    if (!online) { return online.Error(); }
    auto memory = ReadNodes(kMemoryPath);
    if (!memory) { return memory.Error(); }
    auto allowed = AllowedNodes();
    if (!allowed) { return allowed.Error(); }

    std::vector<size_t> nodes;
    for (const auto node : memory.Value()) {
        if (std::find(online.Value().begin(), online.Value().end(), node) == online.Value().end()) {
            continue;
        }
        if (std::find(allowed.Value().begin(), allowed.Value().end(), node) ==
            allowed.Value().end()) {
            continue;
        }
        nodes.push_back(node);
    }
    if (nodes.empty()) { return Status::Error("cache2 numa: no usable NUMA memory node"); }
    std::sort(nodes.begin(), nodes.end());
    return nodes;
}

NodeMask SingleNodeMask(size_t node)
{
    constexpr size_t bitsPerWord = sizeof(unsigned long) * 8;
    const auto words = node / bitsPerWord + 1;
    NodeMask mask{std::vector<unsigned long>(words, 0), 0};
    mask.words[node / bitsPerWord] = 1UL << (node % bitsPerWord);
    /* Linux get_nodes() decrements maxnode before copying the bitmap. */
    mask.maxNode = static_cast<unsigned long>(words * bitsPerWord + 1);
    return mask;
}

}  // namespace

Expected<size_t> RankSliceIndex(size_t rank, size_t rankCount, size_t nodeCount)
{
    if (rankCount == 0) { return Status::InvalidParam("cache2 numa: rank count is zero"); }
    if (nodeCount == 0) { return Status::InvalidParam("cache2 numa: node count is zero"); }
    if (rank >= rankCount) {
        return Status::InvalidParam("cache2 numa: fallback rank({}) is outside rank count({})",
                                    rank, rankCount);
    }
    return rank * nodeCount / rankCount;
}

Expected<Plan> Resolve(std::optional<int32_t> deviceNode, std::optional<size_t> fallbackNumaRank,
                       size_t rankCount)
{
    auto available = AvailableNodes();
    if (!available) { return available.Error(); }
    const auto& nodes = available.Value();

    if (deviceNode.has_value()) {
        if (*deviceNode < 0) {
            return Status::InvalidParam("cache2 numa: negative affinity node({})", *deviceNode);
        }
        const auto node = static_cast<size_t>(*deviceNode);
        if (std::find(nodes.begin(), nodes.end(), node) == nodes.end()) {
            /* A known but unusable affinity node is a real failure: falling back
             * to rank slicing here would hide a broken device probe. */
            return Status::Error(
                fmt::format("cache2 numa: affinity node({}) is not available", node));
        }
        return Plan{Placement::Affinity, *deviceNode};
    }

    if (!fallbackNumaRank.has_value()) {
        return Status::InvalidParam(
            "cache2 numa: no device affinity and no local worker rank to fall back on");
    }
    auto nodeIndex = RankSliceIndex(*fallbackNumaRank, rankCount, nodes.size());
    if (!nodeIndex) { return nodeIndex.Error(); }
    const auto node = nodes[nodeIndex.Value()];
    return Plan{Placement::RankSlice, static_cast<int32_t>(node)};
}

Status BindBeforeTouch(void* base, size_t bytes, const Plan& plan)
{
    if (base == nullptr || bytes == 0) {
        return Status::InvalidParam("cache2 numa: cannot bind an empty segment");
    }
    if (plan.node < 0) { return Status::InvalidParam("cache2 numa: plan carries no node"); }
    const long pageSize = PageSize();
    if (pageSize <= 0) { return Status::Error("cache2 numa: sysconf(_SC_PAGESIZE) failed"); }
    if (reinterpret_cast<uintptr_t>(base) % static_cast<size_t>(pageSize) != 0) {
        return Status::InvalidParam("cache2 numa: segment address is not page-aligned");
    }
    const auto mask = SingleNodeMask(static_cast<size_t>(plan.node));
    const long result = ::syscall(SYS_mbind, base, bytes, MPOL_BIND | MPOL_F_STATIC_NODES,
                                  mask.words.data(), mask.maxNode, 0UL);
    if (result < 0) {
        const auto error = errno;
        return Status::OsApiError(
            fmt::format("cache2 numa: mbind node={} failed: {}", plan.node, std::strerror(error)));
    }
    UC_INFO("cache2 numa bound: node={} placement={} bytes={}", plan.node,
            plan.placement == Placement::Affinity ? "affinity" : "rank-slice", bytes);
    return Status::OK();
}

Status Verify(void* base, size_t bytes, const Plan& plan)
{
    if (base == nullptr || bytes == 0 || plan.node < 0) {
        return Status::InvalidParam("cache2 numa: nothing to verify");
    }
    const long pageSize = PageSize();
    if (pageSize <= 0) { return Status::Error("cache2 numa: sysconf(_SC_PAGESIZE) failed"); }
    const auto pageSizeSz = static_cast<size_t>(pageSize);
    const size_t pageCount = bytes / pageSizeSz;
    if (pageCount == 0) { return Status::OK(); }

    const size_t samples = std::min(pageCount, kMaxVerifyPages);
    std::vector<void*> addresses(samples, nullptr);
    std::vector<int> status(samples, -EIO);
    for (size_t i = 0; i < samples; ++i) {
        const size_t page = samples == pageCount ? i : i * pageCount / samples;
        addresses[i] = static_cast<char*>(base) + page * pageSizeSz;
    }
    const long result =
        ::syscall(SYS_move_pages, 0, samples, addresses.data(), nullptr, status.data(), 0);
    if (result < 0) {
        const auto error = errno;
        /* Diagnostics only: a failed query must not affect service. */
        UC_WARN("cache2 numa verify unavailable: expected_node={} errno={}", plan.node, error);
        return Status::OK();
    }

    std::map<int, size_t> pagesByNode;
    size_t mismatches = 0;
    for (const auto node : status) {
        if (node < 0) { continue; }
        ++pagesByNode[node];
        if (node != plan.node) { ++mismatches; }
    }
    for (const auto& [node, pages] : pagesByNode) {
        UC_INFO("cache2 numa verify: expected_node={} actual_node={} sampled_pages={}", plan.node,
                node, pages);
    }
    if (mismatches != 0) {
        UC_WARN(
            "cache2 numa placement differs from the request: expected_node={} "
            "mismatched_pages={} sampled_pages={}",
            plan.node, mismatches, samples);
    }
    return Status::OK();
}

}  // namespace UC::Cache2::Numa
