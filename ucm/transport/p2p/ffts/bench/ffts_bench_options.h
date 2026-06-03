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
#ifndef UNIFIEDCACHE_TRANSPORT_FFTS_BENCH_OPTIONS_H
#define UNIFIEDCACHE_TRANSPORT_FFTS_BENCH_OPTIONS_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace UC::Transport::Ffts::Bench {

constexpr size_t kKiB = 1024ULL;
constexpr size_t kMiB = 1024ULL * kKiB;
constexpr size_t kGiB = 1024ULL * kMiB;

enum class Scenario {
    All,
    Single,
    Batch,
};

struct Options {
    int32_t deviceId{0};
    size_t warmup{10};
    size_t iterations{100};
    size_t minBytes{4ULL * kKiB};
    size_t maxBytes{256ULL * kMiB};
    Scenario scenario{Scenario::All};
    std::vector<size_t> batchCounts{4, 16, 64, 128};
    std::vector<size_t> batchChunkBytes{4ULL * kKiB, 16ULL * kKiB, 64ULL * kKiB, 256ULL * kKiB, 1ULL * kMiB};
};

Options ParseOptions(int argc, char** argv);

}  // namespace UC::Transport::Ffts::Bench

#endif  // UNIFIEDCACHE_TRANSPORT_FFTS_BENCH_OPTIONS_H
