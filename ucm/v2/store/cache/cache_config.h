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
#pragma once

#include <cstdint>
#include <string>
#include "status/status.h"
#include "type/dictionary.h"

namespace UC::Store {

struct CacheConfig {
    /** Common configuration */
    std::string uniqueId;  /* Unique identifier of the current shared domain. */
    std::string llmRole;   /* Role of the LLM inference process. */
    std::size_t nWorker;   /* Number of worker processes in the current shared domain. */
    int32_t deviceId;      /* ID of the device this process is affined to. */
    std::size_t blockSize; /* Size of a fixed-length block, in bytes. */

    /** Cache-specific configuration */
    std::int64_t capacityGb{128};           /* Total size of the shared pool, in GB. */
    std::size_t blockTotalNumber;           /* Total number of blocks in the shared cache. */
    std::string replacementPolicy{"clock"}; /* Policy for slot replacement and reuse. */
    bool forceHugePage{false};              /* Whether to force-enable huge pages. */
    std::string dmaEngine{};    /* DMA engine kind name; empty selects the build's default. */
    std::size_t nDmaStream{16}; /* Number of concurrent DMA streams. */

    static Expected<CacheConfig> From(const Dictionary& dict)
    {
        CacheConfig cfg;
        DictionaryReader reader{dict, "cache_config"};

        reader.RequiredNonEmpty("unique_id", cfg.uniqueId);
        reader.RequiredEnum("llm_role", cfg.llmRole, {"scheduler", "worker", "both"});
        reader.RequiredPositive("worker_number", cfg.nWorker);
        reader.RequiredInt32("device_id", cfg.deviceId);
        reader.RequiredPositive("block_size", cfg.blockSize);
        reader.OptionalPositive("cache_buffer_capacity_gb", cfg.capacityGb);
        reader.OptionalNonEmpty("cache_replacement_policy", cfg.replacementPolicy);
        reader.OptionalBool("cache_force_huge_page", cfg.forceHugePage);
        reader.OptionalNonEmpty("cache_dma_engine", cfg.dmaEngine);
        reader.OptionalPositive("cache_dma_stream_number", cfg.nDmaStream);

        if (reader.Result().Failure()) { return reader.Result(); }
        cfg.blockTotalNumber = (static_cast<std::size_t>(cfg.capacityGb) << 30) / cfg.blockSize;
        if (cfg.blockTotalNumber == 0) {
            return Status::InvalidParam(
                "cache_config: capacity ({}GB) is smaller than one block ({}B)", cfg.capacityGb,
                cfg.blockSize);
        }
        return cfg;
    }
};

}  // namespace UC::Store
