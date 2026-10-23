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
#include "export/export.h"
#include "status/status.h"
#include "type/block.h"
#include "type/dictionary.h"
#include "type/request.h"

namespace UC::Store {

class Prefetchable;
class TokenAddressed;

struct JobDescriptor {
    /** BlockIds of the blocks to transfer in this job. */
    std::vector<BlockId> blockIds;
    /** Payloads to transfer; payloads[i] corresponds to blockIds[i] (same length). */
    std::vector<BlockPayload> payloads;
    /** Optional: short human-readable description of the job. */
    std::string description;
    /**
     * Optional: prerequisite handle the stream waits on before transfer; 0 when the job
     * has no prerequisite.
     */
    std::uintptr_t prerequisiteHandle{0};
};

/**
 * @brief Base interface of a KV cache storage tier.
 *
 * @note Threading contract: control-plane methods must be called from a
 *       single thread or be externally serialized. The exactly-once
 *       completion semantics of GetFinished()/Wait() implies a single
 *       consumer of completion events.
 */
class UC_EXPORT Store {
public:
    /**
     * @brief Destroys the Store.
     *
     * @note Destruction contract: the caller must Wait() for every in-flight
     *       job before destruction (likewise for UCDestroyStore()); behavior
     *       with in-flight jobs is undefined.
     */
    virtual ~Store() = default;
    /**
     * @brief Sets up and configures the Store instance with the provided configuration.
     *
     * @param config A dictionary containing configuration key-value pairs specific
     *               to this Store instance. The content and structure of the dictionary
     *               may vary depending on the concrete implementation.
     *
     * @return Status indicating the result of the setup operation:
     *         - Status::Ok() if setup was successful
     *         - Error status with appropriate code and message if configuration is
     *           invalid or setup fails
     *
     * @note Implementations should validate all required configuration parameters
     *       and perform any necessary resource allocation or initialization.
     * @note This method is called after object construction but before any processing
     *       operations that depend on the configuration.
     */
    virtual Status Setup(const Dictionary& config) = 0;
    /**
     * @brief Probes the presence of the given blocks in this Store.
     *
     * @param blockIds  Content-addressed ids of the blocks to probe, ordered by
     *                  their position within the request.
     *
     * @return Expected per-block hit vector on success: hits[i] is non-zero
     *         iff blockIds[i] is present, and hits.size() == blockIds.size().
     *         Error status otherwise.
     */
    virtual Expected<std::vector<std::uint8_t>> Lookup(const std::vector<BlockId>& blockIds) = 0;
    /**
     * @brief Prefix-oriented variant of Lookup: reports how far the request
     * is present from its head.
     *
     * @param blockIds  Blocks ordered by their position within the request.
     *
     * @return Expected index of the last block of the leading run of
     *         present blocks: -1 when the first block is absent (empty
     *         input yields -1 as well); blockIds.size() - 1 when every
     *         block is present. Error status otherwise.
     *
     * @note The default implementation derives the answer from a single
     *       Lookup call; override to exploit ordered medium layouts, e.g.
     *       by answering from metadata without probing each block.
     */
    virtual Expected<std::int64_t> LookupOnPrefix(const std::vector<BlockId>& blockIds)
    {
        auto res = Lookup(blockIds);
        if (!res) { return res.Error(); }
        const auto& hits = res.Value();
        for (std::size_t i = 0; i < hits.size(); i++) {
            if (hits[i] == 0) { return static_cast<std::int64_t>(i) - 1; }
        }
        return static_cast<std::int64_t>(hits.size()) - 1;
    }
    /**
     * @brief Reverse variant of Lookup: reports the rightmost present block
     * of the request.
     *
     * @param blockIds  Blocks ordered by their position within the request.
     *
     * @return Expected index of the rightmost present block; -1 when none
     *         is present (empty input yields -1 as well). Error status
     *         otherwise. Contiguity is neither required nor checked:
     *         blocks before the returned index may be absent.
     *
     * @note The default implementation derives the answer from a single
     *       Lookup call; override only if the Store can answer reverse
     *       queries cheaper.
     */
    virtual Expected<std::int64_t> LookupOnReverse(const std::vector<BlockId>& blockIds)
    {
        auto res = Lookup(blockIds);
        if (!res) { return res.Error(); }
        const auto& hits = res.Value();
        for (auto i = hits.size(); i > 0; i--) {
            if (hits[i - 1] != 0) { return static_cast<std::int64_t>(i) - 1; }
        }
        return -1;
    }
    /**
     * @brief Refreshes the recency of the given blocks, for LRU-family eviction
     *        policies implemented by the Store, and extends their retention: a
     *        TTL fact in the context raises each block's deadline to the larger
     *        of the existing one and now + ttl -- deadlines are never shortened.
     *        Blocks that are not cached are ignored.
     *
     * @note Default is a no-op: Stores without recency-based eviction need not
     *       override this.
     */
    virtual void Touch(const std::vector<BlockId>& /*blockIds*/, const RequestContext& /*context*/)
    {
    }

    /**
     * @brief Copies blocks from this Store to the inference side, as an
     * asynchronous job (reload path).
     *
     * @param jobId Caller-allocated handle of the asynchronous job; must be
     *              unique among in-flight jobs.
     * @param job   Store transfer job descriptor: job.blockIds[i] resolves
     *              to its read source on this store's worker side;
     *              job.payloads[i] is the destination.
     *
     * @return Status indicating whether the job was accepted (and started);
     *         error status otherwise, with nothing started.
     */
    virtual Status Load(std::size_t jobId, JobDescriptor job) = 0;
    /**
     * @brief Copies blocks from the inference side into this Store, as an
     * asynchronous job (offload path).
     *
     * @param jobId Caller-allocated handle of the asynchronous job; must be
     *              unique among in-flight jobs.
     * @param job   Store transfer job descriptor: job.blockIds[i] resolves
     *              to its write destination on this store's worker side;
     *              job.payloads[i] is the source.
     *
     * @return Status indicating whether the job was accepted (and started);
     *         error status otherwise, with nothing started.
     */
    virtual Status Dump(std::size_t jobId, JobDescriptor job) = 0;
    /**
     * @brief Returns the jobs finished since the last call and consumes their
     * completion events.
     *
     * @return (jobId, success) pairs, where success tells whether the job's
     *         transfer completed successfully.
     *
     * @note Draining: each completion is reported exactly once across
     *       GetFinished() and Wait(); consumed events are never re-reported.
     */
    virtual std::vector<std::pair<std::size_t, bool>> GetFinished() = 0;
    /**
     * @brief Blocks until every listed job has finished, then returns their
     * (consumed) completion events.
     *
     * @param jobIds Jobs to wait for; ids that are unknown or whose completion
     *               has already been consumed are ignored.
     *
     * @return (jobId, success) pairs for the waited jobs, where success tells
     *         whether the job's transfer completed successfully.
     *
     * @note Completions delivered here are not reported again by GetFinished()
     *       and vice versa (exactly-once).
     */
    virtual std::vector<std::pair<std::size_t, bool>> Wait(
        const std::vector<std::size_t>& jobIds) = 0;
    /**
     * @brief Queries the Prefetchable capability of this Store.
     *
     * @return Pointer to the Prefetchable facet of this object, or nullptr
     *         when this Store does not implement it. The returned pointer
     *         has the lifetime of this Store object.
     */
    virtual Prefetchable* AsPrefetchable() { return nullptr; }
    /**
     * @brief Queries the TokenAddressed capability of this Store.
     *
     * @return Pointer to the TokenAddressed facet of this object, or nullptr
     *         when this Store does not implement it. The returned pointer
     *         has the lifetime of this Store object.
     */
    virtual TokenAddressed* AsTokenAddressed() { return nullptr; }
};

/**
 * @brief Factory entry point type of a Store plugin: the loader dlopen()s the
 *        plugin (RTLD_LOCAL | RTLD_NOW recommended) and resolves the entry
 *        points via dlsym() on the plugin's own handle (never RTLD_DEFAULT).
 *        May be called multiple times; each call returns a new instance, or
 *        nullptr when instance allocation fails.
 *
 * @note Entry point names are uniform across plugins: a binary must link at
 *       most one factory definition. Plugin classes are default-constructible;
 *       configuration is passed afterwards via Setup().
 */
using CreateStoreFn = Store*();
/**
 * @brief Destroys an instance created by the matching create entry point,
 * returning it to the same heap it was created on.
 */
using DestroyStoreFn = void(Store*);

}  // namespace UC::Store
