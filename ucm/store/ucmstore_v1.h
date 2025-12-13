/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#ifndef UNIFIEDCACHE_STORE_V1_H
#define UNIFIEDCACHE_STORE_V1_H

#include "status/status.h"
#include "type/types.h"

namespace UC {

/**
 * @brief Abstract interface for a key-value store that supports
 *        asynchronous load/dump of cached blocks.
 *
 * Thread safety: All public methods must be thread-safe.  Concurrent calls
 * are allowed; implementations are responsible for internal synchronization.
 */
class StoreV1 {
public:
    virtual ~StoreV1() = default;

    /**
     * @brief Check whether the given blocks exist in storage.
     *
     * @param blocks Array of block identifiers to test.
     * @param num Number of block identifiers to test.
     * @return Expected<std::vector<uint8_t>>
     *   - On success: a vector whose i-th element is **true** if blocks[i]
     *     is present, otherwise **false**.
     *   - On failure: appropriate Status code.
     */
    virtual Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId* blocks, size_t num) = 0;

    /**
     * @brief Hint the store to prefetch given blocks into high-speed cache.
     *
     * This call is **non-blocking** and **fire-and-forget**; it returns
     * immediately and carries no completion guarantee. Implementations may
     * ignore the hint if prefetching is not supported or resources are
     * unavailable.
     *
     * @param blocks Array of block identifiers to be prefetched.
     * @param num Number of block identifiers to be prefetched.
     *
     * @note Thread-safe; may be called concurrently with other operations.
     * @note Default implementation does nothing.
     */
    virtual void Prefetch(const Detail::BlockId* blocks, size_t num) = 0;

    /**
     * @brief Start an asynchronous load (storage → device) transfer.
     *
     * @param task Description of shards to be loaded.
     * @return Expected<TaskHandle>
     *   - On success: a task handle that can be passed to Wait() or Check().
     *   - On failure: relevant Status code.
     */
    virtual Expected<Detail::TaskHandle> Load(Detail::TaskDesc task) = 0;

    /**
     * @brief Start an asynchronous dump (device → storage) transfer.
     *
     * @param task Description of shards to be stored.
     * @return Expected<TaskHandle>
     *   - On success: a task handle that can be passed to Wait() or Check().
     *   - On failure: relevant Status code.
     */
    virtual Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task) = 0;

    /**
     * @brief Poll for task completion without blocking.
     *
     * @param taskId Task handle returned by Load() or Dump().
     * @return Expected<bool>
     *   - **true**  if the task has finished (successfully or with an error).
     *   - **false** if the task is still running.
     *   - Any other value indicates an error in the poll itself.
     */
    virtual Expected<bool> Check(Detail::TaskHandle taskId) = 0;

    /**
     * @brief Block until the specified task completes.
     *
     * @param taskId Task handle returned by Load() or Dump().
     * @return Status::OK on successful completion, otherwise an error code
     *         describing the failure.
     */
    virtual Status Wait(Detail::TaskHandle taskId) = 0;

protected:
    /**
     * @brief Protected default constructor.
     *
     * Prevents direct instantiation and enforces derivation.
     */
    StoreV1() = default;
};

}  // namespace UC

#endif
