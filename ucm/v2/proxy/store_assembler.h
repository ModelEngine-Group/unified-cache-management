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

#include <memory>
#include "status/status.h"
#include "store_entry.h"
#include "store_loader.h"
#include "store_peer.h"
#include "store_scheduler.h"
#include "store_worker.h"
#include "type/dictionary.h"

namespace UC::Proxy {

/**
 * @brief Configuration-driven assembly of the store topology root.
 *
 * The user configuration follows a fixed three-level grammar — the root is
 * always a peer, every branch is a pipe, every pipe tier is a store node:
 *
 *   peer      := { routing?: { strategy: <name>, config?: {...} },
 *                  branches: [ pipeNode, ... ] }        # >= 1
 *   pipeNode  := { pipe: [ storeNode, ... ] }           # >= 1
 *   storeNode := { store: <name|path>, config?: {...} }
 *
 * A single-branch peer may omit "routing" (an internal fixed strategy routes
 * every request to branch 0).
 *
 * "runtime" is the flat service dictionary (conceptually "__ucm_runtime__")
 * built programmatically by the embedding layer and never part of user
 * files: every key is merged into every store node's setup config, colliding
 * keys are rejected, and the reserved per-instance key "unique_id" is
 * suffixed with the node's topology path ("@branches[i].pipe[j]") so
 * same-plugin instances receive distinct service identities.
 *
 * User-supplied keys must not start with "__": that namespace is reserved
 * for assembler-injected values (runtime merge, "__ucm_lower_store__" pipe
 * wiring).
 */
class StoreAssembler {
public:
    static Expected<std::unique_ptr<StoreAssembler>> Create(const Dictionary& config,
                                                            const Dictionary& runtime = {});

    StoreScheduler* GetScheduler();
    StoreWorker* GetWorker();

private:
    explicit StoreAssembler(std::unique_ptr<StorePeer> root);
    std::unique_ptr<StorePeer> root_;
};

}  // namespace UC::Proxy
