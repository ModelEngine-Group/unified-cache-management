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
#include "store_assembler.h"
#include <string>
#include <utility>
#include <vector>
#include "routing/strategy.h"
#include "routing/strategy_registry.h"
#include "store_entry.h"
#include "store_peer.h"
#include "store_pipe.h"

namespace UC::Proxy {

namespace {

/* Routing fallback of a single-branch peer: nothing to configure, every
 * request routes to branch 0. */
class FixedRoutingStrategy : public RoutingStrategy {
public:
    Status Setup(const Dictionary& /*config*/, std::size_t /*branchCount*/) override
    {
        return Status::Ok();
    }

    std::size_t Select(const RequestContext& /*request*/) override { return 0; }
};

/* Reserved per-instance key: the assembler appends the node's topology path
 * to the runtime's base value, so same-plugin instances (e.g. two cache
 * tiers) receive distinct service identities. */
std::string SuffixedUniqueId(const std::string& base, std::size_t branch, std::size_t tier)
{
    return base + "@branches[" + std::to_string(branch) + "].pipe[" + std::to_string(tier) + "]";
}

/* User-supplied dictionaries must not carry the reserved "__" namespace:
 * those keys are injected by the assembler (runtime merge, pipe wiring). */
Status ValidateNoReservedKeys(const Dictionary& dict)
{
    Status status = Status::Ok();
    dict.ForEach([&status](const std::string& key, const std::any& value) {
        if (key.rfind("__", 0) == 0) {
            if (status.Success()) {
                status = Status::InvalidParam("config key '{}' is system-reserved", key);
            }
            return;
        }
        if (const auto* nested = std::any_cast<Dictionary>(&value)) {
            if (auto failure = ValidateNoReservedKeys(*nested); failure.Failure()) {
                if (status.Success()) { status = failure; }
            }
            return;
        }
        if (const auto* list = std::any_cast<std::vector<Dictionary>>(&value)) {
            for (const auto& item : *list) {
                if (auto failure = ValidateNoReservedKeys(item); failure.Failure()) {
                    if (status.Success()) { status = failure; }
                }
            }
        }
    });
    return status;
}

/* Merge the flat runtime dictionary into one store node's setup config:
 * colliding keys are rejected up front (service facts are not
 * user-overridable), then every runtime key is merged in and the reserved
 * per-instance key is suffixed with the node's topology path. */
Status MergeRuntime(Dictionary& setupConfig, const Dictionary& runtime, std::size_t branch,
                    std::size_t tier)
{
    Status conflict = Status::Ok();
    runtime.ForEach([&](const std::string& key, const std::any& /*value*/) {
        if (setupConfig.Contains(key)) {
            if (conflict.Success()) {
                conflict = Status::InvalidParam(
                    "runtime key '{}' collides with a store node config key", key);
            }
        }
    });
    if (conflict.Failure()) { return conflict; }
    runtime.ForEach(
        [&](const std::string& key, const std::any& value) { setupConfig.SetAny(key, value); });
    if (runtime.Contains("unique_id")) {
        std::string base;
        if (!runtime.Get<std::string>("unique_id", base)) {
            return Status::InvalidParam("runtime 'unique_id' must be a string");
        }
        setupConfig.Set("unique_id", SuffixedUniqueId(base, branch, tier));
    }
    return Status::Ok();
}

Expected<std::unique_ptr<StoreEntry>> AssemblePipe(const Dictionary& branchConfig,
                                                   const Dictionary& runtime,
                                                   std::size_t branchIndex)
{
    std::vector<Dictionary> tierConfigs;
    if (!branchConfig.Get<std::vector<Dictionary>>("pipe", tierConfigs) || tierConfigs.empty()) {
        return Status::InvalidParam("branch config misses a non-empty 'pipe' list");
    }
    std::vector<std::unique_ptr<LoadedStore>> loaded;
    loaded.reserve(tierConfigs.size());
    for (const auto& tierConfig : tierConfigs) {
        std::string store;
        if (!tierConfig.Get<std::string>("store", store)) {
            return Status::InvalidParam("pipe store entry misses 'store'");
        }
        auto tier = StoreLoader::Load(store);
        if (!tier) { return tier.Error(); }
        loaded.push_back(std::move(tier.Value()));
    }
    /* Bottom-up: every tier is set up after its lower tier, so a chain head
     * probing its lower store at setup time sees a configured instance. */
    for (std::size_t i = loaded.size(); i-- > 0;) {
        Dictionary setupConfig;
        (void)tierConfigs[i].Get<Dictionary>("config", setupConfig);
        const Status merged = MergeRuntime(setupConfig, runtime, branchIndex, i);
        if (merged.Failure()) { return merged; }
        if (i + 1 < loaded.size()) {
            /* Wiring convention: chain heads read "__ucm_lower_store__"
             * (Store::Store*) from their setup dictionary to reach the next
             * tier of the pipe. */
            setupConfig.Set("__ucm_lower_store__", loaded[i + 1]->Store());
        }
        auto setup = loaded[i]->Store()->Setup(setupConfig);
        if (setup.Failure()) { return setup; }
    }
    return std::make_unique<StorePipe>(std::move(loaded));
}

Expected<std::unique_ptr<RoutingStrategy>> AssembleRouting(const Dictionary& config,
                                                           std::size_t branchCount)
{
    Dictionary routingConfig;
    if (!config.Get<Dictionary>("routing", routingConfig)) {
        return Status::InvalidParam("peer config misses a 'routing' dictionary");
    }
    std::string name;
    if (!routingConfig.Get<std::string>("strategy", name)) {
        return Status::InvalidParam("routing config misses 'strategy'");
    }
    auto strategy = RoutingStrategyRegistry::Instance().Create(name);
    if (!strategy) { return strategy.Error(); }
    Dictionary params;
    (void)routingConfig.Get<Dictionary>("config", params);
    std::unique_ptr<RoutingStrategy> instance(strategy.Value());
    auto setup = instance->Setup(params, branchCount);
    if (setup.Failure()) { return setup; }
    return instance;
}

Expected<std::unique_ptr<StorePeer>> AssemblePeer(const Dictionary& config,
                                                  const Dictionary& runtime)
{
    std::vector<Dictionary> branchConfigs;
    if (!config.Get<std::vector<Dictionary>>("branches", branchConfigs) || branchConfigs.empty()) {
        return Status::InvalidParam("peer config misses a non-empty 'branches' list");
    }
    std::vector<std::unique_ptr<StoreEntry>> branches;
    branches.reserve(branchConfigs.size());
    for (std::size_t i = 0; i < branchConfigs.size(); i++) {
        auto branch = AssemblePipe(branchConfigs[i], runtime, i);
        if (!branch) { return branch.Error(); }
        branches.push_back(std::move(branch.Value()));
    }
    std::unique_ptr<RoutingStrategy> routing;
    if (branches.size() > 1 || config.Contains("routing")) {
        auto assembled = AssembleRouting(config, branches.size());
        if (!assembled) { return assembled.Error(); }
        routing = std::move(assembled.Value());
    } else {
        /* Single-branch peer without routing: the internal fixed strategy
         * keeps a trivial deployment free of routing boilerplate. */
        routing = std::make_unique<FixedRoutingStrategy>();
    }
    return std::make_unique<StorePeer>(std::move(branches), std::move(routing));
}

}  // namespace

Expected<std::unique_ptr<StoreAssembler>> StoreAssembler::Create(const Dictionary& config,
                                                                 const Dictionary& runtime)
{
    const Status reserved = ValidateNoReservedKeys(config);
    if (reserved.Failure()) { return reserved; }
    auto peer = AssemblePeer(config, runtime);
    if (!peer) { return peer.Error(); }
    return std::unique_ptr<StoreAssembler>(new StoreAssembler(std::move(peer.Value())));
}

StoreScheduler* StoreAssembler::GetScheduler() { return root_.get(); }

StoreWorker* StoreAssembler::GetWorker() { return root_.get(); }

StoreAssembler::StoreAssembler(std::unique_ptr<StorePeer> root) : root_(std::move(root)) {}

}  // namespace UC::Proxy
