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
#include "routing/strategy.h"
#include "routing/strategy_registry.h"
#include "store_entry.h"
#include "store_peer.h"
#include "store_pipe.h"

namespace UC::Proxy {

namespace {

Expected<std::unique_ptr<StoreEntry>> AssembleEntry(const Dictionary& nodeConfig)
{
    std::string store;
    if (!nodeConfig.Get<std::string>("store", store)) {
        return Status::InvalidParam("store node config misses 'store'");
    }
    Dictionary setupConfig;
    (void)nodeConfig.Get<Dictionary>("config", setupConfig);
    auto loaded = StoreLoader::Load(store);
    if (!loaded) { return loaded.Error(); }
    auto setup = loaded.Value()->Store()->Setup(setupConfig);
    if (setup.Failure()) { return setup; }
    return std::make_unique<StoreEntry>(std::move(loaded.Value()));
}

Expected<std::unique_ptr<StoreEntry>> AssemblePipe(const Dictionary& nodeConfig)
{
    std::vector<Dictionary> storeConfigs;
    if (!nodeConfig.Get<std::vector<Dictionary>>("stores", storeConfigs) || storeConfigs.empty()) {
        return Status::InvalidParam("pipe node config misses a non-empty 'stores' list");
    }
    std::vector<std::unique_ptr<LoadedStore>> loaded;
    loaded.reserve(storeConfigs.size());
    for (const auto& storeConfig : storeConfigs) {
        std::string store;
        if (!storeConfig.Get<std::string>("store", store)) {
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
        (void)storeConfigs[i].Get<Dictionary>("config", setupConfig);
        if (i + 1 < loaded.size()) {
            /* Wiring convention: chain heads read "piped_lower_store"
             * (Store::Store*) from their setup dictionary to reach the next
             * tier of the pipe. */
            setupConfig.Set("piped_lower_store", loaded[i + 1]->Store());
        }
        auto setup = loaded[i]->Store()->Setup(setupConfig);
        if (setup.Failure()) { return setup; }
    }
    return std::make_unique<StorePipe>(std::move(loaded));
}

Expected<std::unique_ptr<StoreEntry>> AssembleBranch(const Dictionary& nodeConfig)
{
    std::string type;
    if (!nodeConfig.Get<std::string>("type", type)) {
        return Status::InvalidParam("branch node config misses 'type'");
    }
    if (type == "store") { return AssembleEntry(nodeConfig); }
    if (type == "pipe") { return AssemblePipe(nodeConfig); }
    if (type == "peer") {
        return Status::InvalidParam("a peer branch cannot nest a peer (flatten to a wider peer)");
    }
    return Status::InvalidParam("unknown branch type '{}'", type);
}

Expected<std::unique_ptr<RoutingStrategy>> AssembleRouting(const Dictionary& config,
                                                           std::size_t branchCount)
{
    Dictionary routingConfig;
    if (!config.Get<Dictionary>("routing", routingConfig)) {
        return Status::InvalidParam("peer node config misses a 'routing' dictionary");
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

Expected<std::unique_ptr<StorePeer>> AssemblePeer(const Dictionary& config)
{
    std::vector<Dictionary> branchConfigs;
    if (!config.Get<std::vector<Dictionary>>("branches", branchConfigs) || branchConfigs.empty()) {
        return Status::InvalidParam("peer node config misses a non-empty 'branches' list");
    }
    std::vector<std::unique_ptr<StoreEntry>> branches;
    branches.reserve(branchConfigs.size());
    for (const auto& branchConfig : branchConfigs) {
        auto branch = AssembleBranch(branchConfig);
        if (!branch) { return branch.Error(); }
        branches.push_back(std::move(branch.Value()));
    }
    auto routing = AssembleRouting(config, branches.size());
    if (!routing) { return routing.Error(); }
    return std::make_unique<StorePeer>(std::move(branches), std::move(routing.Value()));
}

}  // namespace

Expected<std::unique_ptr<StoreAssembler>> StoreAssembler::Create(const Dictionary& config)
{
    std::string type;
    if (!config.Get<std::string>("type", type)) {
        return Status::InvalidParam("config misses 'type'");
    }
    if (type == "store") {
        auto entry = AssembleEntry(config);
        if (!entry) { return entry.Error(); }
        return std::unique_ptr<StoreAssembler>(new StoreAssembler(std::move(entry.Value())));
    }
    if (type == "pipe") {
        auto pipe = AssemblePipe(config);
        if (!pipe) { return pipe.Error(); }
        return std::unique_ptr<StoreAssembler>(new StoreAssembler(std::move(pipe.Value())));
    }
    if (type == "peer") {
        auto peer = AssemblePeer(config);
        if (!peer) { return peer.Error(); }
        return std::unique_ptr<StoreAssembler>(new StoreAssembler(std::move(peer.Value())));
    }
    return Status::InvalidParam("unknown config type '{}'", type);
}

StoreScheduler* StoreAssembler::GetScheduler()
{
    return std::visit([](auto& node) -> StoreScheduler* { return node.get(); }, root_);
}

StoreWorker* StoreAssembler::GetWorker()
{
    return std::visit([](auto& node) -> StoreWorker* { return node.get(); }, root_);
}

StoreAssembler::StoreAssembler(
    std::variant<std::unique_ptr<StoreEntry>, std::unique_ptr<StorePeer>> root)
    : root_(std::move(root))
{
}

}  // namespace UC::Proxy
