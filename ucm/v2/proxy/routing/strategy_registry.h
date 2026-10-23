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

#include <string>
#include <unordered_map>
#include "status/status.h"
#include "strategy.h"

namespace UC::Proxy {

/** @brief Factory entry point type of a RoutingStrategy, registered by name. */
using CreateRoutingStrategyFn = RoutingStrategy*();

/**
 * @brief Compile-time registry of RoutingStrategy implementations: maps
 *        configuration-facing names to factory entry points.
 *
 * Strategies are linked into the same binary as their consumer (the proxy
 * module) -- no extern "C" entry points and no separate destroy function:
 * instances live and die on the same heap as their creator, so the virtual
 * destructor suffices.
 *
 * Thread safety: registration happens during static initialization; Create()
 * may be called from any thread afterwards.
 */
class RoutingStrategyRegistry {
public:
    /** @brief Accesses the process-wide registry. */
    static RoutingStrategyRegistry& Instance()
    {
        static RoutingStrategyRegistry instance;
        return instance;
    }
    /**
     * @brief Registers a factory under the given name.
     *
     * @return true on success; false when the name is already taken (a
     *         duplicate registration is a programming error; the earlier
     *         entry stays in effect).
     */
    bool Register(const std::string& name, CreateRoutingStrategyFn* create)
    {
        return entries_.emplace(name, create).second;
    }
    /**
     * @brief Instantiates the RoutingStrategy registered under name.
     *
     * @return the new instance on success; Status::NotFound when no factory
     *         is registered under name, listing the registered names for
     *         diagnosis.
     */
    Expected<RoutingStrategy*> Create(const std::string& name) const
    {
        auto it = entries_.find(name);
        if (it == entries_.end()) {
            std::string available;
            for (const auto& entry : entries_) {
                if (!available.empty()) { available += ", "; }
                available += entry.first;
            }
            return Status::NotFound("no routing strategy registered under '{}' (available: {})",
                                    name, available);
        }
        return it->second();
    }

private:
    RoutingStrategyRegistry() = default;
    std::unordered_map<std::string, CreateRoutingStrategyFn*> entries_;
};

/**
 * @brief Registers a RoutingStrategy implementation in the compile-time
 *        registry.
 *
 * Invoke exactly once per implementation, at global scope, in the TU that
 * also defines StrategyClass:
 *   UC_PROXY_REGISTER_ROUTING_STRATEGY(TtlRoutingStrategy, "ttl")
 *
 * Requires StrategyClass to be default-constructible; the consumer passes
 * configuration afterwards via Setup(). strategyName is the
 * configuration-facing identifier the factory is looked up by.
 *
 * @note The class name must be passed unqualified (the macro derives a
 *       unique registrar symbol from it).
 * @note Unit tests of class behavior reach the class through the registry
 *       (Create by name), or instantiate it directly when a header exists;
 *       threshold strategies are header-less and register from their .cc.
 */
#define UC_PROXY_REGISTER_ROUTING_STRATEGY(StrategyClass, strategyName)                 \
    namespace {                                                                         \
    struct StrategyClass##Registrar {                                                   \
        StrategyClass##Registrar()                                                      \
        {                                                                               \
            ::UC::Proxy::RoutingStrategyRegistry::Instance().Register(                  \
                strategyName,                                                           \
                []() -> ::UC::Proxy::RoutingStrategy* { return new StrategyClass(); }); \
        };                                                                              \
    };                                                                                  \
    const StrategyClass##Registrar StrategyClass##RegistrarInstance;                    \
    }

}  // namespace UC::Proxy
