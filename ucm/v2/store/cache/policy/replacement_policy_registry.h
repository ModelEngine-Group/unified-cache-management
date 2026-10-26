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
#include "replacement_policy.h"

namespace UC::Store {

/** @brief Factory entry point type of a ReplacementPolicy, registered by name. */
using CreateReplacementPolicyFn = ReplacementPolicy*();

/**
 * @brief Compile-time registry of ReplacementPolicy implementations:
 *        maps configuration-facing names to factory entry points.
 *
 * Policies are linked into the same binary (or plugin) as their
 * consumer -- no extern "C" entry points and no separate destroy
 * function: instances live and die on the same heap as their creator,
 * so the virtual destructor suffices.
 *
 * Thread safety: registration happens during static initialization;
 * Create() may be called from any thread afterwards.
 */
class ReplacementPolicyRegistry {
public:
    /** @brief Accesses the process-wide registry. */
    static ReplacementPolicyRegistry& Instance()
    {
        static ReplacementPolicyRegistry instance;
        return instance;
    }
    /**
     * @brief Registers a factory under the given name.
     *
     * @return true on success; false when the name is already taken (a
     *         duplicate registration is a programming error; the earlier
     *         entry stays in effect).
     */
    bool Register(const std::string& name, CreateReplacementPolicyFn* create)
    {
        return entries_.emplace(name, create).second;
    }
    /**
     * @brief Instantiates the ReplacementPolicy registered under name.
     *
     * @return the new instance on success; Status::NotFound when no
     *         factory is registered under name, listing the registered
     *         names for diagnosis.
     */
    Expected<ReplacementPolicy*> Create(const std::string& name) const
    {
        auto it = entries_.find(name);
        if (it == entries_.end()) {
            std::string available;
            for (const auto& entry : entries_) {
                if (!available.empty()) { available += ", "; }
                available += entry.first;
            }
            return Status::NotFound("no policy registered under '{}' (available: {})", name,
                                    available);
        }
        return it->second();
    }

private:
    ReplacementPolicyRegistry() = default;
    std::unordered_map<std::string, CreateReplacementPolicyFn*> entries_;
};

/**
 * @brief Registers a ReplacementPolicy implementation in the compile-time
 *        registry.
 *
 * Invoke exactly once per implementation, at global scope, in the TU that
 * also defines PolicyClass:
 *   UC_STORE_REGISTER_REPLACEMENT_POLICY(FifoPolicy, "fifo")
 *
 * Requires PolicyClass to be default-constructible; the consumer passes
 * configuration afterwards via Setup(). policyName is the
 * configuration-facing identifier the factory is looked up by.
 *
 * @note The class name must be passed unqualified (the macro derives a
 *       unique registrar symbol from it).
 * @note Unit tests of class behavior should instantiate classes directly,
 *       without this macro.
 */
#define UC_STORE_REGISTER_REPLACEMENT_POLICY(PolicyClass, policyName)                   \
    namespace {                                                                         \
    struct PolicyClass##Registrar {                                                     \
        PolicyClass##Registrar()                                                        \
        {                                                                               \
            ::UC::Store::ReplacementPolicyRegistry::Instance().Register(                \
                policyName,                                                             \
                []() -> ::UC::Store::ReplacementPolicy* { return new PolicyClass(); }); \
        };                                                                              \
    };                                                                                  \
    const PolicyClass##Registrar PolicyClass##RegistrarInstance;                        \
    }

}  // namespace UC::Store
