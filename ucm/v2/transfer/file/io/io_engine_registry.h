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

#include <array>
#include <cstddef>
#include <string>
#include "io_engine.h"
#include "status/status.h"

namespace UC::Transfer {

/** @brief Factory entry point type of an IoEngine, registered by kind. */
using CreateIoEngineFn = IoEngine*();

/**
 * @brief Implementation family of IoEngine; the declaration order
 *        is the default-selection priority, ascending.
 *
 * A build links only the engines it supports (see the file transfer
 * build files), and the registry resolves the default as the
 * highest-priority registered IoEngineKind.
 *
 * @note Maintenance: new engines are inserted before Count and must
 *       gain a IoEngineKindName case (the build keeps that switch
 *       exhaustive); moving an enumerator deliberately re-ranks the
 *       default selection. Numeric values are never persisted --
 *       only the symbols are referenced -- so insertion is safe.
 */
enum class IoEngineKind {
    Psync,
    Aio,

    /** sentinel: the number of kinds, not a kind itself */
    Count
};

/**
 * @brief Configuration- and diagnosis-facing name of an IoEngineKind;
 *        the single source of the name strings.
 */
inline const char* IoEngineKindName(IoEngineKind kind)
{
    switch (kind) {
        case IoEngineKind::Psync: return "psync";
        case IoEngineKind::Aio: return "aio";
        case IoEngineKind::Count: break;
    }
    return "unknown";
}

/**
 * @brief Reverse of IoEngineKindName: the IoEngineKind going by
 *        name.
 *
 * @return the IoEngineKind on success; Status::NotFound listing
 *         the valid names otherwise. An empty name is invalid --
 *         callers wanting this build's default should pass an
 *         empty name to IoEngineRegistry::Create() instead.
 */
inline Expected<IoEngineKind> IoEngineKindFromName(const std::string& name)
{
    for (std::size_t i = 0; i < static_cast<std::size_t>(IoEngineKind::Count); ++i) {
        if (name == IoEngineKindName(static_cast<IoEngineKind>(i))) {
            return static_cast<IoEngineKind>(i);
        }
    }
    std::string valid;
    for (std::size_t i = 0; i < static_cast<std::size_t>(IoEngineKind::Count); ++i) {
        if (!valid.empty()) { valid += ", "; }
        valid += IoEngineKindName(static_cast<IoEngineKind>(i));
    }
    return Status::NotFound("no IO engine kind named '{}' (valid: {})", name, valid);
}

/**
 * @brief Compile-time registry of IoEngine implementations: maps
 *        IoEngineKind to factory entry points, creates engines by
 *        kind or by kind name, and resolves this build's default
 *        engine as the highest-priority registered kind.
 *
 * A build links only the engines it supports, so the default may
 * differ per build while the priority order itself lives once, in
 * IoEngineKind.
 *
 * IO engines are linked into the same binary as their consumer (the
 * file transfer module) -- no extern "C" entry points and no separate
 * destroy function: instances live and die on the same heap as their
 * creator, so the virtual destructor suffices.
 *
 * Thread safety: registration happens during static initialization; Create()
 * may be called from any thread afterwards.
 */
class IoEngineRegistry {
public:
    /** @brief Accesses the process-wide registry. */
    static IoEngineRegistry& Instance()
    {
        static IoEngineRegistry instance;
        return instance;
    }
    /**
     * @brief Registers a factory under the given kind.
     *
     * @return true on success; false when kind is the Count sentinel
     *         or already taken (a duplicate registration is a
     *         programming error; the earlier entry stays in effect).
     */
    bool Register(IoEngineKind kind, CreateIoEngineFn* create)
    {
        const std::size_t index = static_cast<std::size_t>(kind);
        if (index >= kKindCount || slots_[index] != nullptr) { return false; }
        slots_[index] = create;
        return true;
    }
    /**
     * @brief Instantiates the IoEngine whose kind goes by name; an
     *        empty name takes this build's default engine: the
     *        highest-priority registered kind.
     *
     * @return the new instance on success; Status::NotFound when name
     *         denotes no kind (listing the valid names) or its kind has
     *         no factory registered in this build (listing the
     *         registered kinds for diagnosis).
     */
    Expected<IoEngine*> Create(const std::string& name = "") const
    {
        if (name.empty()) { return CreateDefault(); }
        const auto kind = IoEngineKindFromName(name);
        if (!kind.HasValue()) { return kind.Error(); }
        return Create(kind.Value());
    }
    /**
     * @brief Instantiates the IoEngine registered under kind.
     *
     * @return the new instance on success; Status::NotFound when no factory
     *         is registered under kind, listing the registered kinds for
     *         diagnosis.
     */
    Expected<IoEngine*> Create(IoEngineKind kind) const
    {
        const std::size_t index = static_cast<std::size_t>(kind);
        if (index >= kKindCount || slots_[index] == nullptr) {
            return Status::NotFound("no IO engine registered under '{}' (available: {})",
                                    IoEngineKindName(kind), AvailableKinds());
        }
        return slots_[index]();
    }

private:
    IoEngineRegistry() = default;
    /**
     * @brief The empty-name arm of Create: instantiates this build's
     *        default IoEngine, the highest-priority registered kind.
     *
     * @return the new instance on success; Status::NotFound when no
     *         engine is registered in this build.
     */
    Expected<IoEngine*> CreateDefault() const
    {
        for (std::size_t i = kKindCount; i > 0; --i) {
            const CreateIoEngineFn* const slot = slots_[i - 1];
            if (slot != nullptr) { return (*slot)(); }
        }
        return Status::NotFound("no IO engine registered in this build");
    }
    std::string AvailableKinds() const
    {
        std::string available;
        for (std::size_t i = 0; i < kKindCount; ++i) {
            if (slots_[i] == nullptr) { continue; }
            if (!available.empty()) { available += ", "; }
            available += IoEngineKindName(static_cast<IoEngineKind>(i));
        }
        return available;
    }
    static constexpr std::size_t kKindCount = static_cast<std::size_t>(IoEngineKind::Count);
    std::array<CreateIoEngineFn*, kKindCount> slots_ = {};
};

/**
 * @brief Registers an IoEngine implementation in the compile-time
 *        registry.
 *
 * Invoke exactly once per implementation, at global scope, in the TU that
 * also defines EngineClass:
 *   UC_TRANSFER_REGISTER_IO_ENGINE(AioIoEngine, IoEngineKind::Aio)
 *
 * Requires EngineClass to be default-constructible. engineKind is the
 * IoEngineKind the factory is registered under; its position in
 * IoEngineKind decides the default-selection order.
 *
 * @note The class name must be passed unqualified (the macro derives a
 *       unique registrar symbol from it).
 */
#define UC_TRANSFER_REGISTER_IO_ENGINE(EngineClass, engineKind)                               \
    namespace {                                                                               \
    struct EngineClass##Registrar {                                                           \
        EngineClass##Registrar()                                                              \
        {                                                                                     \
            ::UC::Transfer::IoEngineRegistry::Instance().Register(                            \
                engineKind, []() -> ::UC::Transfer::IoEngine* { return new EngineClass(); }); \
        };                                                                                    \
    };                                                                                        \
    const EngineClass##Registrar EngineClass##RegistrarInstance;                              \
    }

}  // namespace UC::Transfer
