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
#include "dma_engine.h"
#include "status/status.h"

namespace UC::Transfer {

/** @brief Factory entry point type of a DmaEngine, registered by kind. */
using CreateDmaEngineFn = DmaEngine*();

/**
 * @brief Implementation family of DmaEngine; the declaration order
 *        is the default-selection priority, ascending.
 *
 * A build links only the engines its runtime environment supports
 * (see the memory transfer build files), and the registry resolves
 * the default as the highest-priority registered DmaEngineKind --
 * different runtime environments therefore pick different defaults
 * from the same order.
 *
 * @note Maintenance: new engines are inserted before Count and must
 *       gain a DmaEngineKindName case (the build keeps that switch
 *       exhaustive); moving an enumerator deliberately re-ranks the
 *       default selection. Numeric values are never persisted --
 *       only the symbols are referenced -- so insertion is safe.
 */
enum class DmaEngineKind {
    Gdr,
    Naive,
    PinBounce,
    FftsBounce,
    Ffts,

    /** sentinel: the number of kinds, not a kind itself */
    Count
};

/**
 * @brief Configuration- and diagnosis-facing name of a DmaEngineKind;
 *        the single source of the name strings.
 */
inline const char* DmaEngineKindName(DmaEngineKind kind)
{
    switch (kind) {
        case DmaEngineKind::Gdr: return "gdr";
        case DmaEngineKind::Naive: return "naive";
        case DmaEngineKind::PinBounce: return "pin-bounce";
        case DmaEngineKind::FftsBounce: return "ffts-bounce";
        case DmaEngineKind::Ffts: return "ffts";
        case DmaEngineKind::Count: break;
    }
    return "unknown";
}

/**
 * @brief Reverse of DmaEngineKindName: the DmaEngineKind going by
 *        name.
 *
 * @return the DmaEngineKind on success; Status::NotFound listing
 *         the valid names otherwise. An empty name is invalid --
 *         callers wanting this build's default should pass an
 *         empty name to DmaEngineRegistry::Create() instead.
 */
inline Expected<DmaEngineKind> DmaEngineKindFromName(const std::string& name)
{
    for (std::size_t i = 0; i < static_cast<std::size_t>(DmaEngineKind::Count); ++i) {
        if (name == DmaEngineKindName(static_cast<DmaEngineKind>(i))) {
            return static_cast<DmaEngineKind>(i);
        }
    }
    std::string valid;
    for (std::size_t i = 0; i < static_cast<std::size_t>(DmaEngineKind::Count); ++i) {
        if (!valid.empty()) { valid += ", "; }
        valid += DmaEngineKindName(static_cast<DmaEngineKind>(i));
    }
    return Status::NotFound("no DMA engine kind named '{}' (valid: {})", name, valid);
}

/**
 * @brief Compile-time registry of DmaEngine implementations: maps
 *        DmaEngineKind to factory entry points, creates engines by
 *        kind or by kind name, and resolves this build's default
 *        engine as the highest-priority registered kind.
 *
 * A build links only the engines its runtime environment supports,
 * so the default differs per runtime environment while the priority
 * order itself lives once, in DmaEngineKind.
 *
 * DMA engines are linked into the same binary as their consumer (the
 * memory transfer module) -- no extern "C" entry points and no separate
 * destroy function: instances live and die on the same heap as their
 * creator, so the virtual destructor suffices.
 *
 * Thread safety: registration happens during static initialization; Create()
 * may be called from any thread afterwards.
 */
class DmaEngineRegistry {
public:
    /** @brief Accesses the process-wide registry. */
    static DmaEngineRegistry& Instance()
    {
        static DmaEngineRegistry instance;
        return instance;
    }
    /**
     * @brief Registers a factory under the given kind.
     *
     * @return true on success; false when kind is the Count sentinel
     *         or already taken (a duplicate registration is a
     *         programming error; the earlier entry stays in effect).
     */
    bool Register(DmaEngineKind kind, CreateDmaEngineFn* create)
    {
        const std::size_t index = static_cast<std::size_t>(kind);
        if (index >= kKindCount || slots_[index] != nullptr) { return false; }
        slots_[index] = create;
        return true;
    }
    /**
     * @brief Instantiates the DmaEngine whose kind goes by name; an
     *        empty name takes this build's default engine: the
     *        highest-priority registered kind.
     *
     * @return the new instance on success; Status::NotFound when name
     *         denotes no kind (listing the valid names) or its kind has
     *         no factory registered in this build (listing the
     *         registered kinds for diagnosis).
     */
    Expected<DmaEngine*> Create(const std::string& name = "") const
    {
        if (name.empty()) { return CreateDefault(); }
        const auto kind = DmaEngineKindFromName(name);
        if (!kind.HasValue()) { return kind.Error(); }
        return Create(kind.Value());
    }
    /**
     * @brief Instantiates the DmaEngine registered under kind.
     *
     * @return the new instance on success; Status::NotFound when no factory
     *         is registered under kind, listing the registered kinds for
     *         diagnosis.
     */
    Expected<DmaEngine*> Create(DmaEngineKind kind) const
    {
        const std::size_t index = static_cast<std::size_t>(kind);
        if (index >= kKindCount || slots_[index] == nullptr) {
            return Status::NotFound("no DMA engine registered under '{}' (available: {})",
                                    DmaEngineKindName(kind), AvailableKinds());
        }
        return slots_[index]();
    }

private:
    DmaEngineRegistry() = default;
    /**
     * @brief The empty-name arm of Create: instantiates this build's
     *        default DmaEngine, the highest-priority registered kind.
     *
     * @return the new instance on success; Status::NotFound when no
     *         engine is registered in this build.
     */
    Expected<DmaEngine*> CreateDefault() const
    {
        for (std::size_t i = kKindCount; i > 0; --i) {
            const CreateDmaEngineFn* const slot = slots_[i - 1];
            if (slot != nullptr) { return (*slot)(); }
        }
        return Status::NotFound("no DMA engine registered in this build");
    }
    std::string AvailableKinds() const
    {
        std::string available;
        for (std::size_t i = 0; i < kKindCount; ++i) {
            if (slots_[i] == nullptr) { continue; }
            if (!available.empty()) { available += ", "; }
            available += DmaEngineKindName(static_cast<DmaEngineKind>(i));
        }
        return available;
    }
    static constexpr std::size_t kKindCount = static_cast<std::size_t>(DmaEngineKind::Count);
    std::array<CreateDmaEngineFn*, kKindCount> slots_ = {};
};

/**
 * @brief Registers a DmaEngine implementation in the compile-time
 *        registry.
 *
 * Invoke exactly once per implementation, at global scope, in the TU that
 * also defines EngineClass:
 *   UC_TRANSFER_REGISTER_DMA_ENGINE(NaiveDmaEngine, DmaEngineKind::Naive)
 *
 * Requires EngineClass to be default-constructible. engineKind is the
 * DmaEngineKind the factory is registered under; its position in
 * DmaEngineKind decides the default-selection order.
 *
 * @note The class name must be passed unqualified (the macro derives a
 *       unique registrar symbol from it).
 */
#define UC_TRANSFER_REGISTER_DMA_ENGINE(EngineClass, engineKind)                               \
    namespace {                                                                                \
    struct EngineClass##Registrar {                                                            \
        EngineClass##Registrar()                                                               \
        {                                                                                      \
            ::UC::Transfer::DmaEngineRegistry::Instance().Register(                            \
                engineKind, []() -> ::UC::Transfer::DmaEngine* { return new EngineClass(); }); \
        };                                                                                     \
    };                                                                                         \
    const EngineClass##Registrar EngineClass##RegistrarInstance;                               \
    }

}  // namespace UC::Transfer
