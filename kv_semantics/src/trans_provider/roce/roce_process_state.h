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
#ifndef KV_TRANS_PROVIDER_ROCE_PROCESS_STATE_H
#define KV_TRANS_PROVIDER_ROCE_PROCESS_STATE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include "roce_verb_qp.h"
#include "trans_provider.h"

namespace kv {
namespace roce {

// One open protection domain: everything a provider needs to talk to one port
// of one adapter.
struct DeviceEntry {
    VerbDevice device;
    // Adapter the device was actually opened on, resolved from the driver when
    // the configuration left the name empty.
    std::string name;
    std::uint32_t port{0};
};

// Memory handed to the client, keyed by the handle it receives back.  A
// registered region owns a verbs access record whose rkey is the token; a bound
// region is pure bookkeeping that carries the peer's token through unchanged
// (mr stays null, so releasing it never touches the device) and points back at
// the registration it was bound to.
struct Registration {
    void* mr{nullptr};
    std::uintptr_t addr{0};
    std::size_t size{0};
    std::uint32_t tokenId{0};
    // Handle of the registration this binding was made from; zero for an entry
    // that RegisterMemory() created.
    MRHandle source{0};
    // Provider that created the entry, so a provider only ever releases what it
    // owns while the rest stays available to the others.
    const void* owner{nullptr};
    // Device whose protection domain produced the access record.  An `rkey` is
    // only valid on that device, so the registration can be bound by a provider
    // of the same (adapter, port) pair and by nobody else.  Null for simulated
    // registrations, which have no access record at all.
    const VerbDevice* device{nullptr};
    // Address space the client registered the region in.  A registration made
    // for a queue pair has to be addressed by the CPU (it posts the SGE itself),
    // so the data path only accepts host registrations.
    TransProvider::MemType memoryType{TransProvider::MemType::MEM_HOST};
};

struct DeviceKey {
    std::string name;
    std::uint32_t port{0};
    // A port has one GID per address it carries, and the queue pairs of the
    // devices below are built around that index, so it has to be part of the
    // identity of a protection domain as well.
    std::uint32_t gidIndex{0};

    bool operator==(const DeviceKey& other) const noexcept
    {
        return port == other.port && gidIndex == other.gidIndex && name == other.name;
    }
};

struct DeviceKeyHash {
    std::size_t operator()(const DeviceKey& key) const noexcept;
};

// State every provider instance in the process shares.
//
// The store side builds one provider per ASU node, so several of them live in
// the same process and talk to the same HCA.  Opening the device once keeps the
// protection domain, and therefore every registered region, usable from all of
// them: a region registered through one provider can be bound by another that
// resolves the address in this table.
struct ProcessState {
    static ProcessState& Instance();

    // Guards devices.  Devices outlive every provider: they are released when
    // the process exits, never when one provider is destroyed.
    std::mutex deviceMutex;
    // Keyed by the adapter, the port and the GID they were opened with, because
    // a protection domain only serves the port and the local address it was
    // created for while a process may well drive several adapters, ports or
    // addresses (one per NUMA node, say).
    std::unordered_map<DeviceKey, std::unique_ptr<DeviceEntry>, DeviceKeyHash> devices;

    // Guards registrations and nextBookkeepingId.
    std::mutex memoryMutex;
    std::unordered_map<MRHandle, Registration> registrations;
    // Simulated registrations, bindings and the shared handle space all draw
    // from one counter so two kinds of handle can never alias each other, not
    // even across providers.
    std::uint64_t nextBookkeepingId{1};
};

}  // namespace roce
}  // namespace kv

#endif  // KV_TRANS_PROVIDER_ROCE_PROCESS_STATE_H
