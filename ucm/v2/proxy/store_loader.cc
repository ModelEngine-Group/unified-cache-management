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
#include "store_loader.h"
#include <dlfcn.h>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace UC::Proxy {

// Defined in the build-generated builtin_stores.cc, compiled from the
// ucm_add_store registrations of this build: built-in name -> plugin path.
const std::vector<std::pair<std::string, std::string>>& BuiltinStoreTable();

namespace {

void DeleteStore(Store::Store* store) { delete store; }

bool IsPathLike(const std::string& nameOrPath)
{
    const bool hasSlash = nameOrPath.find('/') != std::string::npos;
    const bool soSuffix =
        nameOrPath.size() >= 3 && nameOrPath.compare(nameOrPath.size() - 3, 3, ".so") == 0;
    return hasSlash || soSuffix;
}

}  // namespace

LoadedStore::LoadedStore(void* handle, Store::Store* store, Store::DestroyStoreFn* destroy)
    : handle_(handle), store_(store), destroy_(destroy)
{
}

void LoadedStore::Close()
{
    if (destroy_ != nullptr && store_ != nullptr) { destroy_(store_); }
    if (handle_ != nullptr) { dlclose(handle_); }
    handle_ = nullptr;
    store_ = nullptr;
    destroy_ = nullptr;
}

LoadedStore::~LoadedStore() { Close(); }

LoadedStore::LoadedStore(LoadedStore&& other) noexcept
    : handle_(other.handle_), store_(other.store_), destroy_(other.destroy_)
{
    other.handle_ = nullptr;
    other.store_ = nullptr;
    other.destroy_ = nullptr;
}

LoadedStore& LoadedStore::operator=(LoadedStore&& other) noexcept
{
    if (this != &other) {
        Close();
        handle_ = other.handle_;
        store_ = other.store_;
        destroy_ = other.destroy_;
        other.handle_ = nullptr;
        other.store_ = nullptr;
        other.destroy_ = nullptr;
    }
    return *this;
}

std::unique_ptr<LoadedStore> LoadedStore::Adopt(std::unique_ptr<Store::Store> store)
{
    auto* loaded = new (std::nothrow) LoadedStore(nullptr, store.get(), &DeleteStore);
    if (loaded == nullptr) { return nullptr; }
    store.release();
    return std::unique_ptr<LoadedStore>(loaded);
}

Expected<std::unique_ptr<LoadedStore>> StoreLoader::Load(const std::string& nameOrPath)
{
    std::string path = nameOrPath;
    if (!IsPathLike(nameOrPath)) {
        const std::string* builtinPath = nullptr;
        std::string known;
        for (const auto& entry : BuiltinStoreTable()) {
            if (!known.empty()) { known += ", "; }
            known += entry.first;
            if (entry.first == nameOrPath) { builtinPath = &entry.second; }
        }
        if (builtinPath == nullptr) {
            return Status::InvalidParam("unknown built-in store '{}'; known built-ins: {}",
                                        nameOrPath, known);
        }
        path = *builtinPath;
    }
    void* handle = dlopen(path.c_str(), RTLD_LOCAL | RTLD_NOW);
    if (handle == nullptr) {
        const char* error = dlerror();
        return Status::OsApiError("dlopen('{}') failed: {}", path,
                                  error != nullptr ? error : "unknown error");
    }
    auto create = reinterpret_cast<Store::CreateStoreFn*>(dlsym(handle, "UCCreateStore"));
    auto destroy = reinterpret_cast<Store::DestroyStoreFn*>(dlsym(handle, "UCDestroyStore"));
    if (create == nullptr || destroy == nullptr) {
        std::string missing;
        if (create == nullptr) { missing += "UCCreateStore"; }
        if (destroy == nullptr) {
            if (!missing.empty()) { missing += ", "; }
            missing += "UCDestroyStore";
        }
        dlclose(handle);
        return Status::NotFound("factory entry point(s) [{}] not found in '{}'", missing, path);
    }
    auto* store = (*create)();
    if (store == nullptr) {
        dlclose(handle);
        return Status::General("factory returned null instance for '{}'", path);
    }
    auto* loaded = new (std::nothrow) LoadedStore(handle, store, destroy);
    if (loaded == nullptr) {
        destroy(store);
        dlclose(handle);
        return Status::OutOfMemory("failed to allocate LoadedStore for '{}'", path);
    }
    return std::unique_ptr<LoadedStore>(loaded);
}

}  // namespace UC::Proxy
