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
#include <string>
#include "status/status.h"
#include "store.h"

namespace UC::Proxy {

/**
 * @brief Owning handle of one Store plugin instance.
 *
 * Owns both the instance and (when loaded from a plugin) the dlopen handle.
 * Destruction order: the instance is destroyed through the plugin's own
 * UCDestroyStore entry point (same-heap destruction), then the plugin is
 * dlclose()d. Move-only.
 */
class LoadedStore {
public:
    /**
     * @brief Take ownership of a heap Store created by plain `new`, with no
     *        plugin handle behind it (tests and embedding).
     */
    static std::unique_ptr<LoadedStore> Adopt(std::unique_ptr<Store::Store> store);
    ~LoadedStore();
    LoadedStore(LoadedStore&& other) noexcept;
    LoadedStore& operator=(LoadedStore&& other) noexcept;
    LoadedStore(const LoadedStore&) = delete;
    LoadedStore& operator=(const LoadedStore&) = delete;

    /** The owned Store instance; never null after a successful load. */
    Store::Store* Store() const { return store_; }

private:
    friend class StoreLoader;

    LoadedStore(void* handle, Store::Store* store, Store::DestroyStoreFn* destroy);

    void Close();

    void* handle_;
    Store::Store* store_;
    Store::DestroyStoreFn* destroy_;
};

/**
 * @brief Loads a Store plugin and creates one instance from its factory
 *        entry points, by built-in name or by path.
 *
 * Follows the store.h loading contract: dlopen with RTLD_LOCAL | RTLD_NOW,
 * resolve UCCreateStore/UCDestroyStore via dlsym on the plugin's own handle
 * (never RTLD_DEFAULT).
 *
 * Name resolution: a value containing '/' or ending in ".so" is dlopened as
 * an (out-of-tree) plugin path; a plain name is looked up in the built-in
 * table auto-generated from this build's ucm_add_store registrations and
 * the entry's path is dlopened through the same factory chain. Unknown
 * plain names fail with InvalidParam, listing the known built-ins.
 */
class StoreLoader {
public:
    /**
     * @brief Load the plugin at `nameOrPath` and create one Store instance.
     *
     * @return A LoadedStore owning the instance and the plugin handle, or
     *         an error Status (InvalidParam on an unknown built-in name,
     *         OsApiError on dlopen failure, NotFound when the factory entry
     *         points are missing).
     */
    static Expected<std::unique_ptr<LoadedStore>> Load(const std::string& nameOrPath);
};

}  // namespace UC::Proxy
