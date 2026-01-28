/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#ifndef UNIFIEDCACHE_PIPELINE_STORE_CPY_LIBRARY_LOADER_H
#define UNIFIEDCACHE_PIPELINE_STORE_CPY_LIBRARY_LOADER_H

#include <dlfcn.h>
#include <memory>
#include "status/status.h"

namespace UC::PipelineStore {

template <class Interface>
class LibraryLoader {
public:
    LibraryLoader(std::string path, std::string func)
        : path_{std::move(path)}, func_{std::move(func)}
    {
    }
    ~LibraryLoader()
    {
        if (handle_) {
            dlclose(handle_);
            handle_ = nullptr;
        }
    }
    LibraryLoader(LibraryLoader&& other) noexcept
    {
        path_ = std::move(other.path_);
        func_ = std::move(other.func_);
        handle_ = other.handle_;
        other.handle_ = nullptr;
        maker_ = other.maker_;
        other.maker_ = nullptr;
    }
    LibraryLoader& operator=(LibraryLoader&& other) noexcept
    {
        if (this != &other) {
            if (handle_) { dlclose(handle_); }
            path_ = std::move(other.path_);
            func_ = std::move(other.func_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
            maker_ = other.maker_;
            other.maker_ = nullptr;
        }
        return *this;
    }
    Status LoadLibrary()
    {
        handle_ = dlopen(path_.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (!handle_) {
            return Status::Error(fmt::format("failed to load `{}`: {}", path_, dlerror()));
        }
        void* symbol = dlsym(handle_, func_.c_str());
        if (!symbol) {
            return Status::Error(fmt::format("cannot find `{}`: {}", func_, dlerror()));
        }
        maker_ = reinterpret_cast<MakerFn>(symbol);
        return Status::OK();
    }
    std::shared_ptr<Interface> CreateObject() { return std::shared_ptr<Interface>(maker_()); }

private:
    using MakerFn = Interface* (*)();
    std::string path_;
    std::string func_;
    void* handle_{nullptr};
    MakerFn maker_{nullptr};
};

}  // namespace UC::PipelineStore

#endif
