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

#include <iterator>
#include <memory>
#include <vector>
#include "store_entry.h"
#include "store_loader.h"

namespace UC::Proxy {

/* Owns the deeper tiers of a pipe. Listed before StoreEntry in the base list
 * on purpose: bases are destroyed in reverse declaration order, so the head
 * entry is torn down first while its lower tiers are still alive. */
struct StorePipeTailsHolder {
    std::vector<std::unique_ptr<LoadedStore>> piped_;
};

class StorePipe : private StorePipeTailsHolder, public StoreEntry {
public:
    explicit StorePipe(std::vector<std::unique_ptr<LoadedStore>> stores)
        : StoreEntry{std::move(stores.front())}
    {
        piped_.insert(piped_.end(), std::make_move_iterator(stores.begin() + 1),
                      std::make_move_iterator(stores.end()));
    }
};

}  // namespace UC::Proxy
