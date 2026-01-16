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
#ifndef UNIFIEDCACHE_STORE_DETAIL_TEMPLATE_STORE_BINDER_H
#define UNIFIEDCACHE_STORE_DETAIL_TEMPLATE_STORE_BINDER_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "status/status.h"
#include "type/types.h"

namespace UC::Detail {

template <typename Store, typename Config>
class StoreBinder {
    template <typename T>
    struct BufferArrayView {
        const T* data;
        size_t num;
        BufferArrayView(const pybind11::buffer& buffer)
        {
            const auto info = buffer.request(false);
            data = static_cast<const T*>(info.ptr);
            const auto scale = sizeof(T) / info.itemsize;
            num = static_cast<size_t>(info.shape[0]) / scale;
        }
        const T* operator[](size_t i) const noexcept { return data + i; }
    };
    template <typename T>
    struct Buffer2DArrayView {
        const T* data;
        size_t rows, cols;
        Buffer2DArrayView(const pybind11::buffer& buffer)
        {
            const auto info = buffer.request(false);
            data = static_cast<const T*>(info.ptr);
            const auto scale = sizeof(T) / info.itemsize;
            rows = static_cast<size_t>(info.shape[0]) / scale;
            cols = static_cast<size_t>(info.shape[1]) / scale;
        }
        const T* operator[](size_t r) const noexcept { return data + r * cols; }
    };

private:
    std::unique_ptr<Store> store_;

public:
    StoreBinder() : store_{std::make_unique<Store>()} {}
    uintptr_t Self() { return (uintptr_t)(void*)store_.get(); }
    void Setup(const Config& config) { ThrowIfFailed(store_->Setup(config)); }
    pybind11::bytes Lookup(const pybind11::buffer& ids)
    {
        BufferArrayView<BlockId> idArr{ids};
        auto res = store_->Lookup(idArr.data, idArr.num);
        if (res) {
            auto& v = res.Value();
            return pybind11::bytes(reinterpret_cast<const char*>(v.data()), v.size());
        }
        throw std::runtime_error{res.Error().ToString()};
    }
    ssize_t LookupOnPrefix(const pybind11::buffer& ids)
    {
        BufferArrayView<BlockId> idArr{ids};
        auto res = store_->LookupOnPrefix(idArr.data, idArr.num);
        if (res) { return res.Value(); }
        throw std::runtime_error{res.Error().ToString()};
    }
    void Prefetch(const pybind11::buffer& ids)
    {
        BufferArrayView<BlockId> idArr{ids};
        store_->Prefetch(idArr.data, idArr.num);
    }
    TaskHandle Load(const pybind11::buffer& ids, const pybind11::buffer& indexes,
                    const pybind11::buffer& addrs)
    {
        auto desc = MakeTaskDesc(ids, indexes, addrs);
        desc.brief = "Load";
        auto res = store_->Load(std::move(desc));
        if (res) { return res.Value(); }
        throw std::runtime_error{res.Error().ToString()};
    }
    TaskHandle Dump(const pybind11::buffer& ids, const pybind11::buffer& indexes,
                    const pybind11::buffer& addrs)
    {
        auto desc = MakeTaskDesc(ids, indexes, addrs);
        desc.brief = "Dump";
        auto res = store_->Dump(desc);
        if (res) { return res.Value(); }
        throw std::runtime_error{res.Error().ToString()};
    }
    bool Check(TaskHandle taskId)
    {
        auto res = store_->Check(taskId);
        if (res) { return res.Value(); }
        throw std::runtime_error{res.Error().ToString()};
    }
    void Wait(TaskHandle taskId) { ThrowIfFailed(store_->Wait(taskId)); }

protected:
    virtual void ThrowIfFailed(const Status& s)
    {
        if (s.Failure()) [[unlikely]] { throw std::runtime_error{s.ToString()}; }
    }

private:
    TaskDesc MakeTaskDesc(const pybind11::buffer& ids, const pybind11::buffer& indexes,
                          const pybind11::buffer& addrs)
    {
        BufferArrayView<BlockId> idArr{ids};
        BufferArrayView<size_t> idxArr{indexes};
        Buffer2DArrayView<void*> addrArr{addrs};
        if (idArr.num != idxArr.num || idArr.num != addrArr.rows) {
            ThrowIfFailed(
                Status::InvalidParam("invalid dim: {},{},{}", idArr.num, idxArr.num, addrArr.rows));
        }
        TaskDesc desc;
        desc.reserve(idArr.num);
        for (size_t i = 0; i < idArr.num; i++) {
            Shard shard;
            shard.owner = *idArr[i];
            shard.index = *idxArr[i];
            shard.addrs.assign(addrArr[i], addrArr[i] + addrArr.cols);
            desc.push_back(std::move(shard));
        }
        return desc;
    }
};

}  // namespace UC::Detail

#endif
