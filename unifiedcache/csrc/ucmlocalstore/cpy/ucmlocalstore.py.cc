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
#include <list>
#include <future>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "status/status.h"
#include "cache/cache_manager.h"
#include "template/singleton.h"
#include "logger/logger.h"

namespace py = pybind11;

namespace UCM {

struct Id {
    size_t taskId;
    size_t cacheId;
};

class UcmKVStoreBase {
public:
    UcmKVStoreBase() = default;
    virtual ~UcmKVStoreBase() = default;
    virtual int32_t Setup() = 0;
    virtual int32_t AllocBatch(const py::list& blockIds) = 0;
    virtual py::list LookupBatch(const py::list& blockIds) = 0;
    virtual Id LoadToDevice(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList, const std::list<uintptr_t>& addressList,
                        const std::list<size_t>& lengthList) = 0;
    virtual size_t LoadToHost(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList, const std::list<uintptr_t>& addressList,
                        const std::list<size_t>& lengthList) = 0;
    virtual size_t DumpFromDevice(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList, const std::list<uintptr_t>& addressList,
                        const std::list<size_t>& lengthList) = 0;
    virtual size_t DumpFromHost(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList, const std::list<uintptr_t>& addressList,
                        const std::list<size_t>& lengthList) = 0;
    virtual int32_t Wait(const Id id) = 0;
    virtual void CommitBatch(const py::list& blockIds, const bool success) = 0;
};

class UcmLocalStore : public UcmKVStoreBase {
public:
    class Config {
    public:
        size_t capacity;
        size_t cacheSize;
        int32_t transferDeviceId;
        size_t transferIoSize;

        Config(const size_t capacity, const size_t cacheSize)
            : capacity{capacity}, cacheSize{cacheSize},
              transferDeviceId{-1}, transferIoSize{262144}
        {
        }
    };

public:
    UcmLocalStore(const Config &config, void *store)
        : _config{config}, _store{static_cast<UcmKVStoreBase*>(store)}
    {
    }
    int32_t Setup() override;
    int32_t AllocBatch(const py::list& blockIds) override;
    py::list LookupBatch(const py::list& blockIds) override;
    Id LoadToDevice(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList,
                const std::list<uintptr_t>& addressList, const std::list<size_t>& lengthList) override;
    size_t LoadToHost(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList, const std::list<uintptr_t>& addressList,
                const std::list<size_t>& lengthList) override;
    size_t DumpFromDevice(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList, const std::list<uintptr_t>& addressList,
                const std::list<size_t>& lengthList) override;
    size_t DumpFromHost(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList, const std::list<uintptr_t>& addressList,
                const std::list<size_t>& lengthList) override;
    int32_t Wait(const Id id) override;
    void CommitBatch(const py::list& blockIds, const bool success) override;

private:
    Config _config;
    UcmKVStoreBase *_store;
};

void ShowSetupParam(const UcmLocalStore::Config& config)
{
    UCM_INFO("Set UC::Capacity to {}.", config.capacity);
    UCM_INFO("Set UC::CacheSize to {}.", config.cacheSize);
    UCM_INFO("Set UC::TransferDeviceId to {}.", config.transferDeviceId);
    UCM_INFO("Set UC::TransferIoSize to {}.", config.transferIoSize);
}

int32_t UcmLocalStore::Setup()
{
    auto cacheMgr = Singleton<CacheManager>::Instance();
    auto status = cacheMgr->Setup(this->_config.capacity, this->_config.cacheSize,
        this->_config.transferDeviceId, this->_config.transferIoSize);
    if (status.Failure()) {
        UCM_ERROR("Failed({}) to setup LRUCache.", status);
        return status.Underlying();
    }
    ShowSetupParam(this->_config);
    return Status::OK().Underlying();
}

int32_t UcmLocalStore::AllocBatch(const py::list& blockIds)
{
    return this->_store->AllocBatch(blockIds);
}

py::list UcmLocalStore::LookupBatch(const py::list& blockIds)
{
    return this->_store->LookupBatch(blockIds);
}

py::tuple FilterByMissList(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList,
                           const std::list<uintptr_t>& addressList, const std::list<size_t>& lengthList,
                           const std::list<std::string>& misslist)
{
    std::unordered_set<std::string> missSet(misslist.begin(), misslist.end());
    std::list<std::string> missBlockIds;
    std::list<size_t> missOffsets;
    std::list<uintptr_t> missAddresses;
    std::list<size_t> missLengths;
    auto itBlock = blockIdList.begin();
    auto itOffset = offsetList.begin();
    auto itAddr = addressList.begin();
    auto itLength = lengthList.begin();
    for (; itBlock != blockIdList.end(); ++itBlock, ++itOffset, ++itAddr, ++itLength) {
        if (missSet.find(*itBlock) != missSet.end()) {
            missBlockIds.push_back(*itBlock);
            missOffsets.push_back(*itOffset);
            missAddresses.push_back(*itAddr);
            missLengths.push_back(*itLength);
        }
    }
    return py::make_tuple(missBlockIds, missOffsets, missAddresses, missLengths);
}

Id UcmLocalStore::LoadToDevice(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList,
                               const std::list<uintptr_t>& addressList, const std::list<size_t>& lengthList)
{
    auto cacheMgr = Singleton<CacheManager>::Instance();
    std::list<std::string> missList;
    size_t cacheId = cacheMgr->GetCacheId();
    auto itBlock = blockIdList.begin();
    auto itAddr = addressList.begin();
    auto itLen = lengthList.begin();
    auto itOffset = offsetList.begin();
    for (; itBlock != blockIdList.end(); ++itBlock, ++itAddr, ++itLen, ++itOffset) {
        std::string blockId = *itBlock;
        uintptr_t address = *itAddr;
        size_t length = *itLen;
        size_t offset = *itOffset;
        auto status = cacheMgr->Cache2Device(blockId, address, length, offset);
        if (status == Status::NotFound()) {
            missList.push_back(blockId);
        }
    }

    if (missList.empty()) {
        return Id{0, cacheId};
    }

    auto result = FilterByMissList(blockIdList, offsetList,
        addressList, lengthList, missList);
    auto missBlockIds = result[0].cast<std::list<std::string>>();
    auto missOffsets = result[1].cast<std::list<size_t>>();
    auto missAddresses = result[2].cast<std::list<uintptr_t>>();
    auto missLengths = result[3].cast<std::list<size_t>>();
    Id id = this->_store->LoadToDevice(missBlockIds, missOffsets, missAddresses, missLengths);
    size_t taskId = id.taskId;

    auto fut = std::async(std::launch::async, [this, missBlockIds, cacheMgr, id]() {
        std::list<uintptr_t> buffers;
        auto status = cacheMgr->AllocBuffers(missBlockIds, buffers);
        if (status.Failure()) {
            UCM_ERROR("Failed({}) to alloc buffers.", status);
            return;
        }
        std::list<size_t> nfOffsets, nfLengths;
        size_t blockSize = this->_config.cacheSize;
        for (size_t i = 0; i < buffers.size(); ++i) {
            nfOffsets.push_back(0);
            nfLengths.push_back(blockSize);
        }
        this->_store->LoadToHost(missBlockIds, nfOffsets, buffers, nfLengths);
        size_t ret = this->_store->Wait(id);
        if (ret == 0){
            cacheMgr->CommitBuffers(buffers);
        }
        UCM_INFO("Async add blocks to dram success.");
    });
    return Id{taskId, cacheId};
}

size_t UcmLocalStore::LoadToHost(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList,
                                 const std::list<uintptr_t>& addressList, const std::list<size_t>& lengthList)
{
    return this->_store->LoadToHost(blockIdList, offsetList, addressList, lengthList);
}

size_t UcmLocalStore::DumpFromDevice(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList,
                                     const std::list<uintptr_t>& addressList, const std::list<size_t>& lengthList)
{
    return this->_store->DumpFromDevice(blockIdList, offsetList, addressList, lengthList);
}

size_t UcmLocalStore::DumpFromHost(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList,
                                   const std::list<uintptr_t>& addressList, const std::list<size_t>& lengthList)
{
    return this->_store->DumpFromHost(blockIdList, offsetList, addressList, lengthList);
}

int32_t UcmLocalStore::Wait(const Id id)
{
    if (id.cacheId == 0) {
        return this->_store->Wait(id);
    }

    auto cacheMgr = Singleton<CacheManager>::Instance();
    if (id.taskId == 0) {
        return cacheMgr->Wait(id.cacheId).Underlying();
    }

    return cacheMgr->Wait(id.cacheId).Underlying() && this->_store->Wait(id);
}

void UcmLocalStore::CommitBatch(const py::list& blockIds, const bool success)
{
    this->_store->CommitBatch(blockIds, success);
}

} // namespace UCM

PYBIND11_MODULE(ucmlocalstore, module)
{
    auto ucmKVStoreBase = py::class_<UCM::UcmKVStoreBase>(module, "UcmKVStoreBase");
    ucmKVStoreBase.def("Setup", &UCM::UcmKVStoreBase::Setup);
    ucmKVStoreBase.def("AllocBatch", &UCM::UcmKVStoreBase::AllocBatch);
    ucmKVStoreBase.def("LookupBatch", &UCM::UcmKVStoreBase::LookupBatch);
    ucmKVStoreBase.def("LoadToDevice", &UCM::UcmKVStoreBase::LoadToDevice);
    ucmKVStoreBase.def("LoadToHost", &UCM::UcmKVStoreBase::LoadToHost);
    ucmKVStoreBase.def("DumpFromDevice", &UCM::UcmKVStoreBase::DumpFromDevice);
    ucmKVStoreBase.def("DumpFromHost", &UCM::UcmKVStoreBase::DumpFromHost);
    ucmKVStoreBase.def("Wait", &UCM::UcmKVStoreBase::Wait);
    ucmKVStoreBase.def("CommitBatch", &UCM::UcmKVStoreBase::CommitBatch);

    auto ucmLocalStore = py::class_<UCM::UcmLocalStore>(module, "UcmLocalStore");
    ucmLocalStore.def(py::init<const UCM::UcmLocalStore::Config&, void*>());
    ucmLocalStore.def("Setup", &UCM::UcmLocalStore::Setup);
    ucmLocalStore.def("AllocBatch", &UCM::UcmLocalStore::AllocBatch);
    ucmLocalStore.def("LookupBatch", &UCM::UcmLocalStore::LookupBatch);
    ucmLocalStore.def("LoadToDevice", &UCM::UcmLocalStore::LoadToDevice);
    ucmLocalStore.def("LoadToHost", &UCM::UcmLocalStore::LoadToHost);
    ucmLocalStore.def("DumpFromDevice", &UCM::UcmLocalStore::DumpFromDevice);
    ucmLocalStore.def("DumpFromHost", &UCM::UcmLocalStore::DumpFromHost);
    ucmLocalStore.def("Wait", &UCM::UcmLocalStore::Wait);
    ucmLocalStore.def("CommitBatch", &UCM::UcmLocalStore::CommitBatch);

    auto config = py::class_<UCM::UcmLocalStore::Config>(module, "Config");
    config.def(py::init<const size_t, const size_t>(), py::arg("capacity"),
               py::arg("cacheSize"));
    config.def_readwrite("capacity", &UCM::UcmLocalStore::Config::capacity);
    config.def_readwrite("cacheSize", &UCM::UcmLocalStore::Config::cacheSize);
    config.def_readwrite("transferDeviceId", &UCM::UcmLocalStore::Config::transferDeviceId);
    config.def_readwrite("transferIoSize", &UCM::UcmLocalStore::Config::transferIoSize);

    auto id = py::class_<UCM::Id>(module, "Id");
    id.def(py::init<size_t, size_t>());
    id.def_readwrite("taskId", &UCM::Id::taskId);
    id.def_readwrite("cacheId", &UCM::Id::cacheId);
}