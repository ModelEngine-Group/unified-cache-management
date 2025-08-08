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
#include <vector>
#include <future>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "status/status.h"
#include "cache/cache_manager.h"
#include "tsf_task/tsf_task_manager.h"
#include "template/singleton.h"
#include "logger/logger.h"

namespace py = pybind11;

namespace UCM {

class UcmKVStoreBase {
public:
    UcmKVStoreBase() = default;
    virtual ~UcmKVStoreBase() = default;
    virtual int32_t Setup() = 0;
    virtual int32_t AllocBatch(const py::list& blockIds) = 0;
    virtual py::list LookupBatch(const py::list& blockIds) = 0;
    virtual size_t LoadToDevice(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList, const std::list<uintptr_t>& addressList,
                        const std::list<size_t>& lengthList) = 0;
    virtual size_t LoadToHost(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList, const std::list<uintptr_t>& addressList,
                        const std::list<size_t>& lengthList) = 0;
    virtual size_t DumpFromDevice(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList, const std::list<uintptr_t>& addressList,
                        const std::list<size_t>& lengthList) = 0;
    virtual size_t DumpFromHost(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList, const std::list<uintptr_t>& addressList,
                        const std::list<size_t>& lengthList) = 0;
    virtual int32_t Wait(const size_t id) = 0;
    virtual void CommitBatch(const py::list& blockIds, const bool success) = 0;
};

class UcmLocalStore : public UcmKVStoreBase {
public:
    class Config {
    public:
        size_t capacity;
        size_t cacheSize;
        int32_t transferDeviceId;
        size_t transferStreamNumber;
        size_t transferIoSize;
        size_t transferBufferNumber;

        Config(const size_t capacity, const size_t cacheSize)
            : capacity{capacity}, cacheSize{cacheSize}, transferDeviceId{-1},
              transferStreamNumber{32}, transferIoSize{262144}, transferBufferNumber{512}
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
    size_t LoadToDevice(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList,
                const std::list<uintptr_t>& addressList, const std::list<size_t>& lengthList) override;
    size_t LoadToHost(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList, const std::list<uintptr_t>& addressList,
                const std::list<size_t>& lengthList) override;
    size_t DumpFromDevice(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList, const std::list<uintptr_t>& addressList,
                const std::list<size_t>& lengthList) override;
    size_t DumpFromHost(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList, const std::list<uintptr_t>& addressList,
                const std::list<size_t>& lengthList) override;
    int32_t Wait(const size_t id) override;
    void CommitBatch(const py::list& blockIds, const bool success) override;

private:
    Config _config;
    UcmKVStoreBase *_store;
};

void ShowSetupParam(const UcmLocalStore::Config& config)
{
    UCM_INFO("Set UCM::Capacity to {}.", config.capacity);
    UCM_INFO("Set UCM::CacheSize to {}.", config.cacheSize);
    UCM_INFO("Set UCM::DeviceId to {}.", config.transferDeviceId);
    UCM_INFO("Set UCM::StreamNumber to {}.", config.transferStreamNumber);
    UCM_INFO("Set UCM::IOSize to {}.", config.transferIoSize);
    UCM_INFO("Set UCM::BufferNumber to {}.", config.transferBufferNumber);
}

int32_t UcmLocalStore::Setup()
{
    auto status = Singleton<CacheManager>::Instance()->Setup(this->_config.capacity, this->_config.cacheSize);
    if (status.Failure()) {
        UCM_ERROR("Failed({}) to setup LRUCache.", status);
        return status.Underlying();
    }

    status = Singleton<TsfTaskManager>::Instance()->Setup(this->_config.transferDeviceId, this->_config.transferStreamNumber,
        this->_config.transferIoSize, this->_config.transferBufferNumber);
    if (status.Failure()) {
        UCM_ERROR("Failed({}) to setup TsfTaskManager.", status);
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

py::tuple FormatMissLists(const std::list<TsfTask>& tasks)
{
    std::list<std::string> missBlockIds;
    std::list<size_t> missOffsets, missLengths;
    std::list<uintptr_t> missAddresses;
    for (auto& t : tasks) {
        missBlockIds.push_back(t.blockId);
        missOffsets.push_back(t.offset);
        missLengths.push_back(t.length);
        missAddresses.push_back(t.address);
    }
    return py::make_tuple(missBlockIds, missOffsets, missAddresses, missLengths);
}

inline Status SubmitCache(std::list<TsfTask>& missTasks, const std::list<std::string>& blockIdList,
                          const std::list<size_t>& offsetList, const std::list<uintptr_t>& addressList,
                          const std::list<size_t>& lengthList)
{
    std::list<TsfTask> tasks;
    auto cacheMgr = Singleton<CacheManager>::Instance();
    size_t taskId = 0;
    size_t size = 0;
    size_t number = 0;
    auto blockId = blockIdList.begin();
    auto offset = offsetList.begin();
    auto address = addressList.begin();
    auto length = lengthList.begin();
    while ((blockId != blockIdList.end()) && (offset != offsetList.end()) && (address != addressList.end()) &&
           (length != lengthList.end())) {
        TsfTask task(*blockId, *offset, *address, *length);
        auto status = cacheMgr->ReadCache(task);
        if (status.Failure()) {
            return status;
        }
        if (task.buffer == nullptr) {
            missTasks.push_back(std::move(task));
        } else {
            tasks.push_back(std::move(task));
            number++;
            size += *length;
        }
        blockId++;
        offset++;
        address++;
        length++;
    }
    if (tasks.empty()) {
        return Status::OK();
    }
    auto tsfTaskMgr = Singleton<TsfTaskManager>::Instance();
    auto status = tsfTaskMgr->Submit(tasks, size, number, taskId);
    if (status.Failure()) {
        UCM_ERROR("Failed({}) to submit load to device.", status);
        return status;
    }
    status = tsfTaskMgr->Wait(taskId);
    if (status.Failure()) {
        UCM_ERROR("Failed({}) to wait load to device.", status);
        return status;
    }
    return Status::OK();
}

size_t UcmLocalStore::LoadToDevice(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList,
                                   const std::list<uintptr_t>& addressList, const std::list<size_t>& lengthList)
{
    std::list<TsfTask> missTasks;
    auto status = SubmitCache(missTasks, blockIdList, offsetList, addressList, lengthList);
    if (status.Failure()) {
        UCM_ERROR("Failed({}) to submit cache to device.", status);
        return 0;
    }
    if (missTasks.empty()) {
        return 0;
    }
    auto result = FormatMissLists(missTasks);
    auto missBlockIds = result[0].cast<std::list<std::string>>();
    auto missOffsets = result[1].cast<std::list<size_t>>();
    auto missAddresses = result[2].cast<std::list<uintptr_t>>();
    auto missLengths = result[3].cast<std::list<size_t>>();
    size_t storeId = this->_store->LoadToDevice(missBlockIds, missOffsets, missAddresses, missLengths);
    size_t blockSize = this->_config.cacheSize;
    size_t maxBufferNum = std::min(missBlockIds.size(), this->_config.capacity);
    auto t = std::thread([this, missBlockIds, blockSize, maxBufferNum]() {
        std::list<std::string> nfBlockIds;
        auto it = missBlockIds.begin();
        for (size_t i = 0; i < maxBufferNum && it != missBlockIds.end(); ++i, ++it) {
            nfBlockIds.push_back(*it);
        }
        std::list<uintptr_t> buffers;
        auto cacheMgr = Singleton<CacheManager>::Instance();
        auto status = cacheMgr->AllocBuffers(nfBlockIds, buffers);
        if (status.Failure()) {
            UCM_ERROR("Failed({}) to alloc buffers.", status);
            return;
        }
        std::list<size_t> nfOffsets, nfLengths;
        for (size_t i = 0; i < buffers.size(); ++i) {
            nfOffsets.push_back(0);
            nfLengths.push_back(blockSize);
        }
        size_t taskId = this->_store->LoadToHost(nfBlockIds, nfOffsets, buffers, nfLengths);
        size_t ret = this->_store->Wait(taskId);
        if (ret == 0){
            cacheMgr->CommitBuffers(buffers);
        }
        UCM_INFO("Async add blocks to dram success.");
    });
    Singleton<TsfTaskManager>::Instance()->SaveThread(storeId, std::move(t));
    return storeId;
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

int32_t UcmLocalStore::Wait(const size_t storeId)
{
    if (storeId == 0) { return 0; }
    auto tsfTaskMgr = Singleton<TsfTaskManager>::Instance();
    tsfTaskMgr->WaitThread(storeId);
    return this->_store->Wait(storeId);
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
    config.def_readwrite("transferStreamNumber", &UCM::UcmLocalStore::Config::transferStreamNumber);
    config.def_readwrite("transferIoSize", &UCM::UcmLocalStore::Config::transferIoSize);
    config.def_readwrite("transferBufferNumber", &UCM::UcmLocalStore::Config::transferBufferNumber);
}