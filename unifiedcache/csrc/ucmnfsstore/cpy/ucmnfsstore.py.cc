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
#include "ucmnfsstore/ucmnfsstore.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <spdlog/fmt/ranges.h>
#include "status/status.h"
#include "template/singleton.h"
#include "space/space_manager.h"
#include "tsf_task/tsf_task_manager.h"

namespace py = pybind11;

namespace UC {

class UcmKVStoreBase {
public:
    UcmKVStoreBase() = default;
    virtual ~UcmKVStoreBase() = default;
    virtual int32_t Setup() = 0;
    virtual int32_t AllocBatch(const py::list& blockIds) = 0;
    virtual py::list LookupBatch(const py::list& blockIds) = 0;
    virtual size_t LoadToDevice(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList,
                    const std::list<uintptr_t>& addressList, const std::list<size_t>& lengthList) = 0;
    virtual size_t LoadToHost(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList,
                    const std::list<uintptr_t>& addressList, const std::list<size_t>& lengthList) = 0;
    virtual size_t DumpFromDevice(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList,
                    const std::list<uintptr_t>& addressList, const std::list<size_t>& lengthList) = 0;
    virtual size_t DumpFromHost(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList,
                    const std::list<uintptr_t>& addressList, const std::list<size_t>& lengthList) = 0;
    virtual int32_t Wait(const size_t id) = 0;
    virtual void CommitBatch(const py::list& blockIds, const bool success) = 0;
};

class UcmNfsStore : public UcmKVStoreBase {
public:
    class Config {
    public:
        std::vector<std::string> storageBackends;
        size_t kvcacheBlockSize;
        bool transferEnable;
        int32_t transferDeviceId;
        size_t transferStreamNumber;
        size_t transferIoSize;
        size_t transferBufferNumber;
        size_t transferTimeoutMs;

        Config(const std::vector<std::string>& storageBackends, const size_t kvcacheBlockSize,
                const bool transferEnable)
            : storageBackends{storageBackends}, kvcacheBlockSize{kvcacheBlockSize}, transferEnable{transferEnable},
            transferDeviceId{-1}, transferStreamNumber{32}, transferIoSize{262144}, transferBufferNumber{512},
            transferTimeoutMs{30000}
        {
        }
    };

public:
    UcmNfsStore(const Config &config) : _config{config} {}
    int32_t Setup() override;
    int32_t AllocBatch(const py::list& blockIds) override;
    py::list LookupBatch(const py::list& blockIds) override;
    size_t LoadToDevice(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList,
                const std::list<uintptr_t>& addressList, const std::list<size_t>& lengthList) override;
    size_t LoadToHost(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList,
                const std::list<uintptr_t>& addressList, const std::list<size_t>& lengthList) override;
    size_t DumpFromDevice(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList,
                const std::list<uintptr_t>& addressList, const std::list<size_t>& lengthList) override;
    size_t DumpFromHost(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList,
                const std::list<uintptr_t>& addressList, const std::list<size_t>& lengthList) override;
    int32_t Wait(const size_t id) override;
    void CommitBatch(const py::list& blockIds, const bool success) override;

private:
    Config _config;
};

inline void ShowSetupParam(const UcmNfsStore::Config& config)
{
    UC_INFO("Set UC::StorageBackends to {}.", config.storageBackends);
    UC_INFO("Set UC::BlockSize to {}.", config.kvcacheBlockSize);
    UC_INFO("Set UC::TransferEnable to {}.", config.transferEnable);
    UC_INFO("Set UC::DeviceId to {}.", config.transferDeviceId);
    UC_INFO("Set UC::StreamNumber to {}.", config.transferStreamNumber);
    UC_INFO("Set UC::IOSize to {}.", config.transferIoSize);
    UC_INFO("Set UC::BufferNumber to {}.", config.transferBufferNumber);
    UC_INFO("Set UC::TimeoutMs to {}.", config.transferTimeoutMs);
}

int32_t UcmNfsStore::Setup()
{
    auto spaceMgr = Singleton<SpaceManager>::Instance();
    auto status = spaceMgr->Setup(this->_config.storageBackends, this->_config.kvcacheBlockSize);
    if (status.Failure()) {
        UC_ERROR("Failed({}) to setup SpaceManager.", status);
        return status.Underlying();
    }
    if (this->_config.transferEnable) {
        auto taskMgr = Singleton<TsfTaskManager>::Instance();
        status = taskMgr->Setup(this->_config.transferDeviceId, this->_config.transferStreamNumber, this->_config.transferIoSize,
            this->_config.transferBufferNumber, this->_config.transferTimeoutMs, spaceMgr->GetSpaceLayout());
        if (status.Failure()) {
            UC_ERROR("Failed({}) to setup TsfTaskManager.", status);
            return status.Underlying();
        }
    }
    ShowSetupParam(this->_config);
    return Status::OK().Underlying();
}

int32_t UcmNfsStore::AllocBatch(const py::list& blockIds)
{
    int32_t ret = 0;
    for (auto id : blockIds) {
        if ((ret = Alloc(id.cast<std::string>())) != 0) { break; }
    }
    return ret;
}

py::list UcmNfsStore::LookupBatch(const py::list& blockIds)
{
    py::list founds;
    for (auto id : blockIds) { founds.append(Lookup(id.cast<std::string>())); }
    return founds;
}

inline size_t SubmitTsfTasks(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList,
                             const std::list<uintptr_t>& addressList, const std::list<size_t>& lengthList,
                             const TsfTask::Type type, const TsfTask::Location location, const std::string& brief)
{
    std::list<TsfTask> tasks;
    size_t size = 0;
    size_t number = 0;
    auto blockId = blockIdList.begin();
    auto offset = offsetList.begin();
    auto address = addressList.begin();
    auto length = lengthList.begin();
    while ((blockId != blockIdList.end()) && (offset != offsetList.end()) && (address != addressList.end()) &&
           (length != lengthList.end())) {
        tasks.emplace_back(type, location, *blockId, *offset, *address, *length);
        number++;
        size += *length;
        blockId++;
        offset++;
        address++;
        length++;
    }
    return Submit(tasks, size, number, brief);
}

size_t UcmNfsStore::LoadToDevice(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList,
                             const std::list<uintptr_t>& addressList, const std::list<size_t>& lengthList)
{
    return SubmitTsfTasks(blockIdList, offsetList, addressList, lengthList, TsfTask::Type::LOAD,
                          TsfTask::Location::DEVICE, "S2D");
}

size_t UcmNfsStore::LoadToHost(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList,
                               const std::list<uintptr_t>& addressList, const std::list<size_t>& lengthList)
{
    return SubmitTsfTasks(blockIdList, offsetList, addressList, lengthList, TsfTask::Type::LOAD,
                          TsfTask::Location::HOST, "S2H");
}

size_t UcmNfsStore::DumpFromDevice(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList,
                                   const std::list<uintptr_t>& addressList, const std::list<size_t>& lengthList)
{
    return SubmitTsfTasks(blockIdList, offsetList, addressList, lengthList, TsfTask::Type::DUMP,
                          TsfTask::Location::DEVICE, "D2S");
}

size_t UcmNfsStore::DumpFromHost(const std::list<std::string>& blockIdList, const std::list<size_t>& offsetList,
                                 const std::list<uintptr_t>& addressList, const std::list<size_t>& lengthList)
{
    return SubmitTsfTasks(blockIdList, offsetList, addressList, lengthList, TsfTask::Type::DUMP,
                          TsfTask::Location::HOST, "H2S");
}

int32_t UcmNfsStore::Wait(const size_t id) { return Singleton<TsfTaskManager>::Instance()->Wait(id).Underlying(); }

void UcmNfsStore::CommitBatch(const py::list& blockIds, const bool success)
{
    for (auto id : blockIds) { Commit(id.cast<std::string>(), success); }
}

} // namespace UC

PYBIND11_MODULE(ucmnfsstore, module)
{
    module.attr("project") = UC_VAR_PROJECT_NAME;
    module.attr("version") = UC_VAR_PROJECT_VERSION;
    module.attr("commit_id") = UC_VAR_GIT_COMMIT_ID;
    module.attr("build_type") = UC_VAR_BUILD_TYPE;
    auto ucmKVStoreBase = py::class_<UC::UcmKVStoreBase>(module, "UcmKVStoreBase");
    ucmKVStoreBase.def("Setup", &UC::UcmKVStoreBase::Setup);
    ucmKVStoreBase.def("Alloc", &UC::UcmKVStoreBase::AllocBatch);
    ucmKVStoreBase.def("Lookup", &UC::UcmKVStoreBase::LookupBatch);
    ucmKVStoreBase.def("LoadToDevice", &UC::UcmKVStoreBase::LoadToDevice);
    ucmKVStoreBase.def("LoadToHost", &UC::UcmKVStoreBase::LoadToHost);
    ucmKVStoreBase.def("DumpFromDevice", &UC::UcmKVStoreBase::DumpFromDevice);
    ucmKVStoreBase.def("DumpFromHost", &UC::UcmKVStoreBase::DumpFromHost);
    ucmKVStoreBase.def("Wait", &UC::UcmKVStoreBase::Wait);
    ucmKVStoreBase.def("Commit", &UC::UcmKVStoreBase::CommitBatch);

    auto ucmNfsStore = py::class_<UC::UcmNfsStore>(module, "UcmNfsStore");
    ucmNfsStore.def(py::init<const UC::UcmNfsStore::Config&>());
    ucmNfsStore.def("Setup", &UC::UcmNfsStore::Setup);
    ucmNfsStore.def("AllocBatch", &UC::UcmNfsStore::AllocBatch);
    ucmNfsStore.def("LookupBatch", &UC::UcmNfsStore::LookupBatch);
    ucmNfsStore.def("LoadToDevice", &UC::UcmNfsStore::LoadToDevice);
    ucmNfsStore.def("LoadToHost", &UC::UcmNfsStore::LoadToHost);
    ucmNfsStore.def("DumpFromDevice", &UC::UcmNfsStore::DumpFromDevice);
    ucmNfsStore.def("DumpFromHost", &UC::UcmNfsStore::DumpFromHost);
    ucmNfsStore.def("Wait", &UC::UcmNfsStore::Wait);
    ucmNfsStore.def("CommitBatch", &UC::UcmNfsStore::CommitBatch);

    auto config = py::class_<UC::UcmNfsStore::Config>(module, "Config");
    config.def(py::init<const std::vector<std::string>&, const size_t, const bool>(), py::arg("storageBackends"),
               py::arg("kvcacheBlockSize"), py::arg("transferEnable"));
    config.def_readwrite("storageBackends", &UC::UcmNfsStore::Config::storageBackends);
    config.def_readwrite("kvcacheBlockSize", &UC::UcmNfsStore::Config::kvcacheBlockSize);
    config.def_readwrite("transferEnable", &UC::UcmNfsStore::Config::transferEnable);
    config.def_readwrite("transferDeviceId", &UC::UcmNfsStore::Config::transferDeviceId);
    config.def_readwrite("transferStreamNumber", &UC::UcmNfsStore::Config::transferStreamNumber);
    config.def_readwrite("transferIoSize", &UC::UcmNfsStore::Config::transferIoSize);
    config.def_readwrite("transferBufferNumber", &UC::UcmNfsStore::Config::transferBufferNumber);
}
