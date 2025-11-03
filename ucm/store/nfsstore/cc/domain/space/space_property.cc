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
#include "space_property.h"
#include <atomic>
#include <chrono>
#include <thread>
#include "logger/logger.h"
#include "file/file.h"

namespace UC {

static constexpr uint32_t Magic = (('S' << 16) | ('p' << 8) | 1);
static constexpr size_t PropertySize = 256;

struct Property {
    std::atomic<uint32_t> magic;
    uint32_t padding;
    std::atomic<size_t> capacity;
};
static_assert(sizeof(Property) <= PropertySize, "Property take too much space");
static_assert(std::atomic<uint32_t>::is_always_lock_free, "magic must be lock-free");
static_assert(std::atomic<size_t>::is_always_lock_free, "capacity must be lock-free");

inline auto PropertyPtr(void* addr) { return (Property*)addr; }

SpaceProperty::~SpaceProperty()
{
    if (this->addr_) {
        File::MUnmap(this->addr_, PropertySize);
    }
    this->addr_ = nullptr;
}

Status SpaceProperty::Setup(const std::string& PropertyFilePath)
{
    auto file = File::Make(PropertyFilePath);
    if (!file) { return Status::OutOfMemory(); }
    auto flags = IFile::OpenFlag::CREATE | IFile::OpenFlag::EXCL | IFile::OpenFlag::READ_WRITE;
    auto status = file->Open(flags);
    if (status.Success()) { return this->InitShmProperty(file.get()); }
    if (status == Status::DuplicateKey()) { return this->LoadShmProperty(file.get()); }
    return status;
}

void SpaceProperty::IncreaseCapacity(const size_t delta)
{
    PropertyPtr(this->addr_)->capacity += delta;
}

void SpaceProperty::DecreaseCapacity(const size_t delta)
{
    auto property = PropertyPtr(this->addr_);
    auto capacity = property->capacity.load(std::memory_order_acquire);
    while (capacity > delta) {
        if (property->capacity.compare_exchange_weak(capacity, capacity - delta, std::memory_order_acq_rel)) {
            return;
        }
        capacity = property->capacity.load(std::memory_order_acquire);
    }
}

size_t SpaceProperty::GetCapacity() const
{
    return PropertyPtr(this->addr_)->capacity.load(std::memory_order_relaxed);
}

Status SpaceProperty::InitShmProperty(IFile* shmPropertyFile)
{
    auto status = shmPropertyFile->Truncate(PropertySize);
    if (status.Failure()) { return status; }
    status = shmPropertyFile->MMap(this->addr_, PropertySize, true, true, true);
    if (status.Failure()) { return status; }
    std::fill_n((uint8_t*)this->addr_, PropertySize, 0);
    auto property = PropertyPtr(this->addr_);
    property->padding = 0;
    property->capacity = 0;
    property->magic = Magic;
    return Status::OK();
}

Status SpaceProperty::LoadShmProperty(IFile* shmPropertyFile)
{
    auto status = shmPropertyFile->Open(IFile::OpenFlag::READ_WRITE);
    if (status.Failure()) { return status; }
    constexpr auto retryInterval = std::chrono::milliseconds(100);
    constexpr auto maxTryTime = 100;
    auto tryTime = 0;
    IFile::FileStat stat;
    do {
        if (tryTime > maxTryTime) {
            UC_ERROR("Shm file({}) not ready.", shmPropertyFile->Path());
            return Status::Retry();
        }
        std::this_thread::sleep_for(retryInterval);
        status = shmPropertyFile->Stat(stat);
        if (status.Failure()) { return status; }
        tryTime++;
    } while (static_cast<size_t>(stat.st_size) != PropertySize);
    status = shmPropertyFile->MMap(this->addr_, PropertySize, true, true, true);
    if (status.Failure()) { return status; }
    auto property = PropertyPtr(this->addr_);
    tryTime = 0;
    do {
        if (property->magic == Magic) { break; }
        if (tryTime > maxTryTime) {
            UC_ERROR("Shm file({}) not ready.", shmPropertyFile->Path());
            return Status::Retry();
        }
        std::this_thread::sleep_for(retryInterval);
        tryTime++;
    } while (true);
    return Status::OK();
}

} // namespace UC