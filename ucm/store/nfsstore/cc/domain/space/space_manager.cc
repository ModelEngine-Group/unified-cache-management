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
#include "space_manager.h"
#include "file/file.h"
#include "logger/logger.h"
#include "space_shard_temp_layout.h"

namespace UC {

std::unique_ptr<SpaceLayout> MakeSpaceLayout(const bool tempDumpDirEnable)
{
    try {
        if (tempDumpDirEnable) { return std::make_unique<SpaceShardTempLayout>(); }
        return std::make_unique<SpaceShardLayout>();
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to make space layout object.", e.what());
    }
    return nullptr;
}

Status SpaceManager::Setup(const std::vector<std::string>& storageBackends, const size_t blockSize,
                           const bool tempDumpDirEnable, const size_t storageCapacity,
                           const bool recycleEnable, const float recycleThresholdRatio)
{
    if (blockSize == 0) {
        UC_ERROR("Invalid block size({}).", blockSize);
        return Status::InvalidParam();
    }
    this->layout_ = MakeSpaceLayout(tempDumpDirEnable);
    if (!this->layout_) { return Status::OutOfMemory(); }
    auto status = this->layout_->Setup(storageBackends);
    if (status.Failure()) { return status; }
    status = this->property_.Setup(this->layout_->ClusterPropertyFilePath());
    if (recycleEnable && storageCapacity > 0) {
        auto totalBlocks = storageCapacity / blockSize;
        status = this->recycle_.Setup(this->GetSpaceLayout(), totalBlocks, [this] {
            this->property_.DecreaseCapacity(this->blockSize_);
        });
        if (status.Failure()) { return status; }
    }
    if (status.Failure()) { return status; }
    this->blockSize_ = blockSize;
    this->capacity_ = storageCapacity;
    this->recycleEnable_ = recycleEnable;
    this->capacityRecycleThreshold_ = static_cast<size_t>(storageCapacity * recycleThresholdRatio);
    return Status::OK();
}

Status SpaceManager::NewBlock(const std::string& blockId)
{
    Status status = this->CapacityCheck();
    if (status.Failure()) { return status; }
    constexpr auto activated = true;
    auto parent = File::Make(this->layout_->DataFileParent(blockId, activated));
    auto file = File::Make(this->layout_->DataFilePath(blockId, activated));
    if (!parent || !file) {
        UC_ERROR("Failed to new block({}).", blockId);
        return Status::OutOfMemory();
    }
    status = parent->MkDir();
    if (status == Status::DuplicateKey()) { status = Status::OK(); }
    if (status.Failure()) {
        UC_ERROR("Failed({}) to new block({}).", status, blockId);
        return status;
    }
    if ((File::Access(this->layout_->DataFilePath(blockId, false), IFile::AccessMode::EXIST))
            .Success()) {
        status = Status::DuplicateKey();
        UC_ERROR("Failed({}) to new block({}).", status, blockId);
        return status;
    }
    status =
        file->Open(IFile::OpenFlag::CREATE | IFile::OpenFlag::EXCL | IFile::OpenFlag::READ_WRITE);
    if (status.Failure()) {
        if (status != Status::DuplicateKey()) {
            UC_ERROR("Failed({}) to new block({}).", status, blockId);
        }
        return status;
    }
    status = file->Truncate(this->blockSize_);
    if (status.Failure()) {
        UC_ERROR("Failed({}) to new block({}).", status, blockId);
        return status;
    }
    this->property_.IncreaseCapacity(this->blockSize_);
    return Status::OK();
}

Status SpaceManager::CommitBlock(const std::string& blockId, bool success)
{
    const auto activatedParent = this->layout_->DataFileParent(blockId, true);
    const auto activatedFile = this->layout_->DataFilePath(blockId, true);
    const auto archivedParent = this->layout_->DataFileParent(blockId, false);
    auto status = Status::OK();
    do {
        if (!success) { break; }
        if (archivedParent != activatedParent) {
            status = File::MkDir(archivedParent);
            if (status == Status::DuplicateKey()) { status = Status::OK(); }
            if (status.Failure()) { break; }
        }
        const auto archivedFile = this->layout_->DataFilePath(blockId, false);
        status = File::Rename(activatedFile, archivedFile);
    } while (0);
    File::Remove(activatedFile);
    if (!success || archivedParent != activatedParent) { File::RmDir(activatedParent); }
    if (status.Failure()) {
        UC_ERROR("Failed({}) to {} block({}).", status, success ? "commit" : "cancel", blockId);
    }
    this->property_.DecreaseCapacity(this->blockSize_);
    return status;
}

bool SpaceManager::LookupBlock(const std::string& blockId) const
{
    auto path = this->layout_->DataFilePath(blockId, false);
    auto file = File::Make(path);
    if (!file) {
        UC_ERROR("Failed to make file smart pointer, path: {}.", path);
        return false;
    }
    auto s =
        file->Access(IFile::AccessMode::EXIST | IFile::AccessMode::READ | IFile::AccessMode::WRITE);
    if (s.Failure()) {
        if (s != Status::NotFound()) {
            UC_ERROR("Failed to access file, path: {}, errcode: {}.", path, s);
        }
        return false;
    }
    return true;
}

const SpaceLayout* SpaceManager::GetSpaceLayout() const { return this->layout_.get(); }

Status SpaceManager::CapacityCheck()
{
    if (this->capacity_ == 0) { return Status::OK(); }
    
    const size_t used = this->property_.GetCapacity();
    if (this->recycleEnable_ && used >= this->capacityRecycleThreshold_) {
        this->recycle_.Trigger();
    }
    if (used > this->capacity_ - this->blockSize_) {
        UC_ERROR("Capacity is not enough, capacity: {}, current: {}, block size: {}.", 
                 this->capacity_, used, this->blockSize_);
        return Status::NoSpace();
    }
    return Status::OK();
}

} // namespace UC
