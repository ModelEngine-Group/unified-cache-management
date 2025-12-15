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
#include "logger/logger.h"
#include "posix_file.h"

namespace UC::PosixStore {

Status SpaceManager::Setup(const Config& config) { return layout_.Setup(config.backends); }

std::vector<uint8_t> SpaceManager::Lookup(const Detail::BlockId* blocks, size_t num)
{
    std::vector<uint8_t> res(num);
    for (size_t i = 0; i < num; i++) { res[i] = Lookup(blocks + i); }
    return res;
}

uint8_t SpaceManager::Lookup(const Detail::BlockId* block)
{
    const auto& path = layout_.DataFilePath(*block, false);
    PosixFile file{path};
    constexpr auto mode =
        PosixFile::AccessMode::EXIST | PosixFile::AccessMode::READ | PosixFile::AccessMode::WRITE;
    auto s = file.Access(mode);
    if (s.Failure()) {
        if (s != Status::NotFound()) { UC_ERROR("Failed({}) to access file({}).", s, path); }
        return false;
    }
    return true;
}

}  // namespace UC::PosixStore
