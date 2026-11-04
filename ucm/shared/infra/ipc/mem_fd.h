/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fmt/format.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include "status/status.h"

namespace UC {

class MemFd {
    int32_t fd_{-1};
    void* addr_{nullptr};
    size_t size_{0};

public:
    MemFd() = default;
    ~MemFd() { Reset(); }
    MemFd(const MemFd&) = delete;
    MemFd& operator=(const MemFd&) = delete;

    Status Create(const std::string& name, size_t size, uint32_t additionalFlags = 0)
    {
        Reset();
        constexpr uint32_t baseFlags = MFD_ALLOW_SEALING | MFD_CLOEXEC;
        fd_ = ::memfd_create(name.c_str(), baseFlags | additionalFlags);
        if (fd_ < 0) { return OsError("memfd_create failed", errno); }
        if (::ftruncate(fd_, static_cast<off_t>(size)) != 0) {
            const auto status = OsError("ftruncate failed", errno);
            Reset();
            return status;
        }
        if (::fcntl(fd_, F_ADD_SEALS, F_SEAL_SEAL | F_SEAL_SHRINK) != 0) {
            const auto status = OsError("F_ADD_SEALS failed", errno);
            Reset();
            return status;
        }
        auto s = Map(size, "mmap failed");
        if (s.Failure()) { Reset(); }
        return s;
    }

    Status Adopt(int32_t fd, size_t size)
    {
        Reset();
        fd_ = fd;
        auto s = Map(size, "mmap adopt failed");
        if (s.Failure()) { Reset(); }
        return s;
    }

    Status Remap(size_t size)
    {
        if (fd_ < 0) { return Status::InvalidParam("invalid memfd"); }
        if (addr_ != nullptr) {
            ::munmap(addr_, size_);
            addr_ = nullptr;
            size_ = 0;
        }
        return Map(size, "mmap remap failed");
    }

    void* Addr() const { return addr_; }
    int32_t Fd() const { return fd_; }
    size_t Size() const { return size_; }

private:
    Status Map(size_t size, const char* error)
    {
        addr_ = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (addr_ == MAP_FAILED) {
            const auto status = OsError(error, errno);
            addr_ = nullptr;
            return status;
        }
        size_ = size;
        return Status::OK();
    }

    static Status OsError(const char* operation, int error)
    {
        return Status::OsApiError(
            fmt::format("{}: errno={} ({})", operation, error, std::strerror(error)));
    }

    void Reset()
    {
        if (addr_ != nullptr) {
            ::munmap(addr_, size_);
            addr_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        size_ = 0;
    }
};

}  // namespace UC
