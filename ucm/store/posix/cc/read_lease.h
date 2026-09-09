#ifndef UCM_POSIX_READ_LEASE_H
#define UCM_POSIX_READ_LEASE_H

#include <cerrno>
#include <fcntl.h>
#include <map>
#include <unistd.h>
#include "ucmstore_v1.h"
#include "space_layout.h"

namespace UC::PosixStore {

// Linux OFD locks belong to open file descriptions, not process-local counters.
// Never unlink these small, persistent files: all GC/readers must lock the same inode.
// Hash collisions only over-protect unrelated blocks. No data/shard layout is copied.
class PosixReadLease final : public ReadLease {
    class File final {
        int fd_{-1};
        int error_{0};

    public:
        explicit File(const std::string& path)
            : fd_(::open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600))
        {
            if (fd_ < 0) { error_ = errno; }
        }
        ~File()
        {
            if (fd_ >= 0) { ::close(fd_); }
        }
        File(const File&) = delete;
        File& operator=(const File&) = delete;
        bool Valid() const { return fd_ >= 0; }
        int Error() const { return error_; }

        Status Lock(const Detail::BlockId& block, bool writer)
        {
            struct flock lock {};
            lock.l_type = writer ? F_WRLCK : F_RDLCK;
            lock.l_whence = SEEK_SET;
            lock.l_start = static_cast<off_t>(Detail::BlockIdHasher{}(block) & 0x3fffffffffffffffULL);
            lock.l_len = 1;
            if (::fcntl(fd_, F_OFD_SETLK, &lock) == 0) { return Status::OK(); }
            const auto error = errno;
            if (error == EAGAIN || error == EACCES) { return Status::Retry(); }
            return Status::OsApiError(std::to_string(error));
        }
    };
    std::map<std::string, std::unique_ptr<File>> files_;
public:
    static Expected<std::shared_ptr<ReadLease>> Acquire(
        const SpaceLayout& layout, const Detail::BlockId* blocks, size_t num)
    try {
        auto lease = std::make_shared<PosixReadLease>();
        for (size_t i = 0; i < num; ++i) {
            auto status = lease->Lock(layout.StorageBackend(blocks[i]), blocks[i]);
            if (status.Failure()) { return status; }
        }
        return std::static_pointer_cast<ReadLease>(lease);
    } catch (const std::bad_alloc&) {
        return Status::OutOfMemory();
    }

    static Expected<std::shared_ptr<void>> AcquireRemoveGuard(
        const std::string& backend, const Detail::BlockId& block)
    try {
        auto guard = std::make_shared<PosixReadLease>();
        auto status = guard->Lock(backend, block, true);
        if (status.Failure()) { return status; }
        return std::static_pointer_cast<void>(guard);
    } catch (const std::bad_alloc&) {
        return Status::OutOfMemory();
    }

    Status Lock(const std::string& backend, const Detail::BlockId& block, bool writer = false)
    {
        auto path = backend + ".ucm_read_leases";
        auto it = files_.find(path);
        if (it == files_.end()) {
            auto file = std::make_unique<File>(path);
            if (!file->Valid()) {
                return Status::OsApiError(std::to_string(file->Error()));
            }
            it = files_.emplace(path, std::move(file)).first;
        }
        return it->second->Lock(block, writer);
    }
};
}  // namespace UC::PosixStore
#endif
