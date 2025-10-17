

#ifndef UNIFIEDCACHE_FILE_HANDLE_CACHE_H
#define UNIFIEDCACHE_FILE_HANDLE_CACHE_H

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <fcntl.h>
#include <cerrno>
#include <cstring>
#include <cufile.h>
#include "logger/logger.h"
#include <unistd.h>

namespace UC {
class FileHandleCache {
public:
    struct CachedHandle {
        int fd;
        CUfileHandle_t cuHandle;
        std::string filePath;
        int openFlags;
        bool valid;

        CachedHandle()
            : fd(-1), filePath(""), openFlags(0), valid(false) {}
    };

private:
    struct LRUNode {
        std::string key;
        CachedHandle handle;
        LRUNode* prev;
        LRUNode* next;

        LRUNode() : prev(nullptr), next(nullptr) {}
        LRUNode(const std::string& k, CachedHandle&& h)
            : key(k), handle(std::move(h)), prev(nullptr), next(nullptr) {}
    };

public:
    explicit FileHandleCache(size_t maxSize = 65536)
        : maxSize_(maxSize), cacheHits_(0), cacheMisses_(0) {
        head_ = new LRUNode();
        tail_ = new LRUNode();
        head_->next = tail_;
        tail_->prev = head_;
    }

    ~FileHandleCache() {
        Clear();
        delete head_;
        delete tail_;
    }

    FileHandleCache(const FileHandleCache&) = delete;
    FileHandleCache& operator=(const FileHandleCache&) = delete;

    CachedHandle* Get(const std::string& filePath, int flags) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::string key = MakeKey(filePath, flags);
        auto it = cache_.find(key);
        if (it != cache_.end() && it->second->handle.valid) {
            LRUNode* node = it->second;
            MoveToHead(node);
            cacheHits_.fetch_add(1, std::memory_order_relaxed);
            return &node->handle;
        }
        cacheMisses_.fetch_add(1, std::memory_order_relaxed);
        if (cache_.size() >= maxSize_) {
            EvictLRU();
        }
        CachedHandle handle;
        if (!CreateHandle(filePath, flags, handle)) {
            UC_ERROR("Failed to create file handle for {}", filePath);
            return nullptr;
        }
        LRUNode* newNode = new LRUNode(key, std::move(handle));
        AddToHead(newNode);
        cache_[key] = newNode;
        return &newNode->handle;
    }

    void Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        LRUNode* current = head_->next;
        while (current != tail_) {
            LRUNode* next = current->next;
            if (current->handle.valid) {
                DestroyHandle(current->handle);
            }
            delete current;
            current = next;
        }
        head_->next = tail_;
        tail_->prev = head_;
        cache_.clear();
    }

    void GetStats(uint64_t& hits, uint64_t& misses) const {
        hits = cacheHits_.load(std::memory_order_relaxed);
        misses = cacheMisses_.load(std::memory_order_relaxed);
    }

    double GetHitRate() const {
        uint64_t hits = cacheHits_.load(std::memory_order_relaxed);
        uint64_t misses = cacheMisses_.load(std::memory_order_relaxed);
        uint64_t total = hits + misses;
        return total > 0 ? (hits * 100.0 / total) : 0.0;
    }

    size_t GetSize() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.size();
    }

private:
    void AddToHead(LRUNode* node) {
        node->prev = head_;
        node->next = head_->next;
        head_->next->prev = node;
        head_->next = node;
    }

    void RemoveNode(LRUNode* node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }

    void MoveToHead(LRUNode* node) {
        RemoveNode(node);
        AddToHead(node);
    }

    LRUNode* RemoveTail() {
        LRUNode* lastNode = tail_->prev;
        RemoveNode(lastNode);
        return lastNode;
    }

    std::string MakeKey(const std::string& filePath, int flags) const {
        return filePath + "|" + std::to_string(flags);
    }

    bool CreateHandle(const std::string& filePath, int flags, CachedHandle& handle) {
        handle.fd = open(filePath.c_str(), flags, 0644);
        if (handle.fd < 0) {
            UC_ERROR("Failed to open file {}: {}", filePath, strerror(errno));
            return false;
        }
        CUfileDescr_t cfDescr{};
        cfDescr.handle.fd = handle.fd;
        cfDescr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        CUfileError_t err = cuFileHandleRegister(&handle.cuHandle, &cfDescr);
        if (err.err != CU_FILE_SUCCESS) {
            UC_ERROR("Failed to register cuFile handle for {}: error {}",
                     filePath, static_cast<int>(err.err));
            close(handle.fd);
            return false;
        }
        handle.filePath = filePath;
        handle.openFlags = flags;
        handle.valid = true;
        return true;
    }

    void DestroyHandle(CachedHandle& handle) {
        if (!handle.valid) return;
        cuFileHandleDeregister(handle.cuHandle);
        close(handle.fd);
        handle.valid = false;
    }

    void EvictLRU() {
        if (cache_.empty()) return;
        LRUNode* lruNode = RemoveTail();
        DestroyHandle(lruNode->handle);
        cache_.erase(lruNode->key);
        delete lruNode;
    }

private:
    std::unordered_map<std::string, LRUNode*> cache_;
    LRUNode* head_;
    LRUNode* tail_;
    mutable std::mutex mutex_;
    size_t maxSize_;
    std::atomic<uint64_t> cacheHits_;
    std::atomic<uint64_t> cacheMisses_;
};

} // namespace UC

#endif
