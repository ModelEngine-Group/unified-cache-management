#ifndef UCM_SPARSE_KVSTAR_RETRIEVE_RETRIEVE_TASK_SET_H
#define UCM_SPARSE_KVSTAR_RETRIEVE_RETRIEVE_TASK_SET_H

#include <algorithm>
#include <list>
#include <mutex>
#include <shared_mutex>

namespace KVStar {
// 失败任务记录
class RetrieveTaskSet {
    static constexpr size_t nBucket = 8192;
public:
    void Insert(const size_t id)
    {
        auto idx = this->Hash(id);
        std::unique_lock<std::shared_mutex> lk(this->_mutexes[idx]);
        this->_buckets[idx].push_back(id);
    }
    bool Exist(const size_t id)
    {
        auto idx = this->Hash(id);
        std::shared_lock<std::shared_mutex> lk(this->_mutexes[idx]);
        auto bucket = this->_buckets + idx;
        return std::find(bucket->begin(), bucket->end(), id) != bucket->end();
    }
    void Remove(const size_t id)
    {
        auto idx = this->Hash(id);
        std::unique_lock<std::shared_mutex> lk(this->_mutexes[idx]);
        this->_buckets[idx].remove(id);
    }

private:
    size_t Hash(const size_t id) { return id % nBucket; }

private:
    std::shared_mutex _mutexes[nBucket];
    std::list<size_t> _buckets[nBucket];

};

}



#endif //UCM_SPARSE_KVSTAR_RETRIEVE_RETRIEVE_TASK_SET_H