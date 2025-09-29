#ifndef UCM_SPARSE_KVSTAR_RETRIEVE_SINGLETON_H
#define UCM_SPARSE_KVSTAR_RETRIEVE_SINGLETON_H

template <typename T>
class Singleton {
public:
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
    static T* Instance()
    {
        static T t;
        return &t;
    }

private:
    Singleton() = default;
};

#endif //UCM_SPARSE_KVSTAR_RETRIEVE_SINGLETON_H