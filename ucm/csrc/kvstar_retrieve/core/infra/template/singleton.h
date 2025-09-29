#ifndef UCM_SPARSE_KVSTAR_RETRIEVE_SINGLETON_H
#define UCM_SPARSE_KVSTAR_RETRIEVE_SINGLETON_H

template <typename T>
class Singleton {
public:
    Singleton(const Singleton&) = delete; // 单例禁用拷贝或赋值
    Singleton& operator=(const Singleton&) = delete;
    static T* Instance() // 静态函数获取单例实例
    {
        static T t; // 静态局部变量的初始化过程是线程安全的
        return &t;
    }

private:
    Singleton() = default;
};

#endif //UCM_SPARSE_KVSTAR_RETRIEVE_SINGLETON_H