// istats.h
#pragma once
#include <string>
#include <memory>

class IStats {
public:
    virtual ~IStats() = default;
    virtual std::string name() const = 0;
    virtual void update(unordered_map<string, any> params) = 0;
    virtual void reset() = 0;
    virtual std::unique_ptr<IStats> clone() const = 0;
};

// 注册宏：子类需调用此宏声明注册
#define REGISTER_STATS(ClassName) \
    extern template class StatsRegistrar<ClassName>;

// 注册器模板类（用于生成注册代码）
template <typename T>
class StatsRegistrar {
public:
    StatsRegistrar() {
        StatsRegistry::getInstance().registerStats<T>();
    }
};