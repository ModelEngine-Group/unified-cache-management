#ifndef UNIFIEDCACHE_ISTATS_H
#define UNIFIEDCACHE_ISTATS_H

#include <string>
#include <memory>
#include <unordered_map>
#include <vector>

namespace UC::Metrics {

class IStats {
public:
    virtual ~IStats() = default;
    virtual std::string Name() const = 0;
    virtual void Update(const std::unordered_map<std::string, double>& params) = 0;
    virtual void Reset() = 0;
    virtual std::unordered_map<std::string, std::vector<double>> Data() = 0;
};

}  

#endif 