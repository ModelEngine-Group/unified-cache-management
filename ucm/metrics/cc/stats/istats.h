#ifndef UCM_METRICS_ISTATS_H
#define UCM_METRICS_ISTATS_H

#include <string>
#include <memory>
#include <unordered_map>
#include <vector>

class IStats {
public:
    virtual ~IStats() = default;
    virtual std::string name() const = 0;
    virtual void update(const std::unordered_map<std::string, double>& params) = 0;
    virtual void reset() = 0;
    virtual std::unordered_map<std::string, std::vector<double>> data() = 0;
};
#endif 