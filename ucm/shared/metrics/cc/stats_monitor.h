#ifndef UNIFIEDCACHE_MONITOR_H
#define UNIFIEDCACHE_MONITOR_H

#include "stats/istats.h" 
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

namespace UC::Metrics {

class StatsMonitor {
public:
   
    static StatsMonitor& GetInstance() {
        static StatsMonitor inst;
        return inst;
    }

    ~StatsMonitor() = default;

    void CreateStats(const std::string& name);

    std::unordered_map<std::string, std::vector<double>>
        GetStats(const std::string& name);
    
    void ResetStats(const std::string& name);

    std::unordered_map<std::string, std::vector<double>>
        GetStatsAndClear(const std::string& name);

    void UpdateStats(const std::string& name,
                      const std::unordered_map<std::string, double>& params);

    void ResetAllStats();

private:
    std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<IStats>> stats_map_;

    StatsMonitor();
    StatsMonitor(const StatsMonitor&) = delete;
    StatsMonitor& operator=(const StatsMonitor&) = delete;
};

}    

#endif  // UNIFIEDCACHE_MONITOR_H