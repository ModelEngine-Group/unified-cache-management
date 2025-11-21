#ifndef STATS_MONITOR_H
#define STATS_MONITOR_H

#include "stats/istats.h" 
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

class UCMStatsMonitor {
public:
    UCMStatsMonitor();
    ~UCMStatsMonitor() = default;

    // 禁止拷贝
    UCMStatsMonitor(const UCMStatsMonitor&) = delete;
    UCMStatsMonitor& operator=(const UCMStatsMonitor&) = delete;

    /* 业务接口 */
    std::unordered_map<std::string, std::vector<double>>
        getStats(const std::string& name);

    void updateStats(const std::string& name,
                      const std::unordered_map<std::string, double>& params);

    std::unordered_map<std::string, std::unique_ptr<IStats>>
        getStatsSnapshot();

    void resetAllStats();

private:
    std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<IStats>> stats_map_;
};

#endif // STATS_MONITOR_H