#ifndef UCM_METRICS_MONITOR_H
#define UCM_METRICS_MONITOR_H

#include "stats/istats.h" 
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

class UCMStatsMonitor {
public:
   
    static UCMStatsMonitor& getInstance() {
        static UCMStatsMonitor inst;
        return inst;
    }

    ~UCMStatsMonitor() = default;

    void createStats(const std::string& name);

    std::unordered_map<std::string, std::vector<double>>
        getStats(const std::string& name);
    
    void resetStats(const std::string& name);

    std::unordered_map<std::string, std::vector<double>>
        getStatsAndClear(const std::string& name);

    void updateStats(const std::string& name,
                      const std::unordered_map<std::string, double>& params);

    void resetAllStats();

private:
    std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<IStats>> stats_map_;

    UCMStatsMonitor();
    UCMStatsMonitor(const UCMStatsMonitor&) = delete;
    UCMStatsMonitor& operator=(const UCMStatsMonitor&) = delete;
};

#endif  // UCM_METRICS_MONITOR_H