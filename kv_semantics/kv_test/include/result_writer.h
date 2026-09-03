#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include "kv_test_types.h"

namespace UC::KVTest {

class ResultWriter {
public:
    Status Open(const OutputConfig& config);
    Status WriteSummary(const CommandOptions& options, const CommandResult& result);
    Status WriteRealtimeSample(const std::string& csvLine);
    Status WriteLatencySample(const std::string& csvLine);
    Status WriteConsistencyError(const std::string& line);
    Status Close();

private:
    Status WriteHtmlReport(const CommandOptions& options, const CommandResult& result);
    Status WriteReportIndex();
    Status OpenRealtimeFile();
    Status OpenLatencyFile();
    Status OpenConsistencyErrorFile();
    Status RollRealtimeFileIfNeeded(std::uint64_t incomingBytes);

    std::string baseOutputDir_;
    std::string outputDir_;
    std::uint64_t realtimeFileMaxBytes_{0};
    std::uint32_t realtimeFileIndex_{0};
    std::uint64_t realtimeFileBytes_{0};
    std::ofstream realtimeFile_;
    std::ofstream latencyFile_;
    std::ofstream consistencyErrorFile_;
};

}  // namespace UC::KVTest
