#include "result_writer.h"
#include <algorithm>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include "arg_parser.h"
#include "consistency_checker.h"

namespace kv::bench {

namespace {

constexpr int kExitInvalidArgument = 1;

constexpr std::uint64_t kDefaultRealtimeFileMaxBytes = 100ULL * 1024ULL * 1024ULL;

const char* kRealtimeCsvHeader =
    "timestamp_sec,op,bandwidth_value,bandwidth_unit,iops_value,iops_unit,avg_latency_value,"
    "avg_latency_unit,error_count\n";
const char* kLatencyCsvHeader =
    "op,avg_value,avg_unit,min_value,min_unit,max_value,max_unit,p99_9_value,p99_9_unit,"
    "p99_99_value,p99_99_unit,p99_999_value,p99_999_unit\n";

std::string JsonEscape(const std::string& value)
{
    std::ostringstream stream;
    for (const auto ch : value) {
        switch (ch) {
            case '\\': stream << "\\\\"; break;
            case '"': stream << "\\\""; break;
            case '\b': stream << "\\b"; break;
            case '\f': stream << "\\f"; break;
            case '\n': stream << "\\n"; break;
            case '\r': stream << "\\r"; break;
            case '\t': stream << "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    stream << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<int>(static_cast<unsigned char>(ch)) << std::dec
                           << std::setfill(' ');
                } else {
                    stream << ch;
                }
                break;
        }
    }
    return stream.str();
}

std::string JsonString(const std::string& value) { return "\"" + JsonEscape(value) + "\""; }

std::string HtmlEscape(const std::string& value)
{
    std::ostringstream stream;
    for (const auto ch : value) {
        switch (ch) {
            case '&': stream << "&amp;"; break;
            case '<': stream << "&lt;"; break;
            case '>': stream << "&gt;"; break;
            case '"': stream << "&quot;"; break;
            case '\'': stream << "&#39;"; break;
            default: stream << ch; break;
        }
    }
    return stream.str();
}

std::string FormatNumber(double value, int precision = 2)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

std::string FormatLocalTimestamp(const char* format)
{
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm localTime{};
#ifdef _WIN32
    localtime_s(&localTime, &time);
#else
    localtime_r(&time, &localTime);
#endif

    std::ostringstream stream;
    stream << std::put_time(&localTime, format);
    return stream.str();
}

std::string ResultStatusName(const CommandResult& result)
{
    if (!result.status.Ok()) { return "failed"; }
    if (result.taskResult.status.code == kv::StatusCode::PARTIAL_FAILED) {
        return "partial_failed";
    }
    if (!result.taskResult.status.ok()) { return "failed"; }

    const auto failedEntry =
        std::find_if(result.taskResult.entryStatus.begin(), result.taskResult.entryStatus.end(),
                     [](const kv::Status& status) { return !status.ok(); });
    return failedEntry == result.taskResult.entryStatus.end() ? "success" : "partial_failed";
}

std::uint64_t KeyCount(const CommandOptions& options, const CommandResult& result)
{
    if (!options.keys.empty()) { return options.keys.size(); }
    if (options.keyStartSet && options.keyEndSet) { return options.keyEnd - options.keyStart + 1; }
    if (options.count != 0) { return options.count; }
    if (!result.taskResult.entryStatus.empty()) { return result.taskResult.entryStatus.size(); }
    if (!result.queryResult.exists.empty()) { return result.queryResult.exists.size(); }
    return 0;
}

std::filesystem::path BuildOutputPath(const std::string& outputDir, const std::string& fileName)
{
    return std::filesystem::path(outputDir) / fileName;
}

std::string BuildBandwidthSvg(const std::vector<BenchRealtimeSample>& samples)
{
    constexpr double kWidth = 720.0;
    constexpr double kHeight = 220.0;
    constexpr double kPad = 32.0;
    if (samples.empty()) { return "<p class=\"muted\">No realtime samples were recorded.</p>"; }

    double maxBandwidthMiB = 0.0;
    for (const auto& sample : samples) {
        maxBandwidthMiB =
            std::max(maxBandwidthMiB, sample.bandwidthBytesPerSec / (1024.0 * 1024.0));
    }
    if (maxBandwidthMiB <= 0.0) { maxBandwidthMiB = 1.0; }

    std::ostringstream points;
    for (std::size_t index = 0; index < samples.size(); ++index) {
        const auto x =
            samples.size() == 1
                ? kPad
                : kPad + (kWidth - 2.0 * kPad) *
                             (static_cast<double>(index) / static_cast<double>(samples.size() - 1));
        const auto y =
            kHeight - kPad -
            (kHeight - 2.0 * kPad) *
                ((samples[index].bandwidthBytesPerSec / (1024.0 * 1024.0)) / maxBandwidthMiB);
        points << x << ',' << y << ' ';
    }

    std::ostringstream svg;
    svg << "<svg viewBox=\"0 0 720 220\" role=\"img\" aria-label=\"Bandwidth curve\">"
        << "<line x1=\"32\" y1=\"188\" x2=\"688\" y2=\"188\" class=\"axis\"/>"
        << "<line x1=\"32\" y1=\"32\" x2=\"32\" y2=\"188\" class=\"axis\"/>"
        << "<polyline points=\"" << points.str() << "\" class=\"line\"/>"
        << "<text x=\"36\" y=\"24\" class=\"chart-label\">Bandwidth MiB/s, max "
        << FormatNumber(maxBandwidthMiB) << "</text>"
        << "</svg>";
    return svg.str();
}

}  // namespace

Status ResultWriter::Open(const OutputConfig& config)
{
    Close();

    const auto baseDir =
        config.path.empty() ? std::filesystem::path(".") : std::filesystem::path(config.path);
    const auto runDir = baseDir / ("run-" + FormatLocalTimestamp("%Y%m%d-%H%M%S"));

    std::error_code errorCode;
    std::filesystem::create_directories(runDir, errorCode);
    if (errorCode) {
        return Status::Error(
            kExitInvalidArgument,
            "failed to create output directory " + runDir.string() + ": " + errorCode.message());
    }

    outputDir_ = runDir.string();
    baseOutputDir_ = baseDir.string();
    realtimeFileMaxBytes_ = config.realtimeFileMaxBytes == 0 ? kDefaultRealtimeFileMaxBytes
                                                             : config.realtimeFileMaxBytes;
    realtimeFileIndex_ = 0;
    realtimeFileBytes_ = 0;

    return Status::Success();
}

Status ResultWriter::WriteSummary(const CommandOptions& options, const CommandResult& result)
{
    if (outputDir_.empty()) {
        return Status::Error(kExitInvalidArgument, "ResultWriter is not open");
    }

    const auto commandName = CommandTypeName(options.command);
    const auto opName =
        options.command == CommandType::BENCH ? BenchOpTypeName(options.benchOp) : commandName;
    const auto statusName = ResultStatusName(result);
    const auto exitCode = result.status.Ok() ? 0 : result.status.code;
    const auto keyCount = KeyCount(options, result);
    const auto asuStatusCode = AsuStatusCodeName(result.taskResult.status.code);
    const auto summaryTime = FormatLocalTimestamp("%Y-%m-%dT%H:%M:%S%z");

    std::ofstream textFile{BuildOutputPath(outputDir_, "summary.txt")};
    if (!textFile.is_open()) {
        return Status::Error(kExitInvalidArgument, "failed to open summary.txt");
    }
    textFile << "tool: kv-test\n"
             << "command: " << commandName << '\n'
             << "op: " << opName << '\n'
             << "status: " << statusName << '\n'
             << "exit_code: " << exitCode << '\n'
             << "configpath: " << options.configPath << '\n'
             << "key_count: " << keyCount << '\n'
             << "value_size: " << options.valueSize << '\n'
             << "batch_size: " << options.batchSize << '\n'
             << "concurrency: " << options.concurrency << '\n'
             << "duration_sec: " << options.durationSec << '\n'
             << "io_count: " << options.ioCount << '\n'
             << "asu_status_code: " << asuStatusCode << '\n';
    if (!result.status.Ok()) { textFile << "error: " << result.status.message << '\n'; }
    if (!result.taskResult.status.message.empty()) {
        textFile << "asu_message: " << result.taskResult.status.message << '\n';
    }
    if (!textFile.good()) {
        return Status::Error(kExitInvalidArgument, "failed to write summary.txt");
    }

    std::ofstream jsonFile{BuildOutputPath(outputDir_, "summary.json")};
    if (!jsonFile.is_open()) {
        return Status::Error(kExitInvalidArgument, "failed to open summary.json");
    }

    jsonFile << "{\n"
             << "  \"tool\": \"kv-test\",\n"
             << "  \"command\": " << JsonString(commandName) << ",\n"
             << "  \"status\": " << JsonString(statusName) << ",\n"
             << "  \"exit_code\": " << exitCode << ",\n"
             << "  \"configpath\": " << JsonString(options.configPath) << ",\n"
             << "  \"start_time\": null,\n"
             << "  \"end_time\": " << JsonString(summaryTime) << ",\n"
             << "  \"connection\": {\n"
             << "    \"status\": " << JsonString(result.status.Ok() ? "success" : "failed") << ",\n"
             << "    \"asu_status_code\": " << JsonString(asuStatusCode) << "\n"
             << "  },\n"
             << "  \"request\": {\n"
             << "    \"op\": " << JsonString(opName) << ",\n"
             << "    \"key_count\": " << keyCount << ",\n"
             << "    \"value_size\": " << options.valueSize << ",\n"
             << "    \"batch_size\": " << options.batchSize << ",\n"
             << "    \"concurrency\": " << options.concurrency << ",\n"
             << "    \"duration_sec\": " << options.durationSec << ",\n"
             << "    \"io_count\": " << options.ioCount << "\n"
             << "  },\n";
    if (options.command == CommandType::BENCH) {
        jsonFile << "  \"metrics\": {\n"
                 << "    \"bandwidth\": {\"avg\": {\"value\": "
                 << (result.benchMetrics.avgBandwidthBytesPerSec / (1024.0 * 1024.0))
                 << ", \"unit\": \"MiB/s\"}, "
                    "\"realtime_file_pattern\": \"bench-realtime-*.csv\"},\n"
                 << "    \"iops\": {\"avg\": {\"value\": " << result.benchMetrics.avgIops
                 << ", \"unit\": \"1/s\"}, \"avg_batch\": {\"value\": "
                 << result.benchMetrics.avgBatchIops << ", \"unit\": \"1/s\"}},\n"
                 << "    \"latency\": {\"avg\": {\"value\": " << result.benchMetrics.latency.avgUs
                 << ", \"unit\": \"us\"}, \"min\": {\"value\": "
                 << result.benchMetrics.latency.minUs
                 << ", \"unit\": \"us\"}, \"max\": {\"value\": "
                 << result.benchMetrics.latency.maxUs
                 << ", \"unit\": \"us\"}, \"p99_9\": {\"value\": "
                 << result.benchMetrics.latency.p99_9Us
                 << ", \"unit\": \"us\"}, \"p99_99\": {\"value\": "
                 << result.benchMetrics.latency.p99_99Us
                 << ", \"unit\": \"us\"}, \"p99_999\": {\"value\": "
                 << result.benchMetrics.latency.p99_999Us << ", \"unit\": \"us\"}}\n"
                 << "  },\n";
    } else {
        jsonFile << "  \"metrics\": null,\n";
    }

    if (result.consistency.enabled) {
        const auto passRate = result.consistency.checked == 0
                                  ? 0.0
                                  : static_cast<double>(result.consistency.passed) /
                                        static_cast<double>(result.consistency.checked);
        jsonFile << "  \"consistency\": {\"enabled\": true, \"checked\": "
                 << result.consistency.checked << ", \"passed\": " << result.consistency.passed
                 << ", \"failed\": " << result.consistency.failed << ", \"pass_rate\": " << passRate
                 << ", \"key\": " << JsonString(result.consistency.key)
                 << ", \"expected\": " << JsonString(result.consistency.expected)
                 << ", \"actual\": " << JsonString(result.consistency.actual) << "},\n";
    } else {
        jsonFile << "  \"consistency\": " << (options.check ? "{\"enabled\": true}" : "null")
                 << ",\n";
    }
    if (result.status.Ok() && result.taskResult.status.ok()) {
        jsonFile << "  \"error\": null\n";
    } else {
        const auto message =
            !result.status.Ok() ? result.status.message : result.taskResult.status.message;
        jsonFile << "  \"error\": {\n"
                 << "    \"code\": " << JsonString(statusName) << ",\n"
                 << "    \"message\": " << JsonString(message) << ",\n"
                 << "    \"asu_status_code\": " << JsonString(asuStatusCode) << ",\n"
                 << "    \"retryable\": false\n"
                 << "  }\n";
    }
    jsonFile << "}\n";
    if (!jsonFile.good()) {
        return Status::Error(kExitInvalidArgument, "failed to write summary.json");
    }

    if (options.command == CommandType::BENCH) {
        for (const auto& sample : result.benchMetrics.realtimeSamples) {
            std::ostringstream line;
            line << sample.timestampSec << ',' << BenchOpTypeName(sample.op) << ','
                 << (sample.bandwidthBytesPerSec / (1024.0 * 1024.0)) << ",MiB/s," << sample.iops
                 << ",1/s," << sample.avgLatencyUs << ",us," << sample.errorCount;
            auto status = WriteRealtimeSample(line.str());
            if (!status.Ok()) { return status; }
        }

        std::ostringstream latencyLine;
        latencyLine << BenchOpTypeName(result.benchMetrics.op) << ','
                    << result.benchMetrics.latency.avgUs << ",us,"
                    << result.benchMetrics.latency.minUs << ",us,"
                    << result.benchMetrics.latency.maxUs << ",us,"
                    << result.benchMetrics.latency.p99_9Us << ",us,"
                    << result.benchMetrics.latency.p99_99Us << ",us,"
                    << result.benchMetrics.latency.p99_999Us << ",us";
        auto status = WriteLatencySample(latencyLine.str());
        if (!status.Ok()) { return status; }
    }

    if (result.consistency.enabled && result.consistency.failed != 0) {
        std::ostringstream line;
        line << "{\"key\":" << JsonString(result.consistency.key)
             << ",\"expected\":" << JsonString(result.consistency.expected)
             << ",\"actual\":" << JsonString(result.consistency.actual)
             << ",\"message\":" << JsonString(result.status.message) << "}";
        auto status = WriteConsistencyError(line.str());
        if (!status.Ok()) { return status; }
    }

    auto status = WriteHtmlReport(options, result);
    if (!status.Ok()) { return status; }
    return WriteReportIndex();
}

Status ResultWriter::WriteHtmlReport(const CommandOptions& options, const CommandResult& result)
{
    const auto commandName = CommandTypeName(options.command);
    const auto opName = options.command == CommandType::BENCH
                            ? BenchOpTypeName(result.benchMetrics.op)
                            : commandName;
    const auto statusName = ResultStatusName(result);
    const auto keyCount = KeyCount(options, result);
    const auto asuStatusCode = AsuStatusCodeName(result.taskResult.status.code);

    std::ofstream htmlFile{BuildOutputPath(outputDir_, "report.html")};
    if (!htmlFile.is_open()) {
        return Status::Error(kExitInvalidArgument, "failed to open report.html");
    }

    htmlFile
        << "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        << "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        << "<title>kv-test report - " << HtmlEscape(commandName) << "</title>"
        << "<style>"
        << "body{font-family:Arial,sans-serif;margin:0;background:#f6f7f9;color:#1f2933}"
        << "header{background:#17202a;color:white;padding:24px 32px}"
        << "main{padding:24px 32px;max-width:1180px;margin:auto}"
        << ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}"
        << ".card,section{background:white;border:1px solid #d9dee7;border-radius:8px;padding:16px}"
        << ".label{color:#64748b;font-size:12px;text-transform:uppercase}"
        << ".value{font-size:24px;font-weight:700;margin-top:6px}"
        << "table{border-collapse:collapse;width:100%;background:white}"
        << "th,td{border-bottom:1px solid #e5e7eb;padding:9px;text-align:left}"
        << "th{background:#f1f5f9}.ok{color:#047857}.failed{color:#b91c1c}"
        << ".muted{color:#64748b}.axis{stroke:#94a3b8;stroke-width:1}.line{fill:none;stroke:#"
           "2563eb;stroke-width:3}"
        << ".chart-label{fill:#475569;font-size:13px}"
        << "</style></head><body><header><h1>kv-test report</h1><p>" << HtmlEscape(commandName)
        << " / " << HtmlEscape(opName) << "</p></header><main>";

    htmlFile << "<div class=\"grid\">"
             << "<div class=\"card\"><div class=\"label\">Status</div><div class=\"value "
             << (statusName == "success" ? "ok" : "failed") << "\">" << HtmlEscape(statusName)
             << "</div></div>"
             << "<div class=\"card\"><div class=\"label\">Keys</div><div class=\"value\">"
             << keyCount << "</div></div>"
             << "<div class=\"card\"><div class=\"label\">ASU status</div><div class=\"value\">"
             << HtmlEscape(asuStatusCode) << "</div></div>"
             << "<div class=\"card\"><div class=\"label\">Exit code</div><div class=\"value\">"
             << (result.status.Ok() ? 0 : result.status.code) << "</div></div></div>";

    htmlFile << "<section><h2>Request</h2><table><tbody>"
             << "<tr><th>Config</th><td>" << HtmlEscape(options.configPath) << "</td></tr>"
             << "<tr><th>Command</th><td>" << HtmlEscape(commandName) << "</td></tr>"
             << "<tr><th>Operation</th><td>" << HtmlEscape(opName) << "</td></tr>"
             << "<tr><th>Value size</th><td>" << options.valueSize << "</td></tr>"
             << "<tr><th>Batch size</th><td>" << options.batchSize << "</td></tr>"
             << "<tr><th>Concurrency</th><td>" << options.concurrency << "</td></tr>"
             << "<tr><th>Duration sec</th><td>" << options.durationSec << "</td></tr>"
             << "<tr><th>IO count</th><td>" << options.ioCount << "</td></tr>"
             << "</tbody></table></section>";

    if (options.command == CommandType::BENCH) {
        const auto& metrics = result.benchMetrics;
        htmlFile << "<section><h2>Benchmark metrics</h2><div class=\"grid\">"
                 << "<div class=\"card\"><div class=\"label\">Bandwidth</div><div class=\"value\">"
                 << FormatNumber(metrics.avgBandwidthBytesPerSec / (1024.0 * 1024.0))
                 << " MiB/s</div></div>"
                 << "<div class=\"card\"><div class=\"label\">IOPS</div><div class=\"value\">"
                 << FormatNumber(metrics.avgIops) << "</div></div>"
                 << "<div class=\"card\"><div class=\"label\">Batch IOPS</div><div class=\"value\">"
                 << FormatNumber(metrics.avgBatchIops) << "</div></div>"
                 << "<div class=\"card\"><div class=\"label\">Errors</div><div class=\"value\">"
                 << metrics.errorCount << "</div></div></div>"
                 << BuildBandwidthSvg(metrics.realtimeSamples) << "<h3>Latency</h3><table><tbody>"
                 << "<tr><th>Average us</th><td>" << FormatNumber(metrics.latency.avgUs)
                 << "</td></tr><tr><th>Min us</th><td>" << FormatNumber(metrics.latency.minUs)
                 << "</td></tr><tr><th>Max us</th><td>" << FormatNumber(metrics.latency.maxUs)
                 << "</td></tr><tr><th>P99.9 us</th><td>" << FormatNumber(metrics.latency.p99_9Us)
                 << "</td></tr><tr><th>P99.99 us</th><td>" << FormatNumber(metrics.latency.p99_99Us)
                 << "</td></tr><tr><th>P99.999 us</th><td>"
                 << FormatNumber(metrics.latency.p99_999Us)
                 << "</td></tr></tbody></table></section>";
    }

    if (result.consistency.enabled) {
        htmlFile << "<section><h2>Consistency</h2><table><tbody>"
                 << "<tr><th>Checked</th><td>" << result.consistency.checked << "</td></tr>"
                 << "<tr><th>Passed</th><td>" << result.consistency.passed << "</td></tr>"
                 << "<tr><th>Failed</th><td>" << result.consistency.failed << "</td></tr>"
                 << "<tr><th>Key</th><td>" << HtmlEscape(result.consistency.key) << "</td></tr>"
                 << "<tr><th>Expected</th><td>" << HtmlEscape(result.consistency.expected)
                 << "</td></tr><tr><th>Actual</th><td>" << HtmlEscape(result.consistency.actual)
                 << "</td></tr></tbody></table></section>";
    }

    if (!result.status.Ok() || !result.taskResult.status.ok()) {
        htmlFile << "<section><h2>Error</h2><p class=\"failed\">"
                 << HtmlEscape(!result.status.Ok() ? result.status.message
                                                   : result.taskResult.status.message)
                 << "</p></section>";
    }

    htmlFile << "<section><h2>Artifacts</h2><ul>"
             << "<li><a href=\"summary.txt\">summary.txt</a></li>"
             << "<li><a href=\"summary.json\">summary.json</a></li>";
    if (options.command == CommandType::BENCH) {
        htmlFile << "<li><a href=\"bench-realtime-0.csv\">bench-realtime-0.csv</a></li>"
                 << "<li><a href=\"latency.csv\">latency.csv</a></li>";
    }
    if (result.consistency.enabled && result.consistency.failed != 0) {
        htmlFile << "<li><a href=\"consistency_errors.jsonl\">consistency_errors.jsonl</a></li>";
    }
    htmlFile << "</ul></section></main></body></html>";

    if (!htmlFile.good()) {
        return Status::Error(kExitInvalidArgument, "failed to write report.html");
    }
    return Status::Success();
}

Status ResultWriter::WriteReportIndex()
{
    if (baseOutputDir_.empty()) {
        return Status::Error(kExitInvalidArgument, "ResultWriter base output dir is not set");
    }

    std::vector<std::filesystem::directory_entry> reports;
    std::error_code errorCode;
    for (const auto& entry : std::filesystem::directory_iterator(baseOutputDir_, errorCode)) {
        if (errorCode) { break; }
        if (!entry.is_directory()) { continue; }
        const auto reportPath = entry.path() / "report.html";
        if (std::filesystem::exists(reportPath)) { reports.emplace_back(entry); }
    }
    std::sort(reports.begin(), reports.end(), [](const auto& left, const auto& right) {
        return left.path().filename().string() > right.path().filename().string();
    });

    std::ofstream indexFile{BuildOutputPath(baseOutputDir_, "index.html")};
    if (!indexFile.is_open()) {
        return Status::Error(kExitInvalidArgument, "failed to open report index.html");
    }

    indexFile << "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
              << "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
              << "<title>kv-test reports</title><style>"
              << "body{font-family:Arial,sans-serif;margin:0;background:#f6f7f9;color:#1f2933}"
              << "main{padding:24px 32px;max-width:960px;margin:auto}"
              << "table{border-collapse:collapse;width:100%;background:white}"
              << "th,td{border-bottom:1px solid #e5e7eb;padding:10px;text-align:left}"
              << "th{background:#f1f5f9}a{color:#2563eb;text-decoration:none}"
              << "</style></head><body><main><h1>kv-test reports</h1><table>"
              << "<thead><tr><th>Run</th><th>Report</th></tr></thead><tbody>";
    for (const auto& entry : reports) {
        const auto runName = entry.path().filename().string();
        indexFile << "<tr><td>" << HtmlEscape(runName) << "</td><td><a href=\""
                  << HtmlEscape(runName) << "/report.html\">open report</a></td></tr>";
    }
    indexFile << "</tbody></table></main></body></html>";
    if (!indexFile.good()) {
        return Status::Error(kExitInvalidArgument, "failed to write report index.html");
    }
    return Status::Success();
}

Status ResultWriter::WriteRealtimeSample(const std::string& csvLine)
{
    if (!realtimeFile_.is_open()) {
        auto status = OpenRealtimeFile();
        if (!status.Ok()) { return status; }
    }

    const auto lineBytes = csvLine.size() + 1;
    auto status = RollRealtimeFileIfNeeded(lineBytes);
    if (!status.Ok()) { return status; }

    realtimeFile_ << csvLine << '\n';
    if (!realtimeFile_.good()) {
        return Status::Error(kExitInvalidArgument, "failed to write realtime CSV sample");
    }
    realtimeFileBytes_ += lineBytes;
    return Status::Success();
}

Status ResultWriter::WriteLatencySample(const std::string& csvLine)
{
    if (!latencyFile_.is_open()) {
        auto status = OpenLatencyFile();
        if (!status.Ok()) { return status; }
    }

    latencyFile_ << csvLine << '\n';
    if (!latencyFile_.good()) {
        return Status::Error(kExitInvalidArgument, "failed to write latency CSV sample");
    }
    return Status::Success();
}

Status ResultWriter::WriteConsistencyError(const std::string& line)
{
    if (!consistencyErrorFile_.is_open()) {
        auto status = OpenConsistencyErrorFile();
        if (!status.Ok()) { return status; }
    }

    consistencyErrorFile_ << line << '\n';
    if (!consistencyErrorFile_.good()) {
        return Status::Error(kExitInvalidArgument, "failed to write consistency error");
    }
    return Status::Success();
}

Status ResultWriter::Close()
{
    if (realtimeFile_.is_open()) { realtimeFile_.close(); }
    if (latencyFile_.is_open()) { latencyFile_.close(); }
    if (consistencyErrorFile_.is_open()) { consistencyErrorFile_.close(); }

    outputDir_.clear();
    baseOutputDir_.clear();
    realtimeFileBytes_ = 0;
    realtimeFileIndex_ = 0;
    return Status::Success();
}

Status ResultWriter::OpenLatencyFile()
{
    latencyFile_.open(BuildOutputPath(outputDir_, "latency.csv"));
    if (!latencyFile_.is_open()) {
        return Status::Error(kExitInvalidArgument, "failed to open latency.csv");
    }

    latencyFile_ << kLatencyCsvHeader;
    if (!latencyFile_.good()) {
        return Status::Error(kExitInvalidArgument, "failed to write latency.csv");
    }
    return Status::Success();
}

Status ResultWriter::OpenConsistencyErrorFile()
{
    consistencyErrorFile_.open(BuildOutputPath(outputDir_, "consistency_errors.jsonl"));
    if (!consistencyErrorFile_.is_open()) {
        return Status::Error(kExitInvalidArgument, "failed to open consistency_errors.jsonl");
    }
    return Status::Success();
}

Status ResultWriter::OpenRealtimeFile()
{
    if (realtimeFile_.is_open()) { realtimeFile_.close(); }

    const auto fileName = "bench-realtime-" + std::to_string(realtimeFileIndex_) + ".csv";
    realtimeFile_.open(BuildOutputPath(outputDir_, fileName));
    if (!realtimeFile_.is_open()) {
        return Status::Error(kExitInvalidArgument, "failed to open " + fileName);
    }

    realtimeFile_ << kRealtimeCsvHeader;
    if (!realtimeFile_.good()) {
        return Status::Error(kExitInvalidArgument, "failed to write " + fileName);
    }
    realtimeFileBytes_ = std::string(kRealtimeCsvHeader).size();
    return Status::Success();
}

Status ResultWriter::RollRealtimeFileIfNeeded(std::uint64_t incomingBytes)
{
    if (realtimeFileMaxBytes_ == 0 || realtimeFileBytes_ + incomingBytes <= realtimeFileMaxBytes_) {
        return Status::Success();
    }

    ++realtimeFileIndex_;
    return OpenRealtimeFile();
}

}  // namespace kv::bench
