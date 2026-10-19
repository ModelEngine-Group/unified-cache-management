#include "kv_metrics/standalone_metrics_backend.h"
#include <algorithm>
#include <arpa/inet.h>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <netdb.h>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>
#include "kv_metrics/default_metric_descriptors.h"

namespace kv::metrics {
namespace {

constexpr int kListenBacklog = 16;
constexpr std::size_t kMaxRequestBytes = 4096;

std::string Trim(const std::string& value)
{
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) { return {}; }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::string StripComment(const std::string& line)
{
    bool inSingleQuote = false;
    bool inDoubleQuote = false;
    for (std::size_t index = 0; index < line.size(); ++index) {
        if (line[index] == '\'' && !inDoubleQuote) { inSingleQuote = !inSingleQuote; }
        if (line[index] == '"' && !inSingleQuote) { inDoubleQuote = !inDoubleQuote; }
        if (line[index] == '#' && !inSingleQuote && !inDoubleQuote) {
            return line.substr(0, index);
        }
    }
    return line;
}

std::string Unquote(std::string value)
{
    value = Trim(value);
    if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                              (value.front() == '\'' && value.back() == '\''))) {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

bool ParseBuckets(const std::string& value, std::vector<double>& buckets)
{
    auto text = Trim(value);
    if (text.size() < 2 || text.front() != '[' || text.back() != ']') { return false; }
    text = text.substr(1, text.size() - 2);
    std::stringstream stream{text};
    std::string item;
    std::vector<double> parsed;
    try {
        while (std::getline(stream, item, ',')) {
            item = Trim(item);
            if (item.empty()) { continue; }
            std::size_t consumed = 0;
            const auto number = std::stod(item, &consumed);
            if (consumed != item.size() || !std::isfinite(number)) { return false; }
            parsed.emplace_back(number);
        }
    } catch (...) {
        return false;
    }
    if (!std::is_sorted(parsed.begin(), parsed.end()) ||
        std::adjacent_find(parsed.begin(), parsed.end()) != parsed.end()) {
        return false;
    }
    buckets = std::move(parsed);
    return true;
}

bool SplitInlineMapping(const std::string& value,
                        std::unordered_map<std::string, std::string>& fields)
{
    auto text = Trim(value);
    if (text.size() < 2 || text.front() != '{' || text.back() != '}') { return false; }
    text = text.substr(1, text.size() - 2);

    std::vector<std::string> entries;
    std::size_t begin = 0;
    std::size_t bracketDepth = 0;
    bool inSingleQuote = false;
    bool inDoubleQuote = false;
    for (std::size_t index = 0; index <= text.size(); ++index) {
        const char ch = index == text.size() ? ',' : text[index];
        if (ch == '\'' && !inDoubleQuote) { inSingleQuote = !inSingleQuote; }
        if (ch == '"' && !inSingleQuote) { inDoubleQuote = !inDoubleQuote; }
        if (!inSingleQuote && !inDoubleQuote) {
            if (ch == '[') {
                ++bracketDepth;
            } else if (ch == ']') {
                if (bracketDepth == 0) { return false; }
                --bracketDepth;
            } else if (ch == ',' && bracketDepth == 0) {
                entries.emplace_back(text.substr(begin, index - begin));
                begin = index + 1;
            }
        }
    }
    if (inSingleQuote || inDoubleQuote || bracketDepth != 0) { return false; }

    for (const auto& entry : entries) {
        const auto separator = entry.find(':');
        if (separator == std::string::npos) { return false; }
        const auto key = Trim(entry.substr(0, separator));
        if (key.empty() || !fields.emplace(key, Trim(entry.substr(separator + 1))).second) {
            return false;
        }
    }
    return true;
}

bool ParseInlineDescriptor(const std::string& value, MetricType type,
                           std::vector<MetricDescriptor>& descriptors, std::string& error,
                           const std::string& path, std::size_t lineNumber)
{
    std::unordered_map<std::string, std::string> fields;
    if (!SplitInlineMapping(value, fields)) {
        error = "invalid inline metric definition at " + path + ":" + std::to_string(lineNumber);
        return false;
    }
    const auto name = fields.find("name");
    if (name == fields.end() || Unquote(name->second).empty()) {
        error = "metric definition has no name at " + path + ":" + std::to_string(lineNumber);
        return false;
    }

    MetricDescriptor descriptor;
    descriptor.name = Unquote(name->second);
    descriptor.type = type;
    if (const auto documentation = fields.find("documentation"); documentation != fields.end()) {
        descriptor.documentation = Unquote(documentation->second);
    }
    if (const auto buckets = fields.find("buckets");
        buckets != fields.end() && !ParseBuckets(buckets->second, descriptor.buckets)) {
        error = "invalid histogram buckets at " + path + ":" + std::to_string(lineNumber);
        return false;
    }
    descriptors.emplace_back(std::move(descriptor));
    return true;
}

bool IsValidMetricName(const std::string& name)
{
    if (name.empty()) { return false; }
    const auto first = static_cast<unsigned char>(name.front());
    if (!(std::isalpha(first) || name.front() == '_' || name.front() == ':')) { return false; }
    return std::all_of(name.begin() + 1, name.end(),
                       [](unsigned char ch) { return std::isalnum(ch) || ch == '_' || ch == ':'; });
}

std::string EscapeHelp(const std::string& value)
{
    std::string escaped;
    escaped.reserve(value.size());
    for (char ch : value) {
        if (ch == '\\') {
            escaped += "\\\\";
        } else if (ch == '\n') {
            escaped += "\\n";
        } else {
            escaped.push_back(ch);
        }
    }
    return escaped;
}

std::string EscapeLabel(const std::string& value)
{
    std::string escaped;
    escaped.reserve(value.size());
    for (char ch : value) {
        if (ch == '\\') {
            escaped += "\\\\";
        } else if (ch == '"') {
            escaped += "\\\"";
        } else if (ch == '\n') {
            escaped += "\\n";
        } else {
            escaped.push_back(ch);
        }
    }
    return escaped;
}

const char* PrometheusTypeName(MetricType type)
{
    switch (type) {
        case MetricType::COUNTER: return "counter";
        case MetricType::GAUGE: return "gauge";
        case MetricType::HISTOGRAM: return "histogram";
        default: return "untyped";
    }
}

bool LoadYamlDefinitions(const std::string& path, std::string& metricPrefix,
                         std::vector<MetricDescriptor>& descriptors, std::string& error)
{
    std::ifstream input{path};
    if (!input.is_open()) {
        error = "failed to open metrics definition file: " + path;
        return false;
    }

    bool inMetricSection = false;
    MetricType sectionType = MetricType::COUNTER;
    MetricDescriptor* current = nullptr;
    std::string rawLine;
    std::size_t lineNumber = 0;
    while (std::getline(input, rawLine)) {
        ++lineNumber;
        const auto content = Trim(StripComment(rawLine));
        if (content.empty()) { continue; }

        if (content.rfind("metric_prefix:", 0) == 0) {
            metricPrefix = Unquote(content.substr(std::strlen("metric_prefix:")));
            continue;
        }
        if (content == "counter:" || content == "gauge:" || content == "histogram:") {
            inMetricSection = true;
            current = nullptr;
            sectionType = content == "counter:" ? MetricType::COUNTER
                          : content == "gauge:" ? MetricType::GAUGE
                                                : MetricType::HISTOGRAM;
            continue;
        }
        if (!rawLine.empty() && rawLine.front() != ' ' && rawLine.front() != '\t') {
            inMetricSection = false;
            current = nullptr;
            continue;
        }
        if (!inMetricSection) { continue; }

        if (content.rfind("- {", 0) == 0) {
            current = nullptr;
            if (!ParseInlineDescriptor(content.substr(2), sectionType, descriptors, error, path,
                                       lineNumber)) {
                return false;
            }
            continue;
        }
        if (content.rfind("- name:", 0) == 0) {
            MetricDescriptor descriptor;
            descriptor.name = Unquote(content.substr(std::strlen("- name:")));
            descriptor.type = sectionType;
            descriptors.emplace_back(std::move(descriptor));
            current = &descriptors.back();
            continue;
        }
        if (content.front() == '-') {
            error = "invalid metric definition at " + path + ":" + std::to_string(lineNumber);
            return false;
        }
        if (current == nullptr) { continue; }
        if (content.rfind("documentation:", 0) == 0) {
            current->documentation = Unquote(content.substr(std::strlen("documentation:")));
        } else if (content.rfind("buckets:", 0) == 0) {
            if (!ParseBuckets(content.substr(std::strlen("buckets:")), current->buckets)) {
                error = "invalid histogram buckets at " + path + ":" + std::to_string(lineNumber);
                return false;
            }
        }
    }
    return true;
}

class ThreadBufferedMetricsCollector {
public:
    ThreadBufferedMetricsCollector() : generation_(NextCollectorGeneration()) {}
    ~ThreadBufferedMetricsCollector() { StopAggregation(); }

    bool Register(const MetricDescriptor& descriptor, std::string& error)
    {
        if (!IsValidMetricName(descriptor.name)) {
            error = "invalid metric name: " + descriptor.name;
            return false;
        }
        if (descriptor.type == MetricType::HISTOGRAM && descriptor.buckets.empty()) {
            error = "histogram has no buckets: " + descriptor.name;
            return false;
        }
        if (!std::is_sorted(descriptor.buckets.begin(), descriptor.buckets.end()) ||
            std::adjacent_find(descriptor.buckets.begin(), descriptor.buckets.end()) !=
                descriptor.buckets.end()) {
            error = "histogram buckets must be sorted and unique: " + descriptor.name;
            return false;
        }
        const auto metricKey = MetricKey(descriptor.name, descriptor.labels);
        if (metricIds_.find(metricKey) != metricIds_.end()) {
            error = "duplicate metric name and labels: " + descriptor.name;
            return false;
        }
        metricIds_.emplace(metricKey, descriptors_.size());
        descriptors_.emplace_back(descriptor);
        snapshot_.emplace_back(MakeMetricState(descriptor));
        return true;
    }

    bool RegisterMetricLabels(const std::string& name, const MetricLabels& labels)
    {
        const auto base = std::find_if(
            descriptors_.begin(), descriptors_.end(), [&](const MetricDescriptor& descriptor) {
                return descriptor.name == name && descriptor.labels.empty();
            });
        if (base == descriptors_.end()) { return false; }
        auto descriptor = *base;
        descriptor.labels = labels;
        std::string error;
        return Register(descriptor, error);
    }

    void StartAggregation(std::uint32_t intervalMs)
    {
        intervalMs_ = std::chrono::milliseconds{std::max<std::uint32_t>(1, intervalMs)};
        if (aggregatorRunning_.exchange(true, std::memory_order_acq_rel)) { return; }
        aggregator_ = std::thread(&ThreadBufferedMetricsCollector::AggregatorLoop, this);
    }

    void StopAggregation()
    {
        {
            std::lock_guard<std::mutex> lock{wakeupMutex_};
            if (!aggregatorRunning_.exchange(false, std::memory_order_acq_rel)) { return; }
        }
        wakeup_.notify_all();
        if (aggregator_.joinable()) { aggregator_.join(); }
        Aggregate();
    }

    struct SlotBinding final : CachedMetric::Binding {
        explicit SlotBinding(std::size_t slot) : slot(slot) {}
        const std::size_t slot;
    };

    std::size_t ResolveMetric(CachedMetric& metric)
    {
        auto* binding = static_cast<SlotBinding*>(metric.Resolve([&] {
            const auto iter = metricIds_.find(MetricKey(metric.Name(), metric.Labels()));
            return std::make_unique<SlotBinding>(iter == metricIds_.end() ? kInvalidMetricId
                                                                          : iter->second);
        }));
        return binding->slot;
    }

    void UpdateStats(const MetricUpdate* updates, std::size_t count) noexcept
    {
        if (updates == nullptr || count == 0) { return; }
        try {
            auto* buffer = GetThreadBuffer();
            ThreadBuffer::WriteGuard guard{*buffer};
            auto& metrics = buffer->slots[guard.Index()].metrics;
            for (std::size_t index = 0; index < count; ++index) {
                if (updates[index].metric == nullptr) { continue; }
                const auto slot = ResolveMetric(*updates[index].metric);
                if (slot != kInvalidMetricId) {
                    ApplyUpdate(metrics[slot], slot, updates[index].value);
                }
            }
        } catch (...) {
        }
    }

    void Flush() { Aggregate(); }

    std::string Render(const std::string& prefix,
                       const std::map<std::string, std::string>& labels) const
    {
        std::shared_lock<std::shared_mutex> lock{snapshotMutex_};
        std::ostringstream output;
        output << std::setprecision(17);
        for (std::size_t id = 0; id < descriptors_.size(); ++id) {
            const auto& descriptor = descriptors_[id];
            const auto& state = snapshot_[id];
            const auto fullName = prefix + descriptor.name;
            output << "# HELP " << fullName << ' ' << EscapeHelp(descriptor.documentation) << '\n';
            output << "# TYPE " << fullName << ' ' << PrometheusTypeName(descriptor.type) << '\n';
            if (descriptor.type == MetricType::HISTOGRAM) {
                std::uint64_t cumulative = 0;
                for (std::size_t index = 0; index < descriptor.buckets.size(); ++index) {
                    cumulative += state.bucketCounts[index];
                    output << fullName << "_bucket"
                           << RenderLabels(MergeLabels(labels, descriptor.labels), "le",
                                           FormatNumber(descriptor.buckets[index]))
                           << ' ' << cumulative << '\n';
                }
                output << fullName << "_bucket"
                       << RenderLabels(MergeLabels(labels, descriptor.labels), "le", "+Inf") << ' '
                       << state.count << '\n';
                output << fullName << "_sum" << RenderLabels(MergeLabels(labels, descriptor.labels))
                       << ' ' << state.sum << '\n';
                output << fullName << "_count"
                       << RenderLabels(MergeLabels(labels, descriptor.labels)) << ' ' << state.count
                       << '\n';
            } else {
                output << fullName << RenderLabels(MergeLabels(labels, descriptor.labels)) << ' '
                       << state.value << '\n';
            }
        }
        return output.str();
    }

private:
    static std::string MetricKey(const std::string& name, const MetricLabels& labels)
    {
        std::ostringstream output;
        output << name;
        for (const auto& [key, value] : labels) { output << '\x1f' << key << '=' << value; }
        return output.str();
    }

    static MetricLabels MergeLabels(const MetricLabels& base, const MetricLabels& extra)
    {
        auto result = base;
        result.insert(extra.begin(), extra.end());
        return result;
    }
    static std::uint64_t NextCollectorGeneration() noexcept
    {
        static std::atomic<std::uint64_t> generation{1};
        return generation.fetch_add(1, std::memory_order_relaxed);
    }

    struct MetricState {
        double value{0.0};
        std::uint64_t sequence{0};
        bool hasGaugeValue{false};
        double sum{0.0};
        std::uint64_t count{0};
        std::vector<std::uint64_t> bucketCounts;
    };

    struct DeltaSlot {
        std::vector<MetricState> metrics;
    };

    struct ThreadBuffer {
        static constexpr int kNoActiveWriter = -1;

        class WriteGuard {
        public:
            explicit WriteGuard(ThreadBuffer& buffer)
                : buffer_(buffer), index_(buffer_.BeginWrite())
            {
            }

            ~WriteGuard() { buffer_.EndWrite(); }

            WriteGuard(const WriteGuard&) = delete;
            WriteGuard& operator=(const WriteGuard&) = delete;

            int Index() const noexcept { return index_; }

        private:
            ThreadBuffer& buffer_;
            int index_;
        };

        explicit ThreadBuffer(const std::vector<MetricDescriptor>& descriptors)
        {
            for (auto& slot : slots) {
                slot.metrics.reserve(descriptors.size());
                for (const auto& descriptor : descriptors) {
                    slot.metrics.emplace_back(MakeMetricState(descriptor));
                }
            }
        }

        int BeginWrite() noexcept
        {
            while (true) {
                const int index = writeIndex.load(std::memory_order_seq_cst);
                activeWriteIndex.store(index, std::memory_order_seq_cst);
                if (writeIndex.load(std::memory_order_seq_cst) == index) { return index; }
                activeWriteIndex.store(kNoActiveWriter, std::memory_order_seq_cst);
            }
        }

        void EndWrite() noexcept
        {
            activeWriteIndex.store(kNoActiveWriter, std::memory_order_seq_cst);
        }

        int SwitchWriteSlot() noexcept
        {
            return writeIndex.fetch_xor(1, std::memory_order_seq_cst);
        }

        void WaitUntilInactive(int index) const noexcept
        {
            while (activeWriteIndex.load(std::memory_order_seq_cst) == index) {
                std::this_thread::yield();
            }
        }

        std::atomic<int> writeIndex{0};
        std::atomic<int> activeWriteIndex{kNoActiveWriter};
        std::atomic<bool> retired{false};
        DeltaSlot slots[2];
    };

    struct ThreadLocalBufferCache {
        ~ThreadLocalBufferCache() { Retire(); }

        void Bind(const ThreadBufferedMetricsCollector* newOwner, std::uint64_t newOwnerGeneration,
                  std::shared_ptr<ThreadBuffer> newBuffer)
        {
            Retire();
            owner = newOwner;
            ownerGeneration = newOwnerGeneration;
            buffer = std::move(newBuffer);
        }

        void Retire() noexcept
        {
            if (buffer) { buffer->retired.store(true, std::memory_order_release); }
            buffer.reset();
            owner = nullptr;
            ownerGeneration = 0;
        }

        const ThreadBufferedMetricsCollector* owner{nullptr};
        std::uint64_t ownerGeneration{0};
        std::shared_ptr<ThreadBuffer> buffer;
    };

    static constexpr std::size_t kInvalidMetricId = std::numeric_limits<std::size_t>::max();

    static MetricState MakeMetricState(const MetricDescriptor& descriptor)
    {
        MetricState state;
        if (descriptor.type == MetricType::HISTOGRAM) {
            state.bucketCounts.assign(descriptor.buckets.size() + 1, 0);
        }
        return state;
    }

    bool FindMetric(std::string_view name, std::size_t& id) const noexcept
    {
        const auto iter = metricIds_.find(std::string{name});
        if (iter == metricIds_.end()) { return false; }
        id = iter->second;
        return true;
    }

    void ApplyUpdate(MetricState& metric, std::size_t id, double value) noexcept
    {
        if (!std::isfinite(value)) { return; }
        switch (descriptors_[id].type) {
            case MetricType::COUNTER:
                if (value >= 0.0) { metric.value += value; }
                break;
            case MetricType::GAUGE:
                metric.value = value;
                metric.sequence = gaugeSequence_.fetch_add(1, std::memory_order_relaxed) + 1;
                metric.hasGaugeValue = true;
                break;
            case MetricType::HISTOGRAM: {
                metric.sum += value;
                ++metric.count;
                const auto& buckets = descriptors_[id].buckets;
                const auto bucket = static_cast<std::size_t>(
                    std::lower_bound(buckets.begin(), buckets.end(), value) - buckets.begin());
                ++metric.bucketCounts[bucket];
                break;
            }
        }
    }

    ThreadBuffer* GetThreadBuffer()
    {
        static thread_local ThreadLocalBufferCache cache;
        if (cache.owner == this && cache.ownerGeneration == generation_ && cache.buffer) {
            return cache.buffer.get();
        }
        auto buffer = std::make_shared<ThreadBuffer>(descriptors_);
        auto* const rawBuffer = buffer.get();
        {
            std::lock_guard<std::mutex> lock{buffersMutex_};
            buffers_.emplace_back(buffer);
        }
        cache.Bind(this, generation_, std::move(buffer));
        return rawBuffer;
    }

    void AggregatorLoop()
    {
        std::unique_lock<std::mutex> lock{wakeupMutex_};
        while (aggregatorRunning_.load(std::memory_order_acquire)) {
            wakeup_.wait_for(lock, intervalMs_, [this] {
                return !aggregatorRunning_.load(std::memory_order_acquire);
            });
            if (!aggregatorRunning_.load(std::memory_order_acquire)) { break; }
            lock.unlock();
            Aggregate();
            lock.lock();
        }
    }

    void Aggregate()
    {
        std::lock_guard<std::mutex> aggregateLock{aggregateMutex_};
        std::vector<std::shared_ptr<ThreadBuffer>> buffers;
        {
            std::lock_guard<std::mutex> lock{buffersMutex_};
            buffers = buffers_;
        }
        std::vector<ThreadBuffer*> drainedRetiredBuffers;
        {
            std::unique_lock<std::shared_mutex> snapshotLock{snapshotMutex_};
            for (const auto& buffer : buffers) {
                const bool retired = buffer->retired.load(std::memory_order_acquire);
                DrainSlot(*buffer, buffer->SwitchWriteSlot());
                if (retired) {
                    DrainSlot(*buffer, buffer->SwitchWriteSlot());
                    drainedRetiredBuffers.emplace_back(buffer.get());
                }
            }
        }
        if (!drainedRetiredBuffers.empty()) {
            std::lock_guard<std::mutex> lock{buffersMutex_};
            buffers_.erase(std::remove_if(buffers_.begin(), buffers_.end(),
                                          [&](const auto& buffer) {
                                              return std::find(drainedRetiredBuffers.begin(),
                                                               drainedRetiredBuffers.end(),
                                                               buffer.get()) !=
                                                     drainedRetiredBuffers.end();
                                          }),
                           buffers_.end());
        }
    }

    void DrainSlot(ThreadBuffer& buffer, int slot)
    {
        buffer.WaitUntilInactive(slot);
        auto& source = buffer.slots[slot].metrics;
        for (std::size_t id = 0; id < source.size(); ++id) {
            auto& from = source[id];
            auto& to = snapshot_[id];
            switch (descriptors_[id].type) {
                case MetricType::COUNTER:
                    to.value += from.value;
                    from.value = 0.0;
                    break;
                case MetricType::GAUGE:
                    if (from.hasGaugeValue && from.sequence >= to.sequence) {
                        to.value = from.value;
                        to.sequence = from.sequence;
                        to.hasGaugeValue = true;
                    }
                    from.hasGaugeValue = false;
                    from.sequence = 0;
                    break;
                case MetricType::HISTOGRAM:
                    to.sum += from.sum;
                    to.count += from.count;
                    for (std::size_t bucket = 0; bucket < from.bucketCounts.size(); ++bucket) {
                        to.bucketCounts[bucket] += from.bucketCounts[bucket];
                        from.bucketCounts[bucket] = 0;
                    }
                    from.sum = 0.0;
                    from.count = 0;
                    break;
            }
        }
    }

    static std::string FormatNumber(double value)
    {
        std::ostringstream output;
        output << std::setprecision(17) << value;
        return output.str();
    }

    static std::string RenderLabels(const std::map<std::string, std::string>& labels,
                                    const std::string& extraName = {},
                                    const std::string& extraValue = {})
    {
        if (labels.empty() && extraName.empty()) { return {}; }
        std::ostringstream output;
        output << '{';
        bool first = true;
        for (const auto& label : labels) {
            if (!first) { output << ','; }
            output << label.first << "=\"" << EscapeLabel(label.second) << '"';
            first = false;
        }
        if (!extraName.empty()) {
            if (!first) { output << ','; }
            output << extraName << "=\"" << EscapeLabel(extraValue) << '"';
        }
        output << '}';
        return output.str();
    }

    std::vector<MetricDescriptor> descriptors_;
    const std::uint64_t generation_;
    std::unordered_map<std::string, std::size_t> metricIds_;
    mutable std::shared_mutex snapshotMutex_;
    std::vector<MetricState> snapshot_;
    std::atomic<std::uint64_t> gaugeSequence_{0};
    std::mutex buffersMutex_;
    std::vector<std::shared_ptr<ThreadBuffer>> buffers_;
    std::mutex aggregateMutex_;
    std::chrono::milliseconds intervalMs_{500};
    std::atomic<bool> aggregatorRunning_{false};
    std::mutex wakeupMutex_;
    std::condition_variable wakeup_;
    std::thread aggregator_;
};

class MetricsHttpServer {
public:
    MetricsHttpServer(ThreadBufferedMetricsCollector& registry, StandaloneMetricsConfig config)
        : registry_(registry), config_(std::move(config))
    {
    }

    ~MetricsHttpServer() { Stop(); }

    bool Start(std::string& error)
    {
        if (running_.load(std::memory_order_acquire)) { return true; }

        struct addrinfo hints{};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_flags = AI_NUMERICSERV;
        struct addrinfo* addresses = nullptr;
        const auto portText = std::to_string(config_.port);
        const auto rc =
            getaddrinfo(config_.listenAddress.c_str(), portText.c_str(), &hints, &addresses);
        if (rc != 0) {
            error = "failed to resolve metrics listen address: " + std::string{gai_strerror(rc)};
            return false;
        }

        for (auto* address = addresses; address != nullptr; address = address->ai_next) {
            listenFd_ = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
            if (listenFd_ < 0) { continue; }
            int reuse = 1;
            (void)setsockopt(listenFd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
            if (bind(listenFd_, address->ai_addr, address->ai_addrlen) == 0 &&
                listen(listenFd_, kListenBacklog) == 0) {
                break;
            }
            close(listenFd_);
            listenFd_ = -1;
        }
        freeaddrinfo(addresses);

        if (listenFd_ < 0) {
            error = "failed to listen on " + config_.listenAddress + ':' +
                    std::to_string(config_.port) + ": " + std::strerror(errno);
            return false;
        }

        running_.store(true, std::memory_order_release);
        worker_ = std::thread(&MetricsHttpServer::Serve, this);
        return true;
    }

    void Stop()
    {
        if (!running_.exchange(false, std::memory_order_acq_rel)) { return; }
        const int fd = listenFd_;
        if (fd >= 0) {
            (void)shutdown(fd, SHUT_RDWR);
            close(fd);
        }
        if (worker_.joinable()) { worker_.join(); }
        listenFd_ = -1;
    }

private:
    void Serve()
    {
        while (running_.load(std::memory_order_acquire)) {
            const int clientFd = accept(listenFd_, nullptr, nullptr);
            if (clientFd < 0) {
                if (!running_.load(std::memory_order_acquire)) { return; }
                continue;
            }
            Handle(clientFd);
            close(clientFd);
        }
    }

    void Handle(int clientFd)
    {
        timeval timeout{};
        timeout.tv_sec = 2;
        (void)setsockopt(clientFd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        (void)setsockopt(clientFd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

        char request[kMaxRequestBytes + 1] = {};
        const auto received = recv(clientFd, request, kMaxRequestBytes, 0);
        if (received <= 0) { return; }
        request[received] = '\0';

        std::istringstream requestLine{
            std::string{request, static_cast<std::size_t>(received)}
        };
        std::string method;
        std::string path;
        std::string version;
        requestLine >> method >> path >> version;
        (void)version;

        std::string status;
        std::string contentType;
        std::string body;
        if (method != "GET") {
            status = "405 Method Not Allowed";
            contentType = "text/plain; charset=utf-8";
            body = "method not allowed\n";
        } else if (path == config_.metricsPath) {
            status = "200 OK";
            contentType = "text/plain; version=0.0.4; charset=utf-8";
            body = registry_.Render(config_.metricPrefix, config_.constantLabels);
        } else {
            status = "404 Not Found";
            contentType = "text/plain; charset=utf-8";
            body = "not found\n";
        }

        std::ostringstream response;
        response << "HTTP/1.1 " << status << "\r\n"
                 << "Content-Type: " << contentType << "\r\n"
                 << "Content-Length: " << body.size() << "\r\n"
                 << "Connection: close\r\n\r\n"
                 << body;
        const auto text = response.str();
        std::size_t sent = 0;
        while (sent < text.size()) {
            const auto count = send(clientFd, text.data() + sent, text.size() - sent, MSG_NOSIGNAL);
            if (count <= 0) { return; }
            sent += static_cast<std::size_t>(count);
        }
    }

    ThreadBufferedMetricsCollector& registry_;
    StandaloneMetricsConfig config_;
    std::atomic<bool> running_{false};
    int listenFd_{-1};
    std::thread worker_;
};

class StandaloneKvMetricsBackend final : public KvMetricsBackend {
public:
    explicit StandaloneKvMetricsBackend(StandaloneMetricsConfig config) : config_(std::move(config))
    {
    }

    bool Initialize(const std::vector<MetricDescriptor>& descriptors)
    {
        if (config_.metricsPath.empty() || config_.metricsPath.front() != '/') {
            error_ = "metrics path must start with '/'";
            return false;
        }
        collector_ = std::make_unique<ThreadBufferedMetricsCollector>();
        for (const auto& descriptor : descriptors) {
            if (!collector_->Register(descriptor, error_)) {
                Stop();
                return false;
            }
        }
        collector_->StartAggregation(config_.aggregationIntervalMs);
        server_ = std::make_unique<MetricsHttpServer>(*collector_, config_);
        if (!server_->Start(error_)) {
            server_.reset();
            collector_->StopAggregation();
            collector_.reset();
            return false;
        }
        return true;
    }

    void UpdateStats(CachedMetric& metric, double value) noexcept override
    {
        const MetricUpdate update{metric, value};
        UpdateStats(&update, 1);
    }
    void UpdateStats(const MetricUpdate* updates, std::size_t count) noexcept override
    {
        if (collector_) { collector_->UpdateStats(updates, count); }
    }
    bool RegisterMetricLabels(const std::string& name, const MetricLabels& labels) override
    {
        return collector_ != nullptr && collector_->RegisterMetricLabels(name, labels);
    }
    void Flush() override
    {
        if (collector_) { collector_->Flush(); }
    }
    void Stop() override
    {
        if (server_) {
            server_->Stop();
            server_.reset();
        }
        if (collector_) {
            collector_->StopAggregation();
            collector_.reset();
        }
    }
    const std::string& LastError() const noexcept { return error_; }

private:
    StandaloneMetricsConfig config_;
    std::unique_ptr<ThreadBufferedMetricsCollector> collector_;
    std::unique_ptr<MetricsHttpServer> server_;
    std::string error_;
};

}  // namespace

bool SetUpStandaloneMetrics(StandaloneMetricsConfig config, std::string* error)
{
    std::vector<MetricDescriptor> descriptors;
    if (!config.definitionPath.empty()) {
        std::string loadError;
        if (!LoadYamlDefinitions(config.definitionPath, config.metricPrefix, descriptors,
                                 loadError)) {
            if (error != nullptr) { *error = std::move(loadError); }
            return false;
        }
    } else {
        descriptors = DefaultKvMetricDescriptors();
    }
    auto backend = std::make_shared<StandaloneKvMetricsBackend>(std::move(config));
    if (!backend->Initialize(descriptors)) {
        if (error != nullptr) { *error = backend->LastError(); }
        return false;
    }
    if (!InstallBackend(backend, error)) {
        backend->Stop();
        return false;
    }
    return true;
}

}  // namespace kv::metrics
